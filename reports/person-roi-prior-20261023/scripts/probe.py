"""Paired full-frame/person-ROI detector experiment, isolated from production.

Frozen samples: all 57 drink + 33 pickup sources, all 128 platform negatives,
and four known two-person waving controls. Six uniform frames per source.
Reuse the reviewed per-person pipeline's cached detector tracks, not action predictions.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort
import torch
from rtmlib import RTMO

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
DATA = ROOT / 'reports/person-finetune-20261023'
sys.path.insert(0, str(ROOT / 'reports/person-action-20261023/scripts'))
from person_core import OP, associate_poses, crop_person
from demo import content_height
from darkpipe.stages.downscale import DownscaleStage
from darkpipe.stages.enhance_retinexformer import RetinexformerStage
from darkpipe.stages.object_prior import (YoloOnnxDetector, CONTAINER_CLS, _torso,
    DRINK_RADIUS, PICKUP_RADIUS, PICKUP_GROUND_SLACK)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def freeze() -> dict:
    videos = json.loads((DATA / 'source-inventory.json').read_text())['videos']
    groups = {r['video_id']: r['group'] for r in json.loads((DATA / 'splits.json').read_text())['records']}
    waves = {f'real-actiondata-all/挥手{i}.mp4' for i in [3, 4, 30, 31]}
    cases = [dict(v, group=groups[v['id']]) for v in videos
             if v['video_label'] in {'drink', 'pick_up'} or v['processed'] or v['source'] in waves]
    protocol = dict(cases=cases, frame_sampling='six evenly spaced indices from 0 to N-1',
        thresholds=[.25, .1, .05], primary_threshold=.25, person_margin=.15,
        detector_input=640, person_source='frozen per-person tracking manifests at enhanced max_side720',
        variants=['full_enh720', 'person_enh720', 'person_native', 'person_raw720', 'full_native'],
        native_semantics='original raw camera pixels for local files; already processed content for platform clips',
        pose='same full enhanced frame pose shared by every variant, frame quality only; not temporal gate',
        primary_question='Does enhanced-person ROI improve actor-specific object evidence vs enhanced full frame?',
        secondary_question='Do original native pixels preserve additional small-object evidence?',
        metrics='container candidates and person-bound wrist/ankle geometry; NOT object AP or action accuracy',
        statistics='paired recording-family/session group means; bootstrap CI and exact sign test; descriptive only',
        annotation_sha256=sha(DATA / 'annotations.json'), splits_sha256=sha(DATA / 'splits.json'),
        production_changed=False)
    path = OUT / 'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == protocol, 'Refusing to change frozen protocol'
    else:
        write_json(path, protocol)
    return protocol


def detect(session: ort.InferenceSession, frame: np.ndarray) -> np.ndarray:
    canvas, scale, left, top = YoloOnnxDetector._letterbox(frame)
    x = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255
    rows = session.run(None, {session.get_inputs()[0].name: x})[0][0].copy()
    rows[:, [0, 2]] = (rows[:, [0, 2]] - left) / scale
    rows[:, [1, 3]] = (rows[:, [1, 3]] - top) / scale
    rows[:, :4] = np.clip(rows[:, :4], [0]*4, [frame.shape[1], frame.shape[0]]*2)
    return rows[rows[:, 4] > .01]


def map_rows(rows: np.ndarray, roi: np.ndarray, scale_xy: np.ndarray) -> np.ndarray:
    out = rows.copy()
    out[:, :4] += np.tile(roi[:2], 2)
    out[:, :4] /= np.tile(scale_xy, 2)
    return out


def evidence(rows: np.ndarray, pose: np.ndarray | None, bounds: np.ndarray,
             size: np.ndarray) -> dict:
    centers = (rows[:, :2] + rows[:, 2:4]) / 2
    inside = ((centers >= bounds[:2]) & (centers <= bounds[2:])).all(1)
    containers = np.isin(rows[:, 5].astype(int), CONTAINER_CLS)
    result = dict(container_score=float(rows[inside & containers, 4].max(initial=0)),
                  drink_score=0., pickup_score=0., drink_rows=[], pickup_rows=[], pose_usable=False)
    if pose is None:
        return result
    kp = pose.copy(); kp[:, :2] /= size
    torso = _torso(kp)
    if torso is None or torso <= 1e-8 or pose[:, 2].mean() < .5 or (pose[:, 2] > .3).sum() < 10:
        return result
    result['pose_usable'] = True
    for i, row in enumerate(rows):
        if row[5] == 0:
            continue
        center = (row[:2] + row[2:4]) / 2 / size
        bottom = np.array([(row[0]+row[2])/2, row[3]]) / size
        if containers[i] and any(kp[w, 2] > .3 and np.linalg.norm(center-kp[w, :2]) < DRINK_RADIUS*torso for w in [9, 10]):
            result['drink_rows'].append(i)
            result['drink_score'] = max(result['drink_score'], float(row[4]))
        if any(kp[a, 2] > .3 and np.linalg.norm(bottom-kp[a, :2]) < PICKUP_RADIUS*torso
               and bottom[1] >= kp[a, 1]-PICKUP_GROUND_SLACK*torso for a in [15, 16]):
            result['pickup_rows'].append(i)
            result['pickup_score'] = max(result['pickup_score'], float(row[4]))
    return result


def truth_at(item: dict, track_id: int, timestamp: float, annotations: dict) -> str:
    row = annotations.get((item['id'], track_id))
    if row is not None:
        if row['status'] != 'reviewed':
            return 'unknown'
        values = {s['label'] for s in row['segments'] if s['start'] <= timestamp <= s['end']}
        return next(iter(values)) if len(values) == 1 else 'unknown'
    return 'other' if item['video_label'] == 'other' else 'unknown'


def run_case(item: dict, session: ort.InferenceSession, pose_model: RTMO,
             enhancer: RetinexformerStage, annotations: dict) -> dict:
    dest = OUT / 'data' / item['id']; dest.mkdir(parents=True, exist_ok=True)
    target = dest / 'results.json'
    if target.exists():
        return json.loads(target.read_text())
    source = ROOT / item['source']; assert sha(source) == item['sha256']
    cached = json.loads((DATA / 'data' / item['id'] / 'manifest.json').read_text())
    chosen = set(np.linspace(0, item['frames']-1, min(6, item['frames'])).round().astype(int))
    cap = cv2.VideoCapture(str(source)); assert cap.isOpened(), source
    downscale = DownscaleStage(720)
    frames, records, calls = [], [], Counter()
    elapsed = Counter(); count = 0; parity = []
    try:
        while True:
            ok, native = cap.read()
            if not ok:
                break
            fi = count; count += 1
            if fi not in chosen:
                continue
            meta = cached['frames'][fi]
            if item['processed']:
                native = native[:content_height(native.shape[0])]
            small = downscale([native])[0]
            enhanced = small if item['processed'] else enhancer([small])[0]
            H, W = enhanced.shape[:2]; size = np.array([W, H])
            sx = np.array([native.shape[1]/W, native.shape[0]/H])
            fulls = {}
            for variant, image in [('full_enh720', enhanced), ('full_native', native)]:
                start = time.perf_counter(); det = detect(session, image)
                elapsed[variant] += time.perf_counter()-start; calls[variant] += 1
                fulls[variant] = det if variant == 'full_enh720' else map_rows(det, np.zeros(4), sx)
            if not frames:
                ref = YoloOnnxDetector(str(OP/'ckpts'), 'cuda:0'); ref.sess = session
                actual = ref._frames_to_dets([enhanced])[0]
                assert np.allclose(actual, fulls['full_enh720'][fulls['full_enh720'][:, 4] > .25], atol=1e-5)
                parity.append(fi)
            tracks = meta['tracks']; boxes = np.array([t['bbox'] for t in tracks], dtype=np.float32).reshape(-1, 4)
            xy, scores = pose_model(enhanced) if len(tracks) else ([], [])
            poses = associate_poses(boxes, xy, scores)
            cv2.imwrite(str(dest/f'{fi:06d}_native.jpg'), native, [cv2.IMWRITE_JPEG_QUALITY, 97])
            cv2.imwrite(str(dest/f'{fi:06d}_enhanced.png'), enhanced)
            frames.append(dict(frame=fi, luma=meta['luma'], tracks=len(tracks),
                               gate_active=meta['luma'] is None or meta['luma'] >= 75,
                               shape=[H, W], native_shape=list(native.shape[:2])))
            for track, box, pose in zip(tracks, boxes, poses):
                _, bounds = crop_person(enhanced, box)
                native_bounds = np.concatenate([np.floor(bounds[:2]*sx), np.ceil(bounds[2:]*sx)]).astype(int)
                native_bounds = np.clip(native_bounds, [0]*4, [native.shape[1], native.shape[0]]*2)
                variants = dict(fulls)
                for name, image, roi, scale in [
                    ('person_enh720', enhanced, bounds, np.ones(2)),
                    ('person_native', native, native_bounds, sx),
                    ('person_raw720', small, bounds, np.ones(2))]:
                    crop = image[roi[1]:roi[3], roi[0]:roi[2]]
                    if item['processed'] and name == 'person_raw720':
                        variants[name] = variants['person_enh720'].copy()
                        continue
                    start = time.perf_counter(); det = detect(session, crop)
                    elapsed[name] += time.perf_counter()-start; calls[name] += 1
                    variants[name] = map_rows(det, roi, scale)
                rec = dict(video_id=item['id'], source=item['source'], group=item['group'],
                    video_label=item['video_label'], processed=item['processed'], frame=fi,
                    timestamp=fi/item['fps'], track_id=track['track_id'], person_box=box.tolist(),
                    roi=bounds.tolist(), native_roi=native_bounds.tolist(),
                    truth=truth_at(item, track['track_id'], fi/item['fps'], annotations),
                    pose=None if pose is None else pose.tolist(), luma=meta['luma'],
                    gate_active=meta['luma'] is None or meta['luma'] >= 75, variants={})
                for name, rows in variants.items():
                    rec['variants'][name] = dict(detections=rows.tolist(), **evidence(rows, pose, bounds, size))
                records.append(rec)
    finally:
        cap.release()
    assert count == item['frames'] and len(frames) == len(chosen)
    result = dict(item=item, frames=frames, records=records, detector_calls=dict(calls),
                  detector_seconds=dict(elapsed), production_parity_frames=parity,
                  cache_manifest_sha256=sha(DATA/'data'/item['id']/'manifest.json'))
    write_json(target, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = freeze()
    print('[protocol]', len(protocol['cases']), dict(Counter(v['video_label'] for v in protocol['cases'])), flush=True)
    if args.freeze_only:
        return
    torch.set_num_threads(3); cv2.setNumThreads(1)
    options = ort.SessionOptions(); options.intra_op_num_threads = 3; options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(OP/'ckpts/yolo26m_detect.onnx'), sess_options=options,
                                  providers=[('CUDAExecutionProvider', {'device_id': 0})])
    assert 'CUDAExecutionProvider' in session.get_providers()
    pose = RTMO(onnx_model=str(OP/'ckpts/rtmo-s_end2end.onnx'), model_input_size=(640,640),
                backend='onnxruntime', device='cuda:0', score_thr=.3)
    assert 'CUDAExecutionProvider' in pose.session.get_providers()
    enhancer = RetinexformerStage(str(OP/'ckpts'), chunk=1); enhancer.load('cuda:0')
    for _ in range(3):detect(session, np.zeros((404,720,3),np.uint8))
    annotations = {(a['video_id'], a['track_id']): a for a in json.loads((DATA/'annotations.json').read_text())['records']}
    metadata = dict(providers=session.get_providers(), input=session.get_inputs()[0].shape,
                    output=session.get_outputs()[0].shape,
                    class_names=ast.literal_eval(session.get_modelmeta().custom_metadata_map['names']),
                    model_sha256={n:sha(OP/'ckpts'/n) for n in ['yolo26m_detect.onnx','rtmo-s_end2end.onnx','NTIRE.pth']},
                    protocol_sha256=sha(OUT/'protocol.json'), probe_sha256=sha(Path(__file__)))
    write_json(OUT/'model-provenance.json', metadata)
    results = []; start = time.monotonic()
    try:
        for i,item in enumerate(protocol['cases'],1):
            result = run_case(item, session, pose, enhancer, annotations); results.append(result)
            print(f'[probe] {i}/{len(protocol["cases"])} {item["source"]} people_frames={len(result["records"])} elapsed={time.monotonic()-start:.1f}s', flush=True)
    finally:
        enhancer.close()
    write_json(OUT/'completion.json',dict(videos=len(results),frames=sum(len(r['frames'])for r in results),
        person_frames=sum(len(r['records'])for r in results),seconds=time.monotonic()-start,completed=True))


if __name__ == '__main__':
    main()
