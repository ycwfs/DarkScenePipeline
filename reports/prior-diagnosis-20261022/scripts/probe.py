"""Focused detector diagnostic; no production settings or weights are changed.

Run from repository: .venv/bin/python reports/prior-diagnosis-20261022/scripts/probe.py --gpu 1
Counts are detections on sampled frames, NOT accuracy against box/keypoint ground truth.
"""
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont
from rtmlib import RTMO

OUT = Path(__file__).resolve().parents[1]
REPO = OUT.parents[1]
PKG = REPO / 'platform/dark-demo-v2'
sys.path.insert(0, str(PKG))
from darkpipe.stages.downscale import DownscaleStage
from darkpipe.stages.enhance_retinexformer import RetinexformerStage
from darkpipe.stages.object_prior import YoloOnnxDetector, CONTAINER_CLS

CASES = ['喝水16', '捡东西2', '喝水8', '捡东西1', '挥手18']
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
EDGES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11),
         (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


def sampled_frames(path: Path, count: int = 12) -> list:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ids = set(np.linspace(0, n - 1, min(count, n)).round().astype(int))
    frames = []
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f'Decode failed {path}:{i}')
        if i in ids:
            frames.append((i, frame))
    cap.release()
    return frames


def detect(session, frame: np.ndarray) -> np.ndarray:
    canvas, scale, left, top = YoloOnnxDetector._letterbox(frame)
    x = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255
    rows = session.run(None, {session.get_inputs()[0].name: x})[0][0].copy()
    rows[:, [0, 2]] = (rows[:, [0, 2]] - left) / scale
    rows[:, [1, 3]] = (rows[:, [1, 3]] - top) / scale
    rows[:, :4] = np.clip(rows[:, :4], [0] * 4, [frame.shape[1], frame.shape[0]] * 2)
    return rows


def record(name: str, fi: int, variant: str, frame: np.ndarray, rows: np.ndarray,
           pose_data=None, roi=None) -> dict:
    scores = pose_data[1] if pose_data is not None else np.empty((0, 17))
    containers = np.isin(rows[:, 5].astype(int), CONTAINER_CLS)
    persons = rows[:, 5] == 0
    counts = {}
    for threshold in [.25, .1, .05]:
        valid = rows[:, 4] > threshold
        counts[str(threshold)] = dict(person=int((valid & persons).sum()),
            container=int((valid & containers).sum()), non_person=int((valid & ~persons).sum()))
    return dict(video=name, frame=fi, variant=variant, shape=list(frame.shape), roi=roi,
                luma=float((frame.astype(np.float32) @ np.array([.114, .587, .299])).mean()),
                counts=counts, max_container_conf=float(rows[containers, 4].max(initial=0)),
                detections=rows[rows[:, 4] > .01].tolist(),
                pose_mean_scores=scores.mean(1).tolist(),
                pose_scores=scores.tolist(),
                pose_visible_joints=(scores > .3).sum(1).tolist(),
                pose_kpts=pose_data[0].tolist() if pose_data is not None else [])


def panel(frame: np.ndarray, rec: dict) -> Image.Image:
    img = frame.copy()
    for x1, y1, x2, y2, score, cls in rec['detections']:
        if score <= .25:
            continue
        color = (30, 220, 40) if int(cls) == 0 else (0, 180, 255)
        cv2.rectangle(img, (round(x1), round(y1)), (round(x2), round(y2)), color, 2)
        cv2.putText(img, f'{int(cls)} {score:.2f}', (round(x1), max(14, round(y1))),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
    for kp, mean, scores in zip(rec['pose_kpts'], rec['pose_mean_scores'], rec['pose_scores']):
        if mean < .5:
            continue
        for a, b in EDGES:
            if scores[a] <= .3 or scores[b] <= .3:
                continue
            cv2.line(img, tuple(np.rint(kp[a]).astype(int)), tuple(np.rint(kp[b]).astype(int)),
                     (240, 150, 40), 1, cv2.LINE_AA)
    img = cv2.resize(img, (640, round(640 * img.shape[0] / img.shape[1])))
    cell = Image.new('RGB', (640, 440), '#151a24')
    cell.paste(Image.fromarray(img[:, :, ::-1]), (0, 55))
    d = ImageDraw.Draw(cell)
    d.text((8, 4), f"{rec['video']} frame {rec['frame']} / {rec['variant']}",
           font=ImageFont.truetype(FONT, 19), fill='white')
    c = rec['counts']['0.25']
    d.text((8, 29), f"亮度 {rec['luma']:.1f}  人 {c['person']}  容器 {c['container']}  非人 {c['non_person']}",
           font=ImageFont.truetype(FONT, 17), fill='#d1d7e2')
    return cell


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    device = f'cuda:{args.gpu}'
    torch.cuda.set_device(args.gpu)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    ckpts = PKG / 'ckpts'
    session = ort.InferenceSession(str(ckpts / 'yolo26m_detect.onnx'), sess_options=options,
        providers=[('CUDAExecutionProvider', {'device_id': args.gpu})])
    assert 'CUDAExecutionProvider' in session.get_providers()
    pose = RTMO(onnx_model=str(ckpts / 'rtmo-s_end2end.onnx'), model_input_size=(640, 640),
                backend='onnxruntime', device=device, score_thr=.3)
    assert 'CUDAExecutionProvider' in pose.session.get_providers()
    enhancer = RetinexformerStage(str(ckpts), chunk=1)
    enhancer.load(device)
    ds = DownscaleStage(720)
    (OUT / 'visual').mkdir(parents=True, exist_ok=True)
    rows = []
    metadata = dict(gpu=args.gpu, detector_providers=session.get_providers(),
        detector_input=session.get_inputs()[0].shape,
        detector_output=session.get_outputs()[0].shape,
        onnx_metadata=session.get_modelmeta().custom_metadata_map,
        scope='5 selected known problem takes x 12 uniformly sampled frames; no box/keypoint GT; '
              'raw/full resolution and crops are diagnostic variants; production detector is fixed at 640.',
        model_sha256={name: hashlib.sha256((ckpts / name).read_bytes()).hexdigest()
                      for name in ['yolo26m_detect.onnx', 'rtmo-s_end2end.onnx', 'NTIRE.pth']},
        sources={})
    for name in CASES:
        source = REPO / 'real-actiondata-all' / f'{name}.mp4'
        metadata['sources'][name] = dict(path=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        sampled = sampled_frames(source)
        visual_rows = []
        for si, (fi, raw) in enumerate(sampled):
            small = ds([raw])[0]
            enhanced = enhancer([small])[0]
            data = {}
            for variant, frame in [('raw_native', raw), ('raw720', small), ('enhanced720', enhanced)]:
                boxes = detect(session, frame)
                pdata = pose(frame) if variant != 'raw_native' else None
                rec = record(name, fi, variant, frame, boxes, pdata)
                rows.append(rec)
                data[variant] = (frame, rec, boxes)
            # Use a detected person to define an ORIGINAL-resolution ROI, never a synthetic enlargement.
            if name in CASES[:2]:
                b = data['raw_native'][2]
                candidates = b[(b[:, 5] == 0) & (b[:, 4] > .25)]
                if len(candidates):
                    x1, y1, x2, y2 = candidates[np.argmax(candidates[:, 4]), :4]
                    w, h = x2 - x1, y2 - y1
                    for variant, bounds in [('person_crop', (x1-.35*w, y1-.12*h, x2+.35*w, y2+.12*h)),
                        ('interaction_crop', (x1-.15*w, y1-.12*h, x2+.15*w, y1+.58*h) if name == '喝水16'
                         else (x1-.4*w, y1+.35*h, x2+.4*w, y2+.35*h))]:
                        left, top, right, bottom = [int(v) for v in np.clip(bounds, [0]*4, [raw.shape[1], raw.shape[0]]*2)]
                        crop = raw[top:bottom, left:right]
                        rec = record(name, fi, variant, crop, detect(session, crop), roi=[left, top, right, bottom])
                        rows.append(rec)
                        if si in [2, 5, 8]:
                            cv2.imwrite(str(OUT / 'visual' / f'{name}_{fi}_{variant}.jpg'), crop)
            if si in [2, 5, 8]:
                visual_rows.append([panel(data[v][0], data[v][1]) for v in ['raw720', 'enhanced720']])
                for v in ['raw720', 'enhanced720']:
                    cv2.imwrite(str(OUT / 'visual' / f'{name}_{fi}_{v}.jpg'), data[v][0])
        sheet = Image.new('RGB', (1280, 440 * len(visual_rows)))
        for i, cells in enumerate(visual_rows):
            for j, cell in enumerate(cells):
                sheet.paste(cell, (j*640, i*440))
        sheet.save(OUT / 'visual' / f'{name}_comparison.jpg')
        print(f'[done] {name}: {len(sampled)} frames', flush=True)
        (OUT / 'results.json').write_text(json.dumps(dict(metadata=metadata, rows=rows), ensure_ascii=False, indent=2))
    groups = defaultdict(list)
    for r in rows:
        groups[(r['video'], r['variant'])].append(r)
    summary = []
    for (name, variant), group in groups.items():
        r = dict(video=name, variant=variant, frames=len(group),
                 mean_luma=float(np.mean([g['luma'] for g in group])),
                 max_container_conf=max(g['max_container_conf'] for g in group),
                 pose_frames=sum(any(s >= .5 for s in g['pose_mean_scores']) for g in group))
        for threshold in ['0.25', '0.1', '0.05']:
            r[threshold] = {kind: sum(g['counts'][threshold][kind] > 0 for g in group)
                            for kind in ['person', 'container', 'non_person']}
        summary.append(r)
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
