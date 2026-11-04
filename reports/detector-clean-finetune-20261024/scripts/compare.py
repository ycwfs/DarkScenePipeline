"""Frozen detector regression; reuse fixed baseline/pose outputs, never mask exhibits."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
import torch

from prepare_bright import ROOT, OUT, OLD, save, session, YoloOnnxDetector, downscale, iou
from demos import draw_panel, _cjk_font
from person_core import person_boxes, associate_poses
from darkpipe.stages.enhance_retinexformer import RetinexformerStage

BRIGHT = ROOT / 'reports/bright-regression-20261024'
NAMES = ['original', 'previous', 'cleaned']
TITLES = ['原始检测器', '上轮：仅暗场微调', '本轮：清理标注＋亮暗混合']
DEMO_NAMES = {'other95.mp4', '挥手28.mp4', '挥手29.mp4', '追逐25.mp4', '喝水26.mp4',
              '捡东西23.mp4', 'other225.mp4', 'other227.mp4', '捡东西18.mp4', '捡东西20.mp4', '挥手34.mp4'}


def match(truth: list, predictions: np.ndarray, threshold: float = .5) -> dict:
    if not len(truth) or not len(predictions):
        return {}
    overlaps = np.array([[iou(a, b[:4]) for b in predictions] for a in truth])
    ii, jj = linear_sum_assignment(-((overlaps >= threshold) * 10 + overlaps))
    return {int(i): int(j) for i, j in zip(ii, jj) if overlaps[i, j] >= threshold}


def ignored(box: np.ndarray, regions: list) -> bool:
    return any(np.maximum(0, np.minimum(box[2:4], r[2:]) - np.maximum(box[:2], r[:2])).prod()
               / max(np.prod(box[2:4] - box[:2]), 1) >= .5 for r in regions)


def associated(row: dict, name: str) -> tuple:
    boxes = np.asarray(row[name]['boxes'], np.float32).reshape(-1, 6)
    pose = row['fixed_pose']
    assigned = associate_poses(boxes, np.asarray(pose['pose_xy'], np.float32).reshape(-1, 17, 2),
                               np.asarray(pose['pose_scores'], np.float32).reshape(-1, 17))
    return boxes, assigned


def mosaic(image: np.ndarray, row: dict, source: str, domain: str) -> np.ndarray:
    panels = []
    for name, title in zip(NAMES, TITLES):
        boxes, poses = associated(row, name)
        tracks = [SimpleNamespace(tid=i + 1, box=b[:4]) for i, b in enumerate(boxes)]
        panels.append(draw_panel(image, tracks, {i + 1: p for i, p in enumerate(poses)}, title))
    canvas = np.vstack([np.hstack(panels), np.full((48, 2160, 3), (20, 25, 32), np.uint8)])
    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    label = f'{Path(source).name} | F{row["frame"]:04d} L={row["luma"]:.1f} | 同一姿态模型；无平滑；P仅本帧编号'
    if domain == 'dark':
        label += ' | 暗场骨架仅作诊断展示，未更改照度门75'
    ImageDraw.Draw(pil).text((12, 455), label, font=_cjk_font(22), fill='white')
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def detector(path: Path) -> YoloOnnxDetector:
    model = YoloOnnxDetector(str(path.parent), 'cuda:0')
    model.sess = session(path)
    return model


def main() -> None:
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    candidate = OUT / 'models/yolo26m_clean_candidate.onnx'
    selected = json.loads((OUT / 'selection.json').read_text())['selected']
    candidate_sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
    bright_protocol = json.loads((BRIGHT / 'protocol.json').read_text())
    dark_summary = json.loads((OLD / 'demo-summary.json').read_text())['videos']
    videos = [dict(id=v['id'], source=v['source'], domain='bright', frames=v['frames'],
                   audit_frame=v['audit_frame'], indices=v['bright_indices']) for v in bright_protocol['videos']]
    videos += [dict(id=v['id'], source=v['source'], domain='dark', frames=v['frames'],
                    audit_frame=v['frames'] // 2, indices=list(range(v['frames'])))
               for v in dark_summary if v['split'] == 'test' and v['raw_dark_frames']]
    assert len(videos) == 49
    assert sum(len(v['indices']) for v in videos) == 3885
    save(OUT / 'test-protocol.json', dict(videos=videos, selected_checkpoint=selected['path'],
        selected_before_test=True, candidate_sha256=candidate_sha,
        cached_baselines='Unmodified bright-regression and dark-finetune outputs; same per-frame enhancement',
        smoothing=False, runtime_region_mask=False, fixed_pose=str(OLD / 'models/rtmo-s_dark_candidate.onnx')))
    model = detector(candidate)
    enhancer = RetinexformerStage(str(ROOT / 'platform/dark-demo-v2/ckpts'), chunk=1)
    enhancer.load('cuda:0')
    summaries = []
    pixel_checks = []
    for vi, v in enumerate(videos):
        directory = OUT / 'data/test' / v['id']
        directory.mkdir(parents=True, exist_ok=True)
        record_path = directory / 'outputs.json'
        if record_path.exists():
            saved = json.loads(record_path.read_text())
            assert saved['candidate_sha256'] == candidate_sha
            summaries.append(saved['summary'])
            pixel_checks.extend(saved['pixel_checks'])
            continue
        ref_path = (BRIGHT / 'data' / v['id'] / 'outputs.json' if v['domain'] == 'bright'
                    else OLD / 'data' / v['id'] / 'demo.json')
        ref = json.loads(ref_path.read_text())
        cached = {r['frame']: r for r in ref['frames']}
        assert set(v['indices']) == set(cached)
        want_video = Path(v['source']).name in DEMO_NAMES
        cap = cv2.VideoCapture(str(ROOT / v['source']))
        fps = cap.get(cv2.CAP_PROP_FPS)
        encoder = None
        dest = OUT / 'videos' / f'{v["id"]}.mp4'
        if want_video:
            dest.parent.mkdir(parents=True, exist_ok=True)
            encoder = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                '-s', '2160x494', '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'fast',
                '-crf', '20', '-pix_fmt', 'yuv420p', '-threads', '2', '-movflags', '+faststart', str(dest)], stdin=subprocess.PIPE)
        rows = []
        checks = []
        index = 0
        try:
            while True:
                ok, native = cap.read()
                if not ok:
                    break
                if index in cached:
                    old = cached[index]
                    raw = downscale(native)
                    enhanced = enhancer([raw])[0]
                    boxes = person_boxes(model._frames_to_dets([enhanced])[0], confidence=.35)
                    row = dict(frame=index, timestamp=index / fps, luma=old['luma'],
                        original=dict(boxes=old['original']['boxes']), previous=dict(boxes=old['candidate']['boxes']),
                        cleaned=dict(boxes=boxes.tolist()), fixed_pose={key: old['candidate'][key]
                            for key in ('pose_boxes', 'pose_xy', 'pose_scores', 'pose_conf')})
                    if index == v['audit_frame']:
                        cv2.imwrite(str(directory / 'enhanced.png'), enhanced)
                        cv2.imwrite(str(directory / 'raw.png'), raw)
                        poster = OUT / 'posters' / f'{v["id"]}.jpg'
                        poster.parent.mkdir(exist_ok=True)
                        cv2.imwrite(str(poster), mosaic(enhanced, row, v['source'], v['domain']))
                        old_image = BRIGHT / 'data' / v['id'] / 'enhanced.png'
                        if v['domain'] == 'bright':
                            diff = np.abs(enhanced.astype(float) - cv2.imread(str(old_image)).astype(float))
                            checks.append(dict(video_id=v['id'], frame=index, max_pixel_delta=float(diff.max()),
                                               mean_pixel_delta=float(diff.mean())))
                            assert diff.max() <= 1, (v['source'], diff.max())
                    if encoder:
                        encoder.stdin.write(mosaic(enhanced, row, v['source'], v['domain']).tobytes())
                    rows.append(row)
                index += 1
        finally:
            cap.release()
            if encoder:
                encoder.stdin.close()
                if encoder.wait(timeout=60):
                    raise RuntimeError(f'Encoder failed {dest}')
        assert index == v['frames'] and len(rows) == len(v['indices']), (v, index, len(rows))
        if want_video:
            verify = subprocess.run(['ffmpeg', '-v', 'error', '-threads', '2', '-i', str(dest), '-f', 'null', '-'],
                                    capture_output=True, check=True)
            assert not verify.stderr, verify.stderr
            decoded = cv2.VideoCapture(str(dest))
            n = 0
            while decoded.read()[0]:
                n += 1
            decoded.release()
            assert n == len(rows)
        summary = dict(**v, fps=fps, processed_frames=len(rows), video=str(dest.relative_to(OUT)) if want_video else None,
                       full_decode=want_video)
        save(record_path, dict(candidate_sha256=candidate_sha, summary=summary, frames=rows, pixel_checks=checks))
        summaries.append(summary)
        pixel_checks.extend(checks)
        save(OUT / 'replay-summary.json', dict(videos=summaries, pixel_checks=pixel_checks))
        print(f'[replay {vi+1}/{len(videos)}] {v["source"]}: {len(rows)}', flush=True)
    enhancer.close()
    save(OUT / 'replay-summary.json', dict(videos=summaries, pixel_checks=pixel_checks))
    analyze()
    benchmark()


def score(row: dict, truth: list, ignore_regions: list, threshold: float) -> dict:
    result = {}
    for name in NAMES:
        boxes, poses = associated(row, name)
        matches = match(truth, boxes, threshold)
        extras = [j for j, b in enumerate(boxes) if j not in matches.values() and not ignored(b, ignore_regions)]
        result[name] = dict(gt=len(truth), matched=len(matches), predictions=len(boxes), unmatched=len(extras),
            associated_pose10=sum(poses[j] is not None and (poses[j][:, 2] >= .3).sum() >= 10 for j in matches.values()),
            missing=[j for j in range(len(truth)) if j not in matches],
            matched_truth=sorted(matches), unmatched_indices=extras)
        result[name]['associated_pose10'] = int(result[name]['associated_pose10'])
    return result


def analyze() -> None:
    videos = json.loads((OUT / 'test-protocol.json').read_text())['videos']
    labels = {l['video_id']: l for l in json.loads((BRIGHT / 'visual-audit-labels.json').read_text())['labels']}
    dark = json.loads((OLD / 'manifest.json').read_text())['samples']
    dark_labels = {(s['video_id'], s['frame']): s for s in dark if s['split'] == 'test'}
    audits = []
    exhibits = []
    outputs = []
    for v in videos:
        frames = json.loads((OUT / 'data/test' / v['id'] / 'outputs.json').read_text())['frames']
        stats = {name: dict(predictions=0, frames_with_person=0, associated_pose10=0) for name in NAMES}
        fake = {name: dict(detector_frames=[], associated_pose10_frames=[]) for name in NAMES}
        for row in frames:
            for name in NAMES:
                boxes, poses = associated(row, name)
                stats[name]['predictions'] += len(boxes)
                stats[name]['frames_with_person'] += int(len(boxes) > 0)
                stats[name]['associated_pose10'] += int(sum(p is not None and (p[:, 2] >= .3).sum() >= 10 for p in poses))
                if v['domain'] == 'bright' and labels[v['id']]['known_nonperson_boxes']:
                    matched = match(labels[v['id']]['known_nonperson_boxes'], boxes)
                    if matched:
                        fake[name]['detector_frames'].append(row['frame'])
                    if any(poses[j] is not None and (poses[j][:, 2] >= .3).sum() >= 10 for j in matched.values()):
                        fake[name]['associated_pose10_frames'].append(row['frame'])
            truth = None
            ignore_regions = []
            if v['domain'] == 'bright' and row['frame'] == labels[v['id']]['frame']:
                truth = labels[v['id']]['person_boxes']
                ignore_regions = labels[v['id']]['ignore_regions']
            elif (v['id'], row['frame']) in dark_labels:
                s = dark_labels[v['id'], row['frame']]
                truth = [np.array(p['bbox_xyxy']) * np.array([720 / s['width'], 404 / s['height']] * 2)
                         for p in s['people']]
            if truth is not None:
                audits.append(dict(video_id=v['id'], source=v['source'], domain=v['domain'], frame=row['frame'],
                    by_iou={str(t): score(row, truth, ignore_regions, t) for t in (.3, .5, .7)}))
        outputs.append(dict(video_id=v['id'], source=v['source'], domain=v['domain'], frames=len(frames), counts=stats))
        if v['domain'] == 'bright' and labels[v['id']]['known_nonperson_boxes']:
            exhibits.append(dict(video_id=v['id'], source=v['source'], frames=len(frames),
                frame_indices=fake, counts={name: {k: len(vals) for k, vals in value.items()} for name, value in fake.items()}))
    summary = {}
    for domain in ('dark', 'bright'):
        selected = [r for r in audits if r['domain'] == domain]
        assert len(selected) == (44 if domain == 'dark' else 41)
        summary[domain] = {str(t): {name: {key: sum(r['by_iou'][str(t)][name][key] for r in selected)
            for key in ('gt', 'matched', 'predictions', 'unmatched', 'associated_pose10')} for name in NAMES}
            for t in (.3, .5, .7)}
    save(OUT / 'evaluation.json', dict(summary=summary, audits=audits,
        caveat='Auxiliary labels, not independent human GT; dark annotations not exhaustive; pose10 measures coverage only.'))
    save(OUT / 'exhibit-audit.json', dict(videos=exhibits, frames=sum(v['frames'] for v in exhibits),
        totals={name: {k: sum(v['counts'][name][k] for v in exhibits) for k in ('detector_frames', 'associated_pose10_frames')}
                for name in NAMES}, caveat='Fixed known exhibit region for scoring only, not an inference filter; correlated frames.'))
    save(OUT / 'full-replay-statistics.json', dict(videos=outputs,
        caveat='Full video output counts are diagnostic, not detection precision/recall.'))
    print('[evaluation]', json.dumps(summary, ensure_ascii=False), flush=True)


def benchmark() -> None:
    videos = json.loads((BRIGHT / 'protocol.json').read_text())['videos']
    images = [cv2.imread(str(BRIGHT / 'data' / v['id'] / 'enhanced.png')) for v in videos]
    models = {name: detector(path) for name, path in zip(NAMES, [ROOT / 'platform/dark-demo-v2/ckpts/yolo26m_detect.onnx',
        OLD / 'models/yolo26m_dark_candidate.onnx', OUT / 'models/yolo26m_clean_candidate.onnx'])}
    for model in models.values():
        for im in images[:5]:
            model._frames_to_dets([im])
    rows = []
    for repeat in range(5):
        for i, im in enumerate(images):
            order = NAMES[(i + repeat) % 3:] + NAMES[:(i + repeat) % 3]
            for name in order:
                start = time.perf_counter()
                models[name]._frames_to_dets([im])
                rows.append(dict(model=name, repeat=repeat, video_id=videos[i]['id'], ms=(time.perf_counter() - start) * 1000))
    summary = {}
    for name in NAMES:
        values = [r['ms'] for r in rows if r['model'] == name]
        summary[name] = dict(n=len(values), mean=float(np.mean(values)), sd=float(np.std(values, ddof=1)),
                             median=float(np.median(values)), p95=float(np.percentile(values, 95)))
    save(OUT / 'benchmark.json', dict(rows=rows, summary=summary, gpu='RTX3090 physical GPU3',
        warmup=5, scope='Detector preprocess, ONNX and postprocess only; not end-to-end fps', repeats=5))
    print('[benchmark]', summary, flush=True)


if __name__ == '__main__':
    main()
