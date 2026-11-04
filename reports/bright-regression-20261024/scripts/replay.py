"""Frozen bright test replay: two detector/pose checkpoints, four raw combinations."""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT / 'reports/dark-finetune-20261024'
sys.path.insert(0, str(PREVIOUS / 'scripts'))
from inference import Frontend
from demos import downscale, draw_panel, _cjk_font
from prepare import save
from person_core import associate_poses, person_boxes
from darkpipe.stages.enhance_retinexformer import RetinexformerStage

COMBINATIONS = {'original': ('original', 'original'),
                'detector_only': ('candidate', 'original'),
                'pose_only': ('original', 'candidate'),
                'both': ('candidate', 'candidate')}
TITLES = {'original': '原检测 ＋ 原姿态', 'detector_only': '微调检测 ＋ 原姿态',
          'pose_only': '原检测 ＋ 微调姿态', 'both': '微调检测 ＋ 微调姿态'}
DEMO_NAMES = {'挥手28.mp4', '喝水26.mp4', '捡东西23.mp4',
              'other100.mp4', '追逐35.mp4', '挥手45.mp4'}


def combinations(row: dict) -> dict:
    result = {}
    for name, (det, pose) in COMBINATIONS.items():
        boxes = np.asarray(row[det]['boxes'], np.float32).reshape(-1, 6)
        points = np.asarray(row[pose]['pose_xy'], np.float32).reshape(-1, 17, 2)
        scores = np.asarray(row[pose]['pose_scores'], np.float32).reshape(-1, 17)
        result[name] = (boxes, associate_poses(boxes, points, scores))
    return result


def mosaic(image: np.ndarray, row: dict, source: str) -> np.ndarray:
    panels = []
    for name, (boxes, poses) in combinations(row).items():
        tracks = [SimpleNamespace(tid=i + 1, box=b[:4]) for i, b in enumerate(boxes)]
        panels.append(draw_panel(image, tracks, {i + 1: p for i, p in enumerate(poses)}, TITLES[name]))
    canvas = np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:]),
                       np.full((48, 1440, 3), (20, 25, 32), np.uint8)])
    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pil).text((12, 900),
        f'{Path(source).name} | F{row["frame"]:04d} L={row["luma"]:.1f} | 无平滑、无持续时间筛选；P编号仅为本帧序号',
        font=_cjk_font(19), fill=(235, 241, 248))
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def main() -> None:
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    annotations = ROOT / 'reports/dark-annotations-20261023'
    inventory = json.loads((annotations / 'inventory.json').read_text())['raw_videos']
    selected = []
    for video in inventory:
        if video['frozen_split'].get('split') != 'test':
            continue
        scan = json.loads((annotations / 'data' / video['id'] / 'scan.json').read_text())
        indices = [f['frame'] for f in scan['frames'] if f['luma'] >= 75]
        if indices:
            selected.append(dict(**video, bright_indices=indices, audit_frame=indices[len(indices)//2]))
    op = ROOT / 'platform/dark-demo-v2/ckpts'
    paths = {'original': (op / 'yolo26m_detect.onnx', op / 'rtmo-s_end2end.onnx'),
             'candidate': (PREVIOUS / 'models/yolo26m_dark_candidate.onnx',
                           PREVIOUS / 'models/rtmo-s_dark_candidate.onnx')}
    protocol = dict(videos=selected, frames=sum(len(v['bright_indices']) for v in selected),
        thresholds=dict(luma=75, detector=.35, pose=.45, keypoint=.3),
        inference_smoothing=False, duration_confirmation=False, preprocessing='raw720 -> Retinexformer',
        checkpoints={k: [dict(path=str(p.relative_to(ROOT)), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                         for p in ps] for k, ps in paths.items()},
        audit='one deterministic middle bright frame from every frozen test video; assistant visual review, not independent human GT',
        model_selection='fixed previous validation-selected checkpoints; no tuning on these test frames')
    save(OUT / 'protocol.json', protocol)
    models = {name: Frontend(*ps) for name, ps in paths.items()}
    enhancer = RetinexformerStage(str(op), chunk=1)
    enhancer.load('cuda:0')
    summaries = []
    for vi, video in enumerate(selected):
        uid = video['id']
        directory = OUT / 'data' / uid
        directory.mkdir(parents=True, exist_ok=True)
        old = PREVIOUS / 'data' / uid / 'demo.json'
        cached = {r['frame']: r for r in json.loads(old.read_text())['frames']} if old.exists() else {}
        want_video = Path(video['source']).name in DEMO_NAMES
        records_path = directory / 'outputs.json'
        if records_path.exists():
            summaries.append(json.loads(records_path.read_text())['summary'])
            continue
        cap = cv2.VideoCapture(str(ROOT / video['source']))
        fps = cap.get(cv2.CAP_PROP_FPS)
        encoder = None
        dest = OUT / 'videos' / f'{uid}.mp4'
        if want_video:
            encoder = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo',
                '-pix_fmt', 'bgr24', '-s', '1440x940', '-r', str(fps), '-i', '-', '-an',
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p',
                '-threads', '2', '-movflags', '+faststart', str(dest)], stdin=subprocess.PIPE)
        bright = set(video['bright_indices'])
        records = []
        frame = 0
        try:
            while True:
                ok, native = cap.read()
                if not ok:
                    break
                if frame in bright:
                    raw = downscale(native)
                    b, g, r = cv2.mean(raw)[:3]
                    luma = .114*b + .587*g + .299*r
                    assert luma >= 75, (uid, frame, luma)
                    enhanced = enhancer([raw])[0] if (frame not in cached or want_video or frame == video['audit_frame']) else None
                    row = dict(frame=frame, timestamp=frame/fps, luma=luma)
                    if frame in cached:
                        row.update({k: {field: cached[frame][k][field]
                            for field in ('boxes', 'pose_boxes', 'pose_xy', 'pose_scores', 'pose_conf')}
                            for k in models})
                    else:
                        for name, model in models.items():
                            boxes = person_boxes(model.detector._frames_to_dets([enhanced])[0], confidence=.35)
                            pb, xy, cf, score = model.poses(enhanced)
                            row[name] = dict(boxes=boxes.tolist(), pose_boxes=pb.tolist(), pose_xy=xy.tolist(),
                                             pose_scores=cf.tolist(), pose_conf=score.tolist())
                    records.append(row)
                    if frame == video['audit_frame']:
                        cv2.imwrite(str(directory / 'raw.png'), raw)
                        cv2.imwrite(str(directory / 'enhanced.png'), enhanced)
                        cv2.imwrite(str(OUT / 'posters' / f'{uid}.jpg'), mosaic(enhanced, row, video['source']))
                    if encoder:
                        encoder.stdin.write(mosaic(enhanced, row, video['source']).tobytes())
                frame += 1
        finally:
            cap.release()
            if encoder:
                encoder.stdin.close()
                if encoder.wait(timeout=60):
                    raise RuntimeError(f'encode failed {dest}')
        assert frame == video['frames'] and len(records) == len(bright), (uid, frame, len(records))
        if want_video:
            check = subprocess.run(['ffmpeg', '-v', 'error', '-threads', '2', '-i', str(dest), '-f', 'null', '-'], capture_output=True, check=True)
            assert not check.stderr, check.stderr
            checkcap = cv2.VideoCapture(str(dest))
            count = 0
            while checkcap.read()[0]:
                count += 1
            checkcap.release()
            assert count == len(records)
        summary = dict(id=uid, source=video['source'], frames=len(records), fps=fps,
                       original_frames_decoded=frame, reused_outputs=len(cached),
                       video=str(dest.relative_to(OUT)) if want_video else None, full_decode=True)
        save(records_path, dict(summary=summary, frames=records))
        summaries.append(summary)
        save(OUT / 'replay-summary.json', summaries)
        print(f'[{vi+1}/{len(selected)}] {video["source"]}: {len(records)} bright frames; cached={len(cached)}', flush=True)
    enhancer.close()
    save(OUT / 'replay-summary.json', summaries)


if __name__ == '__main__':
    main()
