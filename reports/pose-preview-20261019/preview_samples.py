"""Ten full-video previews using deployed stages, decisions and pose drawing at 0.75."""
from collections import Counter
import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

REPO = Path('/repo')
OP = REPO / 'platform/dark-demo-v2'
OUT = Path('/evidence')
ap = argparse.ArgumentParser()
ap.add_argument('--only', type=int, nargs='*')
args = ap.parse_args()
sys.path.insert(0, str(OP))
from darkpipe.config import PipelineConfig, validate
from darkpipe.media import VideoWriter
from darkpipe.pose_overlay import ActionPoseOverlay
from darkpipe.render import append_label_bar
from darkpipe.stages import build_stages

torch.set_num_threads(4)
cv2.setNumThreads(2)
src = REPO / 'real-actiondata-0909'
choices = [('喝水16', 'drink'), ('喝水17', 'drink'), ('喝水18', 'drink'),
           ('挥手20', 'wave'), ('挥手21', 'wave'), ('挥手22', 'wave'),
           ('捡东西2', 'pick_up'), ('追逐16', 'chase'), ('追逐19', 'chase')]
paths = [(src / (name+'.mp4'), label) for name, label in choices]
# Fixed negative sample from the existing B suite; no label strip/re-enhancement of feedback MP4s.
negative = sorted((src/'other').glob('other*.mp4'))[0]
paths.append((negative, 'other'))
cfg = validate(PipelineConfig(input=str(paths[0][0]), output=str(OUT/'preview.mp4'),
    recognize='behavior', device='cuda:0',
    ckpt_dir=str(OP/'ckpts'), proc_max_side=720, enhance='retinexformer', enhance_chunk=1,
    denoise='fast', sr='bicubic', sr_scale=2, reco_span_sec=1.0,
    reco_min_conf=.75, pose_overlay=True, skel_fusion_alpha=.7, lum_gate=75))
stages, recognizer = build_stages(cfg)
for stage in stages:
    stage.load(cfg.device)
recognizer.load(cfg.device)
assert recognizer.skel is not None, 'Skeleton branch must be active for visual acceptance'
recognizer.warmup()
enhancers = [s for s in stages if not s.post_recognition]
post = [s for s in stages if s.post_recognition]
results = json.loads((OUT/'sample-results.json').read_text()) if args.only else []
(OUT/'videos').mkdir(exist_ok=True)
(OUT/'stills').mkdir(exist_ok=True)

try:
    for number, (path, expected) in enumerate(paths, 1):
        if args.only and number not in args.only:
            continue
        recognizer.reset()
        current = None
        drawer = ActionPoseOverlay()
        cap = cv2.VideoCapture(str(path))
        assert cap.isOpened(), path
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        n_expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        destination = OUT/'videos'/f'{number:02d}_{path.stem}_pose.mp4'
        writer = VideoWriter(str(destination), fps=fps)
        events, draw_counts, draw_ms, branch_reasons = [], [], [], []
        clean_at_best, best, best_rank = None, None, (-1, -1)
        index = 0
        start = time.monotonic()
        try:
            while True:
                ok, frame = cap.read()
                if not ok: break
                ts = index / fps
                chunk = [frame]
                for stage in enhancers: chunk = stage(chunk)
                enhanced = chunk[0]
                ev = recognizer.push(enhanced, index, ts, raw_frame=frame)
                if ev is not None:
                    current = ev
                    events.append(ev.to_dict())
                    branch_reasons.append(recognizer.prior_note)
                for stage in post: chunk = stage(chunk)
                t = time.perf_counter()
                painted = drawer.draw(chunk[0], enhanced, current, ts)
                draw_ms.append((time.perf_counter()-t)*1000)
                draw_counts.append(drawer.last_drawn)
                rendered = append_label_bar(painted, current)
                rank = (drawer.last_drawn, int(ev is not None))
                if best is None or rank > best_rank:
                    best, clean_at_best, best_rank = rendered.copy(), frame.copy(), rank
                writer.write(rendered)
                index += 1
        finally:
            cap.release()
            writer.close()
        assert index == n_expected == writer.count, (path, index, n_expected, writer.count)
        cv2.imwrite(str(OUT/'stills'/f'{number:02d}_{path.stem}_pose.jpg'), best)
        cv2.imwrite(str(OUT/'stills'/f'{number:02d}_{path.stem}_raw.jpg'), clean_at_best)
        rec = dict(number=number, source=str(path.relative_to(REPO)), expected=expected,
                   video=str(destination.relative_to(OUT)), frames=index, fps=fps,
                   duration=index/fps, event_labels=dict(Counter(e['label'] for e in events)),
                   skeleton_active_events=sum(e['skeleton_active'] for e in events),
                   actor_events=sum(bool(e['actor_poses']) for e in events),
                   frames_with_skeleton=sum(n>0 for n in draw_counts), max_people=max(draw_counts),
                   draw_ms_median=float(np.median(draw_ms)), draw_ms_p95=float(np.percentile(draw_ms,95)),
                   events=events, branch_reasons=branch_reasons,
                   elapsed_seconds=time.monotonic()-start)
        results = sorted([r for r in results if r['number'] != number] + [rec], key=lambda r: r['number'])
        (OUT/'sample-results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
        print('[sample]',json.dumps({k:v for k,v in rec.items() if k not in ('events','branch_reasons')},
                                  ensure_ascii=False),flush=True)
finally:
    recognizer.close()
    for stage in stages: stage.close()
