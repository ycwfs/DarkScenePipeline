"""Real-model previews and exact fusion audits, using the selected deployment tree."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
ap = argparse.ArgumentParser()
ap.add_argument('--operator', type=Path, default=ROOT / 'platform/dark-demo-v2')
ap.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[1])
ap.add_argument('--gpu', default='0')
ap.add_argument('--lum-gate', type=float, default=75.)
ap.add_argument('--names', nargs='+', default=['挥手3', '挥手4', '挥手5', '挥手20', '追逐16', '喝水16', '捡东西2', '挥手18'])
args = ap.parse_args()
sys.path.insert(0, str(args.operator))
from darkpipe.config import PipelineConfig, validate
from darkpipe.media import VideoWriter
from darkpipe.pose_overlay import ActionPoseOverlay
from darkpipe.render import append_label_bar
from darkpipe.stages import build_stages
from darkpipe.stages.recognize import reject_probs

torch.set_num_threads(4)
cv2.setNumThreads(2)
out = args.out
(out / 'videos').mkdir(parents=True, exist_ok=True)
(out / 'stills').mkdir(exist_ok=True)
cfg = validate(PipelineConfig(input='preview', output=str(out / 'preview.mp4'),
    recognize='behavior', device=f'cuda:{args.gpu}', ckpt_dir=str(args.operator / 'ckpts'),
    proc_max_side=720, enhance='retinexformer', enhance_chunk=1, denoise='fast',
    sr='bicubic', sr_scale=2, reco_span_sec=1., reco_min_conf=.75,
    pose_overlay=True, skel_fusion_alpha=.7, lum_gate=args.lum_gate))
stages, rec = build_stages(cfg)
for s in stages:
    s.load(cfg.device)
rec.load(cfg.device)
assert rec.skel is not None
rec.warmup()
pre = [s for s in stages if not s.post_recognition]
post = [s for s in stages if s.post_recognition]
last_rgb = None
original_infer = rec._infer


def infer(arr):
    global last_rgb
    last_rgb = original_infer(arr)
    return last_rgb


rec._infer = infer
summary = []
try:
    for name in args.names:
        source = ROOT / 'real-actiondata-all' / f'{name}.mp4'
        if not source.exists():
            source = ROOT / 'real-actiondata-all/other' / f'{name}.mp4'
        cap = cv2.VideoCapture(str(source))
        assert cap.isOpened(), source
        fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        rec.reset()
        drawer = ActionPoseOverlay()
        writer = VideoWriter(str(out / 'videos' / f'{name}_多人骨架.mp4'), fps=fps)
        current, idx = None, 0
        events, counts, labels, costs = [], [], Counter(), []
        best_rank, best = (-1, -1), None
        started = time.monotonic()
        try:
            while True:
                ok, raw = cap.read()
                if not ok:
                    break
                frames = [raw]
                for s in pre:
                    frames = s(frames)
                enhanced = frames[0]
                before = time.monotonic()
                event = rec.push(enhanced, idx, idx/fps, raw_frame=raw)
                if event:
                    costs.append(time.monotonic()-before)
                    current = event
                    info = rec.skel.last_analysis
                    row = event.to_dict()
                    row['rgb_probs'] = last_rgb.tolist()
                    if event.skeleton_active:
                        selected = info['best']
                        candidates = np.flatnonzero(info['valid'])
                        expected_slot = int(candidates[np.argmax(info['slot_p'][candidates].max(1))]) if len(candidates) else -1
                        assert selected == expected_slot == event.skeleton_selected_slot
                        if selected >= 0:
                            ps, pres = info['slot_p'][selected], float(info['presence'][selected])
                        else:
                            ps, pres = np.eye(10, dtype=np.float32)[9], 0.
                        fused = rec.skel.fuse(last_rgb, ps, pres)
                        row['person_probs'] = info['slot_p'].tolist()
                        row['person_valid'] = info['valid'].tolist()
                        row['skeleton_probs'] = ps.tolist()
                        row['pre_reject_probs'] = fused.tolist()
                        for actor in event.actor_poses:
                            probs = info['slot_p'][actor['slot']]
                            assert actor['label'] == rec.labels[int(probs.argmax())]
                            assert actor['selected_for_fusion'] == (actor['slot'] == selected)
                    else:
                        fused = last_rgb
                        assert not event.actor_poses and event.skeleton_selected_slot == -1
                    predicted = reject_probs(fused, rec.labels, .75)
                    assert event.label == rec.display(rec.labels[int(predicted.argmax())])
                    assert abs(event.confidence - float(predicted.max())) < 1e-6
                    events.append(row)
                for s in post:
                    frames = s(frames)
                painted = drawer.draw(frames[0], enhanced, current, idx/fps)
                rendered = append_label_bar(painted, current)
                counts.append(drawer.last_drawn)
                labels.update(p['label'] for p in drawer.last_labels)
                rank = (drawer.last_drawn, int(len({p['label'] for p in drawer.last_labels}) > 1))
                if rank > best_rank:
                    best_rank, best = rank, rendered.copy()
                writer.write(rendered)
                idx += 1
        finally:
            cap.release()
            writer.close()
        assert idx == n == writer.count
        cv2.imwrite(str(out / 'stills' / f'{name}.jpg'), best)
        (out / f'{name}_events.json').write_text(json.dumps(events, ensure_ascii=False, indent=2)+'\n')
        item = dict(name=name, source=str(source), frames=idx, fps=fps, lum_gate=args.lum_gate,
                    video=f'videos/{name}_多人骨架.mp4', events=len(events),
                    active_events=sum(e['skeleton_active'] for e in events),
                    multiperson_events=sum(len(e['actor_poses'])>=2 for e in events),
                    max_people=max(counts), frames_with_pose=sum(c>0 for c in counts),
                    frames_with_multiple_people=sum(c>=2 for c in counts),
                    per_person_labels=dict(labels), final_labels=dict(Counter(e['label'] for e in events)),
                    fusion_verified=True, recognition_seconds_p95=float(np.percentile(costs,95)) if costs else None,
                    elapsed=time.monotonic()-started)
        summary.append(item)
        (out/'sample-results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(item,ensure_ascii=False),flush=True)
finally:
    rec.close()
    for s in stages:
        s.close()
