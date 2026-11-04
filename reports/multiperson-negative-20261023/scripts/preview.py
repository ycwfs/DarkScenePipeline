"""Replay selected processed negatives through the shipped pn2 / per-person alpha path."""
import argparse
from collections import Counter
import hashlib
import json
import sys
import time

import cv2
import numpy as np
import torch

from common import ROOT, OUT, OP, content_height

sys.path.insert(0, str(OP))
from darkpipe.media import VideoWriter
from darkpipe.pose_overlay import ActionPoseOverlay
from darkpipe.render import append_label_bar
from darkpipe.stages.downscale import DownscaleStage
from darkpipe.stages.recognize import BehaviorRecognizer, reject_probs

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--selection', default=str(OUT / 'selection.json'))
args = parser.parse_args()
sources = json.load(open(args.selection))
torch.set_num_threads(4)
cv2.setNumThreads(2)
downscale = DownscaleStage(720)
rec = BehaviorRecognizer(str(OP / 'ckpts'), span_sec=1., reject_tau=.75,
                         skel_fusion_alpha=.7, lum_gate=75, fuse_mode='alpha')
rec.load('cuda:0')
assert rec.skel is not None, 'Actual skeleton inference is required for this preview'
rec.warmup()
captured = {}
original_infer = rec._infer


def infer(arr):
    result = original_infer(arr)
    captured['rgb'] = result.copy()
    return result


rec._infer = infer
for subdir in ('videos', 'stills', 'events'):
    (OUT / subdir).mkdir(exist_ok=True)
summary = []
try:
    for selected in sources:
        path = ROOT / selected['source']
        name = path.stem
        cap = cv2.VideoCapture(str(path))
        assert cap.isOpened(), path
        n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
        h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ch = content_height(h)
        rec.reset()
        drawer = ActionPoseOverlay()
        target = OUT / 'videos' / f'{name}_多人骨架.mp4'
        writer = VideoWriter(str(target), fps=fps)
        events, frame_rows, person_labels = [], [], Counter()
        current, best, idx, best_rank = None, None, 0, (-1, -1)
        started = time.monotonic()
        try:
            while True:
                ok, encoded = cap.read()
                if not ok:
                    break
                content = encoded[:ch]
                model_frame = downscale([content])[0]
                # This is an already enhanced, upscaled recording. The original luma
                # cannot be recovered; use the deployed unavailable-raw-frame policy.
                event = rec.push(model_frame, idx, idx / fps, raw_frame=None)
                if event is not None:
                    assert event.skeleton_active
                    current = event
                    info = rec.skel.last_analysis
                    valid = np.flatnonzero(info['valid'])
                    expected = int(valid[np.argmax(info['slot_p'][valid].max(1))]) if len(valid) else -1
                    assert event.skeleton_selected_slot == info['best'] == expected
                    if expected >= 0:
                        ps, presence = info['slot_p'][expected], float(info['presence'][expected])
                    else:
                        ps = np.eye(len(rec.labels), dtype=np.float32)[rec.labels.index('other')]
                        presence = 0.
                    fused = rec.skel.fuse(captured['rgb'], ps, presence)
                    predicted = reject_probs(fused, rec.labels, .75)
                    assert event.label == rec.display(rec.labels[int(predicted.argmax())])
                    assert abs(event.confidence - float(predicted.max())) < 1e-6
                    for actor in event.actor_poses:
                        p = info['slot_p'][actor['slot']]
                        assert actor['label'] == rec.labels[int(p.argmax())]
                        assert abs(actor['confidence'] - float(p.max())) < 1e-6
                        assert actor['selected_for_fusion'] == (actor['slot'] == expected)
                    assert sum(a['selected_for_fusion'] for a in event.actor_poses) == int(expected >= 0)
                    row = event.to_dict()
                    row.update(rgb_probs=captured['rgb'].tolist(), person_probs=info['slot_p'].tolist(),
                               person_valid=info['valid'].tolist(), skeleton_probs=ps.tolist(),
                               pre_reject_probs=fused.tolist(), raw_luma=None, fusion_verified=True)
                    events.append(row)
                painted = drawer.draw(content, model_frame, current, idx / fps)
                rendered = append_label_bar(painted, current, extra=f'POSES {drawer.last_drawn} | * FUSED')
                writer.write(rendered)
                person_labels.update(a['label'] for a in drawer.last_labels)
                frame_rows.append(dict(frame=idx, timestamp=idx/fps, people=drawer.last_drawn))
                rank = (drawer.last_drawn, len({a['label'] for a in drawer.last_labels}))
                if rank > best_rank:
                    best, best_rank = rendered.copy(), rank
                idx += 1
        finally:
            cap.release()
            writer.close()
        assert idx == n == writer.count and events
        cv2.imwrite(str(OUT / 'stills' / f'{name}.jpg'), best)
        (OUT / 'events' / f'{name}.json').write_text(json.dumps(events, ensure_ascii=False, indent=2) + '\n')
        (OUT / 'events' / f'{name}_frames.json').write_text(json.dumps(frame_rows, indent=2) + '\n')
        counts = [r['people'] for r in frame_rows]
        item = dict(name=name, source=selected['source'], source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    requested=selected.get('requested', False), frames=n, fps=fps, duration=n/fps,
                    source_shape=[w,h], content_height=ch, old_label_bar_removed=h-ch,
                    model_shape=list(model_frame.shape[1::-1]), output_shape=list(rendered.shape[1::-1]),
                    video=str(target.relative_to(OUT)), still=f'stills/{name}.jpg',
                    events=len(events), final_labels=dict(Counter(e['label'] for e in events)),
                    fired_events=sum(e['label'] != 'Other' for e in events),
                    valid_people_per_event=[len(e['actor_poses']) for e in events],
                    max_people=max(counts), frames_with_multiple_people=sum(c>=2 for c in counts),
                    frames_with_at_least_four_people=sum(c>=4 for c in counts),
                    drawn_people_histogram=dict(Counter(counts)),
                    displayed_person_class_observations=dict(person_labels),
                    first_pose_seconds=next((r['timestamp'] for r in frame_rows if r['people']), None),
                    fusion_verified=True, elapsed=time.monotonic()-started,
                    preprocessing='Remove old bar, resize recognition to 720; no repeated enhancement/SR/denoise',
                    raw_luma_available=False, lum_gate_effective='unavailable raw: deployed fail-open policy')
        summary.append(item)
        (OUT / 'sample-results.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(item, ensure_ascii=False), flush=True)
finally:
    rec.close()
