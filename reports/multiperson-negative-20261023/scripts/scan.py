"""Rank the 96 old negative clips by visible, nonduplicate RTMO poses; inspect manually."""
import json
import sys

import cv2
import numpy as np
import torch
from rtmlib import RTMO

from common import ROOT, OUT, OP, SOURCE, content_height

sys.path.insert(0, str(OP))
from darkpipe.stages.downscale import DownscaleStage
from darkpipe.stages.skeleton_aux import duplicate_pose

torch.set_num_threads(4)
cv2.setNumThreads(2)
pose = RTMO(onnx_model=str(OP / 'ckpts/rtmo-s_end2end.onnx'), model_input_size=(640, 640),
            backend='onnxruntime', device='cuda:0', score_thr=.3)
downscale = DownscaleStage(720)
dest = OUT / 'candidates'
dest.mkdir(parents=True, exist_ok=True)
rows = []
for path in sorted(SOURCE.glob('*/*.mp4')):
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), path
    n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    counts, best = [], None
    for idx in sorted({round((n - 1) * f) for f in (.2, .5, .8)}):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        assert ok, (path, idx)
        frame = downscale([frame[:content_height(frame.shape[0])]])[0]
        kpts, scores = pose(frame)
        kept = []
        h, w = frame.shape[:2]
        for i in np.argsort(scores.mean(1))[::-1]:
            if scores[i].mean() < .5 or (scores[i] > .3).sum() < 10:
                continue
            p = np.column_stack([kpts[i] / np.array([w, h]), scores[i]])
            if any(duplicate_pose(p, other) for other in kept):
                continue
            kept.append(p)
            visible = kpts[i][scores[i] > .3].astype(int)
            cv2.rectangle(frame, tuple(visible.min(0)), tuple(visible.max(0)), (40, 230, 60), 2)
        counts.append(len(kept))
        if best is None or len(kept) > best[0]:
            best = len(kept), idx, frame
    cap.release()
    cv2.imwrite(str(dest / f'{path.stem}.jpg'), best[2])
    row = dict(source=str(path.relative_to(ROOT)), name=path.stem, frames=n, fps=fps,
               duration=n / fps, sampled_counts=counts, max_people=max(counts),
               mean_people=float(np.mean(counts)), best_frame=best[1])
    rows.append(row)
    print(f'{len(rows):02d} {path.stem}: {counts}', flush=True)
rows.sort(key=lambda x: (x['max_people'], x['mean_people'], x['duration']), reverse=True)
(OUT / 'candidates.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')
for page in range(2):
    items = rows[page * 16:(page + 1) * 16]
    sheet = np.full((4 * 230, 4 * 400, 3), 25, np.uint8)
    for i, row in enumerate(items):
        img = cv2.imread(str(dest / f"{row['name']}.jpg"))
        thumb = cv2.resize(img, (400, 224))
        y, x = i // 4 * 230, i % 4 * 400
        sheet[y:y+224, x:x+400] = thumb
        title = f"{page*16+i+1}: {row['name'][9:15]} #{row['name'].rsplit('_',1)[-1]} poses={row['sampled_counts']}"
        cv2.rectangle(sheet, (x, y), (x+399, y+25), (15,15,15), -1)
        cv2.putText(sheet, title, (x+5, y+18), cv2.FONT_HERSHEY_SIMPLEX, .48, (255,255,255), 1, cv2.LINE_AA)
    cv2.imwrite(str(OUT / f'contact-{page+1}.jpg'), sheet)
print(json.dumps(rows[:12], ensure_ascii=False, indent=2), flush=True)
