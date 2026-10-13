"""Scan enhanced-840 frames for pose-model disagreements: RTMO sees a person, YOLO26 doesn't.

For a sample of realact cache windows (frame 16 of each): re-read the RAW video frame,
downscale 840 + Retinexformer (exact deploy chain), run both pose models, record per-window
top-person mean-kpt-conf for each. Output: /tmp/darkmiss_conf.json for the figure script.
"""
import glob
import json
import os
import re
import sys

import cv2
import numpy as np

REPO = "/ssd1/wfs/project/lowlight/DarkScenePipeline"
sys.path.insert(0, REPO)
sys.path.insert(0, "/ssd1/wfs/project/lowlight/compare/behavior/skeleton")
sys.path.insert(0, "/ssd1/wfs/project/lowlight/compare/behavior")

from darkpipe.stages.enhance_retinexformer import RetinexformerStage
from realact_cache import resize_max_side
from skeleton_cache import make_pose

CACHE = "/ssd1/wfs/project/lowlight/compare/behavior/cache/realact_rf_v10_full"
SRC_DIRS = [os.path.join(REPO, "real-actiondata-0909"),
            os.path.join(REPO, "real-actiondata-0909", "other"),
            os.path.join(REPO, "real-actiondata")]
# map stem -> video path (first hit wins)
stem_path = {}
for d in SRC_DIRS:
    if not os.path.isdir(d):
        continue
    for p in sorted(glob.glob(os.path.join(d, "*.mp4"))):
        stem_path.setdefault(os.path.splitext(os.path.basename(p))[0], p)

rtmo = make_pose("rtmo", 0.3, "cuda:0")
y26 = make_pose("yolo26", 0.15, "cuda:0")
stage = RetinexformerStage(os.path.join(REPO, "ckpts"))
stage.load("cuda:0")

def confs(bgr):
    out = []
    for fn in (rtmo, y26):
        _, s = fn(bgr)
        out.append(float(s.mean(1).max()) if len(s) else 0.0)   # top-person mean kpt conf
    return out[0], out[1]

rows = []
jobs = []          # (cls, split, stem, idx, npy)
for split in ("train", "val"):
    for npy in sorted(glob.glob(f"{CACHE}/{split}/*/*.npy")):
        cls = os.path.basename(os.path.dirname(npy))
        m = re.fullmatch(r"(.+)_(\d+)\.npy", os.path.basename(npy))
        stem, idx = m.group(1), int(m.group(2))
        jobs.append((cls, split, stem, idx, npy))

print(f"{len(jobs)} windows; {len(stem_path)} takes known", flush=True)
# group by stem, read sequentially
jobs.sort(key=lambda j: (j[2], j[3]))
by_stem = {}
for j in jobs:
    by_stem.setdefault(j[2], []).append(j)

for si, (stem, group) in enumerate(sorted(by_stem.items())):
    path = stem_path.get(stem)
    if path is None:
        print(f"  [skip] no video for {stem}", flush=True)
        continue
    want = {g[3]: g for g in group}
    cap = cv2.VideoCapture(path)
    fi, got = 0, {}
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if fi in want:
            got[fi] = resize_max_side(f, 840)
        fi += 1
    cap.release()
    if not got:
        continue
    enh = stage([got[i] for i in sorted(got)])
    for i, e in zip(sorted(got), enh):
        c1, c2 = confs(e)
        cls, split, _, _, npy = want[i]
        rows.append(dict(cls=cls, split=split, stem=stem, idx=int(i),
                         rtmo=round(c1, 3), y26=round(c2, 3)))
    if si % 10 == 0:
        print(f"  [{si}/{len(by_stem)}] {stem}: {len(got)} frames", flush=True)

with open("/tmp/darkmiss_conf.json", "w") as f:
    json.dump(rows, f)
r = np.array([(x["rtmo"], x["y26"]) for x in rows])
print(f"scanned {len(rows)} frames @840; rtmo>=0.5&y26<0.5: {int(((r[:,0]>=0.5)&(r[:,1]<0.5)).sum())}; "
      f"both>=0.5: {int(((r[:,0]>=0.5)&(r[:,1]>=0.5)).sum())}; "
      f"y26>=0.5&rtmo<0.5: {int(((r[:,1]>=0.5)&(r[:,0]<0.5)).sum())}", flush=True)
