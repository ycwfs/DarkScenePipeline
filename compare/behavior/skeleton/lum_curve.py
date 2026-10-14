"""Brightness-vs-pose/detection degradation curve over ALL field videos.

For every sampled frame of every field take: measure the RAW (pre-enhancement) mean
luma, then push the frame through the deployed chain (840 downscale -> Retinexformer)
and record how the pose model and the object detector perform on the ENHANCED frame --
exactly what serve runs. Binning pose availability by raw luma gives the meeting's
"性能开始下降的亮度点", if one exists.

No keypoint/box ground truth exists, so degradation is measured as AVAILABILITY on takes
that are guaranteed to contain a person (every named action take shows one performing):
  pose_found    best person's mean keypoint score >= TAU (the deployed no-person gate)
  pose_complete that person has >= COMP_MIN_VIS/17 keypoints above POSE_SCORE_THR
  yolo_person   the detector reports >= 1 person box above DET_CONF
otherNN negative clips are excluded from the person curves (a person may legitimately be
absent there) and only contribute the object-count curve.

Per-frame rows land in a JSONL; the summary prints binned rates and two candidate
thresholds (90% and 95% of the bright-scene plateau).

Usage: python -u lum_curve.py <gpu> [--out /tmp/lum_curve.jsonl] [--frames-per-take 12]
"""
import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np

REPO = "/ssd1/wfs/project/lowlight/DarkScenePipeline"
PKG = os.path.join(REPO, "platform", "dark-demo-v2")
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from darkpipe.stages.downscale import DownscaleStage
from darkpipe.stages.enhance_retinexformer import RetinexformerStage
from darkpipe.stages.object_prior import YoloOnnxDetector
from darkpipe.stages.skeleton_aux import (COMP_MIN_VIS, POSE_SCORE_THR, TAU,
                                          SkeletonAux)

W_LUMA = np.array([0.114, 0.587, 0.299], np.float32)      # BGR -> BT.601 luma
DIRS = [os.path.join(REPO, "real-actiondata-all"),
        os.path.join(REPO, "real-actiondata-0909")]


def collect_videos():
    """Union of both field-data dirs, deduped by (basename, size); mp4 only (the .avi
    files in other/ are screen recordings, not camera scenes)."""
    seen, out = set(), []
    for d in DIRS:
        for p in sorted(glob.glob(os.path.join(d, "**", "*.mp4"), recursive=True)):
            key = (os.path.basename(p), os.path.getsize(p))
            if key not in seen:
                seen.add(key)
                out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gpu")
    ap.add_argument("--out", default="/tmp/lum_curve.jsonl")
    ap.add_argument("--frames-per-take", type=int, default=12)
    a = ap.parse_args()
    dev = f"cuda:{a.gpu}"

    vids = collect_videos()
    print(f"[curve] {len(vids)} unique field videos")

    ds = DownscaleStage(840)
    st = RetinexformerStage(os.path.join(PKG, "ckpts"))
    st.load(dev)
    sk = SkeletonAux(os.path.join(PKG, "ckpts"), dev, alpha=0.0, no_person="soft")
    assert not sk.load(), "pose load failed"
    det = YoloOnnxDetector(os.path.join(PKG, "ckpts"), dev)
    r = det.load()
    assert not r, r

    t0 = time.perf_counter()
    n_rows = 0
    with open(a.out, "w", encoding="utf-8") as fo:
        for i, path in enumerate(vids):
            cap = cv2.VideoCapture(path)
            frames = []
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                frames.append(f)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if len(frames) < 3:
                continue
            idx = np.linspace(0, len(frames) - 1, min(a.frames_per_take,
                                                      len(frames))).round().astype(int)
            raw = [frames[j] for j in idx]
            luma = [float((f.astype(np.float32) @ W_LUMA).mean()) for f in raw]
            enhanced = st(ds(raw))
            skeleton = sk._fill_slots(enhanced, 6)            # (T, 6, 17, 3) xy+score
            dets = det(enhanced)
            named = "/other" not in path.replace(os.sep, "/")
            for t in range(len(raw)):
                skt = skeleton[t]
                scores = skt[:, :, 2].mean(1)                 # mean keypoint score per slot
                m = int(np.argmax(scores))
                row = dict(video=os.path.basename(path), named=named, t=int(idx[t]),
                           luma=round(luma[t], 1),
                           pose_score=round(float(scores[m]), 3),
                           pose_found=int(scores[m] >= TAU),
                           pose_complete=int((skt[m, :, 2] > POSE_SCORE_THR).sum()
                                             >= COMP_MIN_VIS),
                           yolo_person=int(bool(len(dets[t])) and
                                           bool((dets[t][:, 5].astype(int) == 0).any())),
                           n_obj=int((dets[t][:, 5].astype(int) != 0).sum()))
                fo.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rows += 1
            if (i + 1) % 50 == 0:
                print(f"[curve] {i + 1}/{len(vids)} videos, {n_rows} frames, "
                      f"{time.perf_counter() - t0:.0f}s", flush=True)
    print(f"[curve] done: {n_rows} frames in {time.perf_counter() - t0:.0f}s -> {a.out}")

    # ---- summary -------------------------------------------------------------------
    rows = [json.loads(line) for line in open(a.out, encoding="utf-8")]
    named = [r for r in rows if r["named"]]
    edges = list(range(0, 181, 15))
    print("\nluma_bin  n     pose_found  pose_complete  yolo_person  n_obj/frame(named)")
    for lo, hi in zip(edges, edges[1:] + [256]):
        b = [r for r in named if lo <= r["luma"] < hi]
        if len(b) < 10:
            continue
        bo = [r for r in rows if lo <= r["luma"] < hi]
        print(f"{lo:3d}-{hi:3d}  {len(b):5d}    "
              f"{np.mean([r['pose_found'] for r in b]):.3f}        "
              f"{np.mean([r['pose_complete'] for r in b]):.3f}          "
              f"{np.mean([r['yolo_person'] for r in b]):.3f}        "
              f"{np.mean([r['n_obj'] for r in bo]):.3f}")
    plateau = np.mean([r["pose_found"] for r in named if r["luma"] >= 120])
    for target, tag in ((0.95, "95%"), (0.90, "90%")):
        thr = next((lo for lo, hi in zip(edges, edges[1:] + [256])
                    if lo >= 15 and np.mean([r["pose_found"] for r in named
                                             if lo <= r["luma"] < hi]) >= target * plateau
                    and len([r for r in named if lo <= r["luma"] < hi]) >= 10), None)
        print(f"pose_found stays >= {tag} of plateau ({plateau:.3f}) down to "
              f"luma {thr if thr is not None else 'no bin drops below (no knee)'}")


if __name__ == "__main__":
    main()
