"""Head-to-head: DA-LLPose teacher (RAW dark frames) vs deployed RTMO-s (enhanced).

For evenly-sampled frames of each take, render two panels of the SAME instant:
  LEFT  raw frame  + teacher kpts (read from dark_infer.py's json, 1920-space,
                    rescaled for display) + teacher mean-kpt score
  RIGHT enhanced(840, Retinexformer) + RTMO-s kpts (the deployed skeleton branch's
        own model, run exactly like pose_vis_samples) + RTMO mean-kpt score
One jpg per take. The question this answers: on our field data, is the teacher's
pseudo-label actually BETTER than what our small model already produces? If not,
distilling RTMO-s from it cannot help.

Usage (darkpipe-dev:dallpose container, one free GPU):
  python -u compare_teacher_rtmo.py --k 4 take1.mp4 take2.mp4 ...
Output: <REPO>/teacher_vs_rtmo/<take>.jpg
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

REPO = "/ssd1/wfs/project/lowlight/DarkScenePipeline"
sys.path.insert(0, REPO)
PKG = os.path.join(REPO, "platform/dark-demo-v2")
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(REPO, "lowlight-skeleton-detection/DA-LLPose/lib"))

from darkpipe.stages.downscale import DownscaleStage
from darkpipe.stages.enhance_retinexformer import RetinexformerStage

TEACHER_JSON_DIR = os.path.join(REPO, "darkpose_teacher_test/raw")
CROWD_EDGES = [(12, 13), (13, 0), (13, 1), (0, 1), (0, 2), (1, 3), (2, 4),
               (3, 5), (0, 6), (1, 7), (6, 7), (6, 8), (7, 9), (8, 10), (9, 11)]
PANEL_W = 840


def score_color(s):
    if s >= 0.5:
        return (0, 200, 0)
    if s >= 0.3:
        return (0, 165, 255)
    return (0, 0, 255)


def draw_14(img, k14):
    """k14: (14,3) in the img's own pixel coords."""
    for i, j in CROWD_EDGES:
        s = min(float(k14[i, 2]), float(k14[j, 2]))
        if s < 0.10:
            continue
        cv2.line(img, tuple(np.round(k14[i, :2]).astype(int)),
                 tuple(np.round(k14[j, :2]).astype(int)), score_color(s), 2, cv2.LINE_AA)
    for x, y, s in k14:
        if float(s) < 0.10:
            continue
        cv2.circle(img, (int(round(x)), int(round(y))), 3, score_color(float(s)), -1,
                   cv2.LINE_AA)


def put(img, txt, org, col):
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--k", type=int, default=4, help="frames per take")
    ap.add_argument("--out", default=os.path.join(REPO, "teacher_vs_rtmo"))
    a = ap.parse_args()

    ds = DownscaleStage(840)
    st = RetinexformerStage(os.path.join(PKG, "ckpts"))
    st.load("cuda")
    from rtmlib import RTMO
    onnx = os.path.join(REPO, "models/pose/rtmo-s_end2end.onnx")
    pose = RTMO(onnx_model=onnx, model_input_size=(640, 640), backend="onnxruntime",
                device="cuda", score_thr=0.0)
    pose(np.zeros((472, 840, 3), np.uint8))            # session init

    os.makedirs(a.out, exist_ok=True)
    for path in a.videos:
        stem = os.path.basename(path).rsplit(".", 1)[0]
        tj = json.load(open(os.path.join(TEACHER_JSON_DIR, stem + ".json")))
        by_idx = {f["i"]: f for f in tj["frames"]}
        cap = cv2.VideoCapture(path)
        rows = []
        for fi in sorted(by_idx)[::max(1, len(by_idx) // a.k)][:a.k]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, raw = cap.read()
            if not ok:
                continue
            # LEFT: raw + teacher (json coords are in the raw frame's own size)
            left = raw.copy()
            rec = by_idx[fi]["people"][0] if by_idx[fi]["people"] else None
            t_score = 0.0
            if rec:
                k14 = np.asarray(rec["kpts14"], np.float32)
                draw_14(left, k14)
                t_score = rec["score"]
            put(left, f"TEACHER raw s={t_score:.2f}", (8, 28),
                (0, 255, 255) if t_score >= 0.2 else (0, 0, 255))
            # RIGHT: enhanced + RTMO (deployed domain)
            enh = st(ds([raw]))[0]
            right = enh.copy()
            r_score, n_p = 0.0, 0
            kpts, scores = pose(enh)
            for m in np.argsort(scores.mean(1))[::-1][:2]:
                draw_14(right, np.concatenate([kpts[m], scores[m][:, None]], 1))
                r_score = max(r_score, float(scores[m].mean()))
                n_p += 1
            put(right, f"RTMO enh s={r_score:.2f} p{n_p}", (8, 28),
                (0, 255, 255) if r_score >= 0.2 else (0, 0, 255))
            h = int(raw.shape[0] * PANEL_W / raw.shape[1])
            left = cv2.resize(left, (PANEL_W, h))
            hh = min(h, right.shape[0])
            left, right = left[:hh], right[:hh]
            rows.append(np.hstack([left, right]))
        cap.release()
        if rows:
            cv2.imwrite(os.path.join(a.out, stem + ".jpg"),
                        np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
            print(f"[cmp] {stem}: {len(rows)} rows -> {stem}.jpg", flush=True)


if __name__ == "__main__":
    main()
