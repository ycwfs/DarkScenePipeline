"""Per-class pose-keypoint visualization on field data, served-chain faithful.

For each class in real-actiondata-all (喝水/挥手/追逐/捡东西 + other), pick N samples evenly
spaced over the sorted file list, then per sample grab K frames evenly over the video and run
the DEPLOYED chain on them: DownscaleStage(840) -> Retinexformer -> RTMO-s pose. Keypoints are
drawn on the enhanced frame (the exact pixels the pose model sees in serving) and tiled into
one contact sheet per take. Score color code: green >=0.5, orange >=0.3, red <0.3 (deployed
POSE_SCORE_THR=0.3 sits at the orange boundary). A per-take summary CSV records person counts
and the deployed-style presence (max over frames/persons of mean keypoint score).

Usage (inside darkpipe-operator container):
  python pose_vis_samples.py [--limit N]
Output: <REPO>/skel_vis_10perclass/<class>/<take>.jpg + summary.csv
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np

REPO = "/ssd1/wfs/project/lowlight/DarkScenePipeline"
sys.path.insert(0, REPO)

from darkpipe.stages.downscale import DownscaleStage
from darkpipe.stages.enhance_retinexformer import RetinexformerStage

DATA = os.path.join(REPO, "real-actiondata-all")
OUT = os.path.join(REPO, "skel_vis_10perclass")
ONNX = os.path.join(REPO, "models", "pose", "rtmo-s_end2end.onnx")
CLASSES = ["other", "喝水", "挥手", "追逐", "捡东西"]
N_PER_CLASS = 10
K_FRAMES = 6
TILE_W = 560
# Same edges skeleton_aux.py uses (COCO-17).
COCO_EDGES = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6),
              (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4),
              (3, 5), (4, 6)]


def score_color(s):
    if s >= 0.5:
        return (0, 200, 0)
    if s >= 0.3:
        return (0, 165, 255)
    return (0, 0, 255)


def draw_pose(img, kpts, scores):
    """In-place COCO-17 skeleton overlay, color-coded by keypoint score."""
    for i, j in COCO_EDGES:
        s = min(float(scores[i]), float(scores[j]))
        if s < 0.10:
            continue
        p1 = tuple(np.round(kpts[i]).astype(int))
        p2 = tuple(np.round(kpts[j]).astype(int))
        cv2.line(img, p1, p2, score_color(s), 2, cv2.LINE_AA)
    for (x, y), s in zip(kpts, scores):
        if float(s) < 0.10:
            continue
        cv2.circle(img, (int(round(x)), int(round(y))), 3, score_color(float(s)), -1,
                   cv2.LINE_AA)


def pick_even(names, n):
    if len(names) <= n:
        return list(names)
    idx = np.linspace(0, len(names) - 1, n).round().astype(int)
    seen, out = set(), []
    for i in idx:                                   # dedupe, keep first-seen order
        if i not in seen:
            seen.add(i)
            out.append(names[i])
    return out


def grab_frames(path, indices):
    """Decode only the requested frame indices (sequential grab is the only way with cv2)."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    want = sorted(set(int(i) for i in indices))
    got, wi, i = {}, 0, 0
    while wi < len(want):
        if not cap.grab():
            break
        if i == want[wi]:
            ok, f = cap.retrieve()
            if ok:
                got[want[wi]] = f
            wi += 1
        i += 1
    cap.release()
    if not total:
        total = i
    return got, total


def count_frames(path):
    """Grab through without retrieving -- the only exact frame count cv2 gives cheaply."""
    cap = cv2.VideoCapture(path)
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N takes per class")
    args = ap.parse_args()

    ds = DownscaleStage(840)
    st = RetinexformerStage(os.path.join(REPO, "ckpts"))
    st.load("cuda")
    from rtmlib import RTMO
    pose = RTMO(onnx_model=ONNX, model_input_size=(640, 640), backend="onnxruntime",
                device="cuda", score_thr=0.0)      # thr at draw time, not det time
    dummy = np.zeros((472, 840, 3), np.uint8)
    pose(dummy)                                    # ONNX session init, untimed

    rows = []
    for cls in CLASSES:
        if cls == "other":
            d = os.path.join(DATA, "other")
            names = sorted(f for f in os.listdir(d) if f.endswith(".mp4"))
        else:
            d = DATA
            names = sorted(f for f in os.listdir(d)
                           if f.startswith(cls) and f.endswith(".mp4"))
        picked = pick_even(names, N_PER_CLASS)
        if args.limit:
            picked = picked[:args.limit]
        od = os.path.join(OUT, cls)
        os.makedirs(od, exist_ok=True)
        print(f"[{cls}] {len(picked)}/{len(names)} takes -> {od}", flush=True)

        for name in picked:
            total = count_frames(os.path.join(d, name))
            if total > 0:
                idx = np.linspace(0, total - 1, min(K_FRAMES, total)).round()
                idx = sorted(set(int(i) for i in idx))
                got, total = grab_frames(os.path.join(d, name), idx)
            else:                                  # undecodable -> will produce empty sheet
                got = {}

            tiles = []
            frame_persons, frame_scores = [], []
            pres = 0.0
            for fi in sorted(got):
                frame = st(ds([got[fi]]))[0]       # served chain: downscale 840 -> enhance
                kpts, scores = pose(frame)
                vis = frame.copy()
                if len(kpts):
                    order = np.argsort(scores.mean(1))[::-1][:2]
                    best = 0.0
                    for m in order:                # same top-2-by-score rule as serving
                        draw_pose(vis, kpts[m], scores[m])
                        best = max(best, float(scores[m].mean()))
                    frame_persons.append(len(order))
                    frame_scores.append(round(best, 3))
                    pres = max(pres, best)
                    cv2.putText(vis, f"p{len(order)} s{best:.2f}", (8, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
                    cv2.putText(vis, f"p{len(order)} s{best:.2f}", (8, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                else:
                    frame_persons.append(0)
                    frame_scores.append(0.0)
                    cv2.putText(vis, "NO PERSON", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (0, 0, 255), 3)
                h = int(vis.shape[0] * TILE_W / vis.shape[1])
                tiles.append(cv2.resize(vis, (TILE_W, h)))
            while len(tiles) < K_FRAMES:
                tiles.append(np.full((int(TILE_W * 9 / 16), TILE_W, 3), 40, np.uint8))

            # 3x2 contact sheet with a header strip (ASCII overlay: cv2 can't draw CJK --
            # the Chinese take name is the .jpg filename itself)
            th = max(t.shape[0] for t in tiles)
            header = np.full((36, TILE_W * 3, 3), 25, np.uint8)
            ascii_name = name.encode("ascii", "replace").decode()
            cv2.putText(header, f"{ascii_name}  frames={total}  presence={pres:.2f}",
                        (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            grid = []
            for r in range(2):
                row = []
                for c in range(3):
                    t = tiles[r * 3 + c]
                    pad = np.full((th, TILE_W, 3), 40, np.uint8)
                    pad[:t.shape[0]] = t
                    row.append(pad)
                grid.append(np.hstack(row))
            sheet = np.vstack([header, np.vstack(grid)])
            cv2.imwrite(os.path.join(od, name.rsplit(".", 1)[0] + ".jpg"),
                        sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
            rows.append([cls, name, total, ",".join(map(str, frame_persons)),
                         ",".join(map(str, frame_scores)), round(pres, 3)])
            print(f"  {name}: frames={total} p={frame_persons} presence={pres:.2f}",
                  flush=True)

    with open(os.path.join(OUT, "summary.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["class", "file", "n_frames", "persons_per_frame",
                    "best_mean_kpt_score_per_frame", "presence"])
        w.writerows(rows)
    print(f"done: {len(rows)} sheets -> {OUT}/summary.csv", flush=True)


if __name__ == "__main__":
    main()
