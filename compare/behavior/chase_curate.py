"""Mine `chase` clips out of HMDB's run/walk pool with a person detector.

No public action dataset has a "chasing" class, so this is the one behavior with no ground
truth anywhere. The labels produced here are WEAK — a geometric proxy for pursuit — and are
reported separately from the other nine, never blended into the headline number.

Pursuit, operationally: two or more people, both moving fast, moving the SAME way (a chase,
not a crossing or two people drifting apart), staying within a bounded distance of each other
for most of the clip. Each condition is scored continuously and combined; clips are ranked and
the top `--keep` become `chase`.

Measured outcome (montage-verified, 780 run/walk clips): 234 hold two people for 60% of
frames, 50 score above zero, 38 pass the floor -> `chase` (28 train / 1 val / 9 test under
HMDB's official splits). That is a SMALL and weakly-supervised class and is reported on its
own, never folded into the headline number. Splits are not rebalanced to pad it: HMDB clips
are cut from feature films and several share a source movie, so re-splitting would leak.

Three outputs, because the interesting decision is what happens to the rejects:
  chase_keep.txt        -> label `chase`
  chase_ambiguous.txt   -> EXCLUDED from training. Multi-person and fast, but the pursuit
                           geometry is unclear; calling these `other` would teach the model
                           that two people running together is ordinary, which is exactly the
                           distinction `chase` rests on.
  everything else       -> `other` (single-person running/walking is ordinary activity)
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import behavior_manifest as BM   # noqa: E402

PERSON = 1              # COCO class id in torchvision detection models
SCORE_TH = 0.75
N_FRAMES = 24           # sampled evenly; consecutive enough for greedy association
MONTAGE = os.path.join(BM.MANIFESTS, "chase_montage")


def load_detector(device):
    from torchvision.models.detection import (FasterRCNN_MobileNet_V3_Large_FPN_Weights,
                                              fasterrcnn_mobilenet_v3_large_fpn)
    w = FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1
    return fasterrcnn_mobilenet_v3_large_fpn(weights=w, box_score_thresh=SCORE_TH) \
        .to(device).eval()


def sample_frames(path, n=N_FRAMES):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if len(frames) < 4:
        return None
    idx = np.linspace(0, len(frames) - 1, min(n, len(frames))).round().astype(int)
    return [frames[i] for i in idx]


@torch.no_grad()
def detect(model, frames, device):
    """-> list per frame of (centroids (k,2), heights (k,)), normalized by frame width."""
    x = [torch.from_numpy(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
         for f in frames]
    out = []
    for s in range(0, len(x), 8):
        batch = [t.to(device) for t in x[s:s + 8]]
        for det in model(batch):
            keep = (det["labels"] == PERSON) & (det["scores"] >= SCORE_TH)
            b = det["boxes"][keep].cpu().numpy()
            h, w = frames[0].shape[:2]
            if len(b) == 0:
                out.append((np.zeros((0, 2)), np.zeros(0)))
                continue
            c = np.stack([(b[:, 0] + b[:, 2]) / 2 / w, (b[:, 1] + b[:, 3]) / 2 / w], 1)
            out.append((c, (b[:, 3] - b[:, 1]) / w))
    return out


def track_two(dets):
    """Greedy nearest-centroid association of the two most persistent people.

    Returns (trackA, trackB) as (F,2) arrays with NaN where a track is missing, plus the
    per-frame person count.
    """
    counts = np.array([len(c) for c, _ in dets])
    F = len(dets)
    A = np.full((F, 2), np.nan)
    B = np.full((F, 2), np.nan)
    heights = []
    prev = None
    for i, (c, h) in enumerate(dets):
        if len(c) == 0:
            continue
        if len(c) == 1:
            order = [0]
        else:
            # two people = the two largest detections (closest to camera, most reliable)
            order = list(np.argsort(-h)[:2])
        pts = c[order]
        heights.append(float(np.mean(h[order])))
        if prev is None or len(pts) == 1:
            A[i] = pts[0]
            if len(pts) > 1:
                B[i] = pts[1]
        else:
            # assign to whichever of the previous two positions is nearer
            d00 = np.linalg.norm(pts[0] - prev[0]) + np.linalg.norm(pts[1] - prev[1])
            d01 = np.linalg.norm(pts[0] - prev[1]) + np.linalg.norm(pts[1] - prev[0])
            if d00 <= d01:
                A[i], B[i] = pts[0], pts[1]
            else:
                A[i], B[i] = pts[1], pts[0]
        prev = (A[i], B[i] if not np.isnan(B[i]).any() else A[i])
    return A, B, counts, (np.mean(heights) if heights else 0.0)


def _disp(track):
    """Net displacement and mean per-frame speed of a track, ignoring gaps."""
    ok = ~np.isnan(track).any(1)
    if ok.sum() < 3:
        return None, 0.0
    p = track[ok]
    steps = np.linalg.norm(np.diff(p, axis=0), axis=1)
    return p[-1] - p[0], float(np.median(steps))


def score_from_features(f):
    """Pure scoring rule over the cached geometry -> score in [0,1].

    MULTIPLICATIVE in speed and heading, deliberately. A weighted sum was tried first and
    ranked 89/126 of its keeps as motionless people: two actors standing in a room have
    perfect heading agreement, ideal separation and two boxes every frame, which is 0.65 of
    an additive score before anyone moves. Worse, clips with heading_cos = -0.9 (walking
    APART) still cleared the floor. Both are necessary conditions for pursuit, so both gate
    the score to zero rather than merely subtracting from it.
    """
    multi, cos = f["frac_multi"], f["heading_cos"]
    rel, sep = f.get("rel_speed", 0.0), f.get("sep_bodies", 0.0)
    if multi < 0.6:
        return 0.0
    # Bands from the measured pool: of 780 run/walk clips only 234 hold two people for 60% of
    # frames, and among those rel_speed is p50=0.025 / p90=0.084. Speed is the binding
    # constraint -- heading is bimodal (p50=-0.02, p70=0.85), so requiring it at all costs
    # almost no recall but removes the pairs that walk apart. The ramp therefore opens at the
    # median of two-person motion and saturates in its top decile.
    s_speed = float(np.clip((rel - 0.025) / 0.075, 0, 1))
    s_head = float(np.clip((cos - 0.35) / 0.5, 0, 1))         # same way, not crossing/parting
    s_sep = float(np.clip((sep - 0.3) / 0.7, 0, 1) * np.clip((4.0 - sep) / 1.5, 0, 1))
    s_multi = float(np.clip((multi - 0.6) / 0.3, 0, 1))
    return s_speed * s_head * (0.5 + 0.3 * s_sep + 0.2 * s_multi)


def score_clip(dets):
    """-> (score in [0,1], features dict). Higher = more like a pursuit."""
    A, B, counts, mean_h = track_two(dets)
    F = len(counts)
    multi = float((counts >= 2).mean())                       # two people present, how often
    dA, sA = _disp(A)
    dB, sB = _disp(B)
    f = dict(frac_multi=multi, n_frames=F, mean_person_h=mean_h,
             speed_a=sA, speed_b=sB, heading_cos=0.0, dist_mean=0.0, dist_std=0.0,
             rel_speed=0.0, sep_bodies=0.0)
    if multi < 0.6 or dA is None or dB is None:
        return 0.0, f

    # heading agreement: both net displacements pointing the same way
    nA, nB = np.linalg.norm(dA), np.linalg.norm(dB)
    cos = float(dA @ dB / (nA * nB)) if nA > 1e-6 and nB > 1e-6 else 0.0
    both = ~np.isnan(A).any(1) & ~np.isnan(B).any(1)
    dist = np.linalg.norm(A[both] - B[both], axis=1)
    # speed relative to person size: a pursuit moves a good fraction of a body height/frame
    f.update(heading_cos=cos, dist_mean=float(dist.mean()) if len(dist) else 0.0,
             dist_std=float(dist.std()) if len(dist) else 0.0,
             rel_speed=float(min(sA, sB) / max(mean_h, 1e-3)),
             sep_bodies=float((dist.mean() if len(dist) else 0.0) / max(mean_h, 1e-3)))
    return score_from_features(f), f


def montage(paths, out_png, cols=6, cell=(160, 120)):
    """Contact sheet of the top candidates for a manual spot-check."""
    rows = (len(paths) + cols - 1) // cols
    sheet = np.zeros((rows * cell[1], cols * cell[0], 3), np.uint8)
    for i, p in enumerate(paths):
        fr = sample_frames(p, 3)
        if not fr:
            continue
        img = cv2.resize(fr[len(fr) // 2], cell)
        r, c = divmod(i, cols)
        sheet[r * cell[1]:(r + 1) * cell[1], c * cell[0]:(c + 1) * cell[0]] = img
        cv2.putText(sheet, os.path.basename(p)[:18], (c * cell[0] + 2, r * cell[1] + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 255), 1)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    cv2.imwrite(out_png, sheet)
    print(f"[montage] {len(paths)} clips -> {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=int, default=60, help="cap on how many -> `chase`")
    ap.add_argument("--keep-min", type=float, default=0.05,
                    help="score floor for `chase`; a fixed top-N alone would admit weak "
                         "clips just to hit the count. 0.05 keeps 38 of 780 -- the pool "
                         "simply does not contain more: only 50 clips score above zero.")
    ap.add_argument("--ambiguous-th", type=float, default=0.005,
                    help="below the floor but above this -> excluded rather than `other`")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rescore", action="store_true",
                    help="re-rank from the cached geometry in chase_scores.json instead of "
                         "re-running the detector (tuning the rule is then free)")
    a = ap.parse_args()

    scores_json = os.path.join(BM.MANIFESTS, "chase_scores.json")
    if a.rescore:
        rows = json.load(open(scores_json))
        for r in rows:
            r["score"] = score_from_features(r)
        print(f"[curate] rescored {len(rows)} cached clips", flush=True)
    else:
        pool = [(p, c, s) for p, c, s in BM.hmdb_clips() if c in BM.CHASE_POOL]
        if a.limit:
            pool = pool[:a.limit]
        print(f"[curate] {len(pool)} run/walk clips", flush=True)

        model = load_detector(a.device)
        rows = []
        for i, (path, cls, split) in enumerate(pool):
            frames = sample_frames(path)
            if frames is None:
                continue
            sc, f = score_clip(detect(model, frames, a.device))
            rows.append(dict(path=path, base=os.path.basename(path), cls=cls, split=split,
                             score=sc, **{k: float(v) for k, v in f.items()}))
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(pool)}  kept-so-far>=0.25: "
                      f"{sum(r['score'] >= 0.25 for r in rows)}", flush=True)

    rows.sort(key=lambda r: -r["score"])
    keep = [r for r in rows if r["score"] >= a.keep_min][:a.keep]
    amb = [r for r in rows[len(keep):] if r["score"] >= a.ambiguous_th]

    os.makedirs(BM.MANIFESTS, exist_ok=True)
    with open(BM.CHASE_KEEP, "w") as fh:
        fh.write("# weak labels: top-ranked pursuit candidates from HMDB run/walk\n")
        fh.write("# score = 0.35*speed + 0.30*heading + 0.20*separation + 0.15*two-person\n")
        for r in keep:
            fh.write(f"{r['base']}\n")
    with open(BM.CHASE_DROP, "w") as fh:
        fh.write("# multi-person and fast but unclear pursuit -> excluded from training\n")
        for r in amb:
            fh.write(f"{r['base']}\n")
    json.dump(rows, open(scores_json, "w"), indent=1)

    by_split = {}
    for r in keep:
        by_split[r["split"]] = by_split.get(r["split"], 0) + 1
    print(f"\n[curate] keep={len(keep)} (score {keep[-1]['score']:.3f}..{keep[0]['score']:.3f}) "
          f"splits={by_split}")
    print(f"[curate] ambiguous={len(amb)} excluded; "
          f"other={len(rows) - len(keep) - len(amb)}")
    montage([r["path"] for r in keep[:36]], os.path.join(MONTAGE, "top36.png"))
    montage([r["path"] for r in rows[-36:]], os.path.join(MONTAGE, "bottom36.png"))


if __name__ == "__main__":
    main()
