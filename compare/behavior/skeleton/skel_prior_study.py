"""Skeleton geometry-prior study: calibrate thresholds for wave/drink/chase posture priors.

For every skeleton window cache file (T, M=2, 17, 3 float32: x/W, y/H in [0,1] + keypoint
score -- the format skeleton_cache.py stores), compute geometric features on the main
person (highest presence) and print per-class percentile tables. The prior rule that ships
in the platform stage (darkpipe/stages/skel_prior.py) uses window_features() verbatim --
the thresholds it carries are chosen from this study's tables, not guessed.

Priors requested (2026-10-07):
  wave  手举过头顶: wrist above the head reference (eyes/nose/ears) for a fraction of frames
  drink 手到脖/面部: wrist within a torso-radius of the head reference
  chase 动作幅度大: joint speed / hip translation / elbow+knee angle change above threshold

Usage (numpy only, any python):
  python3 skel_prior_study.py ../cache/skel_realact_all_v1/val
  python3 skel_prior_study.py ../cache/skel_realact_all_v1/val --out prior_feats_val.jsonl
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

# COCO-17
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK = 11, 12, 13, 14, 15, 16

CONF_THR = 0.3          # same visibility bar as normalize_window
FAST_JOINTS = (L_ELB, R_ELB, L_WRI, R_WRI, L_KNE, R_KNE, L_ANK, R_ANK)


def _angle_deg(a, b, c):
    """Angle at b (degrees) for coordinate rows (..., 2)."""
    u = a - b
    v = c - b
    nu = np.linalg.norm(u, axis=-1)
    nv = np.linalg.norm(v, axis=-1)
    cos = np.clip((u * v).sum(-1) / np.maximum(nu * nv, 1e-6), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def window_features(sk, conf_thr=CONF_THR):
    """sk (T, M, 17, 3) frame-normalized -> dict of window features (nan where unevaluable).

    Main person = highest presence (max-over-frames mean keypoint score), same rule
    merge_persons uses for weighting, so the prior vets the person the ST-GCN mostly heard.
    All lengths are in torso units (mid-shoulder..mid-hip of that person, window median)
    so features are scale- and resolution-invariant.
    """
    T, M, V, _ = sk.shape
    pres = np.array([sk[:, m, :, 2].mean() for m in range(M)] if T else np.zeros(M))
    if not len(pres) or pres.max() <= 0:
        return {}
    m = int(pres.argmax())
    xy, cf = sk[:, m, :, :2], sk[:, m, :, 2]
    vis = cf > conf_thr

    def mid(i, j):                                          # per-frame midpoint or nan
        both = vis[:, i] & vis[:, j]
        out = np.full((T, 2), np.nan, np.float32)
        out[both] = (xy[both, i] + xy[both, j]) / 2
        return out

    sh, hp = mid(L_SHO, R_SHO), mid(L_HIP, R_HIP)
    torso = np.linalg.norm(sh - hp, axis=1)
    scale = float(np.nanmedian(torso)) if np.isfinite(torso).any() else np.nan
    if not np.isfinite(scale) or scale < 1e-3:
        return {}
    f = {"scale": scale}

    # head reference per frame: mid-eye -> nose -> mid-ear
    head = mid(L_EYE, R_EYE)
    fb = mid(L_EAR, R_EAR)
    fill = np.isnan(head[:, 0]) & ~np.isnan(fb[:, 0])
    head[fill] = fb[fill]
    fill = np.isnan(head[:, 0]) & ~np.isnan(xy[:, NOSE, 0])
    head[fill] = xy[:, NOSE][fill]

    # --- wave: 手腕高于头 ---
    wr = np.stack([xy[:, L_WRI], xy[:, R_WRI]], 1)           # (T,2,2)
    wr_vis = vis[:, L_WRI] | vis[:, R_WRI]
    ev = wr_vis & ~np.isnan(head[:, 0])
    raised = np.zeros(T, bool)
    if ev.any():
        d = (head[ev, None, :] - wr[ev])                    # (n,2,2): head - wrist
        above = (d[..., 1] > 0) & (vis[ev][:, [L_WRI, R_WRI]])
        raised[ev] = above.any(1)
        f["wave_gap"] = float(np.max(d[..., 1])) / scale    # + = highest wrist above head
        f["wave_gap_neg"] = float(np.min(d[..., 1])) / scale
    f["wave_frac"] = float(raised[ev].mean()) if ev.sum() else np.nan
    f["wave_n"] = int(ev.sum())

    # --- drink: 手到头部距离 ---
    ev = wr_vis & ~np.isnan(head[:, 0])
    if ev.any():
        dist = np.linalg.norm(wr[ev] - head[ev, None, :], axis=-1) / scale
        f["drink_dmin"] = float(dist.min())
        for r in (0.5, 0.7, 1.0):
            f[f"drink_frac_r{r}"] = float((dist < r).any(1).mean())
    f["drink_n"] = int(ev.sum())

    # --- chase: 运动幅度 ---
    spd = []
    for j in FAST_JOINTS:
        both = vis[1:, j] & vis[:-1, j]
        if both.any():
            spd.append(np.linalg.norm(xy[1:, j] - xy[:-1, j], axis=-1)[both] / scale)
    f["amp_joint"] = float(np.concatenate(spd).mean()) if spd else np.nan
    hp_both = ~np.isnan(hp[1:, 0]) & ~np.isnan(hp[:-1, 0])
    f["amp_hip"] = (float(np.linalg.norm(hp[1:] - hp[:-1], axis=-1)[hp_both].mean() / scale)
                    if hp_both.any() else np.nan)

    def ang_seq(a, b, c):
        ok = vis[:, a] & vis[:, b] & vis[:, c]
        seq = np.full(T, np.nan, np.float32)
        seq[ok] = _angle_deg(xy[ok, a], xy[ok, b], xy[ok, c])
        return seq

    d_ang = []
    for a, b, c in ((L_SHO, L_ELB, L_WRI), (R_SHO, R_ELB, R_WRI),
                    (L_HIP, L_KNE, L_ANK), (R_HIP, R_KNE, R_ANK)):
        s = ang_seq(a, b, c)
        mv = ~np.isnan(s[1:]) & ~np.isnan(s[:-1])
        if mv.any():
            d_ang.append(np.abs(s[1:] - s[:-1])[mv])
    f["ang_amp"] = float(np.concatenate(d_ang).mean()) if d_ang else np.nan
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split_dir", help="骨架缓存 split 目录，如 ../cache/skel_realact_all_v1/val")
    ap.add_argument("--out", default=None, help="把逐窗特征写到该 jsonl（供阈值扫描复用）")
    a = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(a.split_dir, "*", "*.npy")))
    assert paths, f"no windows under {a.split_dir}"
    per_cls = defaultdict(lambda: defaultdict(list))
    rows = []
    for p in paths:
        cls = os.path.basename(os.path.dirname(p))
        f = window_features(np.load(p))
        f["cls"] = cls
        f["stem"] = os.path.basename(p)
        rows.append(f)
        for k, v in f.items():
            if k not in ("cls", "stem") and isinstance(v, float) and np.isfinite(v):
                per_cls[cls][k].append(v)

    keys = ["wave_frac", "wave_gap", "wave_gap_neg",
            "drink_dmin", "drink_frac_r0.5", "drink_frac_r0.7", "drink_frac_r1.0",
            "amp_joint", "amp_hip", "ang_amp"]
    print(f"== {a.split_dir}  ({len(rows)} windows) ==")
    for cls in sorted(per_cls):
        n = len(glob.glob(os.path.join(a.split_dir, cls, "*.npy")))
        print(f"\n-- {cls} (n={n}) --")
        for k in keys:
            v = np.array(per_cls[cls][k])
            if not len(v):
                continue
            q = np.percentile(v, [5, 25, 50, 75, 95])
            print(f"  {k:16s} " + "  ".join(f"p{p}:{x:6.2f}" for p, x in
                                            zip([5, 25, 50, 75, 95], q)))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n逐窗特征 -> {a.out}")


if __name__ == "__main__":
    main()
