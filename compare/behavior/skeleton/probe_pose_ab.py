"""Pose A/B probe: RTMO (skel_v1) vs YOLO26 (skel_y26_v1) on the identical window set.

Both caches were extracted frame-exact from realact_rf_v10_full, so this is a pure array
comparison -- same windows, same frames. Per class over train+val:
  pres%      -- windows whose top person's max mean-kpt-conf >= TAU (gate would fire)
  conf       -- mean of that per-window max (over ALL windows, 0 when never present)
  jitter     -- mean frame-to-frame L2 of normalised kpts (present windows, person 0)
  kpt        -- mean visible-keypoint count at conf>0.3 (present windows)
"""
import glob
import os

import numpy as np

BASE = "/ssd1/wfs/project/lowlight/compare/behavior/cache"
TAU = 0.5
CLASSES = ["wave", "chase", "drink", "pick_up", "other"]
CACHES = {"RTMO-s": os.path.join(BASE, "skel_v1"),
          "YOLO26m": os.path.join(BASE, "skel_y26_v15")}


def stats(sk):
    """sk (T, M, 17, 3) -> pres, conf, jitter, kpt for the top person."""
    p0 = sk[:, 0]                                            # (T,17,3)
    mean_c = p0[:, :, 2].mean(1)                             # (T,)
    pres = float(mean_c.max() >= TAU)
    conf = float(mean_c.max())
    vis = p0[:, :, 2] > 0.3
    if mean_c.max() < TAU:                                   # absent -> jitter undefined
        return pres, conf, np.nan, np.nan
    xy = p0[:, :, :2]
    jit = np.linalg.norm(np.diff(xy, axis=0), axis=2)[1:].mean()   # skip first-frame pop
    return pres, conf, float(jit), float(vis.sum(1).mean())


rows = {}
for cname in CLASSES:
    paths = sorted(glob.glob(os.path.join(BASE, "realact_rf_v10_full", "*", cname, "*.npy")))
    assert paths, cname
    per_model = {}
    for tag, root in CACHES.items():
        S = [stats(np.load(p.replace("realact_rf_v10_full", os.path.basename(root))))
             for p in paths]
        S = np.array(S, np.float64)
        present = S[:, 0] > 0
        per_model[tag] = (100 * S[:, 0].mean(), S[:, 1].mean(),
                          np.nanmean(S[present, 2]) if present.any() else np.nan,
                          np.nanmean(S[present, 3]) if present.any() else np.nan,
                          int(present.sum()), len(paths))
    rows[cname] = per_model

hdr = f"{'class':9s} | " + " | ".join(f"{t:^30s}" for t in CACHES)
print(f"{'':9s} | {'pres%':>5s} {'conf':>5s} {'jitter':>6s} {'kpt':>4s} {'n_win':>9s} | "
      f"{'pres%':>5s} {'conf':>5s} {'jitter':>6s} {'kpt':>4s} {'n_win':>9s}")
for cname in CLASSES:
    a, b = rows[cname]["RTMO-s"], rows[cname]["YOLO26m"]
    print(f"{cname:9s} | {a[0]:5.1f} {a[1]:5.2f} {a[2]:6.3f} {a[3]:4.1f} {a[5]:4d}/{a[4]:<4d} | "
          f"{b[0]:5.1f} {b[1]:5.2f} {b[2]:6.3f} {b[3]:4.1f} {b[5]:4d}/{b[4]:<4d}")
