"""Physically-motivated low-light simulation, to move bright HMDB51 into ARID's domain.

Real dark video is not "bright video times 0.1": the sensor collects few photons, so the
dominant degradation is *shot noise* (Poisson, signal-dependent) on top of read noise, and it
all happens in LINEAR radiance, not in sRGB. So:

    sRGB -> linear (gamma 2.2) -> scale by k -> Poisson(shot) + Gaussian(read) -> sRGB -> uint8

`k` is calibrated against real ARID footage rather than guessed — `--calibrate` fits the k
whose darkened-HMDB mean-luma distribution best matches ARID's (see the printed report).
Per clip we draw k from a range so the training set spans exposures instead of one constant.
"""
import argparse
import glob
import os
import random
import zlib

import cv2
import numpy as np

GAMMA = 2.2
# Photo-electrons at linear 1.0. This sets shot-noise strength, and unlike k/read_noise it was
# NOT fitted -- the calibration matched ARID's mean-luma distribution, which is brightness, not
# SNR. Measured after the fact (BEHAVIOR_REPORT.md 2.2): at ARID's mean luma the simulation at
# 1200 e- carries ~13x ARID's real per-pixel noise, and ~30000 e- would match it. Changing it
# invalidates every cached frame and both trained recognizers, so it stays where the shipped
# models were trained; the gap is documented instead, and it makes the simulated-dark numbers
# pessimistic rather than optimistic.
FULL_WELL = 1200.0

# Fitted against 50 real ARID test clips (mean-luma median 8.47, p10 5.34, p90 28.18) by
# sweeping (k_range, read_noise) — see `--calibrate` and the sweep in BEHAVIOR_REPORT.md.
# These values reproduce median 8.35 / p10 5.65 / p90 26.25 on darkened HMDB, i.e. the whole
# exposure distribution, not just its centre. Read noise matters: at 2.0 e- the simulated
# floor (~11 luma) sits ABOVE ARID's median and no k can reach real darkness.
DEFAULT_K = (0.0015, 0.06)   # log-uniform per clip, ~5.3 stops of exposure spread
READ_NOISE = 1.5             # electrons, Gaussian


def darken_frames(frames, k, rng=None, full_well=FULL_WELL, read_noise=READ_NOISE):
    """frames: (T,H,W,3) uint8 BGR bright -> uint8 BGR dark. One k for the whole clip."""
    rng = rng or np.random.default_rng()
    x = np.asarray(frames, np.float32) / 255.0
    lin = np.power(x, GAMMA) * k                       # sRGB -> linear radiance, attenuated
    e = lin * full_well                                # radiance -> electrons
    e = rng.poisson(np.clip(e, 0, None)).astype(np.float32)          # shot noise
    e += rng.normal(0.0, read_noise, e.shape).astype(np.float32)     # read noise
    lin = np.clip(e / full_well, 0.0, 1.0)
    return (np.power(lin, 1.0 / GAMMA) * 255.0 + 0.5).clip(0, 255).astype(np.uint8)


def clip_k(rng, k_range=DEFAULT_K):
    """Log-uniform draw, so the exposure ladder is even in stops."""
    lo, hi = k_range
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def clip_seed(path):
    """Deterministic per-clip exposure seed, shared by every script that darkens.

    crc32, not hash(): Python randomizes string hashing per process, so hash() would give a
    clip a different exposure in the training cache than at eval time -- a silent train/test
    domain mismatch that no test would catch.
    """
    return zlib.crc32(os.path.basename(path).encode()) & 0x7FFFFFFF


def clip_rng(path, seed=0):
    """The RNG for one clip: same clip -> same noise and exposure, in any process."""
    return np.random.default_rng(seed + clip_seed(path))


def mean_luma(frames):
    a = np.asarray(frames, np.float32)
    y = 0.114 * a[..., 0] + 0.587 * a[..., 1] + 0.299 * a[..., 2]    # BGR
    return float(y.mean())


def _sample_frames(path, n=8):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out, i, want = [], 0, None
    if total > 0:
        want = set(np.linspace(0, total - 1, n).round().astype(int).tolist())
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if want is None or i in want:
            out.append(f)
        i += 1
        if want is None and len(out) >= n:
            break
    cap.release()
    return out


def calibrate(arid_glob, hmdb_glob, n_clips=60, seed=0):
    """Report the k whose darkened-HMDB luma distribution best matches real ARID."""
    rng = np.random.default_rng(seed)
    arid = sorted(glob.glob(arid_glob))
    hmdb = sorted(glob.glob(hmdb_glob))
    random.Random(seed).shuffle(arid)
    random.Random(seed).shuffle(hmdb)
    arid_l = [mean_luma(_sample_frames(p)) for p in arid[:n_clips] if _sample_frames(p)]
    bright = [_sample_frames(p) for p in hmdb[:n_clips]]
    bright = [b for b in bright if b]
    tgt = np.median(arid_l)
    print(f"[calibrate] ARID  n={len(arid_l)}  mean-luma median={tgt:.2f} "
          f"p10={np.percentile(arid_l, 10):.2f} p90={np.percentile(arid_l, 90):.2f}")
    print(f"[calibrate] HMDB  n={len(bright)}  mean-luma median="
          f"{np.median([mean_luma(b) for b in bright]):.2f} (before darkening)")
    best = None
    for k in [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3]:
        lum = [mean_luma(darken_frames(b, k, rng)) for b in bright]
        med = float(np.median(lum))
        err = abs(med - tgt)
        print(f"  k={k:<5} darkened median={med:6.2f}  p10={np.percentile(lum, 10):5.2f} "
              f"p90={np.percentile(lum, 90):5.2f}  |dmedian|={err:.2f}")
        if best is None or err < best[1]:
            best = (k, err)
    print(f"[calibrate] best k={best[0]} (|dmedian|={best[1]:.2f}); "
          f"DEFAULT_K={DEFAULT_K} brackets it to span exposures")
    return best[0]


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--arid-glob",
                    default=os.path.join(here, "..", "arid", "test_clips", "*", "*.mp4"))
    ap.add_argument("--hmdb-glob",
                    default=os.path.join(here, "..", "behavior_data", "hmdb", "test",
                                         "*", "*.mp4"))
    ap.add_argument("--n-clips", type=int, default=60)
    a = ap.parse_args()
    if a.calibrate:
        calibrate(a.arid_glob, a.hmdb_glob, a.n_clips)
