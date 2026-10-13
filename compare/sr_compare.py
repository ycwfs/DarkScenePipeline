"""Head-to-head for the two x2 SR backends, in the domain this pipeline actually runs in.

Public SR benchmarks (Set5/Urban100) are clean daylight photographs. This pipeline feeds SR
the output of a low-light enhancer, which carries amplified sensor noise and washed-out
texture -- a different input distribution, and the one the choice should be made on.

Protocol (standard bicubic-downsample, applied to in-domain frames):
  enhanced dark frame  ->  HR reference
  bicubic /2           ->  LR input
  SR x2                ->  compare against HR

Reported per model: PSNR / SSIM against the HR reference, plus ms/frame at that size, plus
bicubic upsampling as the do-nothing floor. Even sizes only, so the /2 is exact and no model
is charged for a rounding mismatch.
"""
import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DP = os.path.dirname(HERE)
sys.path.insert(0, DP)


def ssim(a, b):
    """Grayscale SSIM, gaussian-weighted 11x11 -- the standard formulation."""
    a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float64)
    b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    k = cv2.getGaussianKernel(11, 1.5)
    win = k @ k.T
    mu_a = cv2.filter2D(a, -1, win)[5:-5, 5:-5]
    mu_b = cv2.filter2D(b, -1, win)[5:-5, 5:-5]
    saa = cv2.filter2D(a * a, -1, win)[5:-5, 5:-5] - mu_a ** 2
    sbb = cv2.filter2D(b * b, -1, win)[5:-5, 5:-5] - mu_b ** 2
    sab = cv2.filter2D(a * b, -1, win)[5:-5, 5:-5] - mu_a * mu_b
    m = ((2 * mu_a * mu_b + C1) * (2 * sab + C2)) / \
        ((mu_a ** 2 + mu_b ** 2 + C1) * (saa + sbb + C2))
    return float(m.mean())


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def collect_hr(clips, n_frames, every, enhance, ckpt_dir, device):
    """Decode frames from real dark clips and enhance them -> the HR references."""
    from darkpipe.stages.enhance_retinexformer import RetinexformerStage
    from darkpipe.stages.enhance_cidnet import CIDNetStage
    stage = (RetinexformerStage if enhance == "retinexformer" else CIDNetStage)(ckpt_dir,
                                                                               chunk=8)
    stage.load(device)
    hr, per_clip = [], max(1, n_frames // len(clips))
    for path in clips:
        cap = cv2.VideoCapture(path)
        raw, i = [], 0
        while len(raw) < per_clip:
            ok, f = cap.read()
            if not ok:
                break
            if i % every == 0:
                h, w = f.shape[:2]
                raw.append(f[: h // 2 * 2, : w // 2 * 2])   # even dims: exact /2
            i += 1
        cap.release()
        # One clip at a time: ARID classes are not all the same resolution, and the enhancer
        # batches into a single tensor.
        if raw:
            hr.extend(stage(raw))
        if len(hr) >= n_frames:
            break
    stage.close()
    return hr[:n_frames]


def run_model(kind, lr, ckpt_dir, device):
    """SR every LR frame one at a time; returns (outputs, ms/frame)."""
    if kind == "bicubic":
        t0 = time.time()
        out = [cv2.resize(f, (f.shape[1] * 2, f.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
               for f in lr]
        return out, (time.time() - t0) / len(lr) * 1000

    import torch
    if kind == "lightsr_x2":
        from darkpipe.stages.sr_lightsr import LightSRStage
        stage = LightSRStage(ckpt_dir, chunk=1)
    else:
        from darkpipe.stages.sr_catanet import CATANetStage
        stage = CATANetStage(ckpt_dir, chunk=1)
    stage.load(device)
    stage([lr[0]])                                  # warm up kernels before timing
    torch.cuda.synchronize()
    t0 = time.time()
    out = [stage([f])[0] for f in lr]
    torch.cuda.synchronize()
    ms = (time.time() - t0) / len(lr) * 1000
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    stage.close()
    torch.cuda.reset_peak_memory_stats()
    return out, ms, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default=os.path.join(HERE, "arid", "test_clips"))
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--every", type=int, default=17)   # spread across the clip, not adjacent
    ap.add_argument("--enhance", default="retinexformer")
    ap.add_argument("--ckpt-dir", default=os.path.join(DP, "ckpts"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "sr_compare.json"))
    a = ap.parse_args()

    clips = sorted(glob.glob(os.path.join(a.clips, "*.mp4"))) or \
        sorted(glob.glob(os.path.join(a.clips, "*", "*.mp4")))
    if not clips:
        sys.exit(f"no clips under {a.clips}")
    clips = clips[::max(1, len(clips) // 12)]      # spread across classes, not one action
    hr = collect_hr(clips, a.frames, a.every, a.enhance, a.ckpt_dir, a.device)
    lr = [cv2.resize(f, (f.shape[1] // 2, f.shape[0] // 2), interpolation=cv2.INTER_CUBIC)
          for f in hr]
    sizes = {f"{f.shape[1]}x{f.shape[0]}" for f in hr}
    print(f"{len(hr)} frames from {len(clips)} clips, HR sizes {sorted(sizes)}, "
          f"enhance={a.enhance}")

    rows = {}
    for kind in ("bicubic", "lightsr_x2", "catanet_x2"):
        res = run_model(kind, lr, a.ckpt_dir, a.device)
        out, ms = res[0], res[1]
        peak = res[2] if len(res) > 2 else 0.0
        ps = [psnr(o, h) for o, h in zip(out, hr)]
        ss = [ssim(o, h) for o, h in zip(out, hr)]
        rows[kind] = dict(psnr=float(np.mean(ps)), ssim=float(np.mean(ss)),
                          ms_per_frame=ms, peak_gib=peak)
        print(f"  {kind:12s} PSNR {np.mean(ps):6.2f} dB   SSIM {np.mean(ss):.4f}   "
              f"{ms:7.1f} ms/frame   peak {peak:.1f} GiB")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(dict(enhance=a.enhance, n_frames=len(hr),
                       hr_size=[hr[0].shape[1], hr[0].shape[0]], models=rows), f, indent=2)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
