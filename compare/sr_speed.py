"""Per-frame cost of the three x2 SR backends, measured through the shipped stage classes.

Going through `BicubicStage` / `LightSRStage` / `CATANetStage` rather than the bare arch means
the numbers include what the pipeline actually does: the autocast policy each neural stage
picks at load (both run a PSNR self-check and may fall back to fp32), the uint8<->tensor
conversions, and the OOM-halving wrapper. That is the cost a `--sr` run pays.

Batch matters differently for the three: lightSR is small enough to amortize across a batch,
CATANet needs ~10 GiB for a single 640x480 frame, so its batch-4 column is expected to OOM,
and bicubic is per-frame CPU work that a batch cannot amortize at all. Serve mode is always
batch 1.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DP = os.path.dirname(HERE)
sys.path.insert(0, DP)

SIZES = [(320, 240), (640, 480)]        # input WxH; output is 2x that
BATCHES = [1, 4]
ITERS, WARMUP = 10, 3
MIN_SEC = 0.5           # bicubic is ~1 ms/frame, so 10 iterations is 10 ms of wall clock and
                        # any one-off cost dominates it; keep going until the timer is trusted


def _make(kind, ckpt_dir, b):
    from darkpipe.stages.sr_bicubic import BicubicStage
    from darkpipe.stages.sr_lightsr import LightSRStage
    from darkpipe.stages.sr_catanet import CATANetStage
    if kind == "bicubic_x2":
        return BicubicStage()                       # no weights, no chunking, no device
    return (LightSRStage if kind == "lightsr_x2" else CATANetStage)(ckpt_dir, chunk=b)


def bench(kind, ckpt_dir, device):
    cuda = kind != "bicubic_x2"                     # bicubic is cv2 on the CPU: no sync, no
    out = {}                                        # peak-memory to read, and no autocast
    for (w, h) in SIZES:
        for b in BATCHES:
            stage = _make(kind, ckpt_dir, b)
            stage.load(device if cuda else "cpu")
            frames = [np.random.randint(0, 255, (h, w, 3), dtype=np.uint8) for _ in range(b)]
            key = f"{w}x{h}_b{b}"
            try:
                if cuda:
                    torch.cuda.reset_peak_memory_stats()
                for _ in range(WARMUP):
                    stage(frames)
                if cuda:
                    torch.cuda.synchronize()
                t0, runs = time.time(), 0
                while runs < ITERS or time.time() - t0 < MIN_SEC:
                    stage(frames)
                    runs += 1
                if cuda:
                    torch.cuda.synchronize()
                ms = (time.time() - t0) / runs / b * 1000
                out[key] = dict(ms_per_frame=round(ms, 2), fps=round(1000.0 / ms, 2),
                                iters=runs)
                if cuda:
                    out[key].update(
                        peak_gib=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
                        autocast=stage.autocast)
                else:
                    out[key].update(device="cpu")
            except torch.cuda.OutOfMemoryError as e:
                # Reported, not swallowed: it is the reason CATANet defaults to --sr-chunk 1.
                out[key] = dict(error="OutOfMemoryError", detail=str(e).split("\n")[0][:90])
            stage.close()
            torch.cuda.empty_cache()
            print(f"  {kind:11s} {key:14s} {out[key]}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default=os.path.join(DP, "ckpts"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "sr_speed.json"))
    a = ap.parse_args()
    torch.cuda.set_device(a.device)
    rows = {k: bench(k, a.ckpt_dir, a.device)
            for k in ("bicubic_x2", "lightsr_x2", "catanet_x2")}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(dict(gpu=torch.cuda.get_device_name(a.device), models=rows), f, indent=2)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
