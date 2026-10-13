"""Params / FLOPs / latency for CIDNet and Retinexformer on one GPU.

Latency: warmup then timed loop, torch.cuda.synchronize around the region,
reported as mean ms/frame and FPS at batch=1. Sizes cover the LOLv2-real
resolution (400x600) and a 256x256 reference. FLOPs via thop (fp32).
"""
import os
import sys
import json
import time

import torch
from thop import profile

HVI_DIR = "/ssd1/wfs/project/lowlight/HVI-CIDNet"
RF_VENDOR = "/ssd1/wfs/project/lowlight/DarkScenePipeline/darkpipe/vendor"
NTIRE = "/ssd1/wfs/project/lowlight/DarkScenePipeline/ckpts/NTIRE.pth"
sys.path.insert(0, HVI_DIR)
sys.path.insert(0, RF_VENDOR)
from net.CIDNet import CIDNet            # noqa: E402
from retinexformer_arch import RetinexFormer  # noqa: E402

SIZES = [(400, 600), (256, 256)]
ITERS = 100
WARMUP = 20


def bench(model, name, dtype):
    out = {}
    model = model.cuda().eval()
    if dtype == torch.float16:
        model = model.half()
    for (h, w) in SIZES:
        try:
            x = torch.rand(1, 3, h, w, device="cuda", dtype=dtype)
            with torch.no_grad():
                for _ in range(WARMUP):
                    model(x)
                torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(ITERS):
                    model(x)
                torch.cuda.synchronize()
                dt = (time.time() - t0) / ITERS
            out[f"{h}x{w}"] = {"ms": round(dt * 1000, 3), "fps": round(1.0 / dt, 1)}
        except Exception as e:
            out[f"{h}x{w}"] = {"error": str(e)[:120]}
    return out


def main():
    res = {}

    # ---- CIDNet ----
    cid = CIDNet()
    n_cid = sum(p.numel() for p in cid.parameters())
    x = torch.rand(1, 3, 256, 256)
    macs_cid, _ = profile(cid, inputs=(x,), verbose=False)
    res["CIDNet"] = {
        "params_M": round(n_cid / 1e6, 3),
        "flops_G_256": round(macs_cid / 1e9, 2),
        "fp32": bench(CIDNet(), "CIDNet", torch.float32),
        "fp16": bench(CIDNet(), "CIDNet", torch.float16),
    }

    # ---- Retinexformer ----
    rf = RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1, 2, 2])
    sd = torch.load(NTIRE, map_location="cpu")
    rf.load_state_dict(sd["params"] if "params" in sd else sd, strict=True)
    n_rf = sum(p.numel() for p in rf.parameters())
    macs_rf, _ = profile(RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1,
                                       num_blocks=[1, 2, 2]), inputs=(x,), verbose=False)
    res["Retinexformer"] = {
        "params_M": round(n_rf / 1e6, 3),
        "flops_G_256": round(macs_rf / 1e9, 2),
        "fp32": bench(RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1,
                                    num_blocks=[1, 2, 2]), "RF", torch.float32),
        "fp16": bench(RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1,
                                    num_blocks=[1, 2, 2]), "RF", torch.float16),
    }

    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
