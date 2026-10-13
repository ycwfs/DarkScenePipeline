"""Clip-level ARID action-recognition eval, A/B over the low-light enhancer.

Protocol (held fixed across enhancers so the ONLY variable is enhancement):
  uniformly sample T=32 frames from each split-1 test clip -> enhance them with the
  chosen stage (cidnet | retinexformer | off) -> VideoMamba preprocessing (resize short
  side 256, center-crop 224, ImageNet norm) -> one VideoMamba forward -> argmax.
Reports top-1 accuracy overall + per class, and dumps per-clip predictions so the two
runs can be joined for agreement analysis.

NOTE: VideoMamba was finetuned on Retinexformer(NTIRE)-enhanced ARID, so this A/B is
biased toward retinexformer by construction (train/test enhancement match). Single
center-crop (no multi-clip TTA), so absolute acc sits a bit under the README's TTA 0.688.
"""
import os
import sys
import json
import time
import argparse

import cv2
import numpy as np
import torch

DP = "/ssd1/wfs/project/lowlight/DarkScenePipeline"
sys.path.insert(0, DP)
from darkpipe.constants import CLASSES, RECO_CFG, CKPT_FILES   # noqa: E402

CKPT_DIR = os.path.join(DP, "ckpts")
CLIPS_ROOT = "/ssd1/wfs/project/lowlight/compare/arid/test_clips"
SPLIT = "/ssd1/wfs/project/lowlight/compare/arid/split1_test.txt"


def sample_indices(n, T):
    if n <= 1:
        return [0] * T
    return [min(int(round(i * (n - 1) / (T - 1))), n - 1) for i in range(T)]


def read_frames(path, T):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    if not frames:
        return None
    return [frames[i] for i in sample_indices(len(frames), T)]


def preprocess_vm(frame_bgr, cfg):
    h, w = frame_bgr.shape[:2]
    s = cfg["resize"] / min(h, w)
    nh, nw = int(round(h * s)), int(round(w * s))
    interp = cv2.INTER_AREA if s < 0.5 else cv2.INTER_LINEAR
    r = cv2.resize(frame_bgr, (nw, nh), interpolation=interp)
    top, left = (nh - cfg["size"]) // 2, (nw - cfg["size"]) // 2
    crop = r[top:top + cfg["size"], left:left + cfg["size"]]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return (rgb - cfg["mean"]) / cfg["std"]


def build_enhancer(name, device, despeckle=True):
    if name == "off":
        return None
    if name == "cidnet":
        from darkpipe.stages.enhance_cidnet import CIDNetStage
        st = CIDNetStage(CKPT_DIR, chunk=32, despeckle=despeckle)
    elif name == "retinexformer":
        from darkpipe.stages.enhance_retinexformer import RetinexformerStage
        st = RetinexformerStage(CKPT_DIR, chunk=32)
    else:
        raise ValueError(name)
    st.load(device)
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enhance", choices=["off", "cidnet", "retinexformer"], required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0, help="cap clips (0 = all)")
    ap.add_argument("--no-despeckle", action="store_true", help="cidnet: disable despeckle guard")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    cfg = RECO_CFG["videomamba"]
    T = cfg["T"]

    from darkpipe.vendor.videomamba import videomamba_tiny
    net = videomamba_tiny(num_classes=len(CLASSES), num_frames=T, img_size=224)
    net.load_state_dict(torch.load(os.path.join(CKPT_DIR, CKPT_FILES["videomamba"]),
                                   map_location="cpu")["model"])
    net = net.to(args.device).eval()
    enhancer = build_enhancer(args.enhance, args.device, despeckle=not args.no_despeckle)

    items = []
    for ln in open(SPLIT):
        ln = ln.strip()
        if ln:
            _, lab, rel = ln.split("\t")
            items.append((rel, int(lab)))
    if args.limit:
        items = items[:args.limit]

    preds = []
    correct = 0
    n = 0
    bad = 0
    t0 = time.time()
    for rel, gt in items:
        path = os.path.join(CLIPS_ROOT, rel)
        frames = read_frames(path, T)
        if frames is None:
            bad += 1
            continue
        if enhancer is not None:
            frames = enhancer(frames)
        arr = np.stack([preprocess_vm(f, cfg) for f in frames])          # T,H,W,C
        x = torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0).contiguous().to(args.device)
        with torch.autocast("cuda", dtype=torch.float16):
            prob = net(x).softmax(1)[0].float().cpu().numpy()
        pred = int(prob.argmax())
        conf = float(prob[pred])
        preds.append({"rel": rel, "gt": gt, "pred": pred, "conf": round(conf, 4)})
        correct += int(pred == gt)
        n += 1
        if n % 100 == 0:
            print(f"  {args.enhance} {n}/{len(items)}  running top1={correct/n:.4f}", flush=True)

    # per-class accuracy
    per_cls = {}
    for c in range(len(CLASSES)):
        idxs = [p for p in preds if p["gt"] == c]
        if idxs:
            per_cls[CLASSES[c]] = round(sum(p["pred"] == c for p in idxs) / len(idxs), 4)
    summary = {
        "enhance": args.enhance, "n": n, "bad": bad,
        "top1": round(correct / max(n, 1), 4),
        "per_class": per_cls,
        "seconds": round(time.time() - t0, 1),
        "preds": preds,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f)
    print(f"[{args.enhance}] top1={summary['top1']}  n={n}  bad={bad}  "
          f"{summary['seconds']}s -> {args.out}")


if __name__ == "__main__":
    main()
