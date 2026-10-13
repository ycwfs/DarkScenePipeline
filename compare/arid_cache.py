"""Pre-enhance ARID clips and cache frames for VideoMamba finetuning/eval.

For each clip: uniformly sample T=32 frames -> enhance with the chosen pipeline stage
(cidnet | retinexformer | off) -> resize short side to 256 (keeping aspect) -> cache as
RGB uint8 npy of shape (32, 256, W, 3), mirroring the clip's relpath. Crop (random 224 for
train, center 224 for eval) + ImageNet norm happen at train/eval time. Shardable across GPUs
via --shard i/n. Enhancement uses the exact pipeline stages so it matches inference.
"""
import os
import sys
import argparse

import cv2
import numpy as np

DP = "/ssd1/wfs/project/lowlight/DarkScenePipeline"
sys.path.insert(0, DP)
from darkpipe.constants import RECO_CFG   # noqa: E402

CKPT_DIR = os.path.join(DP, "ckpts")
T = RECO_CFG["videomamba"]["T"]           # 32
RESIZE = RECO_CFG["videomamba"]["resize"]  # 256


def sample_indices(n, t):
    if n <= 1:
        return [0] * t
    return [min(int(round(i * (n - 1) / (t - 1))), n - 1) for i in range(t)]


def read_frames(path):
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


def short_resize_rgb(bgr):
    h, w = bgr.shape[:2]
    s = RESIZE / min(h, w)
    nh, nw = int(round(h * s)), int(round(w * s))
    interp = cv2.INTER_AREA if s < 0.5 else cv2.INTER_LINEAR
    r = cv2.resize(bgr, (nw, nh), interpolation=interp)
    return cv2.cvtColor(r, cv2.COLOR_BGR2RGB)


def build_enhancer(name, device):
    if name == "off":
        return None
    if name == "cidnet":
        from darkpipe.stages.enhance_cidnet import CIDNetStage
        st = CIDNetStage(CKPT_DIR, chunk=T)
    elif name == "retinexformer":
        from darkpipe.stages.enhance_retinexformer import RetinexformerStage
        st = RetinexformerStage(CKPT_DIR, chunk=T)
    st.load(device)
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_file", required=True)
    ap.add_argument("--clips_root", required=True)
    ap.add_argument("--enhance", choices=["off", "cidnet", "retinexformer"], required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", default="0/1", help="i/n")
    args = ap.parse_args()

    i, nshard = map(int, args.shard.split("/"))
    items = []
    for ln in open(args.split_file):
        ln = ln.strip()
        if ln:
            _, lab, rel = ln.split("\t")
            items.append((rel, int(lab)))
    items = [it for k, it in enumerate(items) if k % nshard == i]

    enh = build_enhancer(args.enhance, args.device)
    done = skipped = bad = 0
    for rel, lab in items:
        out = os.path.join(args.out_dir, rel.replace(".mp4", ".npy"))
        if os.path.exists(out):
            skipped += 1
            continue
        frames = read_frames(os.path.join(args.clips_root, rel))
        if frames is None:
            bad += 1
            continue
        if enh is not None:
            frames = enh(frames)
        arr = np.stack([short_resize_rgb(f) for f in frames]).astype(np.uint8)  # (32,256,W,3)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.save(out, arr)
        done += 1
        if done % 200 == 0:
            print(f"  [{args.enhance} shard{args.shard}] {done} cached", flush=True)
    print(f"[{args.enhance} shard{args.shard}] done={done} skipped={skipped} bad={bad} -> {args.out_dir}")


if __name__ == "__main__":
    main()
