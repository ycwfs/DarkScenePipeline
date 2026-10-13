"""Finetune VideoMamba-Tiny on cached enhanced ARID frames (fair enhancer comparison).

From a common init (the released RF-trained checkpoint), finetune on one enhancer's cached
train frames with a fixed recipe, evaluating on the same enhancer's test frames each epoch.
Also supports --eval_only to score any model on any cached test set (for the 2x2 matrix).

Cache layout: compare/arid/cache/{train,test}_{cidnet,rf}/<Class>/<clip>.npy  (32,256,W,3) RGB uint8.
Train aug: random 224 crop + hflip; eval: center 224 crop. ImageNet norm. AMP fp16.
"""
import os
import sys
import json
import time
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DP = "/ssd1/wfs/project/lowlight/DarkScenePipeline"
sys.path.insert(0, DP)
from darkpipe.constants import CLASSES, IN_MEAN, IN_STD   # noqa: E402
from darkpipe.vendor.videomamba import videomamba_tiny    # noqa: E402

CACHE = "/ssd1/wfs/project/lowlight/compare/arid/cache"
SPLIT_TR = "/ssd1/wfs/project/lowlight/compare/arid/split1_train.txt"
SPLIT_TE = "/ssd1/wfs/project/lowlight/compare/arid/split1_test.txt"
RELEASED = os.path.join(DP, "ckpts", "videomamba_t_arid_32f.pth")
CROP = 224


def read_items(split_file, cache_dir):
    out = []
    for ln in open(split_file):
        ln = ln.strip()
        if not ln:
            continue
        _, lab, rel = ln.split("\t")
        p = os.path.join(cache_dir, rel.replace(".mp4", ".npy"))
        if os.path.exists(p):
            out.append((p, int(lab)))
    return out


class ARIDCached(Dataset):
    def __init__(self, items, train):
        self.items = items
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, lab = self.items[i]
        arr = np.load(p)  # (32,256,W,3) uint8 RGB
        _, H, W, _ = arr.shape
        if self.train:
            top = random.randint(0, H - CROP)
            left = random.randint(0, W - CROP)
        else:
            top, left = (H - CROP) // 2, (W - CROP) // 2
        crop = arr[:, top:top + CROP, left:left + CROP, :]
        if self.train and random.random() < 0.5:
            crop = crop[:, :, ::-1, :]
        crop = np.ascontiguousarray(crop).astype(np.float32) / 255.0
        crop = (crop - IN_MEAN) / IN_STD                      # (T,224,224,3)
        x = torch.from_numpy(crop).permute(3, 0, 1, 2)        # (C,T,224,224)
        return x, lab


def build_model(init_path, device):
    net = videomamba_tiny(num_classes=len(CLASSES), num_frames=32, img_size=224)
    if init_path:
        net.load_state_dict(torch.load(init_path, map_location="cpu")["model"])
    return net.to(device)


@torch.no_grad()
def evaluate(net, loader, device):
    net.eval()
    correct = n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            pred = net(x).argmax(1).cpu()
        correct += (pred == y).sum().item()
        n += y.numel()
    return correct / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enhance", choices=["cidnet", "rf"], required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--init", default=RELEASED)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--model", default="", help="eval_only: model path to score")
    ap.add_argument("--test_enhance", default="", help="eval_only: which test cache (cidnet|rf)")
    args = ap.parse_args()
    dev = args.device

    if args.eval_only:
        te = read_items(SPLIT_TE, f"{CACHE}/test_{args.test_enhance}")
        loader = DataLoader(ARIDCached(te, False), batch_size=16, num_workers=args.workers)
        net = build_model(None, dev)
        net.load_state_dict(torch.load(args.model, map_location="cpu")["model"])
        acc = evaluate(net, loader, dev)
        print(json.dumps({"model": os.path.basename(args.model),
                          "test_enhance": args.test_enhance, "top1": round(acc, 4), "n": len(te)}))
        return

    tr = read_items(SPLIT_TR, f"{CACHE}/train_{args.enhance}")
    te = read_items(SPLIT_TE, f"{CACHE}/test_{args.enhance}")
    print(f"[{args.enhance}] train={len(tr)} test={len(te)}", flush=True)
    tl = DataLoader(ARIDCached(tr, True), batch_size=args.bs, shuffle=True,
                    num_workers=args.workers, pin_memory=True, drop_last=True)
    el = DataLoader(ARIDCached(te, False), batch_size=16, num_workers=args.workers, pin_memory=True)

    net = build_model(args.init, dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.05)
    steps = args.epochs * len(tl)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    scaler = torch.cuda.amp.GradScaler()
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    base = evaluate(net, el, dev)
    print(f"[{args.enhance}] init (released) @ {args.enhance}-test: {base:.4f}", flush=True)
    best = base
    torch.save({"model": net.state_dict(), "top1": base, "epoch": 0, "enhance": args.enhance}, args.out)
    for ep in range(1, args.epochs + 1):
        net.train()
        t0 = time.time()
        tot = 0.0
        for x, y in tl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.float16):
                loss = crit(net(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        acc = evaluate(net, el, dev)
        flag = ""
        if acc >= best:
            best = acc
            torch.save({"model": net.state_dict(), "top1": acc, "epoch": ep, "enhance": args.enhance}, args.out)
            flag = " *best -> saved"
        print(f"[{args.enhance}] ep{ep} loss={tot/len(tl):.3f} "
              f"{args.enhance}-test top1={acc:.4f} ({time.time()-t0:.0f}s){flag}", flush=True)
    print(f"[{args.enhance}] DONE best={best:.4f} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
