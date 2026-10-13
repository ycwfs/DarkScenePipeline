"""Train the COCO-17 ST-GCN on skeleton windows extracted from the realact RGB cache.

Same splits / window alignment as the RGB track (skeleton_cache.py ran on the stored
window frames). From-scratch training: balanced sampler, label smoothing, heavy geometric
augmentation, selection on val macro-F1. The person gate is NOT learned here -- it is a
threshold on per-window presence computed at fusion time.

Usage: python -u skel_train.py [--epochs 80] [--out ../cache/skel_v1/stgcn.pth]
"""
import argparse
import glob
import json
import os
import random
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stgcn import STGCN, augment, merge_persons, normalize_window

CLASSES = ["wave", "throw", "chase", "fall", "fight", "talk", "drink", "pick_up",
           "shake_hands", "other"]
CID = {c: i for i, c in enumerate(CLASSES)}


def items_of(cache, split, min_pres=0.5):
    """Window items; drop no-person windows (all-zero slots) -- they are unclassifiable
    label noise (empty 'chase' vs empty 'other' share the same input); the fusion-time
    presence gate owns the no-person case instead."""
    out = []
    for p in sorted(glob.glob(os.path.join(cache, split, "*", "*.npy"))):
        sk = np.load(p)
        if sk.size and float(sk[..., 2].mean(-1).max()) < min_pres:
            continue
        out.append((p, CID[os.path.basename(os.path.dirname(p))]))
    return out


class SkelDS(Dataset):
    def __init__(self, items, train):
        self.items, self.train = items, train
        self.rng = np.random.RandomState(0)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, lab = self.items[i]
        x, pres = normalize_window(np.load(p))
        if self.train:
            x = augment(x, self.rng)
        return torch.from_numpy(x), torch.tensor(np.asarray(pres), dtype=torch.float32), lab


def balanced_sampler(items):
    cnt = Counter(l for _, l in items)
    return WeightedRandomSampler([1 / cnt[l] for _, l in items], len(items), replacement=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="../cache/skel_v1",
                    help="skeleton cache dir(s), comma-separated to merge")
    ap.add_argument("--out", default="../cache/skel_v1/stgcn.pth")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed), random.seed(a.seed), np.random.seed(a.seed)

    tr, va = [], []
    for cache in a.cache.split(","):            # merge caches (e.g. real + proxy skeletons)
        tr += items_of(cache, "train")
        va += items_of(cache, "val")
    if not tr:
        sys.exit(f"error: 骨架缓存为空，先跑 skeleton_cache.py（{a.cache}）")
    c = Counter(l for _, l in tr)
    print(f"[skelt] train={len(tr)} val={len(va)} per class: "
          + "  ".join(f"{CLASSES[k]}={c[k]}" for k in sorted(c)), flush=True)

    tl = DataLoader(SkelDS(tr, True), a.bs, sampler=balanced_sampler(tr),
                    num_workers=4, pin_memory=True, drop_last=True)
    vl = DataLoader(SkelDS(va, False), 32, num_workers=4, pin_memory=True)

    net = STGCN(num_classes=len(CLASSES)).to(a.device)
    print(f"[skelt] params={sum(p.numel() for p in net.parameters())/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * len(tl))
    crit = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    def evaluate():
        net.eval()
        cm = np.zeros((len(CLASSES),) * 2, np.int64)
        with torch.no_grad():
            for x, pres, y in vl:
                x, pres = x.to(a.device), pres.to(a.device)
                lg = torch.stack([net(x[:, m]) for m in range(x.shape[1])], 1)
                pred = merge_persons(lg, pres).argmax(1).cpu().numpy()
                for t, q in zip(y.numpy(), pred):
                    cm[t, q] += 1
        tp = np.diag(cm).astype(float)
        rec = tp / np.maximum(cm.sum(1), 1e-9)
        prec = tp / np.maximum(cm.sum(0), 1e-9)
        f1 = 2 * rec * prec / np.maximum(rec + prec, 1e-9)
        present = cm.sum(1) > 0
        return float(tp.sum() / max(cm.sum(), 1)), float(f1[present].mean()), cm

    best, best_ep = -1.0, 0
    for ep in range(1, a.epochs + 1):
        net.train()
        t0, tot = time.time(), 0.0
        for x, pres, y in tl:
            x, pres, y = x.to(a.device), pres.to(a.device), y.to(a.device)
            opt.zero_grad(set_to_none=True)
            lg = torch.stack([net(x[:, m]) for m in range(x.shape[1])], 1)
            loss = crit(merge_persons(lg, pres), y)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        top, mf1, cm = evaluate()
        flag = ""
        if mf1 > best:
            best, best_ep = mf1, ep
            torch.save({"model": net.state_dict(), "classes": CLASSES, "epoch": ep,
                        "val_top1": top, "val_macro_f1": mf1}, a.out)
            flag = " *best"
        print(f"[ep{ep}] loss={tot/len(tl):.3f} val top1={top:.4f} mF1={mf1:.4f} "
              f"({time.time()-t0:.0f}s){flag}", flush=True)
        if ep == a.epochs or ep % 10 == 0:
            tp = np.diag(cm).astype(float)
            rec = tp / np.maximum(cm.sum(1), 1e-9)
            f1 = 2 * tp / np.maximum(cm.sum(0) + cm.sum(1), 1e-9)
            print("   " + "  ".join(f"{CLASSES[i]}:r{rec[i]:.2f}/f{f1[i]:.2f}(n{int(cm[i].sum())})"
                                    for i in range(len(CLASSES)) if cm[i].sum()), flush=True)
    print(f"[skelt] DONE best ep{best_ep} val mF1={best:.4f} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
