"""Finetune the shipped behavior head on the real deployment action videos.

The shipped head speaks ten labels; the real footage supervises only the classes it has
videos for (2026-08-20: drink/throw/wave/pick_up/fall from one take each; 2026-08-25 adds
multi-take drink/wave and the first real `chase` data). Every class *without* supervision --
`fight`, `talk`, `shake_hands` and above all `other` -- gets its logits actively pushed down
on every plain cross-entropy step: after a few epochs the model is an N-way classifier
wearing a ten-way head. `other` matters most -- it is what keeps ordinary activity from being
reported as a behavior -- and it is the one with the most to lose.

Two things hold that line, and both are needed:

1. **Learning-without-forgetting.** A frozen copy of the shipped checkpoint scores every batch,
   and a KL term pulls the student's full ten-way distribution back towards it. The five
   unsupervised classes therefore still have a gradient telling them where they used to be.
   `--lwf 0` turns it off, which is how the ablation in the report was produced.

2. **A backbone LR 10x below the head's.** 242 clips from five continuous videos of one scene
   is not enough to move a 3D backbone anywhere good; it is more than enough to overfit it to
   this room's lighting. The head is the part that should absorb a domain this narrow.

Selection is NOT on the new data's accuracy. An epoch that wins the five real classes by
trading away ARID is a regression dressed as an improvement, so the selected epoch has to
clear a floor on the held-out ARID test split (real dark footage, same retinexformer chain,
labelled for drink/pick_up/wave/other) before its realact score counts at all.
"""
import argparse
import copy
import glob
import json
import math
import os
import random
import sys
import time
from collections import Counter

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# cv2 defaults to one thread per core; with several concurrent trainers the dataloader
# workers thrash each other and per-window cost degrades >10x. Keep workers single-threaded.
cv2.setNumThreads(0)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DP = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", ".."))
sys.path.insert(0, DP)
import behavior_manifest as BM                                     # noqa: E402
from darkpipe.constants import IN_MEAN, IN_STD                     # noqa: E402
from darkpipe.vendor.videomamba import videomamba_tiny             # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REALACT = os.path.join(HERE, "cache", "realact_retinexformer")
ARID_CACHE = os.path.normpath(os.path.join(BM.ROOT, "..", "arid", "cache"))
ARID_TEST = os.path.normpath(os.path.join(BM.ROOT, "..", "arid", "split1_test.txt"))
SHIPPED = os.path.join(DP, "ckpts", "videomamba_t_behavior_32f.pth")
CROP, T = 224, 32
# --fullfov: aspect-preserving whole-frame input (no crop). 224 short side -> 400 wide for a
# 16:9 cache frame (456x256), rounded to a multiple of patch 16; 25x14=350 tokens vs 196.
FOV_H, FOV_W = 224, 400
NB = len(BM.BEHAVIORS)


def realact_items(data_dir, split):
    out = []
    for p in sorted(glob.glob(os.path.join(data_dir, split, "*", "*.npy"))):
        out.append((p, BM.BID[os.path.basename(os.path.dirname(p))]))
    return out


def arid_items(limit=0, seed=0):
    """Held-out ARID test clips, retinexformer-cached -- the regression set, never trained on."""
    root = os.path.join(ARID_CACHE, "test_rf")
    out = []
    for ln in open(ARID_TEST):
        ln = ln.strip()
        if not ln:
            continue
        rel = ln.split("\t")[2]
        beh = BM.ARID_TO_BEHAVIOR.get(rel.split("/")[0])
        p = os.path.join(root, rel.replace(".mp4", ".npy"))
        if beh and os.path.exists(p):
            out.append((p, BM.BID[beh]))
    if limit and len(out) > limit:
        out = random.Random(seed).sample(out, limit)
    return out


class ClipCache(Dataset):
    """Identical preprocessing to behavior_train.ClipCache -- same crop, flip and normalize,
    so a checkpoint from here is interchangeable with one from there.

    squash=True is the "no crop" experiment: the whole stored frame is anisotropically
    resized to CROP x CROP instead of cropped, so the left/right quarters that the deployed
    center crop discards stay visible -- at the price of people compressed ~2x horizontally
    and everyone rendered smaller. A checkpoint trained this way is NOT interchangeable with
    the crop chain; deploying it means switching preprocess_frame the same way.

    fov=True keeps the aspect ratio instead of squashing: whole frame resized to FOV_W x FOV_H
    (400x224 for 16:9). Same interchangeability caveat as squash, and the model must be built
    with img_size=(FOV_H, FOV_W) -- see load_net.
    """

    def __init__(self, items, train, squash=False, fov=False):
        self.items, self.train, self.squash, self.fov = items, train, squash, fov

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, lab = self.items[i]
        arr = np.load(p)                                    # (32,256,W,3) uint8 RGB
        _, H, W, _ = arr.shape
        if self.fov:
            crop = np.stack([cv2.resize(f, (FOV_W, FOV_H), interpolation=cv2.INTER_AREA)
                             for f in arr])
        elif self.squash:
            crop = np.stack([cv2.resize(f, (CROP, CROP), interpolation=cv2.INTER_AREA)
                             for f in arr])
        else:
            if self.train:
                top, left = random.randint(0, H - CROP), random.randint(0, W - CROP)
            else:
                top, left = (H - CROP) // 2, (W - CROP) // 2
            crop = arr[:, top:top + CROP, left:left + CROP, :]
        if self.train and random.random() < 0.5:
            crop = crop[:, :, ::-1, :]
        # in-place normalize: each fresh 34MB temporary here is thousands of page faults,
        # and under several concurrent trainers that fault storm degrades the box >10x
        crop = np.ascontiguousarray(crop).astype(np.float32)
        crop /= 255.0
        crop -= IN_MEAN
        crop /= IN_STD
        return torch.from_numpy(crop).permute(3, 0, 1, 2), lab


def balanced_sampler(items):
    cnt = Counter(lab for _, lab in items)
    w = [1.0 / cnt[lab] for _, lab in items]
    return WeightedRandomSampler(w, num_samples=len(items), replacement=True)


def load_net(path, device, fov=False):
    img = (FOV_H, FOV_W) if fov else 224
    net = videomamba_tiny(num_classes=NB, num_frames=T, img_size=img)
    sd = torch.load(path, map_location="cpu")["model"]
    gh, gw = FOV_H // 16, FOV_W // 16
    if fov:
        pe = sd["pos_embed"].float()
        if pe.shape[1] == 1 + gh * gw:
            # checkpoint is already on the fov grid (e.g. init from realact11_fov) -- keep it
            sd["pos_embed"] = pe
        else:
            # crop checkpoints carry a 14x14 grid pos embed; bilinear-fit it onto the fov grid
            cls, sp = pe[:, :1], pe[:, 1:]
            g = int(round(sp.shape[1] ** 0.5))
            sp = sp.reshape(1, g, g, -1).permute(0, 3, 1, 2)
            sp = F.interpolate(sp, size=(gh, gw), mode="bilinear", align_corners=False)
            sp = sp.permute(0, 2, 3, 1).reshape(1, gh * gw, -1)
            sd["pos_embed"] = torch.cat([cls, sp], 1)
    net.load_state_dict(sd, strict=True)
    return net.to(device)


def warmup_cosine(opt, total, warm):
    def f(s):
        if s < warm:
            return (s + 1) / max(1, warm)
        t = (s - warm) / max(1, total - warm)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def metrics(cm):
    tp = np.diag(cm).astype(float)
    f1 = 2 * tp / np.maximum(cm.sum(0) + cm.sum(1), 1e-9)
    present = cm.sum(1) > 0
    return float(tp.sum() / max(cm.sum(), 1)), float(f1[present].mean())


@torch.no_grad()
def evaluate(net, loader, device):
    net.eval()
    cm = np.zeros((NB, NB), np.int64)
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            pred = net(x).argmax(1).cpu().numpy()
        for t, p in zip(y.numpy(), pred):
            cm[t, p] += 1
    return (*metrics(cm), cm)


def report(cm, tag):
    tp = np.diag(cm).astype(float)
    rec = tp / np.maximum(cm.sum(1), 1e-9)
    f1 = 2 * tp / np.maximum(cm.sum(0) + cm.sum(1), 1e-9)
    print(f"  [{tag}] " + "  ".join(
        f"{b}:r{rec[i]:.2f}/f{f1[i]:.2f}(n{int(cm[i].sum())})"
        for i, b in enumerate(BM.BEHAVIORS) if cm[i].sum()), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(DP, "ckpts",
                                                  "videomamba_t_behavior_32f_realact.pth"))
    ap.add_argument("--data", default=REALACT, help="realact_cache.py 的输出目录")
    ap.add_argument("--squash", action="store_true",
                    help="整帧压扁到 224x224 训练/评估（不裁剪），代替 256 短边+224 中心裁剪")
    ap.add_argument("--fullfov", action="store_true",
                    help=f"整帧等比缩放到 {FOV_W}x{FOV_H} 训练/评估（不裁剪、不压扁），"
                         "代替 256 短边+224 中心裁剪")
    ap.add_argument("--init", default=SHIPPED)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5, help="head LR; backbone gets lr*backbone-mult")
    ap.add_argument("--backbone-mult", type=float, default=0.1)
    ap.add_argument("--lwf", type=float, default=2.0, help="0 disables the anti-forgetting term")
    ap.add_argument("--lwf-temp", type=float, default=2.0)
    ap.add_argument("--arid-floor", type=float, default=0.02,
                    help="max macro-F1 the ARID regression set may lose vs the init")
    ap.add_argument("--arid-eval", type=int, default=500,
                    help="per-epoch ARID subsample; 0=all; -1=skip entirely (ARID ungated)")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda:4")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed), random.seed(a.seed), np.random.seed(a.seed)

    tr, va = realact_items(a.data, "train"), realact_items(a.data, "val")
    ar = [] if a.arid_eval < 0 else arid_items(a.arid_eval, a.seed)
    if not tr or not va:
        sys.exit(f"error: 缓存为空，先跑 realact_cache.py（{a.data}）")
    c = Counter(l for _, l in tr)
    # what the cache actually supervises; the complement is what LwF has to protect
    seen = [b for b in BM.BEHAVIORS if c[BM.BID[b]]]
    print(f"[realact] train={len(tr)} val={len(va)} arid_regression={len(ar)} -> "
          f"{os.path.basename(a.out)}")
    print("  train per class: " + "  ".join(f"{b}={c[BM.BID[b]]}" for b in seen), flush=True)

    tl = DataLoader(ClipCache(tr, True, a.squash, a.fullfov), batch_size=a.bs,
                    sampler=balanced_sampler(tr),
                    num_workers=a.workers, pin_memory=True, drop_last=True)
    vl = DataLoader(ClipCache(va, False, a.squash, a.fullfov), batch_size=8,
                    num_workers=a.workers, pin_memory=True)
    al = None if a.arid_eval < 0 else DataLoader(ClipCache(ar, False, a.squash, a.fullfov),
                                                 batch_size=8,
                                                 num_workers=a.workers, pin_memory=True)

    net = load_net(a.init, a.device, fov=a.fullfov)
    teacher = copy.deepcopy(net).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    head = [p for n, p in net.named_parameters() if n.startswith("head.")]
    body = [p for n, p in net.named_parameters() if not n.startswith("head.")]
    opt = torch.optim.AdamW([{"params": body, "lr": a.lr * a.backbone_mult},
                             {"params": head, "lr": a.lr}], weight_decay=0.05)
    sched = warmup_cosine(opt, a.epochs * len(tl), len(tl))
    scaler = torch.amp.GradScaler("cuda")
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    def score():
        rt, rf, rcm = evaluate(net, vl, a.device)
        if al is None:
            return rt, rf, rcm, 0.0, 0.0, None
        at, af, acm = evaluate(net, al, a.device)
        return rt, rf, rcm, at, af, acm

    rt, rf, rcm, at, af, acm = score()
    print(f"[ep0] (init, 未训练) realact top1={rt:.4f} mF1={rf:.4f} | "
          f"ARID top1={at:.4f} mF1={af:.4f}", flush=True)
    report(rcm, "ep0 realact")
    if acm is not None:
        report(acm, "ep0 arid")
    base_rf, base_af = rf, af
    hist = [dict(epoch=0, loss=None, realact_top1=rt, realact_macro_f1=rf,
                 arid_top1=at, arid_macro_f1=af)]
    best, best_ep = -1.0, 0
    for ep in range(1, a.epochs + 1):
        net.train()
        t0, tot, tkl = time.time(), 0.0, 0.0
        for x, y in tl:
            x, y = x.to(a.device, non_blocking=True), y.to(a.device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = net(x)
                loss = crit(logits, y)
                if a.lwf > 0:
                    with torch.no_grad():
                        tl_logits = teacher(x)
                    tau = a.lwf_temp
                    kl = F.kl_div(F.log_softmax(logits / tau, 1),
                                  F.log_softmax(tl_logits / tau, 1),
                                  log_target=True, reduction="batchmean") * tau * tau
                    tkl += kl.item()
                    loss = loss + a.lwf * kl
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        rt, rf, rcm, at, af, acm = score()
        # an epoch only counts if ARID has not fallen more than the floor allows
        ok = af >= base_af - a.arid_floor
        flag = ""
        if ok and rf > best:
            best, best_ep = rf, ep
            torch.save({"model": net.state_dict(), "labels": BM.BEHAVIORS, "epoch": ep,
                        "enhance": "retinexformer", "init": os.path.basename(a.init),
                        "preprocess": ("fullfov%dx%d" % (FOV_H, FOV_W)) if a.fullfov else
                                      ("squash224" if a.squash else "resize256_centercrop224"),
                        "source": "real-actiondata", "seen": seen,
                        "realact_top1": rt, "realact_macro_f1": rf,
                        "arid_top1": at, "arid_macro_f1": af, "lwf": a.lwf}, a.out)
            flag = " *best -> saved"
        elif not ok:
            flag = f" (ARID 掉了 {base_af-af:.4f} > {a.arid_floor}，不保存)"
        hist.append(dict(epoch=ep, loss=tot / len(tl), kl=tkl / len(tl),
                         realact_top1=rt, realact_macro_f1=rf,
                         arid_top1=at, arid_macro_f1=af))
        print(f"[ep{ep}] loss={tot/len(tl):.3f} kl={tkl/max(1,len(tl)):.3f} "
              f"realact top1={rt:.4f} mF1={rf:.4f} | "
              + (f"ARID top1={at:.4f} mF1={af:.4f} " if acm is not None else "(ARID skipped) ")
              + f"({time.time()-t0:.0f}s){flag}", flush=True)
        report(rcm, f"ep{ep} realact")
        if acm is not None:
            report(acm, f"ep{ep} arid")
    if best > 0:
        print(f"[realact] DONE 最佳 ep{best_ep} realact mF1={best:.4f} "
              f"(init {base_rf:.4f}) -> {a.out}", flush=True)
    else:
        print(f"[realact] DONE 没有一个 epoch 同时守住 ARID 且超过 init，未写出权重", flush=True)
    json.dump(hist, open(a.out.replace(".pth", "_hist.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
