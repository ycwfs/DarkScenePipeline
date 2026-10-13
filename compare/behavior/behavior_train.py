"""Finetune VideoMamba-Tiny into the 10-class behavior recognizer.

Same recipe that was already validated on ARID (`compare/arid_train.py`): AdamW 1e-4 / wd
0.05 / cosine / label-smoothing 0.1 / random-224-crop + hflip / AMP fp16, over cached
(32,256,W,3) uint8 RGB frames. Three things differ, all forced by the data:

1. **Init from K400, not from the ARID checkpoint.** The ARID model's head speaks ARID-11 and
   its backbone saw 11 classes of one dataset; K400 (`videomamba_t16_k400_f32_res224.pth`,
   400 classes, already 32 frames at 224) is the broader starting point. Its head is dropped.

2. **Class-balanced sampling.** Class sizes span 28 (`chase`) to 978 (`other`) -- 35x. Plain
   shuffling would spend 50% of every epoch on `other`. A WeightedRandomSampler equalizes the
   expected draws per class; the loss stays unweighted so it is not corrected twice.

3. **Model selection on macro-F1, not top-1.** With `other` at half the validation set, top-1
   rewards a model that answers `other` to everything (0.52 top-1, 0.07 macro-F1). Selection
   runs on the VALIDATION split; test is never touched here.

`--arid` mixes in real dark ARID clips for the four behaviors ARID actually contains
(drink / pick_up / wave / other). Those frames are genuinely dark rather than simulated, so
they anchor the model against any residual mismatch in the darkening model.

`--k700` mixes in darkened Kinetics-700, which is the only source with a named class for all
nine behaviors and therefore the only way `chase` (28 weak HMDB clips) becomes trainable. With
it, `--init behavior` continues from the shipped checkpoint instead of restarting from K400 --
the head is then kept rather than re-initialized, so lower `--lr`, no `--head-lr-mult` and a
shorter warmup are the matching settings.

`--darkact` mixes in real-dark DarkAct (median frame luma 3.5/255, ~3x darker than ARID), the
only source that supplies `talk` on genuinely dark footage. It covers 4 of the 10 behaviors;
`throw`/`chase`/`fall`/`fight`/`shake_hands` get nothing from it and are the regression risk.
"""
import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DP = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", ".."))
sys.path.insert(0, DP)
import behavior_manifest as BM                                     # noqa: E402
from darken import clip_seed                                       # noqa: E402
from darkpipe.constants import IN_MEAN, IN_STD                     # noqa: E402
from darkpipe.vendor.videomamba import videomamba_tiny             # noqa: E402

CACHE = os.path.join(BM.ROOT, "cache")
ARID_CACHE = os.path.normpath(os.path.join(BM.ROOT, "..", "arid", "cache"))
ARID_SPLIT = os.path.normpath(os.path.join(BM.ROOT, "..", "arid", "split1_train.txt"))
K400 = os.path.join(DP, "ckpts", "videomamba_t16_k400_f32_res224.pth")
ARID_INIT = os.path.join(DP, "ckpts", "videomamba_t_arid_32f.pth")
BEHAVIOR_INIT = os.path.join(DP, "ckpts", "videomamba_t_behavior_32f.pth")
INITS = {"k400": K400, "arid": ARID_INIT, "behavior": BEHAVIOR_INIT}
CROP, T = 224, 32
# the two caches were built by different scripts and name the enhancer differently
ARID_TAG = {"retinexformer": "rf", "cidnet": "cidnet"}


def hmdb_items(split, enhance):
    """-> [(npy_path, label_id)] for one split, skipping clips the cache does not have yet."""
    root = os.path.join(CACHE, f"hmdb_dark_{enhance}")
    out, missing = [], 0
    for path, beh, cls, sp in BM.build(split):
        p = os.path.join(root, sp, cls, os.path.basename(path).replace(".mp4", ".npy"))
        if os.path.exists(p):
            out.append((p, BM.BID[beh]))
        else:
            missing += 1
    return out, missing


def k700_items(split, enhance):
    """-> ([(npy_path, label_id)], n_missing) for one K700 split.

    Kinetics is the only source with a class per behavior, so this is where `chase` (550 clips
    against HMDB's 28 weak ones) and the other rare classes actually come from.
    """
    root = os.path.join(CACHE, f"k700_dark_{enhance}")
    out, missing = [], 0
    for path, beh, slug, sp in BM.k700_rows(split):
        p = os.path.join(root, sp, slug, os.path.basename(path).replace(".mp4", ".npy"))
        if os.path.exists(p):
            out.append((p, BM.BID[beh]))
        else:
            missing += 1
    return out, missing


def darkact_items(enhance):
    """-> (train, val, n_missing) over DarkAct's train split. Real dark, never simulated.

    DarkAct ships train/test only, so validation is carved out of TRAIN: 10% by
    `clip_seed(path) % 10 == 0`, the same crc32-of-basename rule the darkening uses. Stable
    across runs and independent of list order, so a clip is never in train one run and val the
    next. The official TEST split is never read here — it is the acceptance set.

    `other` is capped (`DARKACT_OTHER_CAP`); the cache holds all 9040 clips regardless, so
    raising the cap is a retrain and not a rebuild.
    """
    root = os.path.join(CACHE, f"darkact_{enhance}")
    tr, va, missing = [], [], 0
    for path, beh, cls, sp in BM.darkact_rows("train"):
        p = os.path.join(root, sp, cls, os.path.basename(path).replace(".mp4", ".npy"))
        if not os.path.exists(p):
            missing += 1
            continue
        (va if clip_seed(path) % 10 == 0 else tr).append((p, BM.BID[beh]))
    return tr, va, missing


def arid_items(enhance):
    """Real dark supervision for the 4 behaviors ARID covers. Uses ARID's own train split."""
    root = os.path.join(ARID_CACHE, f"train_{ARID_TAG[enhance]}")
    out = []
    for ln in open(ARID_SPLIT):
        ln = ln.strip()
        if not ln:
            continue
        rel = ln.split("\t")[2]
        beh = BM.ARID_TO_BEHAVIOR.get(rel.split("/")[0])
        p = os.path.join(root, rel.replace(".mp4", ".npy"))
        if beh and os.path.exists(p):
            out.append((p, BM.BID[beh]))
    return out


class ClipCache(Dataset):
    def __init__(self, items, train):
        self.items, self.train = items, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, lab = self.items[i]
        arr = np.load(p)                                    # (32,256,W,3) uint8 RGB
        _, H, W, _ = arr.shape
        if self.train:
            top, left = random.randint(0, H - CROP), random.randint(0, W - CROP)
        else:
            top, left = (H - CROP) // 2, (W - CROP) // 2
        crop = arr[:, top:top + CROP, left:left + CROP, :]
        if self.train and random.random() < 0.5:
            crop = crop[:, :, ::-1, :]
        crop = np.ascontiguousarray(crop).astype(np.float32) / 255.0
        crop = (crop - IN_MEAN) / IN_STD
        return torch.from_numpy(crop).permute(3, 0, 1, 2), lab   # (C,T,224,224)


def balanced_sampler(items):
    """Equal expected draws per class, so `chase` (28) is seen as often as `other` (978)."""
    cnt = Counter(lab for _, lab in items)
    w = [1.0 / cnt[lab] for _, lab in items]
    return WeightedRandomSampler(w, num_samples=len(items), replacement=True)


def build_model(device, init=K400, num_classes=len(BM.BEHAVIORS), resume=""):
    net = videomamba_tiny(num_classes=num_classes, num_frames=T, img_size=224)
    if resume:
        net.load_state_dict(torch.load(resume, map_location="cpu")["model"])
    elif init:
        sd = torch.load(init, map_location="cpu")
        sd = sd.get("model", sd)
        # The head is kept only when it already speaks this exact vocabulary. K400 (400
        # classes) and the ARID checkpoint (11) get theirs dropped and re-initialized; a
        # previous behavior checkpoint (10) keeps its trained head, which is the whole point
        # of continuing from one rather than restarting.
        tgt = net.state_dict()
        sd = {k: v for k, v in sd.items()
              if not k.startswith("head.") or tgt[k].shape == v.shape}
        kept_head = any(k.startswith("head.") for k in sd)
        miss, unexp = net.load_state_dict(sd, strict=False)
        assert not unexp, f"unexpected keys in {init}: {unexp[:5]}"
        assert all(k.startswith("head.") for k in miss), f"missing beyond head: {miss[:5]}"
        print(f"[init] {os.path.basename(init)} head="
              f"{'kept' if kept_head else 'reinitialized'}", flush=True)
    return net.to(device)


def param_groups(net, lr, head_mult):
    """A 10x LR on the randomly-initialized head.

    Measured on a 64-clip overfit probe: matched-LR reached loss 1.27 in 8 epochs, the 10x
    head reached 1.09. The backbone arrives pretrained and the head arrives random, so one
    LR cannot suit both -- a slow head also drags the backbone while it catches up.
    """
    head = [p for n, p in net.named_parameters() if n.startswith("head.")]
    body = [p for n, p in net.named_parameters() if not n.startswith("head.")]
    return [{"params": body, "lr": lr}, {"params": head, "lr": lr * head_mult}]


def warmup_cosine(opt, total_steps, warmup_steps):
    """Linear warmup then cosine, per step. Warmup matters here because the head is random:
    the first updates would otherwise push large gradients through a pretrained backbone."""
    def f(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def metrics(cm):
    """(top1, macro_f1) from a confusion matrix. Split out so the union of several validation
    sets is scored on the SUMMED matrix, not on an average of their macro-F1s -- averaging
    would weight a 50-clip source the same as a 1800-clip one."""
    tp = np.diag(cm).astype(float)
    f1 = 2 * tp / np.maximum(cm.sum(0) + cm.sum(1), 1e-9)
    present = cm.sum(1) > 0                       # classes with no support don't count
    # plain floats: numpy scalars in a checkpoint make torch.load(weights_only=True) -- the
    # default since torch 2.6, and what the pipeline uses -- refuse the file outright.
    return float(tp.sum() / max(cm.sum(), 1)), float(f1[present].mean())


@torch.no_grad()
def evaluate(net, loader, device, n_cls=len(BM.BEHAVIORS)):
    """-> (top1, macro_f1, confusion) with confusion[true][pred]."""
    net.eval()
    cm = np.zeros((n_cls, n_cls), np.int64)
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
    ap.add_argument("--enhance", choices=["retinexformer", "cidnet"], required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--arid", action="store_true", help="mix in real-dark ARID clips")
    ap.add_argument("--k700", action="store_true",
                    help="mix in darkened Kinetics-700 clips and validate on its val split too")
    ap.add_argument("--darkact", action="store_true",
                    help="mix in real-dark DarkAct clips (talk/wave/drink/pick_up + capped "
                         "other) and validate on a 10%% carve of its train split")
    ap.add_argument("--init", default="k400",
                    help="k400 | arid | behavior | a checkpoint path (the head is kept only if "
                         "it already has 10 classes)")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--warmup-epochs", type=float, default=1.0)
    ap.add_argument("--head-lr-mult", type=float, default=10.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=8,
                    help="8 is the ceiling on a 24 GB card (13.6 GB measured at bs 8)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    # never the shipped `videomamba_t_behavior_32f.pth`: a finetune that turns out worse must
    # not have overwritten the weight the pipeline is running.
    out = a.out or os.path.join(
        DP, "ckpts", "videomamba_t_behavior_32f_"
        f"{'darkact' if a.darkact else 'k700' if a.k700 else a.enhance}.pth")
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)

    tr, miss_tr = hmdb_items("train", a.enhance)
    va, miss_va = hmdb_items("validation", a.enhance)
    n_hmdb = len(tr)
    if a.arid:
        tr += arid_items(a.enhance)
    n_arid = len(tr) - n_hmdb
    va_k7, va_da, n_k700 = [], [], 0
    if a.k700:
        k_tr, miss_ktr = k700_items("train", a.enhance)
        va_k7, miss_kva = k700_items("val", a.enhance)
        tr += k_tr
        n_k700 = len(k_tr)
        miss_tr, miss_va = miss_tr + miss_ktr, miss_va + miss_kva
    if a.darkact:
        d_tr, va_da, miss_dtr = darkact_items(a.enhance)
        tr += d_tr
        miss_tr += miss_dtr
    if miss_tr or miss_va:
        print(f"[warn] {miss_tr} train / {miss_va} val clips not in the cache yet -- "
              f"training on what exists", flush=True)
    c = Counter(lab for _, lab in tr)
    print(f"[{a.enhance}] train={len(tr)} (hmdb {n_hmdb} + arid {n_arid} + k700 {n_k700} + "
          f"darkact {len(tr)-n_hmdb-n_arid-n_k700}) "
          f"val={len(va)}+{len(va_k7)}+{len(va_da)} -> {os.path.basename(out)}")
    print("  train per class: " + "  ".join(
        f"{b}={c.get(i,0)}" for i, b in enumerate(BM.BEHAVIORS)), flush=True)

    tl = DataLoader(ClipCache(tr, True), batch_size=a.bs, sampler=balanced_sampler(tr),
                    num_workers=a.workers, pin_memory=True, drop_last=True)
    # HMDB-val alone cannot rank models on `chase` (it holds exactly one such clip), so
    # selection runs on the union. Both are still scored separately every epoch so the 0.5176
    # HMDB-val baseline of the shipped checkpoint stays directly comparable.
    vls = [(name, DataLoader(ClipCache(items, False), batch_size=16, num_workers=a.workers,
                             pin_memory=True))
           for name, items in (("hmdb", va), ("k700", va_k7), ("darkact", va_da)) if items]

    def validate():
        """-> (top1, macro_f1, union_cm, [(name, top1, macro_f1)])"""
        per = [(name, *evaluate(net, ld, a.device)) for name, ld in vls]
        cm = sum(p[3] for p in per)
        return (*metrics(cm), cm, [(n, t, f) for n, t, f, _ in per])

    net = build_model(a.device, init=INITS.get(a.init, a.init))
    opt = torch.optim.AdamW(param_groups(net, a.lr, a.head_lr_mult), weight_decay=0.05)
    sched = warmup_cosine(opt, a.epochs * len(tl), int(a.warmup_epochs * len(tl)))
    scaler = torch.amp.GradScaler("cuda")
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    # The init is already a trained behavior model, so score it BEFORE the first optimizer step.
    # That is the bar the finetune has to clear: a too-high LR then shows up as epoch 1 landing
    # below ep0 instead of as a mystery six epochs later.
    top1, mf1, cm, per = validate()
    print(f"[{a.enhance}] ep0 (init, no training) val top1={top1:.4f} macroF1={mf1:.4f}  " +
          "  ".join(f"{n}:top1={t:.4f}/mF1={f:.4f}" for n, t, f in per), flush=True)
    report(cm, "ep0")
    # Several selections, because K700-val outnumbers HMDB-val ~4:1 and therefore dominates the
    # union score -- while acceptance is decided on the per-source TEST splits. An epoch that
    # wins the union by trading away one source would otherwise be the only checkpoint we kept,
    # so each source that has a val set also gets its own. All are seeded from ep0, so no file
    # is written unless it actually improves on the init.
    by0 = {n: f for n, _, f in per}
    sel = {"union": [mf1, out]}
    for name in ("hmdb", "darkact"):
        if name in by0:
            sel[name] = [by0[name], out.replace(".pth", f"_{name}sel.pth")]
    hist = [dict(epoch=0, loss=None, val_top1=top1, val_macro_f1=mf1,
                 per_source={n: dict(top1=t, macro_f1=f) for n, t, f in per})]
    for ep in range(1, a.epochs + 1):
        net.train()
        t0, tot = time.time(), 0.0
        for x, y in tl:
            x, y = x.to(a.device, non_blocking=True), y.to(a.device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = crit(net(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        top1, mf1, cm, per = validate()
        by_src = {n: {"top1": t, "macro_f1": f} for n, t, f in per}
        flag = ""
        for name, (bestv, path) in sel.items():        # selection on VALIDATION only, never test
            score = mf1 if name == "union" else by_src.get(name, {}).get("macro_f1", -1.0)
            if score > bestv:
                sel[name][0] = score
                torch.save({"model": net.state_dict(), "macro_f1": mf1, "top1": top1,
                            "epoch": ep, "enhance": a.enhance, "labels": BM.BEHAVIORS,
                            "init": a.init, "arid": a.arid, "k700": a.k700,
                            "selected_on": name, "val_per_source": by_src}, path)
                flag += f" *best[{name}] -> saved"
        hist.append(dict(epoch=ep, loss=tot / len(tl), val_top1=top1, val_macro_f1=mf1,
                         per_source=by_src))
        print(f"[{a.enhance}] ep{ep} loss={tot/len(tl):.3f} val top1={top1:.4f} "
              f"macroF1={mf1:.4f} ({time.time()-t0:.0f}s){flag}  " +
              "  ".join(f"{n}:mF1={f:.4f}" for n, _, f in per), flush=True)
        report(cm, f"ep{ep}")

    for name, (bestv, path) in sel.items():
        ep0 = (hist[0]["val_macro_f1"] if name == "union"
               else hist[0]["per_source"].get(name, {}).get("macro_f1", -1.0))
        if bestv > ep0:
            print(f"[{a.enhance}] DONE {name}-selected best={bestv:.4f} (ep0 {ep0:.4f}) "
                  f"-> {path}", flush=True)
        else:
            print(f"[{a.enhance}] DONE nothing beat the init on {name} ({ep0:.4f}) -- "
                  f"{os.path.basename(path)} not written", flush=True)
    json.dump(hist, open(out.replace(".pth", "_hist.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
