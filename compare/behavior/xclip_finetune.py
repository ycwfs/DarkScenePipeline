"""Domain-adapt X-CLIP to dark, enhanced video WITHOUT losing the open vocabulary.

The text tower is frozen and its embeddings are precomputed once. Only the video side moves:
the ViT, the temporal MIT, and the small `prompts_generator` that conditions text on the
clip. That is the whole point -- if the text tower were finetuned on ten labels it would stop
being a general text encoder, and `--labels "climbing a fence,loitering"` (and zero-shot
`chase`, which has no real training data) would degrade to noise. Frozen text keeps the model
open-vocabulary: it learns what dark enhanced footage looks like, not what these ten words
mean.

Trains on the same cached frames as `behavior_train.py`, with the same class-balanced
sampler and the same macro-F1 model selection, so the two recognizers are compared on equal
footing. Output is a full HF snapshot directory, so evaluation and the pipeline consume it
through the normal `--xclip-model` path with no special-casing.
"""
import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DP = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", ".."))
sys.path.insert(0, DP)
import behavior_manifest as BM                                      # noqa: E402
import behavior_train as BT                                         # noqa: E402
from darkpipe.constants import CKPT_FILES, RECO_CFG                 # noqa: E402
from darkpipe.stages.recognize import preprocess_frame              # noqa: E402
from darkpipe.stages.recognize_xclip import prompts_for             # noqa: E402

CKPTS = os.path.join(DP, "ckpts")
XCFG = RECO_CFG["xclip"]
CROP, T = XCFG["size"], XCFG["T"]


class Clips(BT.ClipCache):
    """Same cache as the VideoMamba trainer, but X-CLIP's geometry, not VideoMamba's.

    The pipeline's X-CLIP stage resizes the short side to 224 and center-crops 224
    (`RECO_CFG["xclip"]["resize"] == 224`); VideoMamba resizes to 256 and crops 224. Taking
    the inherited 224 crop straight out of the 256-tall cache would train the model on a
    tighter field of view than it ever sees at inference. Measured on the validation split,
    the two framings are not interchangeable: 0.1667 top-1 / 0.2405 macro-F1 for the crop
    versus 0.1489 / 0.2000 for the deployed resize — adapting under the wrong one would tune
    the vision tower for frames the pipeline never produces.

    So the frames go through `preprocess_frame`, the same function the serving stage calls,
    rather than a second copy of the resize/crop/normalize arithmetic that could drift from
    it. The cost is the random crop: only a horizontal flip is left as augmentation.
    """

    def __getitem__(self, i):
        p, lab = self.items[i]
        arr = np.ascontiguousarray(np.load(p)[..., ::-1])   # (32,256,W,3) RGB -> BGR for cv2
        x = np.stack([preprocess_frame(f, XCFG) for f in arr])           # (T,224,224,C)
        if self.train and random.random() < 0.5:
            x = x[:, :, ::-1, :]                            # the one augmentation left: hflip
        return torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2), lab


def text_bank(model, tok, labels, device):
    """Frozen per-prompt text embeddings + the label each prompt belongs to."""
    texts, group = [], []
    for i, lab in enumerate(labels):
        for p in prompts_for(lab):
            texts.append(p)
            group.append(i)
    with torch.no_grad():
        batch = tok(texts, padding=True, return_tensors="pt").to(device)
        emb = model.get_text_features(**batch).float()
    g = torch.tensor(group, device=device)
    return emb, g, torch.bincount(g, minlength=len(labels)).float()


def logits_for(model, x, text, group, gcount):
    """(B,T,C,H,W) -> (B,L) label logits. The differentiable twin of `zero_shot_probs`."""
    B, t = x.shape[:2]
    vout = model.vision_model(pixel_values=x.flatten(0, 1))
    video = model.visual_projection(vout[1])
    video = model.mit(video.view(B, t, -1))[1]
    img = model.prompts_visual_layernorm(vout[0][:, 1:, :])
    img = img @ model.prompts_visual_projection
    img = img.view(B, t, -1, video.shape[-1]).mean(1)
    txt = text.unsqueeze(0).expand(B, -1, -1)
    txt = txt + model.prompts_generator(txt, img)
    video = video / video.norm(p=2, dim=-1, keepdim=True)
    txt = txt / txt.norm(p=2, dim=-1, keepdim=True)
    sim = torch.einsum("bd,bkd->bk", video, txt).float()             # (B, P)
    per_label = torch.zeros(B, len(gcount), device=sim.device, dtype=sim.dtype)
    per_label.index_add_(1, group, sim)
    return (per_label / gcount) * model.logit_scale.exp().float()


def freeze_text(model):
    """-> the trainable parameters. Everything text-side stops learning."""
    for n, p in model.named_parameters():
        p.requires_grad = not (n.startswith("text_model.") or n.startswith("text_projection"))
    return [p for p in model.parameters() if p.requires_grad]


@torch.no_grad()
def evaluate(model, loader, text, group, gcount, device, n_cls):
    model.eval()
    cm = np.zeros((n_cls, n_cls), np.int64)
    for x, y in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            pred = logits_for(model, x.to(device), text, group, gcount).argmax(1).cpu().numpy()
        for t_, p in zip(y.numpy(), pred):
            cm[t_, p] += 1
    tp = np.diag(cm).astype(float)
    f1 = 2 * tp / np.maximum(cm.sum(0) + cm.sum(1), 1e-9)
    present = cm.sum(1) > 0
    return tp.sum() / max(cm.sum(), 1), float(f1[present].mean()), cm


def save_snapshot(model, src, dst):
    """A full HF snapshot, so `--xclip-model <dst>` just works (no bespoke loader)."""
    os.makedirs(dst, exist_ok=True)
    model.save_pretrained(dst)
    for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
              "special_tokens_map.json", "preprocessor_config.json"):
        p = os.path.join(src, f)
        if os.path.exists(p):
            shutil.copy2(p, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enhance", choices=["retinexformer", "cidnet"], required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--model-dir", default=os.path.join(CKPTS, CKPT_FILES["xclip"]))
    ap.add_argument("--arid", action="store_true")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-5, help="low: this is adaptation, not "
                                                          "training -- 1e-4 wrecks CLIP")
    ap.add_argument("--warmup-epochs", type=float, default=0.5)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = a.out or os.path.join(CKPTS, f"xclip_behavior_{a.enhance}")
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)

    from transformers import AutoTokenizer, XCLIPModel
    model = XCLIPModel.from_pretrained(a.model_dir).to(a.device)
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    text, group, gcount = text_bank(model, tok, BM.BEHAVIORS, a.device)
    params = freeze_text(model)

    tr, miss_tr = BT.hmdb_items("train", a.enhance)
    va, miss_va = BT.hmdb_items("validation", a.enhance)
    if a.arid:
        tr += BT.arid_items(a.enhance)
    if miss_tr or miss_va:
        print(f"[warn] {miss_tr} train / {miss_va} val clips missing from the cache", flush=True)
    c = Counter(lab for _, lab in tr)
    print(f"[xclip-ft {a.enhance}] train={len(tr)} val={len(va)} "
          f"trainable={sum(p.numel() for p in params)/1e6:.1f}M -> {out}")
    print("  train per class: " + "  ".join(
        f"{b}={c.get(i,0)}" for i, b in enumerate(BM.BEHAVIORS)), flush=True)

    cnt = Counter(lab for _, lab in tr)
    sampler = WeightedRandomSampler([1.0 / cnt[l] for _, l in tr], len(tr), replacement=True)
    tl = DataLoader(Clips(tr, True), batch_size=a.bs, sampler=sampler,
                    num_workers=a.workers, pin_memory=True, drop_last=True)
    vl = DataLoader(Clips(va, False), batch_size=a.bs, num_workers=a.workers, pin_memory=True)

    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    total, warm = a.epochs * len(tl), int(a.warmup_epochs * len(tl))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else
        0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm)))))
    scaler = torch.amp.GradScaler("cuda")
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    top1, mf1, cm = evaluate(model, vl, text, group, gcount, a.device, len(BM.BEHAVIORS))
    print(f"[xclip-ft {a.enhance}] zero-shot val top1={top1:.4f} macroF1={mf1:.4f}", flush=True)
    best, hist = mf1, [dict(epoch=0, val_top1=top1, val_macro_f1=mf1)]
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for x, y in tl:
            x, y = x.to(a.device, non_blocking=True), y.to(a.device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = crit(logits_for(model, x, text, group, gcount), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        top1, mf1, cm = evaluate(model, vl, text, group, gcount, a.device, len(BM.BEHAVIORS))
        flag = ""
        if mf1 > best:
            best = mf1
            save_snapshot(model, a.model_dir, out)
            flag = " *best -> saved"
        hist.append(dict(epoch=ep, loss=tot / len(tl), val_top1=top1, val_macro_f1=mf1))
        print(f"[xclip-ft {a.enhance}] ep{ep} loss={tot/len(tl):.3f} val top1={top1:.4f} "
              f"macroF1={mf1:.4f} ({time.time()-t0:.0f}s){flag}", flush=True)
        BT.report(cm, f"ep{ep}")

    print(f"[xclip-ft {a.enhance}] DONE best val macroF1={best:.4f} -> {out}", flush=True)
    json.dump(hist, open(os.path.join(CKPTS, f"xclip_behavior_{a.enhance}_hist.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
