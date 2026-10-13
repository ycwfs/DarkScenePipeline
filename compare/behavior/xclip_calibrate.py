"""Fix the `other` class for zero-shot X-CLIP, calibrated on VALIDATION (never on test).

Zero-shot on the nine named behaviors is already strong, but `other` recall was 0.042: CLIP
cannot ground a class defined by negation ("no notable behavior"), so it essentially never
wins the argmax and every ordinary clip is forced into one of the nine.

Two fixes are compared here:
  prompts  describe `other` as concrete ordinary activity (walking, sitting, eating, ...)
           instead of as the absence of the others
  reject   drop `other` from the text side entirely and call it `other` when the best of the
           nine is not confident enough (max prob < tau) — the standard open-set rule, and
           the one that generalises to deployments whose "ordinary" is not HMDB's

The expensive half (the ViT over 32 frames) is run once per clip and cached, so every prompt
set and every tau is scored against the same features for free.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "..", "..")))
import behavior_manifest as BM                                   # noqa: E402
from behavior_eval import CKPTS, RESULTS, ClipSet, arid_rows, metrics   # noqa: E402
from clip_io import to_tensor                                    # noqa: E402
from darkpipe.constants import BEHAVIOR_PROMPTS                  # noqa: E402
from darkpipe.stages.recognize_xclip import XCLIPRecognizer      # noqa: E402

NAMED = BM.BEHAVIORS[:-1]        # the nine; `other` is index 9

# `other` described as what ordinary footage actually contains. Deliberately generic everyday
# activity rather than an enumeration of HMDB's negative classes, which would be test peeking.
OTHER_CONCRETE = [
    "a person walking normally", "a person standing still", "a person sitting down",
    "a person running alone", "people eating a meal", "a person smiling or laughing",
    "a person riding a bicycle", "a person exercising", "a person climbing",
    "a person clapping their hands", "an ordinary everyday activity",
]

VARIANTS = {
    "negation": BEHAVIOR_PROMPTS["other"],     # what constants.py ships today
    "concrete": OTHER_CONCRETE,
}


def encode_split(rows, darken, enhance, device, workers, cache_path):
    """-> (V (N,D), I (N,patches,D), y (N,)), cached to disk keyed by the domain."""
    if os.path.exists(cache_path):
        d = np.load(cache_path)
        print(f"[encode] cached {cache_path}: {d['y'].shape[0]} clips")
        return (torch.from_numpy(d["V"]).to(device), torch.from_numpy(d["I"]).to(device),
                d["y"])
    r = XCLIPRecognizer(CKPTS)
    r.load(device)
    stage = None
    if enhance != "off":
        if enhance == "retinexformer":
            from darkpipe.stages.enhance_retinexformer import RetinexformerStage as S
        else:
            from darkpipe.stages.enhance_cidnet import CIDNetStage as S
        stage = S(CKPTS, chunk=32)
        stage.load(device)

    loader = DataLoader(ClipSet(rows, 32, darken), batch_size=None, num_workers=workers)
    V, I, y = [], [], []
    for i, (frames, label, _) in enumerate(loader):
        if frames.numel() == 0:
            continue
        fl = list(frames.numpy())
        if stage is not None:
            fl = stage(fl)
        v, im = r.encode_video(to_tensor(fl, "xclip"))
        V.append(v.cpu())
        I.append(im.cpu())
        y.append(int(label))
        if (i + 1) % 100 == 0:
            print(f"  encoded {i + 1}/{len(rows)}", flush=True)
    V = torch.cat(V)
    I = torch.cat(I)
    y = np.array(y)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, V=V.numpy(), I=I.numpy(), y=y)
    print(f"[encode] {len(y)} clips -> {cache_path}")
    r.close()
    del r, stage
    torch.cuda.empty_cache()
    return V.to(device), I.to(device), y


def probs_for(reco, V, I, batch=32):
    """Score every cached clip against the recognizer's current label set -> (N, L)."""
    out = []
    for s in range(0, V.shape[0], batch):
        v, im = V[s:s + batch], I[s:s + batch]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            t = reco._text.unsqueeze(0).expand(v.shape[0], -1, -1)
            t = t + reco.net.prompts_generator(t, im)
            vn = v / v.norm(p=2, dim=-1, keepdim=True)
            t = t / t.norm(p=2, dim=-1, keepdim=True)
            sim = torch.einsum("bd,bkd->bk", vn, t).float()
            per = torch.zeros(v.shape[0], len(reco.labels), device=sim.device)
            per.index_add_(1, reco._group, sim)
            per = (per / reco._gcount) * reco.net.logit_scale.exp().float()
            out.append(per.softmax(1).cpu())
    return torch.cat(out).numpy()


def report(name, y, pred):
    m = metrics(y, pred, BM.BEHAVIORS)
    named = [BM.BID[b] for b in NAMED]
    nine = np.array([m["per_class_recall"][i] for i in named
                     if m["support"][i]]).mean()
    print(f"  {name:<34} top1={m['top1']:.4f}  macroF1={m['macro_f1']:.4f}  "
          f"other_recall={m['per_class_recall'][9]:.3f}  mean9_recall={nine:.3f}")
    return m, nine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    ap.add_argument("--source", choices=["hmdb", "arid"], default="hmdb")
    ap.add_argument("--darken", action="store_true")
    ap.add_argument("--enhance", choices=["off", "retinexformer", "cidnet"], default="off")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    rows = arid_rows(a.split) if a.source == "arid" else BM.build(a.split)
    tag = f"{a.source}_{a.split}{'_dark' if a.darken else ''}_{a.enhance}"
    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "behavior_data",
                         "xclip_feat", f"{tag}.npz")
    V, I, y = encode_split(rows, a.darken, a.enhance, a.device, a.workers, cache)
    print(f"[calibrate] {tag}: {len(y)} clips "
          f"({(y == 9).sum()} other, {(y != 9).sum()} named)\n")

    reco = XCLIPRecognizer(CKPTS)
    reco.load(a.device)
    best = None

    print("A. argmax over 10 labels, `other` as a text class")
    for vname, oprompts in VARIANTS.items():
        reco.set_labels(BM.BEHAVIORS, prompts={**BEHAVIOR_PROMPTS, "other": oprompts})
        p = probs_for(reco, V, I)
        m, nine = report(f"other-prompts={vname}", y, p.argmax(1))
        if best is None or m["macro_f1"] > best[0]:
            best = (m["macro_f1"], f"prompts:{vname}", dict(rule="argmax", other=vname))

    print("\nB. argmax over the nine, `other` when max prob < tau (open-set reject)")
    reco.set_labels(NAMED)
    p9 = probs_for(reco, V, I)
    for tau in (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6):
        pred = np.where(p9.max(1) < tau, 9, p9.argmax(1))
        m, nine = report(f"reject tau={tau}", y, pred)
        if m["macro_f1"] > best[0]:
            best = (m["macro_f1"], f"reject:{tau}", dict(rule="reject", tau=tau))

    print("\nC. concrete `other` prompts AND reject (both)")
    reco.set_labels(BM.BEHAVIORS, prompts={**BEHAVIOR_PROMPTS, "other": OTHER_CONCRETE})
    p10 = probs_for(reco, V, I)
    for tau in (0.2, 0.25, 0.3, 0.35, 0.4, 0.5):
        top = p10[:, :9].argmax(1)
        pred = np.where(p10[np.arange(len(p10)), top] < tau, 9, top)
        pred = np.where(p10.argmax(1) == 9, 9, pred)      # text class may also claim `other`
        m, nine = report(f"concrete+reject tau={tau}", y, pred)
        if m["macro_f1"] > best[0]:
            best = (m["macro_f1"], f"concrete+reject:{tau}", dict(rule="both", tau=tau))

    print(f"\n[calibrate] BEST on {a.split}: {best[1]}  macroF1={best[0]:.4f}")
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, f"xclip_calibration_{tag}.json")
    json.dump(dict(split=a.split, tag=tag, best=best[2], best_macro_f1=best[0],
                   other_concrete=OTHER_CONCRETE), open(out, "w"), indent=1)
    print(f"[calibrate] -> {out}")


if __name__ == "__main__":
    main()
