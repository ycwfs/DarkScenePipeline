"""Clip-level evaluation of the behavior recognizers, across models AND frame domains.

Two axes, matching the earlier RF-vs-CIDNet study:
  --model    xclip (zero-shot / finetuned, open vocabulary) | behavior (trained VideoMamba)
  --enhance  off | retinexformer | cidnet     (with --darken, the real pipeline's front end)

Sources: `hmdb` (bright clips, optionally darkened+enhanced), `arid` (genuinely dark footage,
the more trustworthy signal for the three behaviors it actually contains), `k700` (darkened
Kinetics-700 val -- in-domain for any model trained with `--k700`, so it is a sanity check,
never the acceptance criterion) and `darkact` (3738 real-dark clips, the best-powered held-out
set here and the only one covering `talk` on genuinely dark footage).

Decode runs in DataLoader workers, enhancement and the model on the GPU; batch_size is 1
clip (32 frames) so clips of different native resolutions need no collation.

`--cache auto` (the default) reads the darkened+enhanced npy the training cache already
holds instead of re-decoding, re-darkening and re-enhancing every clip -- ~20x faster and,
more importantly, the frames the model was TRAINED on. The exposure draw is keyed off the
filename via `darken.clip_rng`, so the live path reproduces the cached one; `--cache off`
exercises that path end to end. The two agree exactly for `--model behavior` (its 256 short
side is the cache's own); for `--model xclip` they can differ on a borderline clip, because
the cache resizes to 256 and X-CLIP then resizes 256->224, where the live path goes straight
to 224 (measured: 11/12 identical predictions, same top-1).
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "..", "..")))
import behavior_manifest as BM                                   # noqa: E402
from clip_io import read_frames, to_tensor                       # noqa: E402
from darken import DEFAULT_K, READ_NOISE, clip_k, clip_rng, darken_frames  # noqa: E402
from darkpipe.constants import RECO_CFG                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CKPTS = os.path.normpath(os.path.join(HERE, "..", "..", "ckpts"))
RESULTS = os.path.join(HERE, "..", "results")
CACHE = os.path.join(BM.ROOT, "cache")
ARID_CACHE = os.path.normpath(os.path.join(BM.ROOT, "..", "arid", "cache"))
ARID_TAG = {"retinexformer": "rf", "cidnet": "cidnet"}   # the ARID cache names it differently


class ClipSet(Dataset):
    """Returns BGR frames; enhancement happens on the GPU in the main loop.

    Two modes. Live: decode the mp4 and darken it here (workers), enhance in the main loop.
    Cached: `npy` is set, the frames are already darkened+enhanced+resized, and the main loop
    skips the enhancer. Cached frames are stored RGB (that is what the model consumes), so
    they are flipped back to BGR here and the single `to_tensor` path stays shared.
    """

    def __init__(self, rows, T, darken=False, seed=0, span=0.0):
        self.rows, self.T, self.darken, self.seed = rows, T, darken, seed
        self.span = span

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, label = self.rows[i][0], self.rows[i][1]
        npy = self.rows[i][4] if len(self.rows[i]) > 4 else ""
        if npy:
            arr = np.load(npy)[..., ::-1]                     # (T,256,W,3) RGB -> BGR
            return torch.from_numpy(np.ascontiguousarray(arr)), BM.BID[label], path, True
        # `span` (K700 only) must match what the cache used, seed and all, or --cache off would
        # score a different window than --cache auto
        frames = read_frames(path, self.T, span=self.span or None,
                             span_rng=clip_rng(path, seed=1) if self.span else None)
        if frames is None:
            return torch.zeros(0), BM.BID[label], path, False
        arr = np.stack(frames)
        if self.darken:
            # crc32 of the filename, so this reproduces the training cache's exposure exactly
            rng = clip_rng(path, self.seed)
            arr = darken_frames(arr, clip_k(rng), rng, read_noise=READ_NOISE)
        return torch.from_numpy(arr), BM.BID[label], path, False


def attach_cache(rows, source, split, darken, enhance, mode):
    """-> (rows with an npy path appended, cache_root, n_missing). '' root = live decoding."""
    if mode == "off":
        return rows, "", 0
    if source == "hmdb":
        root = os.path.join(CACHE, f"hmdb{'_dark' if darken else ''}_{enhance}")
    elif source == "k700":
        root = os.path.join(CACHE, f"k700{'_dark' if darken else ''}_{enhance}")
    elif source == "darkact":            # real dark, never simulated -> no `_dark` infix
        root = os.path.join(CACHE, f"darkact_{enhance}")
    else:
        tag = ARID_TAG.get(enhance)
        root = os.path.join(ARID_CACHE, f"{split}_{tag}") if tag else ""
    if not root or not os.path.isdir(root):
        if mode == "on":
            sys.exit(f"error: --cache on but no cache at {root or '(none for this combo)'}")
        return rows, "", 0
    out, missing = [], 0
    for r in rows:
        # hmdb, k700 and darkact all cache as {split}/{source_class}/; arid's cache root
        # already has the split in its name
        rel = r[2] if source == "arid" else os.path.join(r[3], r[2])
        p = os.path.join(root, rel, os.path.basename(r[0]).replace(".mp4", ".npy"))
        if not os.path.exists(p):
            # fall back to live decoding for this clip rather than dropping it: silently
            # skipping the clips a half-built cache is missing would bias the test set.
            p, missing = "", missing + 1
        out.append(tuple(r) + (p,))
    return out, root, missing


def arid_rows(split="test"):
    """ARID clips mapped into the behavior vocabulary (drink / pick_up / wave / other)."""
    return BM.arid_rows(split)


def build_model(kind, device, ckpt="", tau=None):
    """-> (recognizer, infer) where `infer` returns the RAW distribution.

    Raw, because the open-set `other` rule is a post-processing threshold: keeping it out of
    the stored numbers lets `--sweep-tau` recalibrate from a saved run without another pass
    over the vision tower (which is 99% of the cost).
    """
    if kind == "xclip":
        from darkpipe.stages.recognize_xclip import XCLIPRecognizer
        r = XCLIPRecognizer(CKPTS, model_dir=ckpt)
        if tau is not None:
            r.reject_tau = tau
        r.load(device)
        return r, lambda a: r.pool(r.score(*r.encode_video(a))).cpu().numpy()
    from darkpipe.stages.recognize import BehaviorRecognizer
    r = BehaviorRecognizer(CKPTS)
    if ckpt:
        r.ckpt = ckpt
    r.load(device)
    return r, lambda a: r._infer(a)


def metrics(y_true, y_pred, labels):
    n = len(labels)
    cm = np.zeros((n, n), int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    support = cm.sum(1)
    recall = np.divide(cm.diagonal(), support, out=np.zeros(n), where=support > 0)
    pred_n = cm.sum(0)
    prec = np.divide(cm.diagonal(), pred_n, out=np.zeros(n), where=pred_n > 0)
    f1 = np.divide(2 * prec * recall, prec + recall, out=np.zeros(n), where=(prec + recall) > 0)
    seen = support > 0
    return dict(top1=float(cm.diagonal().sum() / max(1, cm.sum())),
                macro_f1=float(f1[seen].mean()), per_class_recall=recall.tolist(),
                per_class_f1=f1.tolist(), support=support.tolist(), confusion=cm.tolist())


def sweep_tau(path):
    """Re-threshold a finished run: macro-F1 vs the open-set tau, from the stored raw probs.

    Run this on VALIDATION and apply the winner to test — tau is a fitted parameter, and
    picking it on the test split would be reporting a tuned number as a zero-shot one.
    """
    from darkpipe.stages.recognize_xclip import reject_probs
    m = json.load(open(path))
    if not m.get("probs"):
        sys.exit(f"error: {path} has no stored probabilities (rerun the eval to add them)")
    labels, probs, y = m["labels"], [np.array(p) for p in m["probs"]], m["y_true"]
    print(f"[tau] {m['tag']}  n={len(y)}  (tau used at run time: {m.get('tau')})")
    print(f"  {'tau':>5}{'top1':>9}{'macroF1':>9}{'n_other':>9}")
    best = None
    for tau in np.arange(0.0, 1.0, 0.05):
        pred = [int(reject_probs(p, labels, float(tau)).argmax()) for p in probs]
        s = metrics(y, pred, labels)
        n_other = sum(1 for p in pred if labels[p] == "other")
        print(f"  {tau:>5.2f}{s['top1']:>9.4f}{s['macro_f1']:>9.4f}{n_other:>9}")
        if best is None or s["macro_f1"] > best[1]:
            best = (float(tau), s["macro_f1"], s["top1"])
    print(f"[tau] best tau={best[0]:.2f}  macroF1={best[1]:.4f}  top1={best[2]:.4f}")
    return best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-tau", default="",
                    help="path to a saved eval JSON: re-threshold it instead of running")
    ap.add_argument("--model", choices=["xclip", "behavior"], default="xclip")
    ap.add_argument("--ckpt", default="", help="override model weights / X-CLIP snapshot dir")
    ap.add_argument("--enhance-ckpt-dir", default="",
                    help="override the enhancer's ckpt dir (recognizer stays on the shipped one); "
                         "for A/B-testing a fine-tuned enhancer's effect on recognition")
    ap.add_argument("--source", choices=["hmdb", "arid", "k700", "darkact"], default="hmdb")
    ap.add_argument("--split", default="test",
                    help="k700 has train/val only; darkact has train/test only")
    ap.add_argument("--darken", action="store_true")
    ap.add_argument("--enhance", choices=["off", "retinexformer", "cidnet"], default="off")
    ap.add_argument("--cache", choices=["auto", "on", "off"], default="auto",
                    help="read the prebuilt darkened+enhanced npy instead of redoing that "
                         "work per clip (auto: use it when it exists)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="")
    ap.add_argument("--tau", type=float, default=None,
                    help="open-set threshold for xclip (default: the shipped XCLIP_REJECT_TAU)")
    a = ap.parse_args()
    if a.sweep_tau:
        sweep_tau(a.sweep_tau)
        return

    rows = {"arid": lambda: arid_rows(a.split),
            "k700": lambda: BM.k700_rows(a.split),
            # cap=False: the test split is the dataset's test split, never subsampled
            "darkact": lambda: BM.darkact_rows(a.split, cap=False),
            }.get(a.source, lambda: BM.build(a.split))()
    span = BM.K700_SPAN_SEC if a.source in ("k700", "darkact") else 0.0
    if a.limit:
        rows = rows[::max(1, len(rows) // a.limit)][:a.limit]
    kind = "xclip" if a.model == "xclip" else "behavior"
    T = RECO_CFG[kind]["T"]

    rows, cache_root, n_live = attach_cache(rows, a.source, a.split, a.darken, a.enhance,
                                            "off" if T != 32 else a.cache)
    n_cached = sum(1 for r in rows if len(r) > 4 and r[4])

    model, infer = build_model(a.model, a.device, a.ckpt, a.tau)
    stage = None
    if a.enhance != "off":
        if a.enhance == "retinexformer":
            from darkpipe.stages.enhance_retinexformer import RetinexformerStage as S
        else:
            from darkpipe.stages.enhance_cidnet import CIDNetStage as S
        stage = S(a.enhance_ckpt_dir or CKPTS, chunk=T)
        stage.load(a.device)

    tag = a.tag or f"{a.model}_{a.source}{'_dark' if a.darken else ''}_{a.enhance}"
    print(f"[eval] {tag}: {len(rows)} clips, T={T}, darken={a.darken}, enhance={a.enhance}")
    if cache_root:
        print(f"[eval] cache {n_cached}/{len(rows)} clips <- {cache_root}"
              + (f"  ({n_live} decoded live)" if n_live else ""))

    loader = DataLoader(ClipSet(rows, T, a.darken, span=span), batch_size=None,
                        num_workers=a.workers)
    y_true, y_pred, conf, raw, skipped = [], [], [], [], 0
    for i, (frames, label, path, cached) in enumerate(loader):
        if frames.numel() == 0:
            skipped += 1
            continue
        fl = list(frames.numpy())
        if stage is not None and not cached:      # cached frames are already enhanced
            fl = stage(fl)
        prob = infer(to_tensor(fl, kind))
        raw.append([round(float(v), 5) for v in prob])
        if a.model == "xclip":
            prob = model.reject(prob)             # open-set rule, swept post-hoc via --sweep-tau
        y_true.append(int(label))
        y_pred.append(int(np.argmax(prob)))
        conf.append(float(prob.max()))
        if (i + 1) % 100 == 0:
            acc = np.mean(np.array(y_true) == np.array(y_pred))
            print(f"  {i + 1}/{len(rows)}  running top1={acc:.4f}", flush=True)

    m = metrics(y_true, y_pred, BM.BEHAVIORS)
    m.update(tag=tag, model=a.model, source=a.source, split=a.split, darken=a.darken,
             enhance=a.enhance, ckpt=a.ckpt, enhance_ckpt_dir=a.enhance_ckpt_dir, n=len(y_true), skipped=skipped,
             mean_conf=float(np.mean(conf)) if conf else 0.0, n_cached=n_cached,
             darken_k=list(DEFAULT_K), labels=BM.BEHAVIORS, y_true=y_true, y_pred=y_pred,
             probs=raw, tau=(model.reject_tau if a.model == "xclip" else None))

    print(f"\n[{tag}]  n={m['n']}  top1={m['top1']:.4f}  macroF1={m['macro_f1']:.4f}  "
          f"mean_conf={m['mean_conf']:.3f}  skipped={skipped}")
    print(f"  {'behavior':<13}{'support':>8}{'recall':>9}{'f1':>8}   most-confused-with")
    for i, b in enumerate(BM.BEHAVIORS):
        if not m["support"][i]:
            continue
        row = np.array(m["confusion"][i])
        off = row.copy()
        off[i] = 0
        worst = BM.BEHAVIORS[int(off.argmax())] if off.sum() else "-"
        print(f"  {b:<13}{m['support'][i]:>8}{m['per_class_recall'][i]:>9.3f}"
              f"{m['per_class_f1'][i]:>8.3f}   {worst} ({int(off.max())})")
    print("  predicted-label histogram:",
          dict(Counter(BM.BEHAVIORS[p] for p in y_pred).most_common()))

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, f"behavior_eval_{tag}.json")
    json.dump(m, open(out, "w"), indent=1)
    print(f"[eval] -> {out}")


if __name__ == "__main__":
    main()
