"""Is the DarkAct collapse a content problem or an exposure-range problem?

The shipped recognizer scores below the trivial always-`other` baseline on DarkAct test, and
the cached frames explain why on inspection: after Retinexformer they sit at median luma ~20
against ~90 for ARID and ~78 for darkened HMDB. The model has simply never seen frames this
dark, and the enhancer -- trained on LOL-style input, not on luma-3.5 input -- does not lift
them into its range.

That is a hypothesis, not a finding, so this measures it: re-score a subsample of the SAME
cached clips with a per-clip linear gain that maps each clip's mean luma to a target, and see
whether accuracy recovers. Nothing else changes -- same weights, same frames, same crop.

  * recovers  -> the content is legible and the gap is exposure alignment, which a finetune
                 fixes for free (and a cheap gain stage would fix at inference).
  * does not  -> the frames genuinely lack the signal, and no amount of finetuning on them
                 will produce a usable `talk`/`drink`/`pick_up` on this footage.

Diagnostic only: it is deliberately not a flag on `behavior_eval.py`, because a gain fitted on
the test split is not a result. Report it as a probe, never as a score.
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DP = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", ".."))
sys.path.insert(0, DP)
import behavior_manifest as BM                                     # noqa: E402
from darkpipe.constants import IN_MEAN, IN_STD                     # noqa: E402
from darkpipe.vendor.videomamba import videomamba_tiny            # noqa: E402

CACHE = os.path.join(BM.ROOT, "cache", "darkact_retinexformer")
CROP = 224


def load_net(ckpt, device):
    net = videomamba_tiny(num_classes=len(BM.BEHAVIORS), num_frames=32, img_size=224)
    sd = torch.load(ckpt, map_location="cpu")
    net.load_state_dict(sd.get("model", sd))
    return net.to(device).eval()


def prep(arr, gain):
    """(32,256,W,3) uint8 RGB -> normalized (1,C,T,224,224), center crop, optional gain."""
    _, H, W, _ = arr.shape
    top, left = (H - CROP) // 2, (W - CROP) // 2
    x = arr[:, top:top + CROP, left:left + CROP, :].astype(np.float32)
    if gain != 1.0:
        x = np.clip(x * gain, 0, 255)
    x = ((x / 255.0) - IN_MEAN) / IN_STD
    return torch.from_numpy(x).permute(3, 0, 1, 2).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(DP, "ckpts",
                                                   "videomamba_t_behavior_32f.pth"))
    ap.add_argument("--target-luma", type=float, nargs="+", default=[0, 40, 60, 90],
                    help="per-clip mean luma to gain toward; 0 = no gain (control)")
    ap.add_argument("--limit", type=int, default=800, help="clips, evenly strided over the set")
    ap.add_argument("--max-gain", type=float, default=12.0)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    rows = BM.darkact_rows("test", cap=False)
    rows = rows[::max(1, len(rows) // a.limit)][:a.limit]
    items = []
    for path, beh, cls, sp in rows:
        p = os.path.join(CACHE, sp, cls, os.path.basename(path).replace(".mp4", ".npy"))
        if os.path.exists(p):
            items.append((p, BM.BID[beh]))
    print(f"[probe] {len(items)} clips, ckpt={os.path.basename(a.ckpt)}", flush=True)

    net = load_net(a.ckpt, a.device)
    K = len(BM.BEHAVIORS)
    oi = BM.BID["other"]
    for tgt in a.target_luma:
        cm = np.zeros((K, K), int)
        gains = []
        with torch.no_grad():
            for p, lab in items:
                arr = np.load(p)
                g = 1.0 if not tgt else min(a.max_gain, tgt / max(float(arr.mean()), 1e-6))
                gains.append(g)
                x = prep(arr, g).to(a.device)
                with torch.autocast("cuda", dtype=torch.float16):
                    cm[lab, int(net(x).argmax(1).item())] += 1
        tp = np.diag(cm).astype(float)
        f1 = 2 * tp / np.maximum(cm.sum(0) + cm.sum(1), 1e-9)
        seen = cm.sum(1) > 0
        pos = np.array([i != oi for i in range(K)]) & seen
        hist = Counter({BM.BEHAVIORS[j]: int(n) for j, n in enumerate(cm.sum(0)) if n})
        print(f"  target_luma={str(tgt) or '0':<4} gain median={np.median(gains):5.2f}"
              f"  top1={tp.sum() / cm.sum():.4f}  macroF1={f1[seen].mean():.4f}"
              f"  posF1={f1[pos].mean() if pos.any() else 0:.4f}"
              f"  pred={dict(hist.most_common(4))}", flush=True)
        print("    " + "  ".join(f"{b}:r{tp[i] / max(cm[i].sum(), 1):.2f}(n{int(cm[i].sum())})"
                                 for i, b in enumerate(BM.BEHAVIORS) if cm[i].sum()), flush=True)


if __name__ == "__main__":
    main()
