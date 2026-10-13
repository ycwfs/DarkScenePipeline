"""Darken HMDB/Kinetics-700, push it through the REAL enhancer, cache frames for training.

Same contract as `compare/arid_cache.py` (32 frames, short side 256, RGB uint8 npy, `--shard
i/n`, skip-existing) with one addition: HMDB and K700 are bright, so every clip first goes
through the calibrated low-light simulation in `darken.py`. ARID and DarkAct clips are already
dark and skip it — DarkAct especially, at a median frame luma of 3.5/255, where simulating
low light on top would be physically meaningless.

Order matters and mirrors deployment exactly: darken -> enhance -> resize. The recognizer is
therefore trained on the same frame domain the pipeline hands it at inference, which is the
whole point of the exercise (the earlier ARID study showed a 13-point swing from this alone).

The per-clip exposure `k` is drawn deterministically from the filename, so a clip has the same
simulated exposure in every enhancer's cache — the RF-vs-CIDNet comparison then differs only
by the enhancer, never by the noise draw.

`--span-sec` (used for K700, whose clips are a fixed 10 s against HMDB's 2.6 s median) caches a
contiguous window instead of the whole clip. Its offset comes from a second stream of the same
per-clip seed, so it is deterministic too and independent of the exposure draw.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DP = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", ".."))
sys.path.insert(0, DP)
import behavior_manifest as BM                                   # noqa: E402
from clip_io import read_frames                                  # noqa: E402
from darken import READ_NOISE, clip_k, clip_rng, darken_frames   # noqa: E402
from darkpipe.constants import RECO_CFG                          # noqa: E402

CKPT_DIR = os.path.join(DP, "ckpts")
T = RECO_CFG["behavior"]["T"]           # 32
RESIZE = RECO_CFG["behavior"]["resize"]  # 256
CACHE = os.path.join(BM.ROOT, "cache")


def short_resize_rgb(bgr):
    import cv2
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
        from darkpipe.stages.enhance_cidnet import CIDNetStage as S
    else:
        from darkpipe.stages.enhance_retinexformer import RetinexformerStage as S
    st = S(CKPT_DIR, chunk=T)
    st.load(device)
    return st


def items_for(source, split):
    """-> [(abs_path, relpath_for_cache)]

    Every source keeps its OWN class in the cache path, never the behavior label: remapping a
    class to a different behavior (or re-running chase curation) must not invalidate a cache
    that costs hours to rebuild.
    """
    if source == "arid":
        return [(p, os.path.join(split, cls, os.path.basename(p)))
                for p, _, cls, _ in BM.arid_rows(split)]
    if source == "k700":
        return [(p, os.path.join(sp, slug, os.path.basename(p)))
                for p, _, slug, sp in BM.k700_rows(split)]
    if source == "darkact":                            # cap=False: cache all 9040 train clips,
        return [(p, os.path.join(sp, cls, os.path.basename(p)))   # so raising DARKACT_OTHER_CAP
                for p, _, cls, sp in BM.darkact_rows(split, cap=False)]  # never needs a rebuild
    return [(p, os.path.join(sp, cls, os.path.basename(p)))
            for p, cls, sp in BM.hmdb_clips(split)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["hmdb", "arid", "k700", "darkact"], default="hmdb")
    ap.add_argument("--split", default="", help="blank = all splits")
    ap.add_argument("--enhance", choices=["off", "cidnet", "retinexformer"], required=True)
    ap.add_argument("--no-darken", action="store_true",
                    help="skip the low-light simulation (implied for --source arid/darkact)")
    ap.add_argument("--span-sec", type=float, default=None,
                    help="cache a contiguous window of this many seconds instead of the whole "
                         f"clip (default: {BM.K700_SPAN_SEC} for --source k700/darkact, off "
                         "otherwise)")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", default="0/1", help="i/n")
    a = ap.parse_args()

    darken = not (a.no_darken or a.source in ("arid", "darkact"))
    # DarkAct's median clip is 5.0 s against HMDB's 2.6 s, so it gets the same contiguous window
    # K700 does -- 32 frames over a whole clip would be ~2x slower motion than the model expects.
    span = a.span_sec if a.span_sec is not None else (
        BM.K700_SPAN_SEC if a.source in ("k700", "darkact") else 0.0)
    out_dir = a.out_dir or os.path.join(
        CACHE, f"{a.source}{'_dark' if darken else ''}_{a.enhance}")
    if a.source == "hmdb":
        items = items_for("hmdb", a.split or None)
    elif a.source == "arid":
        items = items_for("arid", a.split or "train")
    elif a.source == "darkact":  # blank means both official splits
        items = [it for sp in ([a.split] if a.split else ["train", "test"])
                 for it in items_for("darkact", sp)]
    else:                        # k700 has no all-splits mode upstream; blank means both
        items = [it for sp in ([a.split] if a.split else ["train", "val"])
                 for it in items_for("k700", sp)]
    i, n = map(int, a.shard.split("/"))
    items = [it for k, it in enumerate(items) if k % n == i]
    print(f"[cache] {len(items)} clips shard={a.shard} darken={darken} "
          f"span={span or 'full'} enhance={a.enhance} -> {out_dir}", flush=True)

    enh = build_enhancer(a.enhance, a.device)
    done = skipped = bad = 0
    for path, rel in items:
        out = os.path.join(out_dir, rel.replace(".mp4", ".npy"))
        if os.path.exists(out):
            skipped += 1
            continue
        # seed=1 keeps the window offset independent of the exposure draw below (seed=0)
        frames = read_frames(path, T, span=span or None,
                             span_rng=clip_rng(path, seed=1) if span else None)
        if frames is None:
            bad += 1
            print(f"  [bad] {path}", flush=True)
            continue
        if darken:
            rng = clip_rng(path)
            frames = list(darken_frames(np.stack(frames), clip_k(rng), rng,
                                        read_noise=READ_NOISE))
        if enh is not None:
            frames = enh(frames)
        arr = np.stack([short_resize_rgb(f) for f in frames]).astype(np.uint8)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.save(out, arr)
        done += 1
        if done % 200 == 0:
            print(f"  [{a.enhance} shard{a.shard}] {done} cached", flush=True)
    print(f"[cache] done={done} skipped={skipped} bad={bad} -> {out_dir}")


if __name__ == "__main__":
    main()
