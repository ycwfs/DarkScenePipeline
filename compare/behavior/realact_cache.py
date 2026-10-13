"""Cache the real deployment action videos as 32-frame clips, through the deployed chain.

The recognizer never sees a raw frame in production: `--enhance retinexformer` runs first, at
`--proc-max-side 840`, and only then does the behavior head resize to a 256 short side. Caching
raw frames would train the head on a distribution it never meets at inference, so this script
reproduces that order exactly -- downscale to 840, enhance, then store at 256.

Filenames carry the label: `喝水3.mp4` supervises `drink`; the trailing number is just the
take. How a class is split depends on how many takes it has:

  * **>= MIN_VIDEO_SPLIT takes -> whole videos are held out.** Windows are dense (stride 8 of
    32 frames), so any split *within* a video puts a window's near-duplicate neighbour in the
    other half, scoring memorisation as generalisation. With separate takes available, val can
    be takes the model never saw a frame of. The *highest-numbered* takes go to val, not a
    random sample: the numbers track recording order (the 2026-08-25 batch spans two sessions
    an hour apart), so this is a temporal holdout at the take level too.
  * **Fewer takes -> the original TEMPORAL split.** One ~20-30 s video per class was all the
    2026-08-20 delivery had: train takes the first 70% of the video, val the last 30%, with a
    32-frame gap so no frame is shared. Same scene, person and lighting on both sides, which
    is why those classes' val numbers are an upper bound, not a field estimate.
"""
import argparse
import os
import re
import sys
from collections import defaultdict

import cv2
import numpy as np

DP = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", ".."))
sys.path.insert(0, DP)
from darkpipe.stages.enhance_retinexformer import RetinexformerStage      # noqa: E402

# Filename prefix (as delivered) -> the BEHAVIORS key it supervises. All already exist in the
# shipped label set, so this is a finetune of the current head, not a new vocabulary.
# 追逐 arrived with the 2026-08-25 batch -- the first real training data the class has ever
# had (its previous coverage was detector-curated weak labels, F1 ~0).
VIDEO_BEHAVIOR = {
    "喝水": "drink",
    "抛物": "throw",
    "挥手": "wave",
    "捡东西": "pick_up",
    "摔倒": "fall",
    "追逐": "chase",
    "other": "other",    # on-site no-action takes (2026-09-14, real-actiondata-0909/other/)
}
T, STRIDE, STORE_SHORT = 32, 8, 256
MIN_VIDEO_SPLIT = 4              # takes needed before whole-video holdout beats temporal split


def resize_max_side(frame, max_side):
    h, w = frame.shape[:2]
    if max(h, w) <= max_side:
        return frame
    s = max_side / max(h, w)
    return cv2.resize(frame, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)


def resize_short_side(frame, short):
    h, w = frame.shape[:2]
    s = short / min(h, w)
    return cv2.resize(frame, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)


def read_processed(path, stage, proc_side, chunk=32):
    """-> list of (256, W, 3) uint8 RGB frames, enhanced exactly as deployment would."""
    cap = cv2.VideoCapture(path)
    out, buf = [], []

    def flush():
        if not buf:
            return
        for f in stage(buf):
            out.append(cv2.cvtColor(resize_short_side(f, STORE_SHORT), cv2.COLOR_BGR2RGB))
        buf.clear()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        buf.append(resize_max_side(frame, proc_side))
        if len(buf) >= chunk:
            flush()
    flush()
    cap.release()
    return out


def windows(n, lo, hi):
    """Start indices of 32-frame windows fully inside [lo, hi)."""
    return list(range(lo, hi - T + 1, STRIDE))


def collect(src, behaviors):
    """-> {behavior: [(stem, take_number_or_None, path)] sorted by take}, from src/*.mp4."""
    groups = defaultdict(list)
    import glob as _glob
    paths = sorted(_glob.glob(os.path.join(src, "*.mp4"))
                   + _glob.glob(os.path.join(src, "other", "*.mp4")))
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        m = re.fullmatch(r"(\D+)(\d*)", stem)
        base = m.group(1) if m else stem
        if base not in VIDEO_BEHAVIOR:
            print(f"[cache] 跳过（文件名不在 VIDEO_BEHAVIOR 里）: {os.path.basename(path)}")
            continue
        beh = VIDEO_BEHAVIOR[base]
        if behaviors and beh not in behaviors:
            continue
        num = int(m.group(2)) if m and m.group(2) else None
        groups[beh].append((stem, num, path))
    for beh in groups:                       # bare take first, then by number = recording order
        groups[beh].sort(key=lambda t: (t[1] is not None, t[1] or 0))
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(DP, "real-actiondata"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "cache", "realact_rf_v2"))
    ap.add_argument("--ckpt-dir", default=os.path.join(DP, "ckpts"))
    ap.add_argument("--proc-side", type=int, default=840, help="deployed --proc-max-side")
    ap.add_argument("--train-frac", type=float, default=0.7, help="temporal split (few-take classes)")
    ap.add_argument("--val-videos-frac", type=float, default=0.25,
                    help="fraction of takes held out whole (many-take classes)")
    ap.add_argument("--behaviors", default="",
                    help="comma list to restrict to (for sharding the cache build across GPUs)")
    ap.add_argument("--shard", default="",
                    help="i/n — process only every n-th video of each class. Split decisions are "
                         "always made on the FULL class list, so shards stay consistent.")
    ap.add_argument("--device", default="cuda:4")
    a = ap.parse_args()
    behaviors = {s.strip() for s in a.behaviors.split(",") if s.strip()}

    groups = collect(a.src, behaviors)
    if not groups:
        sys.exit(f"error: {a.src} 下没有可识别的视频")

    stage = RetinexformerStage(a.ckpt_dir)
    stage.load(a.device)
    total = {"train": 0, "val": 0}

    def save(split, beh, stem, frames, lo, hi):
        d = os.path.join(a.out, split, beh)
        os.makedirs(d, exist_ok=True)
        ws = windows(len(frames), lo, hi)
        for s in ws:
            np.save(os.path.join(d, f"{stem}_{s:05d}.npy"), np.stack(frames[s:s + T]))
        total[split] += len(ws)
        return len(ws)

    shard = tuple(int(x) for x in a.shard.split("/")) if a.shard else None

    for beh, vids in sorted(groups.items()):
        if len(vids) >= MIN_VIDEO_SPLIT:
            n_val = max(1, round(len(vids) * a.val_videos_frac))
            split_of = {stem: ("val" if i >= len(vids) - n_val else "train")
                        for i, (stem, _, _) in enumerate(vids)}
            if shard:                        # after split_of: shards share one split decision
                vids = vids[shard[0]::shard[1]]
            for stem, _, path in vids:
                frames = read_processed(path, stage, a.proc_side)
                split = split_of[stem]
                k = save(split, beh, stem, frames, 0, len(frames))
                print(f"[cache] {stem}({beh}) {len(frames)} 帧 -> {split} {k}", flush=True)
        else:
            if shard:
                vids = vids[shard[0]::shard[1]]
            for stem, _, path in vids:
                frames = read_processed(path, stage, a.proc_side)
                n = len(frames)
                cut = int(n * a.train_frac)
                # the gap is one full window, so no train window shares a frame with a val one
                kt = save("train", beh, stem, frames, 0, cut)
                kv = save("val", beh, stem, frames, min(cut + T, n), n)
                print(f"[cache] {stem}({beh}) {n} 帧（时间切分）-> train {kt} / val {kv}",
                      flush=True)
    stage.close()
    print(f"[cache] 合计 train={total['train']} val={total['val']} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
