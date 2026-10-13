"""Build RGB caches for the 5 skeleton-missing behaviors from K700 + HMDB.

The skeleton-aux branch only had real field supervision for wave/chase/drink/pick_up (+other).
This gives the ST-GCN proxy supervision for throw/fall/fight/talk/shake_hands by caching
K700+HMDB clips through the SAME preprocessing the VideoMamba RGB track used -- darken sim ->
Retinexformer -> short-256 resize (deterministic per-clip exposure and K700 span offset, via
the shared helpers in behavior_cache.py / darken.py) -- so the pose model sees the same frame
domain the deployed enhancer produces.

Sampling (deterministic, seed 0): K700 train capped per behavior, K700 val capped smaller;
HMDB train+validation go to train, HMDB test to val. HMDB's run/walk stay out of `other`
(they are the chase pool); proxy `other` draws from K700's capped other rows only.

Usage: python -u proxy_rgb_cache.py <device> [--shard i/n] [--cap 450]
"""
import argparse
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # compare/behavior
import behavior_cache as BC                        # noqa: E402
import behavior_manifest as BM                     # noqa: E402
from clip_io import read_frames                    # noqa: E402
from darken import READ_NOISE, clip_k, clip_rng, darken_frames  # noqa: E402

NEW_CLASSES = ["throw", "fall", "fight", "talk", "shake_hands"]
K700_VAL_CAP = 100
OTHER_TRAIN_CAP = 300
OTHER_VAL_CAP = 60


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("device")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--cap", type=int, default=450)
    ap.add_argument("--out", default=os.path.join(BC.CACHE, "proxy_rgb_v1"))
    a = ap.parse_args()

    rng0 = random.Random(0)
    rows = {"train": [], "val": []}
    per = {}
    for sp, cap in (("train", a.cap), ("val", K700_VAL_CAP)):
        pool = {b: [] for b in NEW_CLASSES}
        others = []
        for p, beh, _, _ in BM.k700_rows(sp):
            if beh in pool:
                pool[beh].append(p)
            elif beh == "other":
                others.append(p)
        for b in NEW_CLASSES:
            take = sorted(pool[b])
            rng0.shuffle(take)
            rows[sp] += [(p, b) for p in take[:cap]]
            per[(b, sp)] = (min(len(take), cap), len(take))
        oc = OTHER_TRAIN_CAP if sp == "train" else OTHER_VAL_CAP
        take_o = sorted(others)
        rng0.shuffle(take_o)
        rows[sp] += [(p, "other") for p in take_o[:oc]]
        per[("other", sp)] = (oc, len(others))
    for sp, dst in (("train", "train"), ("validation", "train"), ("test", "val")):
        for p, cls, _ in BM.hmdb_clips(sp):
            beh = BM.HMDB_TO_BEHAVIOR.get(cls)
            if beh in NEW_CLASSES:
                rows[dst].append((p, beh))

    items = [it for sp in ("train", "val") for it in rows[sp]]
    i, n = map(int, a.shard.split("/"))
    items = [it for k, it in enumerate(items) if k % n == i]
    print(f"[proxy] {len(items)} clips shard={a.shard} cap={a.cap} -> {a.out}", flush=True)
    print("[proxy] K700 caps: " + "  ".join(
        f"{b}={per[(b, 'train')][0]}/{per[(b, 'train')][1]}" for b in NEW_CLASSES), flush=True)

    enh = BC.build_enhancer("retinexformer", a.device)
    t0 = done = skipped = bad = 0
    for path, beh in items:
        # cache key must not collide between K700 and HMDB copies of the same basename
        tag = "k7" if "/k700/" in path.replace(os.sep, "/") else "hm"
        out = os.path.join(a.out, beh, f"{tag}_{os.path.basename(path)[: -4]}.npy")
        if os.path.exists(out):
            skipped += 1
            continue
        is_k700 = tag == "k7"
        span = BM.K700_SPAN_SEC if is_k700 else 0.0
        frames = read_frames(path, BC.T, span=span or None,
                             span_rng=clip_rng(path, seed=1) if span else None)
        if frames is None:
            bad += 1
            print(f"  [bad] {path}", flush=True)
            continue
        rng = clip_rng(path)
        frames = darken_frames(np.stack(frames), clip_k(rng), rng, read_noise=READ_NOISE)
        frames = enh(list(frames))
        arr = np.stack([BC.short_resize_rgb(f) for f in frames]).astype(np.uint8)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.save(out, arr)
        done += 1
        if done % 100 == 0:
            print(f"  [proxy shard{a.shard}] {done} cached "
                  f"({(time.time()-t0)/max(done,1):.2f}s/clip)", flush=True)
    print(f"[proxy] DONE done={done} skipped={skipped} bad={bad} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
