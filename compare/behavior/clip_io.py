"""Clip -> model-ready tensor, shared by the eval, cache and training scripts.

Preprocessing is imported from darkpipe (`preprocess_frame`) rather than reimplemented, so
what a model is evaluated/trained on is bit-identical to what the live pipeline feeds it.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "..", "..")))
from darkpipe.constants import RECO_CFG                      # noqa: E402
from darkpipe.stages.recognize import preprocess_frame       # noqa: E402


def _window(total, fps, span, rng):
    """Frame bounds [lo, hi) of a contiguous `span`-second window, or the whole clip.

    Kinetics-700 clips are a fixed 10 s where HMDB's median is 2.6 s, so sampling n frames
    across a whole K700 clip would train the recognizer on ~4x slower motion than it sees at
    inference. `span` cuts a clip down to the same temporal footprint before sampling.
    """
    if not span or fps <= 0:
        return 0, total
    win = int(round(span * fps))
    if win >= total:
        return 0, total
    off = int(rng.integers(0, total - win + 1)) if rng is not None else (total - win) // 2
    return off, off + win


def read_frames(path, n, jitter=False, rng=None, span=None, span_rng=None):
    """n frames sampled uniformly across the clip (or a `span`-second window of it), BGR uint8.

    `jitter` picks a random offset inside each of the n temporal segments (TSN-style training
    augmentation); otherwise segment centers are used, which is the deterministic eval path.

    `span` (seconds) restricts sampling to one contiguous window, positioned by `span_rng` — a
    separate generator from `rng` so a clip's window offset and its exposure draw stay
    independent. Without `span` the behaviour is unchanged.
    """
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if total <= 0:                                   # some HMDB mp4s report 0 -> decode all
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        if not frames:
            return None
        lo, hi = _window(len(frames), fps, span, span_rng)
        idx = np.linspace(lo, hi - 1, n).round().astype(int)
        return [frames[i] for i in idx]

    lo, hi = _window(total, fps, span, span_rng)
    edges = np.linspace(lo, hi, n + 1)
    if jitter:
        rng = rng or np.random.default_rng()
        idx = [int(min(hi - 1, rng.integers(int(edges[i]), max(int(edges[i]) + 1,
                                                               int(edges[i + 1])))))
               for i in range(n)]
    else:
        idx = [int(min(hi - 1, (edges[i] + edges[i + 1]) / 2)) for i in range(n)]

    out, want = [], sorted(set(idx))
    got, k, i = {}, 0, 0
    while k < len(want):
        ok, f = cap.read()
        if not ok:
            break
        if i == want[k]:
            got[i] = f
            k += 1
        i += 1
    cap.release()
    if not got:
        return None
    last = got[min(got)]
    for j in idx:                                    # tolerate short/truncated decodes
        last = got.get(j, last)
        out.append(last)
    return out


def to_tensor(frames_bgr, kind="xclip"):
    """[BGR uint8] -> (T,H,W,C) float32, normalized for `kind`."""
    cfg = RECO_CFG[kind]
    return np.stack([preprocess_frame(f, cfg) for f in frames_bgr])


def load_clip(path, kind="xclip", jitter=False, rng=None, span=None, span_rng=None):
    cfg = RECO_CFG[kind]
    frames = read_frames(path, cfg["T"], jitter=jitter, rng=rng, span=span, span_rng=span_rng)
    return None if frames is None else to_tensor(frames, kind)
