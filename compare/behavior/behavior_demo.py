"""End-to-end annotated demo clips: one per behavior, through the shipped CLI.

This is the artifact you watch rather than read. For each behavior it takes a held-out test
clip, darkens it with the same calibrated simulation the training cache used, and runs the
real `darkpipe` entry point over it -- enhancer, recognizer and label bar included. Nothing
here reimplements the pipeline, so what the video shows is what the pipeline does.

Clip choice is deterministic (first by sorted filename within the behavior), NOT "a clip the
model gets right". The printed table and `demo_index.json` report the predicted label next to
the true one, mistakes included; picking winners would make the reel a slideshow of the
model's best moments rather than evidence.
"""
import argparse
import json
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DP = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", ".."))
sys.path.insert(0, DP)
import behavior_manifest as BM                                       # noqa: E402
import darken as DK                                                  # noqa: E402
from darkpipe.constants import BEHAVIOR_DISPLAY                      # noqa: E402

PY = os.path.join(DP, ".venv", "bin", "python")
RESULTS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "results"))
WORK = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", "behavior_data", "demo"))


def darken_clip(src, dst):
    """Whole-clip darkening at the clip's own calibrated exposure -> dst, returns (k, frames)."""
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        return None, 0
    rng = DK.clip_rng(src)                     # same exposure this clip had in training
    k = DK.clip_k(rng)
    dark = DK.darken_frames(np.stack(frames), k, rng)
    h, w = dark.shape[1:3]
    wr = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in dark:
        wr.write(f)
    wr.release()
    return k, len(dark)


def run_pipeline(src, out, events, enhance, recognize, device, extra=()):
    cmd = [PY, "-m", "darkpipe.cli", "--mode", "offline", "--input", src, "--output", out,
           "--enhance", enhance, "--sr", "off", "--recognize", recognize, "--device", device,
           "--ckpt-dir", os.path.join(DP, "ckpts"), "--events-json", events, *extra]
    p = subprocess.run(cmd, cwd=DP, capture_output=True, text=True)
    if p.returncode != 0:
        return None, (p.stderr or p.stdout).strip().splitlines()[-1][:200]
    return json.load(open(events)) if os.path.exists(events) else [], ""


def majority(evs):
    if not evs:
        return "none", 0.0
    labs = [e["label"] for e in evs]
    top = max(set(labs), key=labs.count)
    conf = float(np.mean([e["confidence"] for e in evs if e["label"] == top]))
    return top, conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enhance", default="retinexformer", choices=["retinexformer", "cidnet"])
    ap.add_argument("--recognize", default="behavior", choices=["behavior", "xclip"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()

    os.makedirs(WORK, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    rows = BM.build(a.split)
    by_lab = {}
    for path, lab, cls, sp in sorted(rows, key=lambda r: os.path.basename(r[0])):
        by_lab.setdefault(lab, (path, cls))

    index, hits = [], 0
    for lab in BM.BEHAVIORS:
        if lab not in by_lab:
            print(f"  {lab:<12} no clip in the {a.split} split")
            continue
        src, cls = by_lab[lab]
        dark = os.path.join(WORK, f"{lab}_dark.mp4")
        out = os.path.join(RESULTS, f"demo_{lab}_{a.recognize}.mp4")
        ev_json = os.path.join(WORK, f"{lab}_events.json")
        k, n = darken_clip(src, dark)
        if not n:
            print(f"  {lab:<12} unreadable: {src}")
            continue
        evs, err = run_pipeline(dark, out, ev_json, a.enhance, a.recognize, a.device)
        if evs is None:
            print(f"  {lab:<12} FAILED {err}")
            continue
        pred, conf = majority(evs)
        # events carry the human-readable label the bar shows ("picking up things"), not the
        # raw class name ("pick_up") -- compare in the same space the renderer uses.
        ok = pred == BEHAVIOR_DISPLAY.get(lab, lab)
        hits += ok
        index.append(dict(behavior=lab, hmdb_class=cls, source=os.path.basename(src),
                          frames=n, darken_k=round(k, 5), events=len(evs),
                          predicted=pred, confidence=round(conf, 3), correct=bool(ok),
                          video=os.path.relpath(out, RESULTS)))
        print(f"  {lab:<12} k={k:<8.5f} {n:>3}f  {len(evs):>2} events  "
              f"-> {pred:<12} {conf:.2f}  {'ok' if ok else 'MISS'}   {os.path.basename(out)}")

    p = os.path.join(RESULTS, f"demo_index_{a.recognize}.json")
    json.dump(index, open(p, "w"), indent=1)
    print(f"\n  {hits}/{len(index)} demo clips labelled correctly (single clip per behavior "
          f"-- illustrative, not a metric; see the eval tables)")
    print(f"[demo] -> {p}")


if __name__ == "__main__":
    main()
