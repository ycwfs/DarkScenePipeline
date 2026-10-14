"""Gate sweep + luma analysis over a prior_eval.py JSONL.

The run must have been executed with --lum-gate 0 (gate off) so every window carries its
raw three-step outcome; the step-1 gate is then swept OFFLINE: for a candidate threshold
t, windows with lum < t replay as RGB-only (p_rgb + reject_tau -- exactly the deployed
below-gate path), windows with lum >= t keep their recorded max_prior vector. The gate
changes nothing above it, so the mix is exact.

Prints:
  1. per-suite luma distribution (the field-data picture the meeting asked for);
  2. suite A binned by luma: RGB-only top1 vs max_prior top1 and skeleton reliability
     (mean qualifying persons) -- where the skeleton stops earning its keep;
  3. the sweep table: for each t, all four suites' headline numbers, with the deployed
     stack's alpha baselines alongside;
  4. the 两部曲 sweep (2026-10-14): alpha blend + gate -- below t pure RGB, above t the
     recorded a0.80/a0.70 blend, per t.

Usage: python lum_sweep.py [runs.jsonl]
"""
import json
import sys
from collections import defaultdict

import numpy as np

LABELS = ["wave", "throw", "chase", "fall", "fight", "talk", "drink", "pick_up",
          "shake_hands", "other"]
OTHER = LABELS.index("other")
REJECT = 0.65


def fired(p):
    return LABELS[int(np.argmax(p))] != "other" and float(np.max(p)) >= REJECT


def rgb_only(row):
    """The deployed below-gate path: the RGB vector + the open-set reject, reproducing
    recognize.reject_probs exactly (kept: named entries, other = max(1-named, named+eps))."""
    p = np.array(row["rgb"], float)
    named = np.delete(p, OTHER)
    nm = float(named.max())
    if nm < REJECT and int(np.argmax(p)) != OTHER:
        q = p.copy()
        q[OTHER] = max(1.0 - nm, nm + 1e-6)
        return q
    return p


def outcome(row, t):
    return rgb_only(row) if row["lum"] < t else np.array(row["mp"])


def alpha_outcome(row, t, key):
    """The meeting "两部曲" (2026-10-14 instruction.md revision): below the gate pure RGB,
    above it the RECORDED alpha blend (a0.80 / a0.70). Exact offline mix, same argument as
    `outcome`: the gate changes nothing at or above it, and the blend is per-window."""
    return rgb_only(row) if row["lum"] < t else np.array(row[key])


def main(path):
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    print(f"{len(rows)} windows from {path}\n")

    print("== 1. luma distribution (RAW pre-enhance window mean luma, 0-255) ==")
    for suite in "ABCP":
        sub = [r["lum"] for r in rows if r["suite"] == suite]
        if not sub:
            continue
        q = np.percentile(sub, [0, 10, 25, 50, 75, 90, 100])
        print(f"  {suite}: min {q[0]:5.1f}  p10 {q[1]:5.1f}  p25 {q[2]:5.1f}  "
              f"p50 {q[3]:5.1f}  p75 {q[4]:5.1f}  p90 {q[5]:5.1f}  max {q[6]:5.1f}")

    sub = [r for r in rows if r["suite"] == "A"]
    if sub:
        print("\n== 2. suite A by luma bin (skeleton reliability vs RGB-only accuracy) ==")
        edges = [0, 15, 30, 45, 60, 75, 90, 256]      # raw-luma bins, knee at 75
        print("  bin          n   nvalid  top1(rgb-only)  top1(max_prior)")
        for lo, hi in zip(edges, edges[1:]):
            b = [r for r in sub if lo <= r["lum"] < hi]
            if not b:
                continue
            tr = np.mean([LABELS[int(np.argmax(rgb_only(r)))] == r["expect"] for r in b])
            tm = np.mean([LABELS[int(np.argmax(outcome(r, 0)))] == r["expect"] for r in b])
            print(f"  {lo:3d}-{hi:3d}  {len(b):4d}   "
                  f"{np.mean([r['nvalid'] for r in b]):5.2f}      {tr:8.3f}      {tm:8.3f}")

    print("\n== 3. gate sweep (t = below-gate windows replay as RGB-only) ==")
    print("  t      A top1   A mF1    B fired   P fired   C fired")
    sub_b = [r for r in rows if r["suite"] == "B"]
    sub_p = [r for r in rows if r["suite"] == "P"]
    sub_c = [r for r in rows if r["suite"] == "C"]
    for t in [0] + [30, 40, 50, 60, 65, 70, 75, 80, 90, 105, 120, 135, 150]:
        line = f" {t:4d} " if t else "  off "
        if sub:
            cm = np.zeros((10, 10), np.int64)
            for r in sub:
                cm[LABELS.index(r["expect"]), int(np.argmax(outcome(r, t)))] += 1
            tp = np.diag(cm).astype(float)
            f1 = 2 * tp / np.maximum(cm.sum(0) + cm.sum(1), 1e-9)
            line += f" {tp.sum() / cm.sum():.4f} {f1[cm.sum(1) > 0].mean():.4f}"
        else:
            line += "   -      -"
        fb = sum(fired(outcome(r, t)) for r in sub_b)
        fp = sum(fired(outcome(r, t)) for r in sub_p)
        fc = sum(fired(outcome(r, t)) for r in sub_c)
        print(line + f"   {fb:3d}/{len(sub_b):<3d}  {fp:3d}/{len(sub_p):<3d}  "
                     f"{fc:2d}/{len(sub_c)}")

    print("\n  baselines from the same run: alpha0.8 / alpha0.7")
    for key in ("a0.80", "a0.70"):
        if sub:
            cm = np.zeros((10, 10), np.int64)
            for r in sub:
                cm[LABELS.index(r["expect"]), int(np.argmax(r[key]))] += 1
            tp = np.diag(cm).astype(float)
            f1 = 2 * tp / np.maximum(cm.sum(0) + cm.sum(1), 1e-9)
            a = f"top1={tp.sum() / cm.sum():.4f} mF1={f1[cm.sum(1) > 0].mean():.4f}"
        else:
            a = "-"
        fb = sum(fired(np.array(r[key])) for r in sub_b)
        fp = sum(fired(np.array(r[key])) for r in sub_p)
        fc = sum(fired(np.array(r[key])) for r in sub_c)
        print(f"  {key}: {a}  B {fb}/{len(sub_b)}  P {fp}/{len(sub_p)}  C {fc}/{len(sub_c)}")

    print("\n== 4. 两部曲 sweep (alpha blend + gate: below t pure RGB, above t the"
          " recorded blend) ==")
    for key, al in (("a0.80", "0.80"), ("a0.70", "0.70")):
        print(f"  -- alpha={al} --")
        print("  t      A top1   A mF1    B fired   P fired   C fired")
        for t in [0] + [30, 40, 50, 60, 65, 70, 75, 80, 90, 105, 120, 135, 150]:
            line = f" {t:4d} " if t else "  off "
            if sub:
                cm = np.zeros((10, 10), np.int64)
                for r in sub:
                    cm[LABELS.index(r["expect"]),
                       int(np.argmax(alpha_outcome(r, t, key)))] += 1
                tp = np.diag(cm).astype(float)
                f1 = 2 * tp / np.maximum(cm.sum(0) + cm.sum(1), 1e-9)
                line += f" {tp.sum() / cm.sum():.4f} {f1[cm.sum(1) > 0].mean():.4f}"
            else:
                line += "   -      -"
            fb = sum(fired(alpha_outcome(r, t, key)) for r in sub_b)
            fp = sum(fired(alpha_outcome(r, t, key)) for r in sub_p)
            fc = sum(fired(alpha_outcome(r, t, key)) for r in sub_c)
            print(line + f"   {fb:3d}/{len(sub_b):<3d}  {fp:3d}/{len(sub_p):<3d}  "
                         f"{fc:2d}/{len(sub_c)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/prior_eval_runs_v2.jsonl")
