"""Gate sweep + luma analysis over a prior_eval.py JSONL.

Requires schema_version=2 from prior_eval.py: full precision probabilities and luma,
with explicit ungated vectors. Old rounded files cannot reproduce threshold decisions
exactly and must be regenerated. The original run may use any lum_gate.

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
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[3] / "platform/dark-demo-v2"
sys.path.insert(0, str(PKG))
from darkpipe.stages.recognize import reject_probs

LABELS = ["wave", "throw", "chase", "fall", "fight", "talk", "drink", "pick_up",
          "shake_hands", "other"]
OTHER = LABELS.index("other")
REJECT = 0.65


def fired(p):
    return LABELS[int(np.argmax(p))] != "other" and float(np.max(p)) >= REJECT


def validate_row(row):
    if (row.get("schema_version") != 2 or row.get("probability_dtype") != "float32"
            or not all(k in row for k in ("rgb", "mp_ungated", "a0.70_ungated",
                                         "a0.80_ungated", "lum", "reject_tau", "no_person"))):
        raise ValueError("旧版/不完整 JSONL 无法精确回放；请用新版 prior_eval.py 重新导出")
    if row["reject_tau"] != REJECT:
        raise ValueError("当前报表仅支持生产阈值 0.65；请以此阈值重新导出")


def rgb_only(row):
    return reject_probs(np.asarray(row["rgb"], dtype=np.float32), LABELS, row["reject_tau"])


def outcome(row, t):
    validate_row(row)
    return rgb_only(row) if row["lum"] < t else np.asarray(row["mp_ungated"], np.float32)


def alpha_outcome(row, t, key):
    validate_row(row)
    return rgb_only(row) if row["lum"] < t else np.asarray(row[key + "_ungated"], np.float32)


def main(path):
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    for row in rows:
        validate_row(row)
    print(f"{len(rows)} windows from {path}\n")
    print("no_person policies:", sorted({r["no_person"] for r in rows}))

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
