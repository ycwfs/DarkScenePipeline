"""Significance testing over saved `behavior_eval.py` runs. Point estimates decide nothing.

    python behavior_sig.py A.json B.json [C.json ...]        # tags taken from each file
    python behavior_sig.py --name OLD=a.json --name NEW=b.json

Every pair gets a paired bootstrap CI on macro-F1 and an exact McNemar test on top-1, both
computed over the per-clip `y_true`/`y_pred` the evaluator already stores. Paired, because the
runs score the SAME clips: an unpaired comparison throws away exactly the correlation that
makes small differences resolvable.

This exists because a point-estimate gate on HMDB test (611 clips, 9-34 per class) produced a
per-class "regression" that a second finetune was then spent chasing. It was noise -- every
difference between the three checkpoints came back p > 0.25, including in classes whose mapping
was never touched. Run this before reporting a winner or diagnosing a per-class delta.

`--per-class` adds a McNemar test per behavior (predicted-as-c vs actually-c, i.e. whether the
two models disagree on that class more than chance), for checking that classes a training set
could not teach did not regress.
"""
import argparse
import itertools
import json
import math
import os

import numpy as np

BOOT = 2000
SEED = 0


def macro_f1(yt, yp, K):
    """Macro-F1 over classes PRESENT in `yt` — matches `behavior_eval.metrics`."""
    f = []
    for c in range(K):
        tp = int(np.sum((yp == c) & (yt == c)))
        fn = int(np.sum((yp != c) & (yt == c)))
        if tp + fn == 0:                       # class absent from this resample: skip, do not
            continue                           # score it 0 (that would penalise rare classes)
        fp = int(np.sum((yp == c) & (yt != c)))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f)) if f else 0.0


def binom_two_sided(k, n):
    """Exact two-sided binomial test against p=0.5. Symmetric, so twice the smaller tail.

    Exact rather than the chi-square approximation because the discordant counts here are often
    small enough (n01+n10 < 25) that the approximation is not trustworthy, and `math.comb` makes
    the exact form free — scipy is not installed in this venv.
    """
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def mcnemar(correct_a, correct_b):
    """-> (b fixes, b breaks, p). Discordant pairs only; concordant ones carry no information."""
    n01 = int((~correct_a & correct_b).sum())
    n10 = int((correct_a & ~correct_b).sum())
    return n01, n10, binom_two_sided(n01, n01 + n10)


def load(spec):
    name, _, path = spec.rpartition("=")
    d = json.load(open(path))
    return (name or d.get("tag") or os.path.basename(path)[:-5]), d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="eval JSONs, optionally NAME=path.json")
    ap.add_argument("--per-class", action="store_true",
                    help="also McNemar each behavior separately (regression check)")
    ap.add_argument("--boot", type=int, default=BOOT)
    a = ap.parse_args()

    got = dict(load(s) for s in a.runs)
    if len(got) < 2:
        raise SystemExit("error: need at least two runs to compare")
    first = next(iter(got.values()))
    yt, labels = np.array(first["y_true"]), first["labels"]
    K = len(labels)
    for name, d in got.items():
        # The bootstrap indexes both runs with the same resample, so a differing clip order
        # would silently pair clip i of one run with a different clip of the other.
        if not np.array_equal(yt, np.array(d["y_true"])):
            raise SystemExit(f"error: {name} scored a different clip set/order than {list(got)[0]}")
    yp = {n: np.array(d["y_pred"]) for n, d in got.items()}
    src = f"{first.get('source', '?')} {first.get('split', '')}".strip()

    print(f"=== {src}  n={len(yt)}  runs: {', '.join(got)}")
    print(f"  {'run':<22}{'top1':>9}{'macroF1':>9}")
    for n in got:
        print(f"  {n:<22}{float((yp[n] == yt).mean()):>9.4f}{macro_f1(yt, yp[n], K):>9.4f}")

    print(f"\n--- macro-F1, {a.boot}-resample paired bootstrap")
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(yt), size=(a.boot, len(yt)))
    for x, y in itertools.combinations(got, 2):
        d = np.array([macro_f1(yt[i], yp[y][i], K) - macro_f1(yt[i], yp[x][i], K) for i in idx])
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = 2 * min(float((d <= 0).mean()), float((d >= 0).mean()))
        print(f"  {y} - {x} = {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  p={p:.3f}"
              f"  {'SIGNIFICANT' if p < 0.05 else 'ns'}")

    print("\n--- top-1, McNemar exact")
    for x, y in itertools.combinations(got, 2):
        n01, n10, p = mcnemar(yp[x] == yt, yp[y] == yt)
        print(f"  {y} vs {x}: fixes {n01}, breaks {n10} (net {n01 - n10:+d} of {len(yt)})"
              f"  p={p:.4f}  {'SIGNIFICANT' if p < 0.05 else 'ns'}")

    if a.per_class:
        for x, y in itertools.combinations(got, 2):
            print(f"\n--- per-class recall, {y} vs {x} (McNemar on that class's clips)")
            print(f"  {'behavior':<13}{'sup':>6}{'  ' + x[:9]:>11}{'  ' + y[:9]:>11}"
                  f"{'fix/brk':>10}{'p':>8}")
            for c, b in enumerate(labels):
                m = yt == c
                if not m.sum():
                    continue
                ca, cb = (yp[x][m] == c), (yp[y][m] == c)
                n01, n10, p = mcnemar(ca, cb)
                flag = "  <-- SIGNIFICANT" if p < 0.05 else ""
                print(f"  {b:<13}{int(m.sum()):>6}{ca.mean():>11.3f}{cb.mean():>11.3f}"
                      f"{f'+{n01}/-{n10}':>10}{p:>8.3f}{flag}")


if __name__ == "__main__":
    main()
