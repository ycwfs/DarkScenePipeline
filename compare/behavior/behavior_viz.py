"""Figures for the behavior study, read straight from the eval JSONs (no hardcoded numbers).

Three artifacts:
  behavior_confusion_<tag>.png   row-normalized confusion, one per eval run
  behavior_per_class_f1.png      per-class F1 across the compared models
  behavior_model_compare.png     macro-F1 / top-1 summary

Row-normalized because the support ranges from 9 (`chase`) to 339 (`other`); a raw-count
heatmap would show one bright column and nothing else.
"""
import argparse
import glob
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.normpath(os.path.join(HERE, "..", "results"))

SETS = {"arid_real": "ARID test, real dark video (1768 clips, 4 behaviors)",
        "hmdb_bright": "HMDB test, original bright video (611 clips)",
        "hmdb_dark": "HMDB test, darkened + enhanced (611 clips)"}


def set_of(tag):
    """Which test set a run was scored on -- runs are only comparable within one."""
    if "arid" in tag:
        return "arid_real"
    return "hmdb_bright" if "bright" in tag else "hmdb_dark"


def load(patterns):
    out = []
    for pat in patterns:
        for p in sorted(glob.glob(os.path.join(RESULTS, pat))):
            m = json.load(open(p))
            if "confusion" in m:
                out.append(m)
    return out


def confusion_fig(m):
    labels = m["labels"]
    cm = np.array(m["confusion"], float)
    sup = cm.sum(1, keepdims=True)
    norm = np.divide(cm, sup, out=np.zeros_like(cm), where=sup > 0)
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    im = ax.imshow(norm, cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([f"{b} (n={int(sup[i, 0])})" for i, b in enumerate(labels)])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.55 else "black")
    ax.set_title(f"{m['tag']}   n={m['n']}  top-1={m['top1']:.3f}  macro-F1={m['macro_f1']:.3f}",
                 fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="fraction of the true class")
    fig.tight_layout()
    out = os.path.join(RESULTS, f"behavior_confusion_{m['tag']}.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def per_class_fig(runs, title, name):
    """Per-behavior F1 for runs that share a test set.

    Callers must group by support first: the x labels carry each behavior's clip count, and
    one shared axis for ARID (4 behaviors, 1768 clips) and HMDB (10, 611) would print one
    set of counts under bars measured on the other.
    """
    labels = runs[0]["labels"]
    x = np.arange(len(labels))
    w = 0.8 / len(runs)
    fig, ax = plt.subplots(figsize=(12, 5))
    for k, m in enumerate(runs):
        ax.bar(x + k * w - 0.4 + w / 2, m["per_class_f1"], w, label=m["tag"])
    sup = runs[0]["support"]
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\nn={sup[i]}" for i, b in enumerate(labels)], fontsize=9)
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    out = os.path.join(RESULTS, name)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def compare_fig(runs, name="behavior_model_compare.png"):
    tags = [m["tag"] for m in runs]
    x = np.arange(len(tags))
    fig, ax = plt.subplots(figsize=(1.9 * len(tags) + 3, 4.8))
    for k, (key, colour) in enumerate((("top1", "#8a8f98"), ("macro_f1", "#4c78c9"))):
        v = [m[key] for m in runs]
        b = ax.bar(x + (k - 0.5) * 0.38, v, 0.38, label=key, color=colour)
        for r, val in zip(b, v):
            ax.text(r.get_x() + r.get_width() / 2, val + 0.008, f"{val:.3f}",
                    ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", "\n") for t in tags], fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    ax.set_title("Behavior recognition: top-1 vs macro-F1\n"
                 "top-1 is dominated by `other` (55% of the test split) — macro-F1 is the "
                 "metric that reflects the nine behaviors", fontsize=11)
    fig.tight_layout()
    out = os.path.join(RESULTS, name)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", nargs="*", default=["behavior_eval_*.json"],
                    help="eval JSONs to plot, relative to compare/results")
    a = ap.parse_args()
    runs = load(a.glob)
    if not runs:
        raise SystemExit(f"no eval JSONs matched {a.glob} in {RESULTS}")
    for m in runs:
        print("saved", confusion_fig(m))

    # Validation runs exist only to calibrate tau; putting them on the headline chart next to
    # test runs would read as if the models had been scored on eight test sets, not three.
    test = [m for m in runs if "_val_" not in m["tag"]]
    groups = {}
    for m in test:
        groups.setdefault(SETS.get(set_of(m["tag"]), set_of(m["tag"])), []).append(m)
    for key, g in groups.items():
        print("saved", per_class_fig(
            g, f"Per-behavior F1 — {key}\n`chase` carries weak, detector-curated labels and "
               f"is reported separately", f"behavior_per_class_f1_{set_of(g[0]['tag'])}.png"))
    print("saved", compare_fig(test))


if __name__ == "__main__":
    main()
