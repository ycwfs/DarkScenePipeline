"""Aggregate the ARID recognition A/B: off vs CIDNet vs Retinexformer.

Reads the three per-clip JSON dumps, joins by clip, and reports overall + per-class
top-1, enhancer agreement, win/lose breakdown, and a grouped per-class bar chart.
"""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/ssd1/wfs/project/lowlight/compare/results"
CLASSES = ["Drink", "Jump", "Pick", "Pour", "Push", "Run", "Sit", "Stand", "Turn", "Walk", "Wave"]


def load(name):
    with open(f"{R}/arid_{name}.json") as f:
        d = json.load(f)
    d["by_rel"] = {p["rel"]: p for p in d["preds"]}
    return d


def main():
    off, cid, rf = load("off"), load("cidnet"), load("rf")
    print(f"{'enhancer':<16}{'top1':>8}{'mean_conf':>12}{'n':>7}")
    for name, d in [("off (dark)", off), ("cidnet", cid), ("retinexformer", rf)]:
        mc = np.mean([p["conf"] for p in d["preds"]])
        print(f"{name:<16}{d['top1']:>8.4f}{mc:>12.4f}{d['n']:>7}")

    # join on clips present in both cid & rf
    rels = [r for r in cid["by_rel"] if r in rf["by_rel"]]
    cc = rc = only_c = only_r = both_wrong = agree = 0
    for r in rels:
        pc, pr = cid["by_rel"][r], rf["by_rel"][r]
        gt = pc["gt"]
        c_ok, r_ok = pc["pred"] == gt, pr["pred"] == gt
        agree += pc["pred"] == pr["pred"]
        cc += c_ok and r_ok
        only_c += c_ok and not r_ok
        only_r += r_ok and not c_ok
        both_wrong += not c_ok and not r_ok
    n = len(rels)
    print(f"\njoined clips: {n}")
    print(f"prediction agreement (cidnet==rf): {agree/n:.4f}")
    print(f"both correct:      {cc:>5} ({cc/n:.3f})")
    print(f"cidnet-only right: {only_c:>5} ({only_c/n:.3f})")
    print(f"rf-only right:     {only_r:>5} ({only_r/n:.3f})")
    print(f"both wrong:        {both_wrong:>5} ({both_wrong/n:.3f})")

    # per-class table
    print(f"\n{'class':<8}{'off':>8}{'cidnet':>8}{'rf':>8}")
    cc_pc, rf_pc, off_pc = [], [], []
    for ci, cname in enumerate(CLASSES):
        o = off["per_class"].get(cname, 0.0)
        c = cid["per_class"].get(cname, 0.0)
        r = rf["per_class"].get(cname, 0.0)
        off_pc.append(o); cc_pc.append(c); rf_pc.append(r)
        print(f"{cname:<8}{o:>8.3f}{c:>8.3f}{r:>8.3f}")

    # grouped bar chart
    x = np.arange(len(CLASSES)); w = 0.27
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - w, off_pc, w, label=f"off (dark)  {off['top1']:.3f}", color="#8a8f98")
    ax.bar(x, cc_pc, w, label=f"HVI-CIDNet  {cid['top1']:.3f}", color="#e08a3c")
    ax.bar(x + w, rf_pc, w, label=f"Retinexformer  {rf['top1']:.3f}", color="#4c78c9")
    ax.set_xticks(x); ax.set_xticklabels(CLASSES, rotation=30)
    ax.set_ylabel("top-1 accuracy"); ax.set_ylim(0, 1.0)
    ax.set_title("ARID split-1 test: VideoMamba action recognition by low-light enhancer "
                 f"(n={rf['n']} clips)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{R}/arid_recognition_by_class.png", dpi=110)
    print(f"\nsaved {R}/arid_recognition_by_class.png")


if __name__ == "__main__":
    main()
