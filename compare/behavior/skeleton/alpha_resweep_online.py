"""Replay deployed fuse+reject at several alphas from reco_video_vis.py CSVs.

The CSV stores per-window rgb_vec / skel_vec / presence. Fusion (skeleton_aux.fuse,
TAU=0.5) is linear in alpha, so decisions at alpha in {0.9, 0.8, 0.7} are recomputed
exactly without re-running any model:
    pres >= 0.5:  fused = a*rgb + (1-a)*skel
    pres <  0.5:  fused = a*rgb + (1-a)*onehot(other)      (soft no-person)
    dec    = named argmax if its prob >= 0.65 else other    (reject_tau)
At alpha=1.0 the recompute is checked against the CSV's own fused_vec/dec columns.

Usage: python3 alpha_resweep_online.py <csv_dir>
"""
import csv
import glob
import os
import sys

LABELS = ["wave", "throw", "chase", "fall", "fight",
          "talk", "drink", "pick_up", "shake_hands", "other"]
OTHER = LABELS.index("other")
TAU = 0.5
REJECT = 0.65


def dec_of(fused):
    named = max((p, i) for i, p in enumerate(fused) if i != OTHER)
    if named[0] >= REJECT:
        return LABELS[named[1]], named[0]
    return "other", fused[OTHER]


def fused_at(a, rgb, skel, pres):
    if pres >= TAU:
        return [a * r + (1 - a) * s for r, s in zip(rgb, skel)]
    out = [0.0] * len(LABELS)
    out[OTHER] = 1.0
    return [a * r + (1 - a) * o for r, o in zip(rgb, out)]


def main():
    csv_dir = sys.argv[1]
    alphas = [1.0, 0.9, 0.8, 0.7]
    n_fire = {a: 0 for a in alphas}
    n_clip_fire = {a: 0 for a in alphas}
    n_clip = 0
    n_win = 0
    bad = 0
    for path in sorted(glob.glob(os.path.join(csv_dir, "*.csv"))):
        stem = os.path.basename(path)[:-4]
        per_alpha_fire = {a: [] for a in alphas}
        for row in csv.DictReader(open(path, encoding="utf-8-sig")):
            n_win += 1
            rgb = [float(x) for x in row["rgb_vec"].split()]
            sk = row["skel_vec"]
            skel = [float(x) for x in sk.split()] if sk != "-" else None
            pres = float(row["presence"])
            # sanity at alpha=1 against the tool's own columns
            if skel is not None:
                f1 = fused_at(1.0, rgb, skel, pres)
                ref = [float(x) for x in row["fused_vec"].split()]
                if any(abs(x - y) > 2e-3 for x, y in zip(f1, ref)):
                    bad += 1
            rec = {"stem": stem, "frame": row["frame"], "t": row["t_s"],
                   "pres": pres, "rgb_top": row["rgb_top"], "rgb_p": row["rgb_p"]}
            for a in alphas:
                if skel is None:
                    lbl, p = (row["dec_top"], row["dec_p"])  # alpha>=0.999 pure RGB
                else:
                    lbl, p = dec_of(fused_at(a, rgb, skel, pres))
                if lbl != "other":
                    n_fire[a] += 1
                    per_alpha_fire[a].append((row["frame"], lbl, p, pres))
        n_clip += 1
        fire08 = per_alpha_fire[0.8]
        if fire08:
            f, l, p, pr = fire08[0]
            print(f"{stem}: fire@0.8 {len(fire08)}  first f{f} {l} {p:.3f} pres={pr:.2f}")
        for a in alphas:
            if per_alpha_fire[a]:
                n_clip_fire[a] += 1
    print(f"\nwindows={n_win}  clips={n_clip}  alpha1-recompute-mismatches={bad}")
    for a in alphas:
        print(f"fires  alpha={a:g}: windows {n_fire[a]}  clips {n_clip_fire[a]}/{n_clip}")


if __name__ == "__main__":
    main()
