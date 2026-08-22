#!/usr/bin/env python3
"""Grid comparison: NTIRE baseline, the real-deploy fine-tune, and every other official
Retinexformer pretrained checkpoint shipped in ckpts/ (SID/SMID/SDSD_indoor/SDSD_outdoor/
LOL_v1/LOL_v2_real/LOL_v2_synthetic/FiveK) -- all share NTIRE's exact architecture
(n_feat=40, stage=1, num_blocks=[1,2,2]), so they're drop-in loadable through the same
RetinexformerStage. One frame per scene, one grid per scene, stacked into a single PNG.

Loads each checkpoint's model ONCE and reuses it across all scenes (rather than reloading
per scene x variant), since that's the expensive part.

    python finetune_pretrained_compare.py --out out.png
"""
import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DSP = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
DATA = os.path.normpath(os.path.join(DSP, "..", "real-development-data", "输入暗源"))
CKPTS = os.path.join(DSP, "ckpts")

from darkpipe.config import PipelineConfig, validate  # noqa: E402
from darkpipe.stages import build_stages  # noqa: E402
from darkpipe.stages.downscale import DownscaleStage  # noqa: E402

SCENES = [
    ("报告厅", os.path.join(DATA, "输入暗源-报告厅.mp4"), 50),
    ("展厅5屏", os.path.join(DATA, "输入暗源-展厅5屏.mp4"), 50),
    ("展厅1屏", os.path.join(DATA, "输入暗源-展厅1屏.mp4"), 50),
    ("展厅内入口", os.path.join(DATA, "输入暗源-展厅内入口.mp4"), 50),
]

# name -> ckpt_dir. First two are the two already compared; the rest are the other
# official Retinexformer checkpoints, wired in via a throwaway dir containing a
# NTIRE.pth symlink (RetinexformerStage always loads "<ckpt_dir>/NTIRE.pth").
ALT_DATASETS = ["LOL_v1", "LOL_v2_real", "LOL_v2_synthetic", "FiveK",
                 "SID", "SMID", "SDSD_indoor", "SDSD_outdoor"]


def alt_ckpt_dir(name):
    d = f"/tmp/ckpt_cmp_{name}"
    os.makedirs(d, exist_ok=True)
    link = os.path.join(d, "NTIRE.pth")
    if not os.path.exists(link):
        os.symlink(os.path.join(CKPTS, f"{name}.pth"), link)
    return d


def variants():
    v = [("baseline (NTIRE)", CKPTS),
         ("fine-tuned (real-deploy)", os.path.join(DSP, "ckpts_realdeploy_ft"))]
    for name in ALT_DATASETS:
        v.append((name, alt_ckpt_dir(name)))
    return v


def load_stage(ckpt_dir, device):
    cfg = validate(PipelineConfig(input="unused.mp4", enhance="retinexformer", sr="off",
                                  recognize="off", ckpt_dir=ckpt_dir, device=device,
                                  output="/tmp/unused.mp4"))
    stages, _ = build_stages(cfg)
    stages[0].load(device)
    return stages[0]


def read_frame(video, idx, proc_max_side):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, f = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"error: could not read frame {idx} from {video}")
    return DownscaleStage(proc_max_side)([f])[0] if proc_max_side else f


PANEL_W = 260
BAR_H = 20


def label(img, text, w):
    bar = np.zeros((BAR_H, w, 3), np.uint8)
    cv2.putText(bar, text, (6, BAR_H - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(DSP, "..", "real-development-data",
                                                    "finetune_pretrained_compare.png"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cols", type=int, default=5)
    a = ap.parse_args()

    small_by_scene = {name: read_frame(video, idx, 1280) for name, video, idx in SCENES}

    results = {name: {} for name, _, _ in SCENES}
    for vname, ckpt_dir in variants():
        print(f"[load] {vname} <- {ckpt_dir}")
        stage = load_stage(ckpt_dir, a.device)
        for sname, _, _ in SCENES:
            enh = stage([small_by_scene[sname]])[0]
            results[sname][vname] = enh
        stage.close()

    vnames = [v for v, _ in variants()]
    rows_out = []
    for sname, _, _ in SCENES:
        tiles = []
        for vname in vnames:
            img = results[sname][vname]
            th = cv2.resize(img, (PANEL_W, int(img.shape[0] * PANEL_W / img.shape[1])))
            tiles.append(label(th, vname, PANEL_W))
        # pad to a multiple of --cols with blank tiles
        while len(tiles) % a.cols:
            tiles.append(np.zeros_like(tiles[0]))
        grid_rows = [np.hstack(tiles[i:i + a.cols]) for i in range(0, len(tiles), a.cols)]
        grid = np.vstack(grid_rows)
        title = np.zeros((26, grid.shape[1], 3), np.uint8)
        cv2.putText(title, sname, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        rows_out.append(np.vstack([title, grid]))

    sep = np.full((8, rows_out[0].shape[1], 3), 40, np.uint8)
    out = rows_out[0]
    for r in rows_out[1:]:
        out = np.vstack([out, sep, r])

    out_path = os.path.normpath(a.out)
    cv2.imwrite(out_path, out)
    print(f"wrote {out_path}  {out.shape[1]}x{out.shape[0]}")


if __name__ == "__main__":
    main()
