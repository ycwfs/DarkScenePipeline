#!/usr/bin/env python3
"""Side-by-side visual comparison: baseline NTIRE.pth vs the real-deploy fine-tune.

For each scene: full enhanced frame (baseline | fine-tuned) on top, a 2x zoomed center
crop of the same frame below it (where the noise difference is easiest to see). One row
per scene, stacked into a single composite PNG.

    python finetune_visual_compare.py --out out.png
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from denoise_probe import enhanced_frames  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DSP = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
DATA = os.path.normpath(os.path.join(DSP, "..", "real-development-data", "输入暗源"))
CKPT_BASE = os.path.join(DSP, "ckpts")
CKPT_FT = os.path.join(DSP, "ckpts_realdeploy_ft")

SCENES = [
    ("报告厅", os.path.join(DATA, "输入暗源-报告厅.mp4"), 50),
    ("展厅5屏", os.path.join(DATA, "输入暗源-展厅5屏.mp4"), 50),
    ("展厅1屏", os.path.join(DATA, "输入暗源-展厅1屏.mp4"), 50),
    ("展厅内入口", os.path.join(DATA, "输入暗源-展厅内入口.mp4"), 50),
]

PANEL_W = 480
BAR_H = 28
ZOOM = 2


def label(img, text, w):
    bar = np.zeros((BAR_H, w, 3), np.uint8)
    cv2.putText(bar, text, (8, BAR_H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def center_crop(img, cw, ch):
    h, w = img.shape[:2]
    cw, ch = min(cw, w), min(ch, h)
    x0 = (w - cw) // 2
    y0 = (h - ch) // 2
    return img[y0:y0 + ch, x0:x0 + cw]


def make_row(scene, video, idx, device):
    _, _, enh_base = enhanced_frames(video, [idx], 1280, device, CKPT_BASE)
    _, _, enh_ft = enhanced_frames(video, [idx], 1280, device, CKPT_FT)
    b, f = enh_base[0], enh_ft[0]

    full_b = cv2.resize(b, (PANEL_W, int(b.shape[0] * PANEL_W / b.shape[1])))
    full_f = cv2.resize(f, (PANEL_W, int(f.shape[0] * PANEL_W / f.shape[1])))
    full = np.hstack([label(full_b, f"{scene}  baseline (NTIRE)", PANEL_W),
                       label(full_f, f"{scene}  fine-tuned (real-deploy)", PANEL_W)])

    crop_b = center_crop(b, PANEL_W // ZOOM, 200)
    crop_f = center_crop(f, PANEL_W // ZOOM, 200)
    crop_b = cv2.resize(crop_b, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST)
    crop_f = cv2.resize(crop_f, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST)
    crop = np.hstack([label(crop_b, "zoom 2x (baseline)", PANEL_W),
                       label(crop_f, "zoom 2x (fine-tuned)", PANEL_W)])

    return np.vstack([full, crop])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "..", "..", "..",
                                                    "real-development-data", "finetune_before_after.png"))
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    rows = [make_row(scene, video, idx, a.device) for scene, video, idx in SCENES]
    sep = np.full((6, rows[0].shape[1], 3), 40, np.uint8)
    out = rows[0]
    for r in rows[1:]:
        out = np.vstack([out, sep, r])

    out_path = os.path.normpath(a.out)
    cv2.imwrite(out_path, out)
    print(f"wrote {out_path}  {out.shape[1]}x{out.shape[0]}")


if __name__ == "__main__":
    main()
