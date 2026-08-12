"""Colour restoration that keeps the real colours, as the alternative to re-colourising.

The colourisation model scored 66.3 deg of hue deviation from the raw frame's own colours
because it never sees them -- it takes L and invents ab. Everything here works on the ab
channels the enhancer already produced, so the hue the camera actually recorded is what
gets amplified or re-balanced.

  sat xN        scale (a,b) about neutral in Lab. Scaling both by the same factor leaves
                atan2(b,a) -- the hue -- exactly unchanged, so this can only make existing
                colour stronger, never different. Chroma is the one thing it changes.
  shades-of-gray  Minkowski-norm illuminant estimate (p=6), the standard robust version of
                grey-world: divide each channel by its p-norm and rebalance. This removes a
                global cast, which is a hue change -- deliberately, since a cast is exactly
                what "the whites came out blue" means.
"""
import argparse
import os
import sys

import cv2
import numpy as np


def boost_chroma(bgr, k):
    """Scale chroma about neutral in Lab. Hue is invariant under this by construction."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 1] = np.clip((lab[..., 1] - 128.0) * k + 128.0, 0, 255)
    lab[..., 2] = np.clip((lab[..., 2] - 128.0) * k + 128.0, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def shades_of_gray(bgr, p=6):
    """Illuminant estimate by Minkowski p-norm; p=1 is grey-world, p=inf is white-patch."""
    f = bgr.astype(np.float32)
    est = np.power(np.power(f, p).mean(axis=(0, 1)), 1.0 / p)     # per-channel norm, BGR
    est = est / (est.mean() + 1e-6)
    return np.clip(f / (est + 1e-6), 0, 255).astype(np.uint8)


def chroma(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    return float(np.hypot(lab[..., 1] - 128, lab[..., 2] - 128).mean())


def hue_dev(ref_bgr, out_bgr, min_chroma=8.0):
    """Mean hue deviation (degrees) from `ref`, over pixels where ref has real colour."""
    a = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    b = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    mask = np.hypot(a[..., 1] - 128, a[..., 2] - 128) > min_chroma
    if mask.sum() < 50:
        return float("nan")
    ha = np.arctan2(a[..., 2] - 128, a[..., 1] - 128)[mask]
    hb = np.arctan2(b[..., 2] - 128, b[..., 1] - 128)[mask]
    return float(np.abs(np.degrees(np.arctan2(np.sin(hb - ha), np.cos(hb - ha)))).mean())


def label(img, text):
    bar = np.full((26, img.shape[1], 3), 20, np.uint8)
    cv2.putText(bar, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1,
                cv2.LINE_AA)
    return np.vstack([bar, img])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="/ssd1/wfs/project/lowlight/compare/arid/test_clips/"
                                       "Drink/Drink_3_20.mp4")
    ap.add_argument("--frames", default="10,25,45")
    ap.add_argument("--every", type=int, default=0, help="metrics over every Nth frame")
    ap.add_argument("--out", default="/tmp/colortest/restore_sheet.png")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    sys.path.insert(0, "/ssd1/wfs/project/lowlight/DarkScenePipeline")
    from darkpipe.config import PipelineConfig, validate
    from darkpipe.stages import build_stages

    want = set(int(x) for x in a.frames.split(","))
    cap = cv2.VideoCapture(a.video)
    shown, allraw, i = [], [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i in want:
            shown.append(f)
        if a.every and i % a.every == 0:
            allraw.append(f)
        i += 1
    cap.release()

    cfg = validate(PipelineConfig(input=a.video, enhance="retinexformer", sr="off",
                                  recognize="off", ckpt_dir="/ssd1/wfs/project/lowlight/"
                                  "DarkScenePipeline/ckpts", device=a.device,
                                  output="/tmp/unused.mp4"))
    stages, _ = build_stages(cfg)
    for s in stages:
        s.load(a.device)

    variants = [
        ("enhanced (baseline)", lambda e: e),
        ("+ sat x1.8", lambda e: boost_chroma(e, 1.8)),
        ("+ sat x2.6", lambda e: boost_chroma(e, 2.6)),
        ("+ shades-of-gray", lambda e: shades_of_gray(e)),
        ("+ SoG + sat x2.2", lambda e: boost_chroma(shades_of_gray(e), 2.2)),
    ]

    rows = []
    for raw, enh in zip(shown, stages[0](list(shown))):
        panels = [label(raw, "raw (dark)")] + [label(fn(enh), n) for n, fn in variants]
        rows.append(np.hstack(panels))
    sheet = np.vstack(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    cv2.imwrite(a.out, sheet)
    print(f"[out] {a.out}  {sheet.shape[1]}x{sheet.shape[0]}")

    metric_raw = allraw or shown
    enh_all = stages[0](list(metric_raw))
    print(f"\n  {'方法':<24}{'色相偏离原始':>14}{'平均彩度':>10}")
    for name, fn in variants:
        hs, cs = [], []
        for r, e in zip(metric_raw, enh_all):
            o = fn(e)
            d = hue_dev(r, o)
            if not np.isnan(d):
                hs.append(d)
            cs.append(chroma(o))
        print(f"  {name:<24}{np.mean(hs):>12.1f}°{np.mean(cs):>12.1f}")
    print(f"  {'(参考) 增强+eccv16 上色':<24}{'66.3':>12}°{'24.2':>12}")


if __name__ == "__main__":
    main()
