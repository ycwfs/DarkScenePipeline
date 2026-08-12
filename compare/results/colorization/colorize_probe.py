"""First look: richzhang/colorization as a colour-restoration step after low-light enhancement.

What this actually does matters for reading the output. The model takes ONLY the L
(luminance) channel and predicts ab; it never sees the input's colour. So applied after the
enhancer it does not *correct* a colour cast -- it discards whatever colour the enhancer
produced and paints in what it thinks a scene of that luminance should look like.

Panels, left to right:
  raw          the dark frame as the camera gave it
  enhanced     Retinexformer (what ships today)
  enh+eccv16   enhanced luminance, ECCV16 colours
  enh+sig17    enhanced luminance, SIGGRAPH17 colours
  raw+sig17    control: colourise WITHOUT enhancing first, to separate what the enhancer
               contributes from what the colouriser does
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, "/ssd1/wfs/project/lowlight/colorization")
from colorizers import eccv16, postprocess_tens, preprocess_img, siggraph17  # noqa: E402


def to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def colorize(model, rgb, device):
    """rgb uint8 HxWx3 -> colourised uint8, at the original resolution."""
    tens_l_orig, tens_l_rs = preprocess_img(rgb, HW=(256, 256))
    with torch.no_grad():
        ab = model(tens_l_rs.to(device)).cpu()
    out = postprocess_tens(tens_l_orig, ab)          # float RGB in [0,1]
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def label(img, text):
    """Caption strip above each panel, so a saved sheet is self-describing."""
    bar = np.full((28, img.shape[1], 3), 20, np.uint8)
    cv2.putText(bar, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1,
                cv2.LINE_AA)
    return np.vstack([bar, img])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="/ssd1/wfs/project/lowlight/compare/arid/test_clips/"
                                       "Drink/Drink_3_20.mp4")
    ap.add_argument("--frames", default="10,25,40")
    ap.add_argument("--ckpt-dir", default="/ssd1/wfs/project/lowlight/DarkScenePipeline/ckpts")
    ap.add_argument("--out", default="/tmp/colortest/sheet.png")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    sys.path.insert(0, "/ssd1/wfs/project/lowlight/DarkScenePipeline")
    from darkpipe.config import PipelineConfig, validate
    from darkpipe.stages import build_stages

    want = sorted(int(x) for x in a.frames.split(","))
    cap = cv2.VideoCapture(a.video)
    raws, i = [], 0
    while len(raws) < len(want):
        ok, f = cap.read()
        if not ok:
            break
        if i in want:
            raws.append(f)
        i += 1
    cap.release()
    if not raws:
        sys.exit(f"error: no frames read from {a.video}")

    cfg = validate(PipelineConfig(input=a.video, enhance="retinexformer", sr="off",
                                  recognize="off", ckpt_dir=a.ckpt_dir, device=a.device,
                                  output="/tmp/unused.mp4"))
    stages, _ = build_stages(cfg)
    for s in stages:
        s.load(a.device)
    enhanced = stages[0](list(raws))

    m_eccv = eccv16(pretrained=True).eval().to(a.device)
    m_sig = siggraph17(pretrained=True).eval().to(a.device)

    rows = []
    for raw, enh in zip(raws, enhanced):
        r_rgb, e_rgb = to_rgb(raw), to_rgb(enh)
        panels = [
            label(r_rgb, "raw (dark)"),
            label(e_rgb, "enhanced (retinexformer)"),
            label(colorize(m_eccv, e_rgb, a.device), "enhanced + eccv16"),
            label(colorize(m_sig, e_rgb, a.device), "enhanced + siggraph17"),
            label(colorize(m_sig, r_rgb, a.device), "raw + siggraph17 (no enhance)"),
        ]
        rows.append(np.hstack(panels))
    sheet = np.vstack(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    cv2.imwrite(a.out, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print(f"[out] {a.out}  {sheet.shape[1]}x{sheet.shape[0]}")

    # A number to go with the picture: how far each version moves the colour, measured as
    # mean chroma in Lab. The enhancer's own output is the baseline to beat or match.
    for name, imgs in (("enhanced", [to_rgb(e) for e in enhanced]),
                       ("enh+eccv16", [colorize(m_eccv, to_rgb(e), a.device) for e in enhanced]),
                       ("enh+sig17", [colorize(m_sig, to_rgb(e), a.device) for e in enhanced])):
        chroma = []
        for im in imgs:
            lab = cv2.cvtColor(im, cv2.COLOR_RGB2LAB).astype(np.float32)
            chroma.append(float(np.hypot(lab[..., 1] - 128, lab[..., 2] - 128).mean()))
        print(f"[chroma] {name:<12} 平均彩度 {np.mean(chroma):6.2f}")


if __name__ == "__main__":
    main()
