"""Whole-pipeline visualization (native pixel sizes, like the repo's stage_progression).

Row of panels for one frame of a dark ARID clip:
  1. dark input 320x240 | 2. HVI-CIDNet | 3. Retinexformer | 4. CIDNet +lightSR x2 640x480
  | 5. Retinexformer +lightSR x2 640x480
Plus a final-result strip: the annotated (label-bar) frame from each full pipeline output.
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DP = "/ssd1/wfs/project/lowlight/DarkScenePipeline"
sys.path.insert(0, DP)
from darkpipe.stages.enhance_cidnet import CIDNetStage          # noqa: E402
from darkpipe.stages.enhance_retinexformer import RetinexformerStage  # noqa: E402
from darkpipe.stages.sr_lightsr import LightSRStage             # noqa: E402

CKPT = os.path.join(DP, "ckpts")
DEV = "cuda:3"
W = "/ssd1/wfs/project/lowlight/compare/arid"
OUT = "/ssd1/wfs/project/lowlight/compare/results"
CLIP = f"{W}/clips/Sit_10_1.mp4"
FRAME_IDX = 60
PAD = 10
HEADER = 52


def get_frame(path, idx):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    cap.release()
    if not ok:
        cap = cv2.VideoCapture(path)
        while True:
            ok, f2 = cap.read()
            if not ok:
                break
            fr = f2
        cap.release()
    return fr


def font(sz):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", sz)
    except Exception:
        return ImageFont.load_default()


def compose(panels, out_path, title=""):
    # panels: list of (label, bgr_img)
    imgs = [cv2.cvtColor(p[1], cv2.COLOR_BGR2RGB) for p in panels]
    hmax = max(im.shape[0] for im in imgs)
    total_w = sum(im.shape[1] for im in imgs) + PAD * (len(imgs) + 1)
    canvas = Image.new("RGB", (total_w, hmax + HEADER + PAD * 2), (15, 15, 15))
    draw = ImageDraw.Draw(canvas)
    if title:
        draw.text((PAD, 6), title, fill=(120, 220, 120), font=font(16))
    x = PAD
    for (label, _), im in zip(panels, imgs):
        draw.text((x, HEADER - 20), label, fill=(255, 230, 0), font=font(15))
        canvas.paste(Image.fromarray(im), (x, HEADER))
        x += im.shape[1] + PAD
    canvas.save(out_path)
    print("saved", out_path, canvas.size)


def main():
    frame = get_frame(CLIP, FRAME_IDX)  # BGR 320x240

    cid_raw = CIDNetStage(CKPT, chunk=1, despeckle=False); cid_raw.load(DEV)
    cid = CIDNetStage(CKPT, chunk=1, despeckle=True); cid.load(DEV)
    rf = RetinexformerStage(CKPT, chunk=1); rf.load(DEV)
    sr = LightSRStage(CKPT, chunk=1); sr.load(DEV)

    cid_raw_out = cid_raw([frame])[0]
    cid_out = cid([frame])[0]
    rf_out = rf([frame])[0]
    cid_sr = sr([cid_out])[0]
    rf_sr = sr([rf_out])[0]

    compose(
        [("1. dark input  320x240", frame),
         ("2. HVI-CIDNet raw  320x240", cid_raw_out),
         ("3. HVI-CIDNet despeckled  320x240", cid_out),
         ("4. Retinexformer  320x240", rf_out),
         ("5. HVI-CIDNet + lightSR x2  640x480", cid_sr),
         ("6. Retinexformer + lightSR x2  640x480", rf_sr)],
        f"{OUT}/stage_progression_compare.png",
        title="DarkScenePipeline stages on a dark ARID 'Sit' frame  (native pixel sizes; "
              "panel 2 shows CIDNet's raw dark-region color speckle, panel 3 the despeckle guard)")

    # final annotated frames (label bar) from the full pipeline outputs
    fc = get_frame(f"{W}/out_cidnet.mp4", 64)
    fr = get_frame(f"{W}/out_rf.mp4", 64)
    compose(
        [("Final: HVI-CIDNet -> SR -> VideoMamba", fc),
         ("Final: Retinexformer -> SR -> VideoMamba", fr)],
        f"{OUT}/final_annotated_compare.png",
        title="Final pipeline output frame (enhanced + x2 SR + recognition label bar)")


if __name__ == "__main__":
    main()
