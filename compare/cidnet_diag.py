"""Diagnose CIDNet speckle on extremely dark ARID frames: sweep weight / gating / gamma."""
import os, sys
import cv2, numpy as np, torch
from PIL import Image, ImageDraw, ImageFont

DP = "/ssd1/wfs/project/lowlight/DarkScenePipeline"
sys.path.insert(0, DP)
from darkpipe.vendor.cidnet import CIDNet
HVI = "/ssd1/wfs/project/lowlight/HVI-CIDNet/weights"
DEV = "cuda:3"

W = {
    "generalization": f"{DP}/ckpts/CIDNet_generalization.pth",
    "syn_wo_perc": f"{HVI}/LOLv2_syn/wo_perc.pth",
    "syn_w_perc": f"{HVI}/LOLv2_syn/w_perc.pth",
}

def load(w):
    m = CIDNet().to(DEV).eval()
    sd = torch.load(W[w], map_location="cpu")
    sd = sd["params"] if isinstance(sd, dict) and "params" in sd else sd
    m.load_state_dict(sd, strict=True)
    return m

def lum_desat(rgb, v0, v1):
    """Scale saturation by a smooth ramp of brightness: gray in dark (unreliable-color)
    regions, full color where V>=v1. Removes dark-region color speckle, keeps real color."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    v = hsv[:, :, 2] / 255.0
    scale = np.clip((v - v0) / (v1 - v0), 0.0, 1.0)
    hsv[:, :, 1] *= scale
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def run(m, x, gated, gated2, alpha_s, alpha, gamma, median=0, desat=None):
    m.trans.gated = gated; m.trans.gated2 = gated2
    m.trans.alpha_s = alpha_s; m.trans.alpha = alpha
    with torch.no_grad():
        y = m((x ** gamma) if gamma != 1.0 else x)
    y = torch.clamp(y, 0, 1)[0].permute(1, 2, 0).cpu().numpy()
    out = (y * 255).round().astype(np.uint8)  # RGB
    if desat:
        out = lum_desat(out, *desat)
    if median:
        out = cv2.medianBlur(out, median)
    return out

def main():
    cap = cv2.VideoCapture("/ssd1/wfs/project/lowlight/compare/arid/clips/Sit_10_1.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 60); ok, fr = cap.read(); cap.release()
    rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).to(DEV).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    configs = [
        ("a_s=1.0 raw (speckle)", dict(w="generalization", gated=False, gated2=False, alpha_s=1.0, alpha=1.0, gamma=1.0, median=0, desat=None)),
        ("lumdesat 0.15-0.45", dict(w="generalization", gated=False, gated2=False, alpha_s=1.0, alpha=1.0, gamma=1.0, median=0, desat=(0.15, 0.45))),
        ("lumdesat 0.20-0.55", dict(w="generalization", gated=False, gated2=False, alpha_s=1.0, alpha=1.0, gamma=1.0, median=0, desat=(0.20, 0.55))),
        ("lumdesat 0.25-0.60 +med3", dict(w="generalization", gated=False, gated2=False, alpha_s=1.0, alpha=1.0, gamma=1.0, median=3, desat=(0.25, 0.60))),
        ("lumdesat 0.20-0.55 +med3", dict(w="generalization", gated=False, gated2=False, alpha_s=1.0, alpha=1.0, gamma=1.0, median=3, desat=(0.20, 0.55))),
        ("a_s=0.0 gray (clean ref)", dict(w="generalization", gated=True, gated2=False, alpha_s=0.0, alpha=1.0, gamma=1.0, median=0, desat=None)),
    ]
    panels = [("0. dark input", rgb)]
    cache = {}
    for label, c in configs:
        if c["w"] not in cache:
            cache[c["w"]] = load(c["w"])
        out = run(cache[c["w"]], x, c["gated"], c["gated2"], c["alpha_s"], c["alpha"], c["gamma"], c["median"], c["desat"])
        panels.append((label, out))

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
    except Exception:
        font = ImageFont.load_default()
    pw = rgb.shape[1]; ph = rgb.shape[0]; PAD = 8; HDR = 20
    cols = len(panels)
    canvas = Image.new("RGB", (cols * pw + PAD * (cols + 1), ph + HDR + PAD * 2), (15, 15, 15))
    d = ImageDraw.Draw(canvas)
    x0 = PAD
    for label, im in panels:
        d.text((x0, 3), label, fill=(255, 230, 0), font=font)
        canvas.paste(Image.fromarray(im), (x0, HDR))
        x0 += pw + PAD
    canvas.save("/ssd1/wfs/project/lowlight/compare/results/cidnet_diag.png")
    print("saved cidnet_diag.png")

if __name__ == "__main__":
    main()
