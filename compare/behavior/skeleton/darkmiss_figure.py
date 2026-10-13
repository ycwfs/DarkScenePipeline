"""Render 3-panel comparison rows: RAW dark | RTMO-s on enhanced | YOLO26m on enhanced."""
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = "/ssd1/wfs/project/lowlight/DarkScenePipeline"
sys.path.insert(0, REPO)
sys.path.insert(0, "/ssd1/wfs/project/lowlight/compare/behavior/skeleton")
sys.path.insert(0, "/ssd1/wfs/project/lowlight/compare/behavior")

from darkpipe.stages.enhance_retinexformer import RetinexformerStage
from realact_cache import resize_max_side
from skeleton_cache import make_pose
from stgcn import COCO_EDGES

SRC_DIRS = [os.path.join(REPO, "real-actiondata-0909"),
            os.path.join(REPO, "real-actiondata-0909", "other"),
            os.path.join(REPO, "real-actiondata")]
CLS_ZH = {"wave": "挥手", "drink": "喝水", "pick_up": "捡东西", "chase": "追逐",
          "other": "无动作负样本"}
TAG = {"miss": ("YOLO26 漏检", (255, 120, 90)),
       "agree": ("两者一致检出", (120, 220, 120)),
       "reverse": ("YOLO26 检出 / RTMO 漏检", (120, 200, 255))}

# (class, stem, frame_idx, kind) -- from the 840-scale scan, one take each
PICKS = [
    ("wave",    "挥手15", 0,   "miss"),
    ("drink",   "喝水15", 40,  "miss"),
    ("pick_up", "捡东西10", 40, "miss"),
    ("chase",   "追逐10", 8,   "miss"),
    ("other",   "other26", 16, "miss"),
    ("other",   "other80", 8,  "agree"),
    ("wave",    "挥手11", 192, "reverse"),
    ("pick_up", "捡东西8", 24,  "reverse"),
]

FONT = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 24)
F_TITLE = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 26)
FONT_S = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 21)

stem_path = {}
for d in SRC_DIRS:
    if os.path.isdir(d):
        for p in sorted(os.listdir(d)):
            if p.endswith(".mp4") and os.path.splitext(p)[0] not in stem_path:
                stem_path[os.path.splitext(p)[0]] = os.path.join(d, p)

rtmo = make_pose("rtmo", 0.3, "cuda:0")
y26 = make_pose("yolo26", 0.15, "cuda:0")
stage = RetinexformerStage(os.path.join(REPO, "ckpts"))
stage.load("cuda:0")

PW, GAP, PAD, CAP = 470, 4, 8, 44


def draw_skel(img, kpts, scores, thr=0.3):
    for i in range(len(kpts)):
        col = (0, 210, 255) if i == 0 else (255, 140, 0)
        pts = kpts[i]
        for a, b in COCO_EDGES:
            if scores[i, a] > thr and scores[i, b] > thr:
                cv2.line(img, tuple(np.round(pts[a]).astype(int)),
                         tuple(np.round(pts[b]).astype(int)), col, 2, cv2.LINE_AA)
        for j in range(17):
            if scores[i, j] > thr:
                cv2.circle(img, tuple(np.round(pts[j]).astype(int)), 3,
                           (90, 255, 90), -1, cv2.LINE_AA)


def strip(text, rgb, w=PW, h=CAP, font=FONT):
    img = Image.new("RGB", (w, h), (18, 18, 22))
    ImageDraw.Draw(img).text((8, (h - font.size) // 2 - 2), text, font=font, fill=rgb)
    return img


rows = []
for cls, stem, idx, kind in PICKS:
    cap = cv2.VideoCapture(stem_path[stem])
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, raw = cap.read()
    cap.release()
    assert ok, (stem, idx)
    raw840 = resize_max_side(raw, 840)
    enh = stage([raw840])[0]
    k1, s1 = rtmo(enh)
    k2, s2 = y26(enh)
    c1 = float(s1.mean(1).max()) if len(s1) else 0.0
    c2 = float(s2.mean(1).max()) if len(s2) else 0.0
    luma = int(cv2.cvtColor(raw840, cv2.COLOR_BGR2GRAY).mean())

    ph = round(PW * enh.shape[0] / enh.shape[1])
    sc_r = cv2.resize(raw840, (PW, ph), interpolation=cv2.INTER_AREA)
    full1 = enh.copy()                     # draw at INFERENCE scale, then shrink (coords match)
    full2 = enh.copy()
    draw_skel(full1, k1, s1)
    draw_skel(full2, k2, s2)
    sc_e1 = cv2.resize(full1, (PW, ph), interpolation=cv2.INTER_AREA)
    sc_e2 = cv2.resize(full2, (PW, ph), interpolation=cv2.INTER_AREA)
    rows.append((cls, stem, idx, kind, luma, c1, c2,
                 sc_r[:, :, ::-1], sc_e1[:, :, ::-1], sc_e2[:, :, ::-1], ph))
    print(f"{cls} {stem}_{idx:05d} luma={luma} rtmo={c1:.2f} y26={c2:.2f}", flush=True)

ph = rows[0][-1]
W = PAD + PW + GAP + PW + GAP + PW + PAD
ROW_H = CAP + ph + CAP + PAD
H = 52 + len(rows) * ROW_H + PAD
canvas = Image.new("RGB", (W, H), (30, 30, 34))
dr = ImageDraw.Draw(canvas)
dr.text((PAD, 10), "同一帧 · 同一部署增强链（840 宽）    左：原始暗帧    中：RTMO-s    右：YOLO26m-pose",
        font=F_TITLE, fill=(255, 255, 255))

x0, x1, x2 = PAD, PAD + PW + GAP, PAD + PW + GAP + PW + GAP
y = 52
for cls, stem, idx, kind, luma, c1, c2, a_raw, a_rt, a_yo, ph in rows:
    tag, tcol = TAG[kind]
    canvas.paste(strip(f"{CLS_ZH[cls]}  ·  {stem}.mp4 第 {idx} 帧  ·  原图均亮 {luma}/255  ·  {tag}",
                       tcol, w=W - PAD * 2), (PAD, y))
    y += CAP
    canvas.paste(Image.fromarray(a_raw), (x0, y))
    canvas.paste(Image.fromarray(a_rt), (x1, y))
    canvas.paste(Image.fromarray(a_yo), (x2, y))
    y += ph
    conf1 = f"person conf = {c1:.2f}" if c1 > 0 else "NO PERSON"
    conf2 = f"person conf = {c2:.2f}" if c2 > 0 else "NO PERSON"
    good1 = c1 >= 0.5
    good2 = c2 >= 0.5
    canvas.paste(strip(f"RTMO-s   {conf1}",
                       (120, 220, 120) if good1 else (255, 120, 90)), (x1, y))
    canvas.paste(strip(f"YOLO26m   {conf2}",
                       (120, 220, 120) if good2 else (255, 120, 90)), (x2, y))
    canvas.paste(strip("（增强前画面）", (150, 150, 150)), (x0, y))
    y += CAP + PAD
canvas.save("/tmp/darkmiss_compare.png")
print("saved /tmp/darkmiss_compare.png", canvas.size)
