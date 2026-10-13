"""Per-class decision visualisation for the skeleton-aux fusion (alpha=0.85, conf>=0.65).

For each real-actiondata class (one IR take each) run the full deployed chain
(downscale 840 -> Retinexformer -> sliding 1 s windows @0.5 s) and produce, under
viz/<class>/:
  prob_chart.png -- grouped bars per window: P(decided class) under RGB-only,
                   skeleton-only and the alpha-fusion; red line = fire threshold
                   0.65; green group background = FIRED; grey = gated (no person);
                   footnote shows each model's own argmax.
  skel_video.mp4 -- non-overlapping 1 s windows, enhanced frames with the RTMO
                   skeleton drawn and the fused decision overlaid.

Chase note (user): single-person semantics -- the M=2 slot stays empty for solo
runs and contributes zero weight; no two-person requirement anywhere.

Usage: python -u viz_class.py <gpu> [--skel-ckpt ../cache/skel_v1/stgcn.pth] [--alpha 0.85]
"""
import argparse
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import torch

for _f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",):
    if os.path.exists(_f):
        fm.fontManager.addfont(_f)      # TTC registers as family "Noto Sans CJK JP"
# glyph-level fallback ONLY works via a font.family list (mpl >=3.6), not font.sans-serif
plt.rcParams["font.family"] = ["DejaVu Sans", "Noto Sans CJK JP", "Droid Sans Fallback"]
plt.rcParams["axes.unicode_minus"] = False

REPO = "/ssd1/wfs/project/lowlight/DarkScenePipeline"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fusion_eval as FE
from darkpipe.stages.downscale import DownscaleStage
from darkpipe.stages.enhance_retinexformer import RetinexformerStage
from fusion_eval import LABELS, Models, ODIR, TAKES, read_video, windows_of
from stgcn import COCO_EDGES

OTHER = LABELS.index("other")
CONF = 0.65

TAKES_ALL = [("挥手22", "wave"), ("捡东西2", "pick_up"), ("追逐16", "chase"), ("喝水16", "drink")]
SRC = FE.SRC


def draw_skeleton(frame, kpts, scores, thr=0.3):
    """Overlay COCO-17 stick figures (bright joints, coloured bones) in place."""
    for k, (i, j) in enumerate(COCO_EDGES):
        if scores[i] > thr and scores[j] > thr:
            col = (0, 200, 255) if k % 2 == 0 else (255, 160, 0)
            cv2.line(frame, tuple(np.round(kpts[i]).astype(int)),
                     tuple(np.round(kpts[j]).astype(int)), col, 2, cv2.LINE_AA)
    for p in range(len(kpts)):
        if scores[p] > thr:
            cv2.circle(frame, tuple(np.round(kpts[p]).astype(int)), 3, (80, 255, 80), -1, cv2.LINE_AA)


def decide(pr, ps, pres, alpha):
    """-> (pf, cls_name, conf, fired, gated)."""
    if pres < FE.TAU:
        pf = np.eye(len(LABELS))[OTHER]
        gated = True
    else:
        pf = alpha * pr + (1 - alpha) * ps
        gated = False
    cls, conf = int(pf.argmax()), float(pf.max())
    return pf, LABELS[cls], conf, cls != OTHER and conf >= CONF, gated


def chart(rows, out_png, title):
    """rows: dicts(t, pr, ps, pf, cls, conf, fired, gated, cls_rgb, cls_skel)."""
    xs = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(2.0 * len(rows) + 2.5, 5.6))
    w = 0.27
    for i, r in enumerate(rows):
        vals = [r["pr"][r["ci"]], 0.0 if r["gated"] else r["ps"][r["ci"]], r["pf"][r["ci"]]]
        for k, (v, c, lab) in enumerate(zip(vals, ["#1f77b4", "#ff7f0e", "#2ca02c"],
                                            ["RGB", "skeleton", "fused"])):
            ax.bar(i + (k - 1) * w, v, w, color=c, label=lab if i == 0 else None,
                   alpha=1.0 if r["fired"] else 0.55,
                   hatch="///" if r["gated"] and k == 1 else None, edgecolor="k", linewidth=0.4)
            ax.text(i + (k - 1) * w, v + 0.015, f"{v:.2f}", rotation=90, ha="center",
                    va="bottom", fontsize=6, color="#333333")
        note = (f"无人 -> other" if r["gated"] else
                f"{r['cls']}" + ("  FIRE" if r["fired"] else ""))
        ax.text(i, 1.055, note, ha="center", va="bottom", fontsize=9,
                fontweight="bold" if r["fired"] else "normal",
                color="#b22222" if r["fired"] else ("dimgrey" if r["gated"] else "black"))
        ax.text(i, 1.005, f"RGB:{r['cls_rgb']} 骨架:{r['cls_skel']}",
                ha="center", va="bottom", fontsize=7, color="dimgrey")
    ax.axhline(CONF, color="red", ls="--", lw=1, label=f"conf {CONF}")
    ax.set_xticks(xs, [r["t"] for r in rows])
    ax.set_ylim(0, 1.24)
    ax.set_ylabel("P(判定类别)")
    ax.set_xlabel("窗口起始时间 (s)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gpu")
    ap.add_argument("--skel-ckpt", default="../cache/skel_v1/stgcn.pth")
    ap.add_argument("--alpha", type=float, default=0.85)
    a = ap.parse_args()
    FE.dev = dev = f"cuda:{a.gpu}"                    # Models.* read the module global

    ds = DownscaleStage(840)
    st = RetinexformerStage(os.path.join(REPO, "ckpts"))
    st.load(dev)
    M = Models(dev, a.skel_ckpt)

    for take, cls in TAKES_ALL:
        frames, fps = read_video(os.path.join(SRC, take + ".mp4"))
        frames = st(ds(frames))                       # enhanced BGR 840x472
        n = len(frames)
        H, W = frames[0].shape[:2]
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viz", cls)
        os.makedirs(out_dir, exist_ok=True)

        # ---- probability chart over the sliding stream --------------------------------
        rows = []
        for wname, i0, i1 in windows_of(n, fps):
            win = frames[i0:i1]
            pr = M.rgb_probs(win)
            ps, pres = M.skel_probs(win)
            pf, cname, conf, fired, gated = decide(pr, ps, pres, a.alpha)
            rows.append(dict(t=wname, ci=LABELS.index(cname), pr=pr, ps=ps, pf=pf,
                             cls=cname, conf=conf, fired=fired, gated=gated,
                             cls_rgb=LABELS[int(pr.argmax())],
                             cls_skel=LABELS[int(ps.argmax())]))
        fires = [r["t"] for r in rows if r["fired"]]
        chart(rows, os.path.join(out_dir, "prob_chart.png"),
              f"{cls}（{take}.mp4）  α={a.alpha}  触发阈值={CONF}  "
              f"判定：{len(fires)}/{len(rows)} 窗触发"
              + (f"  [{','.join(fires)}]" if fires else ""))

        # ---- skeleton+decision video (non-overlapping 1 s windows) --------------------
        vw = cv2.VideoWriter(os.path.join(out_dir, "skel_video.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        w = max(2, int(round(fps)))
        for s in range(0, n - w + 1, w):
            win = frames[s:s + w]
            pr = M.rgb_probs(win)
            ps, pres = M.skel_probs(win)
            pf, cname, conf, fired, gated = decide(pr, ps, pres, a.alpha)
            col = (0, 90, 255) if fired else ((160, 160, 160) if gated else (0, 170, 0))
            txt = (f"NO PERSON -> other" if gated else
                   (f"{cname}  {conf:.2f}  FIRE" if fired else f"hold  ({cname} {conf:.2f})"))
            for f in win:
                fr = f.copy()
                kpts, scores = M.pose(fr)
                for i in range(len(kpts)):
                    draw_skeleton(fr, kpts[i], scores[i])
                cv2.rectangle(fr, (0, H - 46), (330, H), (30, 30, 30), -1)
                cv2.putText(fr, txt, (8, H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 2)
                cv2.rectangle(fr, (0, 0), (320, 32), (30, 30, 30), -1)
                cv2.putText(fr, f"skeleton-aux  a={a.alpha}  enhanced view",
                            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
                vw.write(fr)
        vw.release()
        nf = sum(1 for r in rows if r["fired"])
        print(f"[viz] {cls}/{take}: {len(rows)} sliding windows, {nf} fire, "
              f"chart+video -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
