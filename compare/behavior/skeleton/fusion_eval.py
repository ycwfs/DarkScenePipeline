"""Fusion A/B: RGB full-FOV model vs skeleton ST-GCN vs person-gate vs weighted fusion.

Suites (all sharing the deployed enhance chain):
  A. val       -- the realact val split (window-aligned: RGB cache + skeleton cache)
  B. fp65      -- the 65 no-action/person-walk clips, sliding 1 s @ 0.5 s + whole window
  C. positives -- the 4 IR night takes, sliding windows

Reports, per fusion weight alpha (p = alpha*p_rgb + (1-alpha)*p_skel, after the person
gate forces 'other' on no-person windows): top1/mF1 on val, fired-window counts on fp65,
correct-window counts on positives.

Usage: python -u fusion_eval.py <gpu> --skel-ckpt ../cache/skel_v1/stgcn.pth
"""
import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
import torch

REPO = "/ssd1/wfs/project/lowlight/DarkScenePipeline"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from darkpipe.constants import IN_MEAN, IN_STD, RECO_CFG
from darkpipe.stages.downscale import DownscaleStage
from darkpipe.stages.enhance_retinexformer import RetinexformerStage
from darkpipe.vendor.videomamba import videomamba_tiny

from stgcn import STGCN, merge_persons, normalize_window

LABELS = ["wave", "throw", "chase", "fall", "fight", "talk", "drink", "pick_up",
          "shake_hands", "other"]
RGB_CKPT = os.path.join(REPO, "ckpts", "videomamba_t_behavior_32f_realact11_fov.pth")
CACHE_DIR = "/ssd1/wfs/project/lowlight/compare/behavior/cache"
POSE_ONNX = os.path.join(REPO, "models", "pose", "rtmo-s_end2end.onnx")
ODIR = os.path.join(REPO, "real-actiondata-0909", "other")
SRC = os.path.join(REPO, "real-actiondata-0909")
TAKES = [("挥手22", "wave"), ("捡东西2", "pick_up"), ("追逐16", "chase"), ("喝水16", "drink")]
FOV_H, FOV_W = 224, 400
TAU = 0.5                                     # person gate on window-max mean-kpt score


def read_video(path):
    cap = cv2.VideoCapture(path)
    frames, fps = [], cap.get(cv2.CAP_PROP_FPS) or 25
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames, fps


class Models:
    def __init__(self, dev, skel_ckpt, pose="rtmo", pose_thr=0.3, rgb_ckpt=RGB_CKPT):
        self.rgb = videomamba_tiny(num_classes=10, num_frames=32, img_size=(FOV_H, FOV_W))
        self.rgb.load_state_dict(torch.load(rgb_ckpt, map_location="cpu")["model"], strict=True)
        self.rgb = self.rgb.to(dev).eval()
        self.skel = STGCN(num_classes=10)
        self.skel.load_state_dict(torch.load(skel_ckpt, map_location="cpu")["model"])
        self.skel = self.skel.to(dev).eval()
        from skeleton_cache import make_pose
        self.pose = make_pose(pose, pose_thr, device=dev)

    @torch.no_grad()
    def rgb_probs(self, frames_rgb):
        arr = np.stack([self._fov(f) for f in
                        [frames_rgb[i] for i in
                         np.linspace(0, len(frames_rgb) - 1, 32).round().astype(int)]])
        x = torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0).contiguous().to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            return self.rgb(x).softmax(1)[0].float().cpu().numpy()

    @staticmethod
    def _fov(f_bgr):
        r = cv2.resize(f_bgr, (FOV_W, FOV_H), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return (rgb - IN_MEAN) / IN_STD

    @torch.no_grad()
    def skel_probs(self, frames_bgr):
        """frames are ENHANCED bgr -> pose per frame -> ST-GCN probs + presence."""
        T = len(frames_bgr)
        sk = np.zeros((T, 2, 17, 3), np.float32)
        H, W = frames_bgr[0].shape[:2]
        for t in range(T):
            kpts, scores = self.pose(frames_bgr[t])
            if len(kpts) == 0:
                continue
            per = scores.mean(1)
            for m, i in enumerate(np.argsort(per)[::-1][:2]):
                sk[t, m, :, :2] = kpts[i] / np.array([W, H], np.float32)
                sk[t, m, :, 2] = scores[i]
        x, pres = normalize_window(sk)
        xt = torch.from_numpy(x).unsqueeze(0).to(dev)
        lg = torch.stack([self.skel(xt[:, m]) for m in range(xt.shape[1])], 1)[0]
        p = merge_persons(lg, torch.tensor(pres).unsqueeze(0).to(dev))[0].softmax(0)
        return p.float().cpu().numpy(), float(pres.max())


def windows_of(n, fps):
    w = max(2, int(round(1.0 * fps)))
    wins = [("whole", 0, n)]
    i = 0
    while i + w <= n:
        wins.append((f"t{i / fps:.1f}", i, i + w))
        i += max(1, int(round(0.5 * fps)))
    return wins


def fuse(pr, ps, alpha):
    return alpha * pr + (1 - alpha) * ps


def fired(p, conf=0.65):
    return LABELS[int(p.argmax())] != "other" and float(p.max()) >= conf


def main():
    global dev
    ap = argparse.ArgumentParser()
    ap.add_argument("gpu")
    ap.add_argument("--skel-ckpt", default="../cache/skel_v1/stgcn.pth")
    ap.add_argument("--skel-cache", default="../cache/skel_v1",
                    help="skeleton cache for suite A (must match the pose model)")
    ap.add_argument("--rgb-ckpt", default=RGB_CKPT,
                    help="VideoMamba checkpoint (default realact11_fov)")
    ap.add_argument("--val-cache", default=os.path.join(CACHE_DIR, "realact_rf_v10_full"),
                    help="RGB cache dir whose val split suite A reads")
    ap.add_argument("--pose", choices=("rtmo", "yolo26"), default="rtmo")
    ap.add_argument("--pose-thr", type=float, default=None,
                    help="det/kpt score threshold (default 0.3 rtmo / 0.15 yolo26)")
    ap.add_argument("--alphas", default="1.0,0.7,0.5")
    ap.add_argument("--suites", default="ABC", choices=("A", "ABC", "B", "C", "BC"),
                    help="which suites to run (A=val, B=fp, C=positives)")
    a = ap.parse_args()
    dev = f"cuda:{a.gpu}"
    thr = a.pose_thr if a.pose_thr is not None else (0.3 if a.pose == "rtmo" else 0.15)
    alphas = [float(x) for x in a.alphas.split(",")]

    ds = DownscaleStage(840)
    st = RetinexformerStage(os.path.join(REPO, "ckpts"))
    st.load(dev)
    M = Models(dev, a.skel_ckpt, pose=a.pose, pose_thr=thr, rgb_ckpt=a.rgb_ckpt)

    # ---- A. val split (window-aligned caches) ---------------------------------------
    val = sorted(glob.glob(os.path.join(a.val_cache, "val", "*", "*.npy"))) \
        if "A" in a.suites else []
    rows = []
    for p in val:
        lab = os.path.basename(os.path.dirname(p))
        arr = np.load(p)                                    # (32,256,456,3) uint8 RGB
        pr = M.rgb_probs(np.ascontiguousarray(arr[..., ::-1]))   # _fov expects BGR
        skel_p = os.path.join(a.skel_cache, "val", lab, os.path.basename(p))
        x, pres = normalize_window(np.load(skel_p))
        xt = torch.from_numpy(x).unsqueeze(0).to(dev)
        with torch.no_grad():
            lg = torch.stack([M.skel(xt[:, m]) for m in range(xt.shape[1])], 1)[0]
        ps = merge_persons(lg, torch.tensor(pres).unsqueeze(0).to(dev))[0].softmax(0)
        rows.append((lab, pr, ps.float().cpu().numpy(), float(pres.max())))
    print(f"\n== A. val ({len(rows)} windows)  gate TAU={TAU} ==")
    for alpha in alphas:
        cm = np.zeros((10, 10), np.int64)
        for lab, pr, ps, pres in rows:
            p = fuse(pr, ps, alpha) if pres >= TAU else \
                np.eye(10)[LABELS.index("other")]
            cm[LABELS.index(lab), int(p.argmax())] += 1
        tp = np.diag(cm).astype(float)
        f1 = 2 * tp / np.maximum(cm.sum(0) + cm.sum(1), 1e-9)
        present = cm.sum(1) > 0
        print(f"  alpha={alpha:.2f}: top1={tp.sum()/cm.sum():.4f} "
              f"mF1={f1[present].mean():.4f}")
        print("    " + "  ".join(f"{LABELS[i]}:f{f1[i]:.2f}(n{int(cm[i].sum())})"
                                 for i in range(10) if cm[i].sum()))

    # ---- B. 65-clip FP suite ---------------------------------------------------------
    if "B" not in a.suites:
        print("\n== B. fp suite (skipped) ==")
    else:
        src_of = {r["clip"]: r["source"] for r in csv.DictReader(
            open(os.path.join(ODIR, "other_segments.csv")))}
        clips = sorted(f for f in os.listdir(ODIR)
                       if f.startswith("other") and f.endswith(".mp4"))
        print(f"\n== B. fp suite ({len(clips)} clips, sliding 1s @0.5s + whole) ==")
        for alpha in alphas:
            agg = defaultdict(lambda: [0, 0])
            for name in clips:
                frames, fps = read_video(os.path.join(ODIR, name))
                frames = st(ds(frames))
                n = len(frames)
                for wname, i0, i1 in windows_of(n, fps):
                    win = frames[i0:i1]
                    pr = M.rgb_probs(win)
                    ps, pres = M.skel_probs(win)
                    p = fuse(pr, ps, alpha) if pres >= TAU else \
                        np.eye(10)[LABELS.index("other")]
                    a_ = agg[src_of[name]]
                    a_[0] += 1
                    a_[1] += fired(p)
            tot_w = sum(v[0] for v in agg.values())
            tot_f = sum(v[1] for v in agg.values())
            print(f"  alpha={alpha:.2f}: fired {tot_f}/{tot_w}")
            for s, (nw, nf) in sorted(agg.items()):
                if nf:
                    print(f"     {s}: {nf}/{nw}")

    # ---- C. positives ----------------------------------------------------------------
    if "C" not in a.suites:
        print("\n== C. positives (skipped) ==")
        return
    print("\n== C. positives (4 IR takes) ==")
    for alpha in alphas:
        line = []
        for stem, expect in TAKES:
            frames, fps = read_video(os.path.join(SRC, stem + ".mp4"))
            frames = st(ds(frames))
            n = len(frames)
            hits = 0
            tot = 0
            for wname, i0, i1 in windows_of(n, fps):
                pr = M.rgb_probs(frames[i0:i1])
                ps, pres = M.skel_probs(frames[i0:i1])
                p = fuse(pr, ps, alpha) if pres >= TAU else \
                    np.eye(10)[LABELS.index("other")]
                tot += 1
                hits += fired(p) and LABELS[int(p.argmax())] == expect
            line.append(f"{stem} {hits}/{tot}")
        print(f"  alpha={alpha:.2f}: " + "  ".join(line))


if __name__ == "__main__":
    main()
