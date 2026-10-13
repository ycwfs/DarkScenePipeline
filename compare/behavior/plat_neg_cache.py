"""Cache the platform-collected false-positive clips as `other` negatives.

2026-10-12 delivery: 96 clips the OPERATOR saved on the platform at skel_fusion_alpha=1,
all verified non-actions (museum-hall visitors drinking/waving at nothing). Their whole
point is that the DEPLOYED stack fired on them, so they are the exact domain the RGB head
must learn to keep quiet -- the "other needs a footing inside the firing domain" lesson
from negatives_cache.py, one level harder: these windows are not merely from a firing
camera, they ARE firing windows (41/96 reproduce locally at alpha=1).

Two deliberate departures from realact_cache.read_processed:

  * NO re-enhancement. These pixels already went through the deployed chain once
    (retinexformer@840 -> SR x2 -> H.264). Feeding them through Retinexformer again would
    train on a double-enhanced distribution the recognizer never sees at serve time.
  * The bottom 76 px are cropped (the burned-in label bar, measured y>=944 of 1020). The
    bar exists only in saved clips, never in serve-time recognition input.

Windows: T=32 stride 16, same as the deployed recognition stride, so every firing moment
lands inside at least one window. Each window is hardlinked into val/other as well --
this cache trains the fit-round protocol (train=val), and the real referees are the
external suites (fp65 / positives / platform replay), not this val.
"""
import argparse
import glob
import os

import cv2
import numpy as np

T, STRIDE, STORE_SHORT, BAR_PX = 32, 16, 256, 76
DEFAULT_SRC = ("/ssd1/wfs/project/lowlight/DarkScenePipeline/"
               "online-negative-sample/20261012_083305_78")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--out", required=True, help="缓存目录（写 <out>/{train,val}/other/）")
    ap.add_argument("--split", default="train", choices=("train", "val"),
                    help="which split of an EXISTING merged cache to extend in place")
    a = ap.parse_args()

    vids = sorted(glob.glob(os.path.join(a.src, "*", "*.mp4")))
    if not vids:
        raise SystemExit(f"error: {a.src} 下没有 mp4")
    out_dir = os.path.join(a.out, a.split, "other")
    os.makedirs(out_dir, exist_ok=True)
    if a.split == "train":                                   # fit-round: train=val
        os.makedirs(os.path.join(a.out, "val", "other"), exist_ok=True)
    total = 0
    for path in vids:
        stem = os.path.splitext(os.path.basename(path))[0]
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            f = f[:f.shape[0] - BAR_PX]                      # drop the burned label bar
            s = STORE_SHORT / min(f.shape[:2])
            f = cv2.resize(f, (int(round(f.shape[1] * s)), STORE_SHORT),
                           interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release()
        n = len(frames)
        saved = 0
        for s0 in range(0, n - T + 1, STRIDE):
            name = f"platneg_{stem}_{s0:05d}.npy"
            p = os.path.join(out_dir, name)
            np.save(p, np.stack(frames[s0:s0 + T]))
            if a.split == "train":                           # fit-round: train=val
                vp = os.path.join(a.out, "val", "other", name)
                if not os.path.exists(vp):
                    os.link(p, vp)
            saved += 1
        total += saved
        print(f"[platneg] {stem} {n}f -> {saved} 窗", flush=True)
    print(f"[platneg] 合计 {total} 窗 -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
