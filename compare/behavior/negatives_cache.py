"""Cache person-free surveillance clips as `other` negatives for the realact finetune.

2026-09-09 lesson (realact6, rejected): finetuning on the new IR-night camera's action
takes ALONE made night false positives worse (72 -> 196 named events), because every
IR-domain window in the cache was a positive -- the domain itself became evidence for the
action classes, and the LwF teacher's own false predictions on IR input got distilled into
the student. `other` needs a footing inside the exact deployment domains that false-fire.

Inputs are 30 s 1080p grabs from the reservoir cameras (hydraulic/), static dam scenes with
no people: IR night mode at 00:00/03:00 and plain daytime. Same chain as realact_cache.py
(downscale 840 -> retinexformer -> short side 256, RGB). Windows are non-overlapping
(hop 40 frames > window 32) -- a static scene yields ~19 windows/clip, and with 20 clips
that is ~350 `other` windows, the same order as each action class.

Held-out probes: --val clips are written under val/other and NOT into train/, so the
best-epoch selection (which scores every class present in val) sees FP suppression on an
unseen night, not just action recall.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from realact_cache import read_processed, T                                # noqa: E402

DP = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", ".."))
H = "/ssd1/wfs/project/lowlight/hydraulic"
TRAIN_CLIPS = [
    # IR night, 2026-08-21 00:00/03:00, five stations (+ Lushan 04:30)
    f"{H}/dark-data/mp4/桃子冲水库雨水情_桃子冲水库雨水情_20260821000000_20260821000030.mp4",
    f"{H}/dark-data/mp4/桃子冲水库雨水情_桃子冲水库雨水情_20260821030000_20260821030030.mp4",
    f"{H}/dark-data/mp4/胡家塔水库雨水情_胡家塔水库雨水情_20260821000000_20260821000030.mp4",
    f"{H}/dark-data/mp4/胡家塔水库雨水情_胡家塔水库雨水情_20260821030000_20260821030030.mp4",
    f"{H}/dark-data/mp4/许山水库雨水情_许山水库雨水情_20260821000000_20260821000030.mp4",
    f"{H}/dark-data/mp4/许山水库雨水情_许山水库雨水情_20260821030000_20260821030030.mp4",
    f"{H}/dark-data/mp4/鲁山水库雨水情_鲁山水库雨水情_20260821000000_20260821000030.mp4",
    f"{H}/dark-data/mp4/鲁山水库雨水情_鲁山水库雨水情_20260821030000_20260821030030.mp4",
    f"{H}/dark-data/mp4/鲁山水库雨水情_鲁山水库雨水情_20260821043000_20260821043030.mp4",
    f"{H}/dark-data/mp4/鲫鱼沟水库雨水情_鲫鱼沟水库雨水情_20260821000000_20260821000030.mp4",
    f"{H}/dark-data/mp4/鲫鱼沟水库雨水情_鲫鱼沟水库雨水情_20260821030000_20260821030030.mp4",
    # day, 2026-08-21 06:00/12:00 (same static dams; Xushan 08-27 day clips are held out)
    f"{H}/dark-data/mp4/桃子冲水库雨水情_桃子冲水库雨水情_20260821060000_20260821060030.mp4",
    f"{H}/dark-data/mp4/桃子冲水库雨水情_桃子冲水库雨水情_20260821120000_20260821120030.mp4",
    f"{H}/dark-data/mp4/胡家塔水库雨水情_胡家塔水库雨水情_20260821060000_20260821060030.mp4",
    f"{H}/dark-data/mp4/胡家塔水库雨水情_胡家塔水库雨水情_20260821120000_20260821120030.mp4",
    f"{H}/dark-data/mp4/许山水库雨水情_许山水库雨水情_20260821060000_20260821060030.mp4",
    f"{H}/dark-data/mp4/许山水库雨水情_许山水库雨水情_20260821120000_20260821120030.mp4",
    f"{H}/dark-data/mp4/鲁山水库雨水情_鲁山水库雨水情_20260821120000_20260821120030.mp4",
    f"{H}/dark-data/mp4/鲫鱼沟水库雨水情_鲫鱼沟水库雨水情_20260821060000_20260821060030.mp4",
    f"{H}/dark-data/mp4/鲫鱼沟水库雨水情_鲫鱼沟水库雨水情_20260821120000_20260821120030.mp4",
]
VAL_CLIPS = [
    # unseen night + unseen day on the 08-27 delivery -- the generalization probes
    f"{H}/rgb-dark-data/mp4/许山水库雨水情_许山水库雨水情_20260827000000_20260827000030.mp4",
    f"{H}/rgb-dark-data/mp4/许山水库雨水情_许山水库雨水情_20260827030000_20260827030030.mp4",
    f"{H}/rgb-dark-data/mp4/许山水库雨水情_许山水库雨水情_20260827060000_20260827060030.mp4",
    f"{H}/rgb-dark-data/mp4/许山水库雨水情_许山水库雨水情_20260827080000_20260827080030.mp4",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="缓存目录（写 <out>/{train,val}/other/）")
    ap.add_argument("--hop", type=int, default=40, help="窗口起点间隔（帧）；>32 即不重叠")
    ap.add_argument("--ckpt-dir", default=os.path.join(DP, "ckpts"))
    ap.add_argument("--device", default="cuda:2")
    a = ap.parse_args()

    from darkpipe.stages.enhance_retinexformer import RetinexformerStage
    stage = RetinexformerStage(a.ckpt_dir)
    stage.load(a.device)
    total = {"train": 0, "val": 0}
    for split, clips in (("train", TRAIN_CLIPS), ("val", VAL_CLIPS)):
        d = os.path.join(a.out, split, "other")
        os.makedirs(d, exist_ok=True)
        for path in clips:
            stem = os.path.basename(path).replace("水库雨水情_", "")[:40]
            frames = read_processed(path, stage, 840)
            n = len(frames)
            saved = 0
            for s in range(0, n - T + 1, a.hop):
                np.save(os.path.join(d, f"{stem}_{s:05d}.npy"), np.stack(frames[s:s + T]))
                saved += 1
            total[split] += saved
            print(f"[neg] {stem} {n} 帧 -> {split}/other {saved}", flush=True)
    stage.close()
    print(f"[neg] 合计 train={total['train']} val={total['val']} -> {a.out}")


if __name__ == "__main__":
    main()
