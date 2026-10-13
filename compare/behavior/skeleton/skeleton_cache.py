"""Extract COCO-17 skeletons for every window of an RGB realact cache via RTMO or YOLO26.

Runs the pose model directly on the stored (32,256,W,3) uint8 RGB frames -- they are
already downscaled+enhanced, so the skeleton windows are frame-exact aligned with the RGB
track (same windows, same order). Per window the top-2 persons by mean keypoint score are
stored as (32, M=2, 17, 3) float32: x/W, y/H in [0,1] plus the keypoint score; a missing
person slot stays all zero. The keypoint scores double as the person-gate signal.

Usage: python -u skeleton_cache.py --rgb-cache ../cache/realact_rf_v10_full --out ../cache/skel_v1
       python -u skeleton_cache.py --pose yolo26 --out ../cache/skel_y26_v1
"""
import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np

REPO = "/ssd1/wfs/project/lowlight/DarkScenePipeline"
POSE_ONNX = os.path.join(REPO, "models", "pose", "rtmo-s_end2end.onnx")
YOLO_PTS = "/ssd1/wfs/project/lowlight/yolo26m-pose.pt"


def make_pose(name, score_thr, device="cuda:0"):
    """-> fn(bgr HxW) -> (kpts_px (N,17,2) float32, scores (N,17) float32)."""
    if name == "rtmo":
        from rtmlib import RTMO
        model = RTMO(onnx_model=POSE_ONNX, model_input_size=(640, 640),
                     backend="onnxruntime", device=device, score_thr=score_thr)

        def fn(bgr):
            return model(bgr)
    elif name == "yolo26":
        from ultralytics import YOLO
        model = YOLO(YOLO_PTS)
        model.to(device)
        model(np.zeros((480, 640, 3), np.uint8), verbose=False,
              device=int(device.split(":")[-1]) if ":" in device else 0)   # torch warmup

        def fn(bgr):
            r = model(bgr, verbose=False, conf=score_thr)[0]
            k = r.keypoints
            if k is None or k.xyn is None or len(k.xyn) == 0:
                return np.zeros((0, 17, 2), np.float32), np.zeros((0, 17), np.float32)
            xy = k.xy.float().cpu().numpy()                          # (N,17,2) px
            sc = k.conf.float().cpu().numpy().reshape(len(xy), 17)   # (N,17)
            return xy.astype(np.float32), sc.astype(np.float32)
    else:
        raise ValueError(name)
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb-cache", default="../cache/realact_rf_v10_full")
    ap.add_argument("--out", default="../cache/skel_v1")
    ap.add_argument("--pose", choices=("rtmo", "yolo26"), default="rtmo")
    ap.add_argument("--score-thr", type=float, default=0.3)
    ap.add_argument("--max-persons", type=int, default=2)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", default="0/1", help="i/n")
    a = ap.parse_args()

    model = make_pose(a.pose, a.score_thr, a.device)

    items = []
    for split in ("train", "val"):
        for p in sorted(glob.glob(os.path.join(a.rgb_cache, split, "*", "*.npy"))):
            items.append((split, p))
    i, n = map(int, a.shard.split("/"))
    items = [it for k, it in enumerate(items) if k % n == i]
    print(f"[skel:{a.pose}] {len(items)} windows shard={a.shard} -> {a.out}", flush=True)

    t0, done = time.time(), 0
    conf_hist = []                                          # per-window top person score
    for split, path in items:
        cls = os.path.basename(os.path.dirname(path))
        out_dir = os.path.join(a.out, split, cls)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, os.path.basename(path))
        if os.path.exists(out_path):
            continue
        arr = np.load(path)                                 # (32,H,W,3) uint8 RGB
        T, H, W = arr.shape[:3]
        out = np.zeros((T, a.max_persons, 17, 3), np.float32)
        for t in range(T):
            bgr = cv2.cvtColor(arr[t], cv2.COLOR_RGB2BGR)
            kpts, scores = model(bgr)                       # (N,17,2) px, (N,17)
            if len(kpts) == 0:
                continue
            per = scores.mean(1)                            # person-level score
            order = np.argsort(per)[::-1][:a.max_persons]
            for m, i in enumerate(order):
                out[t, m, :, :2] = kpts[i] / np.array([W, H], np.float32)
                out[t, m, :, 2] = scores[i]
        np.save(out_path, out)
        conf_hist.append(float(out[:, 0, :, 2].max()))
        done += 1
        if done % 100 == 0:
            print(f"[skel:{a.pose}] {done}/{len(items)}  {(time.time()-t0)/done:.2f}s/win", flush=True)

    conf_hist = np.array(conf_hist)
    print(f"[skel:{a.pose}] DONE {done} windows in {time.time()-t0:.0f}s; "
          f"window top-conf percentiles "
          + " ".join(f"p{p}={np.percentile(conf_hist, p):.2f}" for p in (1, 5, 25, 50, 75)),
          flush=True)


if __name__ == "__main__":
    main()
