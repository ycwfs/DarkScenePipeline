"""Run the DA-LLPose (ECCV'24) low-light pose teacher on OUR field videos.

The inference recipe is copied verbatim from the repo's own test loop
(tools/valid_test.py): multi-scale {0.5,1,2} + flip-test, DEKR root-regression
decoding, pose NMS (NMS_THRE 0.15), heatmap matching (MATCH_HMP) -- the exact
settings the README gives for ExLPose-test evaluation. Input frames go through
the same colour path as their test loader (cv2 BGR -> RGB, HWC uint8, ImageNet
normalisation), so what the teacher sees here is what it saw at eval time.

Output per video: one json (per-frame CrowdPose-14 people in original pixel
coords, plus the COCO-17 mapping used by our RTMO-s pipeline: 12 limb/torso
points map 1:1, nose <- head is an approximation, COCO eyes/ears have no
counterpart and are emitted v=0/unlabeled), and optional score-coloured
visualisation jpgs for eyeballing quality before trusting it as pseudo-GT.

Usage (darkpipe-dev:dallpose container, one free GPU):
  python -u tools/dark_infer.py --videos a.mp4 b.mp4 --out OUTDIR
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
import torchvision

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "lib"))
PKG = "/ssd1/wfs/project/lowlight/DarkScenePipeline/platform/dark-demo-v2"

import models                                                # noqa: E402  (lib/models)
from config import cfg, update_config                        # noqa: E402
from core.inference import get_multi_stage_outputs, aggregate_results  # noqa: E402
from core.nms import pose_nms                                # noqa: E402
from core.match import match_pose_to_heatmap                 # noqa: E402
from utils.transforms import (resize_align_multi_scale,      # noqa: E402
                              get_final_preds, get_multi_scale_size)

# CrowdPose-14 (this repo's convention, see lib/dataset/transforms/build.py):
# 0/1 L/R shoulder, 2/3 L/R elbow, 4/5 L/R wrist, 6/7 L/R hip, 8/9 L/R knee,
# 10/11 L/R ankle, 12 head, 13 neck.  Skeleton edges for drawing:
CROWD_EDGES = [(12, 13), (13, 0), (13, 1), (0, 1), (0, 2), (1, 3), (2, 4),
               (3, 5), (0, 6), (1, 7), (6, 7), (6, 8), (7, 9), (8, 10), (9, 11)]
# crowd14 index -> COCO-17 index (our RTMO-s convention)
CROWD2COCO = {0: 5, 1: 6, 2: 7, 3: 8, 4: 9, 5: 10, 6: 11, 7: 12, 8: 13,
              9: 14, 10: 15, 11: 16, 12: 0}          # nose <- head (approx)
COCO_NAMES = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_sho", "r_sho",
              "l_el", "r_el", "l_wr", "r_wr", "l_hip", "r_hip", "l_knee",
              "r_knee", "l_ank", "r_ank"]


def to_coco17(kpts14):
    """(14,3) x,y,score -> (17,3) x,y,v in COCO order; v: 2 labeled, 0 absent."""
    out = np.zeros((17, 3), np.float32)
    for c, coco_i in CROWD2COCO.items():
        out[coco_i, :2] = kpts14[c, :2]
        out[coco_i, 2] = 2 if float(kpts14[c, 2]) > 0.05 else 0
    return out


def score_color(s):
    if s >= 0.5:
        return (0, 200, 0)
    if s >= 0.3:
        return (0, 165, 255)
    return (0, 0, 255)


def draw_crowd(img, kpts14):
    for i, j in CROWD_EDGES:
        s = min(float(kpts14[i, 2]), float(kpts14[j, 2]))
        if s < 0.10:
            continue
        cv2.line(img, tuple(np.round(kpts14[i, :2]).astype(int)),
                 tuple(np.round(kpts14[j, :2]).astype(int)), score_color(s), 2,
                 cv2.LINE_AA)
    for x, y, s in kpts14:
        if float(s) < 0.10:
            continue
        cv2.circle(img, (int(round(x)), int(round(y))), 3, score_color(float(s)),
                   -1, cv2.LINE_AA)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cfg", default=os.path.join(REPO, "experiments/exlpose/test_config.yaml"))
    ap.add_argument("--model", default=os.path.join(REPO, "model/exlpose/model_final.pth"))
    ap.add_argument("--stride", type=int, default=1, help="run every Nth frame")
    ap.add_argument("--limit-frames", type=int, default=0, help="0 = whole video")
    ap.add_argument("--vis-frames", type=int, default=6,
                    help="dump visualised jpgs for the first N person-frames (0=off)")
    ap.add_argument("--kpt-thr", type=float, default=0.30,
                    help="joint score threshold for presence stats")
    ap.add_argument("--enhance", action="store_true",
                    help="feed frames through the served chain (downscale 840 + "
                         "Retinexformer) instead of raw frames -- the pixel domain "
                         "our RTMO-s actually sees at serve time; output coords are "
                         "then in the 840-side frame")
    return ap.parse_args()


def main():
    a = parse_args()
    class _A:
        pass
    _na = _A()
    _na.cfg = a.cfg
    _na.opts = []
    update_config(cfg, _na)
    # README's ExLPose-test command line settings, applied directly:
    cfg.defrost()
    cfg.GPUS = (0,)
    cfg.TEST.MODEL_FILE = a.model
    cfg.TEST.SCALE_FACTOR = [0.5, 1, 2]
    cfg.TEST.FLIP_TEST = True
    cfg.TEST.NMS_THRE = 0.15
    cfg.TEST.MATCH_HMP = True
    cfg.DATASET.INPUT_SIZE = cfg.DATASET.INPUT_SIZE   # 512, unchanged
    cfg.freeze()

    model = eval("models." + cfg.MODEL.NAME + ".get_pose_net")(cfg, is_train=False)
    sd = torch.load(cfg.TEST.MODEL_FILE, map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=True)
    model.cuda().eval()
    print(f"[dark_infer] model loaded: {cfg.MODEL.NAME} "
          f"joints={cfg.MODEL.NUM_JOINTS} from {os.path.basename(cfg.TEST.MODEL_FILE)}",
          flush=True)

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    ])

    os.makedirs(a.out, exist_ok=True)
    if a.enhance:
        sys.path.insert(0, PKG)
        from darkpipe.stages.downscale import DownscaleStage
        from darkpipe.stages.enhance_retinexformer import RetinexformerStage
        ds = DownscaleStage(840)
        enh = RetinexformerStage(os.path.join(PKG, "ckpts"))
        enh.load("cuda")
        print("[dark_infer] enhance chain ON: downscale 840 + Retinexformer", flush=True)

    for path in a.videos:
        stem = os.path.basename(path).rsplit(".", 1)[0]
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames = []
        vis_left = a.vis_frames
        i = 0
        n_person_frames = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            take = (i % a.stride == 0)
            if take:
                if a.enhance:
                    bgr = enh(ds([bgr]))[0]
                image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)   # same as their loader
                base_size, center, scale = get_multi_scale_size(
                    image, cfg.DATASET.INPUT_SIZE, 1.0, 1.0)
                with torch.no_grad():
                    heatmap_sum = 0
                    poses = []
                    scale_resized = None
                    for scale in sorted(cfg.TEST.SCALE_FACTOR, reverse=True):
                        image_resized, center, scale_resized = resize_align_multi_scale(
                            image, cfg.DATASET.INPUT_SIZE, scale, 1.0)
                        image_resized = transforms(image_resized).unsqueeze(0).cuda()
                        heatmap, posemap = get_multi_stage_outputs(
                            cfg, model, image_resized, cfg.TEST.FLIP_TEST)
                        heatmap_sum, poses = aggregate_results(
                            cfg, heatmap_sum, poses, heatmap, posemap, scale)
                    heatmap_avg = heatmap_sum / len(cfg.TEST.SCALE_FACTOR)
                    poses, scores = pose_nms(cfg, heatmap_avg, poses)
                    if len(scores) > 0:
                        if cfg.TEST.MATCH_HMP:
                            poses = match_pose_to_heatmap(cfg, poses, heatmap_avg)
                        final_poses = get_final_preds(
                            poses, center, scale_resized, base_size)
                    else:
                        final_poses = []

                people = []
                top2 = sorted(
                    range(len(final_poses)),
                    key=lambda k: -float(np.mean(final_poses[k][:, 2])))[:2]
                for k in top2:                       # keep top-2, like serving
                    k14 = np.asarray(final_poses[k][:, :3], np.float32)
                    people.append(dict(kpts14=k14.tolist(),
                                       coco17=to_coco17(k14).tolist(),
                                       score=float(np.mean(k14[:, 2])),
                                       root_score=float(scores[k])))
                if people:
                    n_person_frames += 1
                    if vis_left > 0:
                        vis = bgr.copy()
                        for p in people:
                            draw_crowd(vis, np.asarray(p["kpts14"]))
                        cv2.putText(vis, f"f{i} n{len(people)} "
                                    f"s{people[0]['score']:.2f}", (8, 28),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
                        cv2.putText(vis, f"f{i} n{len(people)} "
                                    f"s{people[0]['score']:.2f}", (8, 28),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                        vd = os.path.join(a.out, stem + "_vis")
                        os.makedirs(vd, exist_ok=True)
                        cv2.imwrite(os.path.join(vd, f"{i:06d}.jpg"), vis,
                                    [cv2.IMWRITE_JPEG_QUALITY, 88])
                        vis_left -= 1
                frames.append(dict(i=i, t=round(i / fps, 3), people=people))
            i += 1
            if a.limit_frames and i >= a.limit_frames:
                break
        cap.release()
        with open(os.path.join(a.out, stem + ".json"), "w", encoding="utf-8") as f:
            json.dump(dict(video=os.path.basename(path), fps=fps, n_read=i,
                           n_person_frames=n_person_frames, frames=frames),
                      f, ensure_ascii=False)
        print(f"[dark_infer] {stem}: {i} frames read, {n_person_frames} person-frames"
              f" -> {stem}.json", flush=True)


if __name__ == "__main__":
    main()
