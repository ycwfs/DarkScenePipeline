"""Unified metrics over an output folder vs GT folder.

PSNR/SSIM use the exact functions from HVI-CIDNet/measure.py so numbers are
directly comparable to the paper table. LPIPS uses AlexNet (cached weights).
Computes BOTH variants in one pass:
  raw        : output as-is
  gt_mean    : output rescaled so its gray mean matches the GT gray mean
               (the LLFlow/KinD/Retinexformer convention, paired sets only)
Filenames in out_dir must match filenames in gt_dir.
Prints a JSON line: {"raw": {...}, "gt_mean": {...}, "n": N}
"""
import os
import sys
import glob
import json
import argparse

import cv2
import numpy as np
import torch
import lpips
from PIL import Image


def ssim(prediction, target):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(target, ref):
    img1 = np.array(target, dtype=np.float64)
    img2 = np.array(ref, dtype=np.float64)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            return np.array([ssim(img1[:, :, i], img2[:, :, i]) for i in range(3)]).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def calculate_psnr(target, ref):
    img1 = np.array(target, dtype=np.float32)
    img2 = np.array(ref, dtype=np.float32)
    diff = img1 - img2
    return 10.0 * np.log10(255.0 * 255.0 / (np.mean(np.square(diff)) + 1e-8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--im_dir", required=True, help="glob or folder of outputs")
    ap.add_argument("--gt_dir", required=True, help="GT folder (trailing slash ok)")
    args = ap.parse_args()

    im_glob = args.im_dir
    if os.path.isdir(im_glob):
        im_glob = os.path.join(im_glob, "*")
    gt_dir = args.gt_dir if args.gt_dir.endswith("/") else args.gt_dir + "/"

    loss_fn = lpips.LPIPS(net='alex')
    loss_fn.cuda()

    acc = {"raw": {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0},
           "gt_mean": {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}}
    n = 0
    for item in sorted(glob.glob(im_glob)):
        if not item.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            continue
        name = os.path.basename(item)
        im1 = Image.open(item).convert('RGB')
        gt_path = os.path.join(gt_dir, name)
        if not os.path.exists(gt_path):
            # try alternate extension
            stem = os.path.splitext(name)[0]
            cand = glob.glob(os.path.join(gt_dir, stem + ".*"))
            if not cand:
                continue
            gt_path = cand[0]
        im2 = Image.open(gt_path).convert('RGB')
        (w, h) = im2.size
        im1 = im1.resize((w, h))
        im1 = np.array(im1)
        im2 = np.array(im2)
        n += 1

        for variant in ("raw", "gt_mean"):
            a = im1.astype(np.float32)
            if variant == "gt_mean":
                mean_restored = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY).mean()
                mean_target = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY).mean()
                a = np.clip(im1 * (mean_target / (mean_restored + 1e-8)), 0, 255)
            acc[variant]["psnr"] += calculate_psnr(a, im2)
            acc[variant]["ssim"] += calculate_ssim(a, im2)
            ex_p0 = lpips.im2tensor(a.astype(np.uint8)).cuda()
            ex_ref = lpips.im2tensor(im2).cuda()
            acc[variant]["lpips"] += loss_fn.forward(ex_ref, ex_p0).item()
        torch.cuda.empty_cache()

    for variant in acc:
        for k in acc[variant]:
            acc[variant][k] = float(acc[variant][k]) / max(n, 1)
    acc["n"] = n
    print(json.dumps(acc))


if __name__ == "__main__":
    main()
