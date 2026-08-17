"""Low-light enhancement stage: HVI-CIDNet (CVPR2025), fp32, chunked.

Uses the vendored CIDNet (darkpipe/vendor/cidnet). Runtime settings mirror the
official "arbitrary image" inference path (eval_hf.py): gated + gated2 enabled,
alpha_s = alpha_i = 1.0, gamma = 1.0, reflect-pad to a multiple of 8. Runs in fp32
— CIDNet's HVI color-space transform uses trig ops that are numerically happier in
fp32 and gain little from fp16 (see the LOLv2 speed benchmark).
"""
import os

import cv2
import numpy as np
import torch

from ..constants import CKPT_FILES
from ..utils import free_device_cache, bgr_batch_to_tensor, chunked, reflect_pad_to, tensor_to_bgr_list
from .base import FrameStage


def _despeckle(bgr, lo, hi, median):
    """Luminance-guided desaturation + optional median, to suppress CIDNet's dark-region
    color speckle on extreme-dark input. CIDNet's HVI->RGB transform amplifies the color
    (H,V) channels where intensity is near zero (PHVIT divides by a tiny color_sensitive),
    yielding spurious saturated pixels. Color is only trustworthy where the enhanced pixel
    is bright enough, so scale saturation by a smooth ramp of output value; median removes
    residual isolated specks. Intensity/structure is untouched."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    v = hsv[:, :, 2] / 255.0
    hsv[:, :, 1] *= np.clip((v - lo) / (hi - lo), 0.0, 1.0)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return cv2.medianBlur(out, median) if median else out


class CIDNetStage(FrameStage):
    name = "enhance:cidnet"

    def __init__(self, ckpt_dir: str, chunk: int = 32,
                 alpha_s: float = 1.0, alpha_i: float = 1.0, gamma: float = 1.0,
                 despeckle: bool = True, desat_lo: float = 0.20, desat_hi: float = 0.55,
                 median: int = 3):
        self.ckpt = os.path.join(ckpt_dir, CKPT_FILES["cidnet"])
        self.chunk = chunk
        self.alpha_s = alpha_s
        self.alpha_i = alpha_i
        self.gamma = gamma
        self.despeckle = despeckle
        self.desat_lo = desat_lo
        self.desat_hi = desat_hi
        self.median = median
        self.net = None
        self.device = "cuda"

    def load(self, device: str) -> None:
        from ..vendor.cidnet import CIDNet
        net = CIDNet()
        sd = torch.load(self.ckpt, map_location="cpu")
        sd = sd["params"] if isinstance(sd, dict) and "params" in sd else sd
        net.load_state_dict(sd, strict=True)
        net.trans.gated = True
        net.trans.gated2 = True
        net.trans.alpha_s = self.alpha_s
        net.trans.alpha = self.alpha_i
        self.device = device
        self.net = net.to(device).eval()   # fp32

    @torch.inference_mode()
    def _batch(self, frames: list) -> list:
        x = bgr_batch_to_tensor(frames, self.device, torch.float32)
        x, (h, w) = reflect_pad_to(x, 8)
        if self.gamma != 1.0:
            x = x ** self.gamma
        y = self.net(x)[:, :, :h, :w]
        batch = tensor_to_bgr_list(y)
        if self.despeckle:
            batch = [_despeckle(f, self.desat_lo, self.desat_hi, self.median) for f in batch]
        return batch

    def __call__(self, frames: list) -> list:
        return chunked(self, frames, self._batch)

    def close(self) -> None:
        self.net = None
        free_device_cache(self.device)
