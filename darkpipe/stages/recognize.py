"""Sliding-window action recognizers: R(2+1)D-18 and VideoMamba-T/32f.

Frames are preprocessed once at push() time (resize short side, center crop, normalize)
and kept in a ring buffer; every `stride` pushes after the window fills, one forward
produces a RecognitionEvent. INTER_AREA is used for strong downscales (scale < 0.5,
e.g. 480px SR frames -> 128) to avoid aliasing; INTER_LINEAR (the training resize)
otherwise.
"""
import os
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn

from ..constants import CKPT_FILES, CLASSES, RECO_CFG
from .base import RecognitionEvent, Recognizer


class _WindowRecognizer(Recognizer):
    kind = "r3d"

    def __init__(self, ckpt_dir: str, stride: int | None = None):
        self.cfg = RECO_CFG[self.kind]
        self.window = self.cfg["T"]
        self.stride = stride or self.window // 2
        self.ckpt = os.path.join(ckpt_dir, CKPT_FILES[self.kind])
        self.buf = deque(maxlen=self.window)
        self._filled_pushes = 0  # pushes seen since the window first filled
        self.net = None
        self.device = "cuda"
        self.name = self.kind

    def _build(self):
        raise NotImplementedError

    def load(self, device: str) -> None:
        self.device = device
        self.net = self._build().to(device).eval()

    def _preprocess(self, frame_bgr):
        c = self.cfg
        h, w = frame_bgr.shape[:2]
        s = c["resize"] / min(h, w)
        nh, nw = int(round(h * s)), int(round(w * s))
        interp = cv2.INTER_AREA if s < 0.5 else cv2.INTER_LINEAR
        r = cv2.resize(frame_bgr, (nw, nh), interpolation=interp)
        top, left = (nh - c["size"]) // 2, (nw - c["size"]) // 2
        crop = r[top:top + c["size"], left:left + c["size"]]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return (rgb - c["mean"]) / c["std"]  # HWC float32

    @torch.inference_mode()
    def push(self, frame_bgr, frame_index: int, timestamp: float):
        self.buf.append(self._preprocess(frame_bgr))
        if len(self.buf) < self.window:
            return None
        fire = self._filled_pushes % self.stride == 0  # first full window, then every stride
        self._filled_pushes += 1
        if not fire:
            return None
        arr = np.stack(self.buf)  # T,H,W,C
        x = torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0).contiguous().to(self.device)
        with torch.autocast("cuda", dtype=torch.float16):
            prob = self.net(x).softmax(1)[0].float().cpu().numpy()
        order = prob.argsort()[::-1][:3]
        return RecognitionEvent(
            frame_index=frame_index, timestamp=timestamp,
            label=CLASSES[int(order[0])], confidence=float(prob[order[0]]),
            topk=[(CLASSES[int(i)], float(prob[i])) for i in order],
            model=self.kind, window=self.window)

    def close(self) -> None:
        self.net = None
        self.buf.clear()
        torch.cuda.empty_cache()


class R3DRecognizer(_WindowRecognizer):
    kind = "r3d"

    def _build(self):
        from torchvision.models.video import r2plus1d_18
        net = r2plus1d_18(weights=None)
        net.fc = nn.Linear(net.fc.in_features, len(CLASSES))
        net.load_state_dict(torch.load(self.ckpt, map_location="cpu")["model"])
        return net


class VideoMambaRecognizer(_WindowRecognizer):
    kind = "videomamba"

    def _build(self):
        from ..vendor.videomamba import videomamba_tiny
        net = videomamba_tiny(num_classes=len(CLASSES), num_frames=self.window, img_size=224)
        net.load_state_dict(torch.load(self.ckpt, map_location="cpu")["model"])
        return net
