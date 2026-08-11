"""Sliding-window action recognizers: R(2+1)D-18, VideoMamba-T/32f (ARID-11) and
VideoMamba-T/32f behavior (the 9 common behaviors + `other`).

Frames are preprocessed once at push() time (resize short side, center crop, normalize)
and kept in a ring buffer; every `stride` pushes after the window fills, one forward
produces a RecognitionEvent. INTER_AREA is used for strong downscales (scale < 0.5,
e.g. 480px SR frames -> 128) to avoid aliasing; INTER_LINEAR (the training resize)
otherwise.

Two window policies:
  span_sec=None (default)  the model sees the last `window` PROCESSED frames — the policy
                           both ARID checkpoints were validated under.
  span_sec=S               the buffer is trimmed by timestamp and `window` frames are
                           sampled uniformly from the last S seconds, so the emitted label
                           always describes the past <= S s regardless of pipeline speed.
"""
import os
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn

from ..constants import BEHAVIOR_DISPLAY, BEHAVIORS, CKPT_FILES, CLASSES, RECO_CFG
from .base import RecognitionEvent, Recognizer


def preprocess_frame(frame_bgr, cfg):
    """BGR uint8 -> normalized HWC float32, exactly as the recognizers see it at serve time.

    Module-level so the offline cache/eval/training scripts import the SAME function and the
    training domain cannot silently drift from the serving domain.
    """
    h, w = frame_bgr.shape[:2]
    s = cfg["resize"] / min(h, w)
    nh, nw = int(round(h * s)), int(round(w * s))
    interp = cv2.INTER_AREA if s < 0.5 else cv2.INTER_LINEAR
    r = cv2.resize(frame_bgr, (nw, nh), interpolation=interp)
    top, left = (nh - cfg["size"]) // 2, (nw - cfg["size"]) // 2
    crop = r[top:top + cfg["size"], left:left + cfg["size"]]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return (rgb - cfg["mean"]) / cfg["std"]


def reject_probs(p, labels, tau):
    """Open-set rule: unless a NAMED behavior clears `tau`, call it `other`.

    Reported `other` confidence is 1 - max(named), i.e. "probability it is none of the nine" —
    the quantity the rule actually decided on. The untouched distribution still reaches the
    caller through the top-k list.

    Module-level so the calibration script can sweep tau over stored raw probabilities instead
    of re-running the vision tower, without reimplementing (and drifting from) the rule.
    """
    if "other" not in labels or tau <= 0:
        return p
    o = labels.index("other")
    named = np.delete(p, o)
    if named.max() < tau and p.argmax() != o:
        p = p.copy()
        # max() keeps `other` the argmax even for a tau above 0.5
        p[o] = max(1.0 - float(named.max()), float(named.max()) + 1e-6)
    return p


class _WindowRecognizer(Recognizer):
    kind = "r3d"
    labels = CLASSES

    def __init__(self, ckpt_dir: str, stride: int | None = None,
                 span_sec: float | None = None, reject_tau: float = 0.0):
        self.cfg = RECO_CFG[self.kind]
        self.window = self.cfg["T"]
        self.stride = stride or self.window // 2
        self.span = span_sec
        # Below this, a named behaviour is reported as `other` instead (see reject_probs).
        # 0 disables. Only meaningful for label sets that contain `other` -- the ARID heads
        # have no such class, and reject_probs leaves their distribution untouched.
        self.reject_tau = reject_tau
        self.ckpt = os.path.join(ckpt_dir, CKPT_FILES[self.kind])
        self.buf = deque()  # (preprocessed HWC float32, timestamp)
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
        return preprocess_frame(frame_bgr, self.cfg)

    def _trim(self):
        """Bound the buffer: `window` frames, or the last `span` seconds (>= 1 frame)."""
        if self.span is None:
            while len(self.buf) > self.window:
                self.buf.popleft()
        else:
            t_new = self.buf[-1][1]
            while len(self.buf) > 1 and t_new - self.buf[0][1] > self.span:
                self.buf.popleft()

    def _ready(self):
        """Enough history to predict from.

        Under `span`, readiness canNOT be `covered >= span`: `_trim` has just guaranteed
        `covered <= span`, so that test only passes on exact float equality -- which offline
        timestamps (i/fps) happen to hit and wall-clock serve timestamps never do. That is why
        serve mode used to sit at "recognizing..." forever while offline emitted events from
        the identical recognizer. The buffer can only ever be one inter-frame gap short of the
        span (the frame `_trim` dropped was just beyond it), so a stream that has run for
        `span` seconds is ready, and the <= span guarantee is kept exactly.
        """
        if self.span is None:
            return len(self.buf) >= self.window
        if len(self.buf) < 2:
            return False
        covered = self.buf[-1][1] - self.buf[0][1]
        gap = covered / (len(self.buf) - 1)                 # mean inter-frame interval
        return covered >= self.span - gap or len(self.buf) >= self.window

    def _window_frames(self):
        """`window` frames: the buffer itself, or uniformly resampled from the span."""
        n = len(self.buf)
        if self.span is None or n == self.window:
            return [f for f, _ in self.buf]
        if n == 1:
            return [self.buf[0][0]] * self.window
        idx = np.linspace(0, n - 1, self.window).round().astype(int)
        return [self.buf[i][0] for i in idx]

    @torch.no_grad()
    def _infer(self, arr):
        """arr: (T,H,W,C) float32 -> probability vector over self.labels.

        Guarded here, not only in `push()`: the eval scripts call `_infer` directly, and
        without this every clip would build (and keep) an autograd graph.
        """
        x = torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0).contiguous().to(self.device)
        with torch.autocast("cuda", dtype=torch.float16):
            return self.net(x).softmax(1)[0].float().cpu().numpy()

    @torch.inference_mode()
    def push(self, frame_bgr, frame_index: int, timestamp: float):
        self.buf.append((self._preprocess(frame_bgr), timestamp))
        self._trim()
        if not self._ready():
            return None
        fire = self._filled_pushes % self.stride == 0  # first full window, then every stride
        self._filled_pushes += 1
        if not fire:
            return None
        prob = self._infer(np.stack(self._window_frames()))
        prob = reject_probs(prob, self.labels, self.reject_tau)
        order = prob.argsort()[::-1][:min(3, len(self.labels))]
        return RecognitionEvent(
            frame_index=frame_index, timestamp=timestamp,
            label=self.display(self.labels[int(order[0])]), confidence=float(prob[order[0]]),
            topk=[(self.display(self.labels[int(i)]), float(prob[i])) for i in order],
            model=self.kind, window=self.window)

    @staticmethod
    def display(label: str) -> str:
        return BEHAVIOR_DISPLAY.get(label, label)

    def reset(self) -> None:
        self.buf.clear()
        self._filled_pushes = 0

    def close(self) -> None:
        self.net = None
        self.buf.clear()
        torch.cuda.empty_cache()


class R3DRecognizer(_WindowRecognizer):
    kind = "r3d"
    labels = CLASSES

    def _build(self):
        from torchvision.models.video import r2plus1d_18
        net = r2plus1d_18(weights=None)
        net.fc = nn.Linear(net.fc.in_features, len(self.labels))
        net.load_state_dict(torch.load(self.ckpt, map_location="cpu")["model"])
        return net


class VideoMambaRecognizer(_WindowRecognizer):
    kind = "videomamba"
    labels = CLASSES

    def _build(self):
        from ..vendor.videomamba import videomamba_tiny
        net = videomamba_tiny(num_classes=len(self.labels), num_frames=self.window,
                              img_size=224)
        net.load_state_dict(torch.load(self.ckpt, map_location="cpu")["model"])
        return net


class BehaviorRecognizer(VideoMambaRecognizer):
    """Same VideoMamba-T/32f backbone, retrained head over BEHAVIORS (9 behaviors + other)."""
    kind = "behavior"
    labels = BEHAVIORS
