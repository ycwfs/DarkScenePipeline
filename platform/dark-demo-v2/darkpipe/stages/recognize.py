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
import time
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn

from ..constants import BEHAVIOR_DISPLAY, BEHAVIORS, CKPT_FILES, CLASSES, RECO_CFG
from ..utils import free_device_cache
from .base import RecognitionEvent, Recognizer


def preprocess_frame(frame_bgr, cfg):
    """BGR uint8 -> normalized HWC float32, exactly as the recognizers see it at serve time.

    Module-level so the offline cache/eval/training scripts import the SAME function and the
    training domain cannot silently drift from the serving domain.

    cfg["fov"] = (H, W) switches to the whole-frame pipeline the realact11_fov lineage was
    trained with (aspect-preserving resize to 400x224, no crop): the crop chain feeds the net
    the central 49%x88% and leaves the frame edges blind, which is exactly what the full-frame
    checkpoints were retrained to stop doing. INTER_AREA unconditionally, matching the
    training resize (realact_train.ClipCache / fusion_eval.Models._fov).
    """
    fov = cfg.get("fov")
    if fov:
        r = cv2.resize(frame_bgr, (fov[1], fov[0]), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return (rgb - cfg["mean"]) / cfg["std"]
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
        # Cumulative seconds inside the three parts of a firing window, so a deployment can say
        # WHERE the inference time goes instead of only how much of it there is: building the
        # window (an np.stack of 32x224x224x3 float32 = 18.4 MB), marshalling it onto the card,
        # and the forward itself. Exists because "recognition costs 312 ms" does not say whether
        # to spend effort on the model or on the data path -- and on a card that recognition has
        # to itself, host-side copies are the difference between 39% and 25% occupancy.
        # Written only by the thread that calls push(); readers diff against their own previous
        # snapshot, which is the same lock-free arrangement as server._Stat.
        self.t_parts = {"window": 0.0, "marshal": 0.0, "forward": 0.0, "n": 0}
        self.fov = None  # (H, W) whole-frame input, set from the checkpoint metadata in load()
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

        `.to(device)` comes BEFORE the permute, which is the same order the enhance path
        already uses (utils.bgr_batch_to_tensor). The other way round -- permute, contiguous,
        then transfer -- runs the transpose on the CPU, and for a 32x224x224x3 float32 window
        that is an 18.4 MB strided copy whose innermost source step is 12 bytes, so it touches
        roughly three times the bytes it writes and does it single-threaded. Transferring first
        makes the H2D copy one contiguous block and leaves the transpose to the GPU. It also
        keeps the recognition thread off the CPU that output_loop needs for JPEG encoding --
        the two contend (see the encode 11 -> 22 ms note in server.process_loop).

        The split billed below is host-side wall clock, so `forward` also absorbs whatever the
        marshal deferred: the GPU-side transpose is async and `.cpu()` is the only sync point.
        That is the honest attribution -- `marshal` is what this thread spent, not what the copy
        cost end to end.
        """
        t0 = time.perf_counter()
        x = torch.from_numpy(arr).to(self.device).permute(3, 0, 1, 2).unsqueeze(0).contiguous()
        t1 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16):
            prob = self.net(x).softmax(1)[0].float().cpu().numpy()
        t2 = time.perf_counter()
        self.t_parts["marshal"] += t1 - t0
        self.t_parts["forward"] += t2 - t1
        self.t_parts["n"] += 1
        return prob

    def timings(self):
        """Snapshot of the cumulative per-part seconds (see self.t_parts). Diff two of these."""
        return dict(self.t_parts)

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
        t0 = time.perf_counter()
        arr = np.stack(self._window_frames())
        self.t_parts["window"] += time.perf_counter() - t0
        prob = self._infer(arr)
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

    def warmup(self) -> None:
        """First inference measured 4611 ms in a deployment (cuDNN autotune + lazy CUDA module
        load + fp16 autocast setup), against a 235 ms steady state. Serve mode pays that with
        the stream frozen: the first telemetry window showed fps=1.1, latency=4739ms,
        wait_gpu=94%. A dummy window here moves the cost to startup, where nothing is watching.

        Deliberately touches neither self.buf nor _filled_pushes, so the stride phase a real
        stream sees is exactly as if warmup had never run.

        Runs under a device guard because this is called on the LOADING thread while the real
        inferences happen on the recognition worker: cuDNN autotune results and cuBLAS
        workspaces are cached per device, so warming them up on the process default card would
        leave the first real window on the recognition card to pay the cost anyway.
        """
        h, w = self.cfg.get("fov") or (self.cfg["size"], self.cfg["size"])
        arr = np.zeros((self.window, h, w, 3), np.float32)
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            with torch.cuda.device(self.device):
                self._infer(arr)
        else:
            self._infer(arr)

    def close(self) -> None:
        self.net = None
        self.buf.clear()
        free_device_cache(self.device)


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

    def _apply_preprocess_meta(self, meta):
        """Select the input pipeline from the checkpoint's own metadata.

        realact_train.py stamps every checkpoint with "preprocess": either
        "fullfov<H>x<W>" (whole-frame 400x224 lineage) or "resize256_centercrop224". Older
        checkpoints carry no key at all -- they are crop-chain, the historical default. The
        point of reading it here rather than adding a config knob: the weight and its input
        pipeline are one fact, and a deployment that could set them independently could set
        them inconsistently (a crop-fed fov model silently loses most of its accuracy).
        """
        meta = str(meta)
        if meta.startswith("fullfov"):
            h, w = (int(x) for x in meta[len("fullfov"):].split("x", 1))
            self.fov = (h, w)
            self.cfg = dict(self.cfg, fov=(h, w))

    def load(self, device: str) -> None:
        self.device = device
        meta = torch.load(self.ckpt, map_location="cpu")
        self._apply_preprocess_meta(meta.get("preprocess", ""))
        self.net = self._build(meta["model"]).to(device).eval()

    def _build(self, sd):
        from ..vendor.videomamba import videomamba_tiny
        net = videomamba_tiny(num_classes=len(self.labels), num_frames=self.window,
                              img_size=self.fov or 224)
        net.load_state_dict(sd)
        return net


class BehaviorRecognizer(VideoMambaRecognizer):
    """Same VideoMamba-T/32f backbone, retrained head over BEHAVIORS (9 behaviors + other).

    v2: optionally fused with a COCO-17 skeleton branch (RTMO-s pose + ST-GCN, see
    skeleton_aux.py). Every firing window additionally gets the skeleton branch's 10-way
    probability; the reported distribution is `alpha * p_rgb + (1 - alpha) * p_skel`. A
    window in which the pose model sees no person takes the `no_person` policy (see
    skeleton_aux.NO_PERSON_MODES): force `other`, let the absent skeleton vote `other` with
    weight (1-alpha), or let the skeleton abstain -- the brake exists because on empty scenes
    the RGB branch occasionally reports a behaviour at 0.85+ while the skeleton branch,
    having no person, cannot; the softer settings exist because the pose model can also miss
    a REAL person (extreme dark, occlusion), and a hard `other` would kill those windows no
    matter how confident the RGB branch is. `skel_fusion_alpha >= 0.999` (or a failed/absent
    skeleton setup -- missing rtmlib, missing onnxruntime CUDA, weights not shipped) falls
    back to pure RGB, the historical behaviour, so the branch can only ever degrade to what
    shipped before it, never break it.
    """
    kind = "behavior"
    labels = BEHAVIORS

    def __init__(self, ckpt_dir: str, stride: int | None = None,
                 span_sec: float | None = None, reject_tau: float = 0.0,
                 skel_fusion_alpha: float = 0.7, skel_no_person: str = "soft"):
        super().__init__(ckpt_dir, stride=stride, span_sec=span_sec, reject_tau=reject_tau)
        self.skel_alpha = float(skel_fusion_alpha)
        self.skel_no_person = skel_no_person
        self.skel = None               # skeleton_aux.SkeletonAux once load() succeeds
        self.rawbuf = deque()          # raw enhanced BGR frames, 1:1 with self.buf
        self.t_parts["skel"] = 0.0     # skeleton branch cost per firing window (see t_parts)

    def load(self, device: str) -> None:
        super().load(device)
        if self.skel_alpha >= 0.999:
            print("[recognize] skel_fusion_alpha>=0.999，骨架分支关闭，纯 RGB 识别")
            return
        from .skeleton_aux import SkeletonAux
        aux = SkeletonAux(os.path.dirname(self.ckpt), device, self.skel_alpha,
                          no_person=self.skel_no_person)
        reason = aux.load()
        if reason:
            # Fail-soft: the skeleton branch is an accuracy add-on, not a load-bearing one.
            # A site without rtmlib/onnxruntime must still run, exactly as it did before v2.
            print(f"[recognize] [warn] {reason}")
            return
        self.skel = aux
        from .skeleton_aux import NO_PERSON_ZH, TAU
        print(f"[recognize] 骨架分支已启用 RTMO-s+ST-GCN 融合 α={self.skel_alpha:g} "
              f"(无人窗策略={NO_PERSON_ZH.get(self.skel_no_person, self.skel_no_person)}"
              f"，TAU={TAU:g}，姿态 {aux.pose_ms:.1f} ms/帧)")

    def _trim(self):
        n0 = len(self.buf)
        super()._trim()
        while len(self.rawbuf) > len(self.buf):     # keep the raw ring 1:1 with buf
            self.rawbuf.popleft()

    def _raw_window(self):
        """The `_window_frames` policy applied to the raw BGR ring (same indices either way:
        both rings are appended and trimmed in lockstep, so they always have equal length)."""
        n = len(self.rawbuf)
        if self.span is None or n == self.window:
            return list(self.rawbuf)
        if n == 1:
            return [self.rawbuf[0]] * self.window
        idx = np.linspace(0, n - 1, self.window).round().astype(int)
        return [self.rawbuf[i] for i in idx]

    @torch.inference_mode()
    def push(self, frame_bgr, frame_index: int, timestamp: float):
        if self.skel is not None:
            self.rawbuf.append(frame_bgr)           # enhanced frame as received, BGR uint8 --
        self.buf.append((self._preprocess(frame_bgr), timestamp))   # the pose model's domain
        self._trim()
        if not self._ready():
            return None
        fire = self._filled_pushes % self.stride == 0
        self._filled_pushes += 1
        if not fire:
            return None
        t0 = time.perf_counter()
        arr = np.stack(self._window_frames())
        self.t_parts["window"] += time.perf_counter() - t0
        prob = self._infer(arr)
        if self.skel is not None and len(self.rawbuf) == len(self.buf):
            t0 = time.perf_counter()
            ps, pres = self.skel.probs(self._raw_window())
            self.t_parts["skel"] += time.perf_counter() - t0
            prob = self.skel.fuse(prob, ps, pres)
        prob = reject_probs(prob, self.labels, self.reject_tau)
        order = prob.argsort()[::-1][:min(3, len(self.labels))]
        return RecognitionEvent(
            frame_index=frame_index, timestamp=timestamp,
            label=self.display(self.labels[int(order[0])]), confidence=float(prob[order[0]]),
            topk=[(self.display(self.labels[int(i)]), float(prob[i])) for i in order],
            model=self.kind, window=self.window)

    def reset(self) -> None:
        super().reset()
        self.rawbuf.clear()

    def warmup(self) -> None:
        super().warmup()
        if self.skel is not None:
            # RTMO's first call pays onnxruntime session init; the ST-GCN's first CUDA forward
            # pays cuBLAS/cuDNN setup. Both belong at startup with the RGB warmup, not in the
            # first firing window of a live stream.
            self.skel.warmup()

    def close(self) -> None:
        self.skel = None                             # drops the onnxruntime session too
        self.rawbuf.clear()
        super().close()
