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


def frame_luma(frame_bgr):
    """Mean BT.601 luma (0-255) of one BGR uint8 frame -- the RAW-scene brightness the
    max_prior illuminance gate keys on (default 75, the field-measured pose/detection
    degradation knee, 2026-10-14). Retinexformer whitens its output -- every field suite's
    ENHANCED windows land at luma 66-180 regardless of how dark the scene was -- so a gate
    on the enhanced frame cannot see darkness at all; the gate only works on the frame
    BEFORE enhancement. Callers hand the pre-enhancement frame (post-downscale) in through
    push(raw_frame=...) and this measures it. Channel means first (cv2.mean -- 0.1 ms at
    840p, where numpy's axis reduction costs 10 ms), weighted sum second: identical result
    to weighting every pixel."""
    b, g, r = cv2.mean(frame_bgr)[:3]
    return float(0.114 * b + 0.587 * g + 0.299 * r)


def gate_luma(luma_window, lum_gate):
    """The illuminance gate, shared by BOTH decision modes (the 2026-10-14 "两部曲" runs it
    ahead of the alpha blend, max_prior as its step 1): returns the blocking RAW
    pre-enhancement window luma when the gate engages, else None -- gate off (`lum_gate`
    <= 0), no raw luma supplied for the window (fail open, the pre-gate behaviour), or the
    window is at/above the gate. Module-level so the offline eval stack (prior_eval.py,
    which invokes _decide_max_prior with a plain attribute bag as `self`) runs the exact
    same gate math without inheriting from this class."""
    if not lum_gate or lum_gate <= 0 or not luma_window:
        return None
    vals = [v for v in luma_window if v is not None]
    if not vals:
        return None
    lum = float(np.mean(vals))
    return None if lum >= lum_gate else lum


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


def prior_reject(p, label):
    """Step-3 veto: the preliminary result failed its necessary condition -> report `other`.

    The named entries are KEPT and `other` is raised just above the rejected winner, so the
    event's top-k still shows what almost fired ("喝水 0.90" under "其他 0.90") -- a rule
    rejection is not the model's confusion, and hiding the near-miss would make platform
    acceptance of the veto impossible to audit. `other` becoming the argmax is all the
    decision machinery needs (clips and the label bar key on the top-1 label)."""
    o = BEHAVIORS.index("other")
    i = BEHAVIORS.index(label)
    p = p.copy()
    p[o] = max(float(p[o]), float(p[i]) + 1e-6)
    return p


class BehaviorRecognizer(VideoMambaRecognizer):
    """Same VideoMamba-T/32f backbone, retrained head over BEHAVIORS (9 behaviors + other).

    v2: optionally fused with a COCO-17 skeleton branch (RTMO-s pose + ST-GCN, see
    skeleton_aux.py). Two decision modes:

      fuse_mode="alpha" (DEFAULT -- every field suite prefers it: max_prior loses on all
        four, val top1 0.70 vs 0.98 and platform replay 141-177 vs 10 false fires): the
        v2.0 weighted blend p = alpha*p_rgb + (1-alpha)*p_skel with the no-person policy.
        2026-10-14 "两部曲" (instruction.md revision): the illuminance gate below now runs
        ahead of the blend as well, so the shipped default (alpha + lum_gate=75) IS the
        meeting's two-step -- pure RGB in genuinely dark scenes (where pose availability
        collapses), weighted fusion wherever the skeleton branch is reliable.
      fuse_mode="max_prior" (the meeting "three-step", v2 of 2026-10-14, opt-in): step 1
        is the SAME illuminance gate -- the RAW pre-enhancement scene brightness
        (push(raw_frame=...), default 75 = the field-measured degradation knee): below
        `lum_gate` only the RGB branch runs (the skeleton and object detector are unstable
        there, so no second judgment either); above it,
        both branches classify, the skeleton side recognizing up to MAX_PERSONS skeletons
        individually (SkeletonAux.analyze). Step 2 picks the skeleton result: among
        qualifying persons, one whose top-1 is NOT other and beats `skel_tau`; if none,
        the skeleton side abstains and the RGB result stands. Higher-confidence branch
        wins as the preliminary result; a winner below `act_tau` reports other outright,
        before any prior. Step 3: preliminary results for pick_up / wave / drink /
        shake_hands face the object/geometry prior gate (object_prior.PRIOR_ACTIONS) and
        a failed condition reports `other` (prior_reject).

    Either way a failed/absent skeleton setup (missing rtmlib, missing onnxruntime CUDA,
    weights not shipped) degrades to pure RGB -- byte-for-byte the v1 behaviour -- so the
    branch can only ever degrade to what shipped before it, never break it.
    """
    kind = "behavior"
    labels = BEHAVIORS

    def __init__(self, ckpt_dir: str, stride: int | None = None,
                 span_sec: float | None = None, reject_tau: float = 0.0,
                 skel_fusion_alpha: float = 0.7, skel_no_person: str = "soft",
                 fuse_mode: str = "alpha", lum_gate: float = 75.0,
                 act_tau: float = 0.65, skel_tau: float = 0.65):
        super().__init__(ckpt_dir, stride=stride, span_sec=span_sec, reject_tau=reject_tau)
        self.skel_alpha = float(skel_fusion_alpha)
        self.skel_no_person = skel_no_person
        self.fuse_mode = fuse_mode if fuse_mode in ("alpha", "max_prior") else "alpha"
        # Illuminance gate (lum_gate, 0 disables it -- below the gate only the RGB branch
        # runs in EITHER decision mode, see gate_luma; default 75 is the RAW
        # pre-enhancement luma where field-wide pose/detection availability collapses, and
        # it keys on the raw_frame=... push() input -- enhanced luma is useless here),
        # plus the max_prior-only thresholds: the activation threshold (a lower-confidence
        # winner reports other before any prior runs) and the skeleton-side recognition
        # threshold (a person's named top-1 must beat it to be eligible as the skeleton
        # result). INDEPENDENT knobs per the meeting's wording; act_tau/skel_tau default
        # to the open-set reject_tau value.
        self.lum_gate = float(lum_gate)
        self.act_tau = float(act_tau)
        self.skel_tau = float(skel_tau)
        self.skel = None               # skeleton_aux.SkeletonAux once load() succeeds
        self.prior = None              # object_prior.PriorGate in max_prior mode
        self.prior_note = ""           # why the last firing window was vetoed/passed, if any
        self.rawbuf = deque()          # raw enhanced BGR frames, 1:1 with self.buf
        # RAW-scene (pre-enhancement) frame luma, 1:1 with rawbuf; None where the caller
        # passed no raw_frame (a caller that predates the gate -- the gate then skips, i.e.
        # fails open, keeping the old behaviour byte-for-byte).
        self.lumbuf = deque()
        self.t_parts["skel"] = 0.0     # skeleton branch cost per firing window (see t_parts)
        self.t_parts["prior"] = 0.0    # object-detection prior gate cost

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
        if self.fuse_mode == "max_prior":
            from .object_prior import PriorGate
            self.prior = PriorGate(os.path.dirname(self.ckpt), device)
            reason = self.prior.load()      # never fatal: object priors fail open, wave
            if reason:                      # (geometry) keeps working without the detector
                print(f"[recognize] [warn] {reason}")
            else:
                print(f"[recognize] 物体先验已启用 (yolo26m 检测 {self.prior.det.det_ms:.1f} ms/帧)")
            from .skeleton_aux import (COMP_FRAC, COMP_MIN_VIS, MAX_PERSONS, TAU)
            gate = (f"照度门 {self.lum_gate:.0f}（量提亮前原图亮度，低于则仅用RGB）"
                    if self.lum_gate > 0 else "照度门关")
            print(f"[recognize] 决策模式=max_prior：{gate}，双分支分类取高分，骨架窗内最多 "
                  f"{MAX_PERSONS} 人各自识别(质量≥{TAU:g} 且 ≥{COMP_MIN_VIS}/17 点完整率"
                  f"≥{COMP_FRAC:g})，非other且>{self.skel_tau:g} 者为骨架结果，"
                  f"胜者置信度<{self.act_tau:g} 直接other；捡东西/挥手/喝水/握手过先验"
                  f"二次判断(不满足→other)")
            return
        from .skeleton_aux import NO_PERSON_ZH, TAU
        gate = (f"，照度门 {self.lum_gate:.0f}（量提亮前原图亮度，低于门限的窗仅用RGB"
                f"、不做骨架融合，两部曲）" if self.lum_gate > 0 else "，照度门关")
        print(f"[recognize] 骨架分支已启用 RTMO-s+ST-GCN 融合 α={self.skel_alpha:g} "
              f"(无人窗策略={NO_PERSON_ZH.get(self.skel_no_person, self.skel_no_person)}"
              f"，TAU={TAU:g}，姿态 {aux.pose_ms:.1f} ms/帧){gate}")

    def _trim(self):
        n0 = len(self.buf)
        super()._trim()
        while len(self.rawbuf) > len(self.buf):     # keep the raw ring 1:1 with buf
            self.rawbuf.popleft()
            self.lumbuf.popleft()                   # and the luma ring 1:1 with the raw one

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

    def _luma_window(self):
        """The same index selection applied to the RAW-scene luma ring (1:1 with the raw
        frame ring, so the indices are identical). A None entry means the caller did not
        supply a pre-enhancement frame for that push; the gate then skips (fails open)."""
        n = len(self.lumbuf)
        if n != len(self.rawbuf):
            return [None] * self.window
        if self.span is None or n == self.window:
            return list(self.lumbuf)
        if n == 1:
            return [self.lumbuf[0]] * self.window
        idx = np.linspace(0, n - 1, self.window).round().astype(int)
        return [self.lumbuf[i] for i in idx]

    def _decide_max_prior(self, p_rgb, info, raw_window, luma_window):
        """The meeting three-step v2 (2026-10-14 revision). Returns (reported distribution,
        note).

        Step 1  illuminance gate: below `lum_gate` only the RGB branch runs -- no skeleton,
                no object detector, no priors. `lum_gate <= 0` disables the gate. The gate
                measures the RAW pre-enhancement scene luma (`luma_window`, per-frame values
                from frame_luma on the caller's raw_frame) -- enhanced luma is whitened by
                Retinexformer and cannot see darkness. Windows with no supplied raw luma
                skip the gate (fail open).
        Step 2  skeleton side: among `valid` persons, eligible = top-1 is not other AND
                beats `skel_tau`; the strongest eligible person is THE skeleton result,
                otherwise the skeleton side abstains (an "other" skeleton must not out-vote
                the RGB branch by confidence -- abstention, not a competing vote). The
                higher-confidence branch's top-1 is the preliminary result; below
                `act_tau` it reports other outright, before any prior runs.
        Step 3  PRIOR_ACTIONS preliminary results face the prior gate; a failed necessary
                condition reports other (prior_reject keeps the near-miss in the top-k).
        """
        o = self.labels.index("other")
        lum = gate_luma(luma_window, self.lum_gate)
        if lum is not None:
            return p_rgb, (f"照度 {lum:.0f}（提亮前原图）< 门限 "
                           f"{self.lum_gate:.0f}，仅用RGB结果")
        skel_p = None
        if info["p"] is not None:
            cand = [m for m in np.flatnonzero(info["valid"])
                    if int(info["slot_p"][m].argmax()) != o
                    and float(info["slot_p"][m].max()) > self.skel_tau]
            if cand:
                m = cand[int(np.argmax(info["slot_p"][cand].max(1)))]
                skel_p = info["slot_p"][m]
        if skel_p is None or float(p_rgb.max()) >= float(skel_p.max()):
            init, init_label = p_rgb, self.labels[int(p_rgb.argmax())]
        else:
            init, init_label = skel_p, self.labels[int(skel_p.argmax())]
        if float(init.max()) < self.act_tau:
            out = np.zeros_like(init)
            out[o] = 1.0
            return out, f"置信度 {float(init.max()):.2f} < 启动阈值 {self.act_tau:g}，直接判为other"
        from .object_prior import PRIOR_ACTIONS
        if init_label in PRIOR_ACTIONS and self.prior is not None:
            t0 = time.perf_counter()
            ok, reason = self.prior.check(init_label, info["sk"], info["valid"], raw_window)
            self.t_parts["prior"] += time.perf_counter() - t0
            if not ok:
                return prior_reject(init, init_label), f"先验否决 {init_label}: {reason}"
            return init, f"先验通过 {init_label}: {reason}"
        return init, ""

    @torch.inference_mode()
    def push(self, frame_bgr, frame_index: int, timestamp: float, raw_frame=None):
        """`frame_bgr` is the enhanced frame the model sees; `raw_frame` is the same moment's
        PRE-enhancement frame (post-downscale), whose luma the illuminance gate measures
        (both decision modes). Optional for backward compatibility -- without it the gate
        just never engages (fail open), exactly the pre-gate behaviour."""
        if self.skel is not None:
            self.rawbuf.append(frame_bgr)           # enhanced frame as received, BGR uint8 --
            # the pose model's domain; and the RAW-scene brightness, if the caller kept the
            # pre-enhancement frame around
            self.lumbuf.append(frame_luma(raw_frame) if raw_frame is not None else None)
        self.buf.append((self._preprocess(frame_bgr), timestamp))
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
        self.prior_note = ""
        if self.skel is not None and len(self.rawbuf) == len(self.buf):
            raw = self._raw_window()
            lums = self._luma_window()
            lum = gate_luma(lums, self.lum_gate)
            if lum is not None:
                # 两部曲 (2026-10-14 instruction.md revision): the gate runs ahead of BOTH
                # decision modes -- below it the skeleton branch does not even load, pure
                # RGB, the same below-gate behaviour max_prior's step 1 has. Skipping the
                # pose run also gives dark scenes its ~9 ms/frame back.
                self.prior_note = (f"照度 {lum:.0f}（提亮前原图）< 门限 "
                                   f"{self.lum_gate:.0f}，仅用RGB结果")
            else:
                t0 = time.perf_counter()
                if self.fuse_mode == "max_prior":
                    info = self.skel.analyze(raw)
                    self.t_parts["skel"] += time.perf_counter() - t0
                    prob, self.prior_note = self._decide_max_prior(prob, info, raw, lums)
                else:
                    ps, pres = self.skel.probs(raw)
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
        self.lumbuf.clear()

    def warmup(self) -> None:
        super().warmup()
        if self.skel is not None:
            # RTMO's first call pays onnxruntime session init; the ST-GCN's first CUDA forward
            # pays cuBLAS/cuDNN setup. Both belong at startup with the RGB warmup, not in the
            # first firing window of a live stream.
            self.skel.warmup()
        if self.prior is not None:
            # The YOLO onnxruntime session init was already priced by PriorGate.load()'s
            # probe; this absorbs the first REAL-size letterbox+inference of the gate.
            self.prior.warmup()

    def close(self) -> None:
        self.skel = None                             # drops the onnxruntime session too
        self.prior = None
        self.rawbuf.clear()
        self.lumbuf.clear()
        super().close()
