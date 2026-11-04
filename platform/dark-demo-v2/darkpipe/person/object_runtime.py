"""On-demand seven-class object evidence on five exact tracked enhanced frames.

Frames are shared in the frontend history, not duplicated for every person's 32 RGB
samples. Only five references travel with each task. Model errors propagate; missing
or ambiguous image evidence abstains, as in the calibrated policy.
"""
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from darkpipe.constants import BEHAVIORS
from darkpipe.stages.object_prior import YoloOnnxDetector
from .core import iou
from .object_prior import allows_object_prior, filter_object_evidence

OBJECT_MODEL = 'yolo26m_objects_20261026.onnx'
OBJECT_CLASSES = ('bottle', 'cup', 'phone', 'camera', 'paper', 'portable_object', 'robot_or_exhibit')
SAMPLE_POSITIONS = (0, 8, 16, 23, 31)


@dataclass(frozen=True)
class EvidenceFrame:
    index: int
    pixels: np.ndarray
    source_shape: tuple


class FrameHistory:
    """Time-, count- and byte-bounded frame cache, separate from per-person RGB crops."""
    def __init__(self, max_frames: int = 120, max_bytes: int = 128*1024*1024):
        self.frames = OrderedDict()
        self.bytes = 0
        self.max_frames, self.max_bytes = max_frames, max_bytes

    def clear(self) -> None:
        self.frames.clear()
        self.bytes = 0

    def put(self, index: int, stamp: float, pixels: np.ndarray | None, source_shape: tuple) -> None:
        if pixels is not None:
            if index in self.frames:
                raise ValueError('Duplicate evidence frame index')
            self.frames[index] = (stamp, EvidenceFrame(index, pixels, tuple(source_shape)))
            self.bytes += pixels.nbytes
        while self.frames and (stamp-next(iter(self.frames.values()))[0] > 1.2 or
                len(self.frames) > self.max_frames or self.bytes > self.max_bytes):
            _, (_, frame) = self.frames.popitem(last=False)
            self.bytes -= frame.pixels.nbytes

    def select(self, window: tuple | list) -> tuple:
        indices = dict.fromkeys(window[i].frame_index for i in SAMPLE_POSITIONS)
        return tuple(self.frames[i][1] for i in indices if i in self.frames)


def regions(box: np.ndarray, pose: np.ndarray | None, shape: tuple, action: str) -> list:
    """Frozen calibration geometry, evaluated in original-source coordinates."""
    margin = (box[2:]-box[:2])*.3
    rect = np.r_[np.floor(np.maximum(0, box[:2]-margin)),
                 np.ceil(np.minimum([shape[1], shape[0]], box[2:]+margin))].astype(int)
    output = [rect]
    if pose is not None:
        indices = [0, 1, 2, 3, 4, 9, 10] if action == 'drink' else [9, 10, 13, 14]
        points = pose[indices]
        points = points[points[:, 2] >= .5, :2]
        if len(points) >= 2:
            padding = max(24., (box[3]-box[1])*.12)
            lo = np.floor(np.maximum(rect[:2], points.min(0)-padding)).astype(int)
            hi = np.ceil(np.minimum(rect[2:], points.max(0)+padding)).astype(int)
            if np.all(hi-lo >= 24) and np.prod(hi-lo) < np.prod(rect[2:]-rect[:2])*.8:
                output.append(np.r_[lo, hi])
    return output


class PersonObjectPrior:
    def __init__(self, ckpt_dir: str, device: str, threshold: float, lum_gate: float):
        import onnxruntime as ort
        self.threshold, self.lum_gate = threshold, lum_gate
        self.stats = Counter()
        self.cache = OrderedDict()  # small detection arrays only, never image tensors
        self.detector = YoloOnnxDetector(ckpt_dir, device, min_conf=0.)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.detector.sess = ort.InferenceSession(str(Path(ckpt_dir)/OBJECT_MODEL),
            sess_options=options,
            providers=[('CUDAExecutionProvider', {'device_id': int(device.split(':')[-1])})])
        if self.detector.sess.get_providers()[0] != 'CUDAExecutionProvider':
            raise RuntimeError('Object detector unexpectedly fell back to CPU')
        # A configured required model cannot silently disappear or disable itself under load.
        self.detector._frames_to_dets([np.full((320, 240, 3), 114, np.uint8)]*2)

    def _view(self, key: tuple, image: np.ndarray) -> list:
        if key in self.cache:
            self.cache.move_to_end(key)
            self.stats['reused_crops'] += 1
            return self.cache[key]
        # Direct call deliberately propagates ORT/CUDA failures (legacy __call__ fails open).
        detections = self.detector._frames_to_dets([image])[0]
        detections = detections[detections[:, 4] >= self.threshold].tolist()
        self.cache[key] = detections
        while len(self.cache) > 512:
            self.cache.popitem(last=False)
        self.stats['inferred_crops'] += 1
        return detections

    def apply(self, task, row: dict) -> None:
        label = row['label']
        samples = []
        if label in ('drink', 'pick_up') and allows_object_prior(row['raw_luma'], self.lum_gate):
            lookup = {s.frame_index: s for s in task.window}
            for frame in task.prior_frames:
                sample = lookup[frame.index]
                entry = dict(frame=frame.index, timestamp=sample.timestamp,
                    source_shape=list(frame.source_shape[:2]), raw_luma=sample.luma,
                    binding_valid=sample.binding_valid, available=False,
                    person_roi=None, detections=[])
                samples.append(entry)
                if (not sample.binding_valid or sample.person_box is None or
                        not allows_object_prior(sample.luma, self.lum_gate)):
                    continue
                h, w = frame.pixels.shape[:2]
                sh, sw = frame.source_shape[:2]
                ratio = np.array([w/sw, h/sh])
                pose = sample.pose.copy().astype(float)
                pose[:, :2] *= [sw, sh]
                rects = regions(sample.person_box.astype(float)/np.tile(ratio, 2),
                                pose, frame.source_shape, label)
                entry['person_roi'] = rects[0].tolist()
                entry['regions'] = [r.tolist() for r in rects]
                boxes = []
                for rect in rects:
                    lo = np.maximum(0, np.floor(rect[:2]*ratio)).astype(int)
                    hi = np.minimum([w, h], np.ceil(rect[2:]*ratio)).astype(int)
                    if np.any(hi <= lo):
                        continue
                    # Generation/session separation prevents stale evidence after reconnect/ID reuse.
                    key = (*task.key, frame.index, *lo.tolist(), *hi.tolist())
                    dets = self._view(key, frame.pixels[lo[1]:hi[1], lo[0]:hi[0]])
                    entry['available'] = True
                    for det in dets:
                        box = np.asarray(det, float).copy()
                        box[:4] = (box[:4]+np.tile(lo, 2))/np.tile(ratio, 2)
                        boxes.append(box.tolist())
                for box in sorted(boxes, key=lambda b: -b[4]):
                    if not any(int(box[5]) == int(b[5]) and
                               iou(np.asarray(box[:4]), np.asarray(b[:4])) > .5
                               for b in entry['detections']):
                        entry['detections'].append(box)
        result = filter_object_evidence(label, samples, row['raw_luma'], self.threshold, self.lum_gate)
        result.update(model=OBJECT_MODEL, samples=samples,
                      coordinate_space='source_pixels', sample_positions=list(SAMPLE_POSITIONS))
        row['object_prior'] = result
        row['action_label'] = label
        row['action_confidence'] = row['confidence']
        self.stats[result['reason']] += 1
        if result['label'] != label:
            row['action_decision_scores'] = row['decision_scores'][:]
            # Deterministic veto scores, not a newly estimated model posterior.
            row['decision_scores'] = [float(c == 'other') for c in BEHAVIORS]
            row['label'], row['confidence'] = 'other', 1.
            self.stats['filtered_windows'] += 1

    def close(self) -> None:
        self.detector.sess = None
        self.cache.clear()
