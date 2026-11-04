"""Per-person temporal inputs, association, fusion and visualization."""
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import sys

import cv2
import numpy as np
from .assignment import linear_sum_assignment

from darkpipe.constants import BEHAVIORS, BEHAVIOR_ZH
from darkpipe.pose_overlay import EDGES
from darkpipe.render import _cjk_font
from darkpipe.stages.recognize import gate_luma, reject_probs

DEFAULT_ACTION_THRESHOLD = 0.925


def iou(a: np.ndarray, b: np.ndarray) -> float:
    overlap = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    area = np.maximum(0, a[2:] - a[:2]).prod() + np.maximum(0, b[2:] - b[:2]).prod()
    return float(overlap / max(area - overlap, 1e-6))


def crop_person(frame: np.ndarray, box: np.ndarray, shape=(224, 400), margin=.15) -> tuple:
    """Pad each tracked person's own ROI, retaining aspect ratio and all visible limbs."""
    h, w = frame.shape[:2]
    xy1, xy2 = box[:2], box[2:]
    size = xy2 - xy1
    lo = np.maximum(0, np.floor(xy1 - size * margin)).astype(int)
    hi = np.minimum([w, h], np.ceil(xy2 + size * margin)).astype(int)
    if np.any(hi <= lo):
        raise ValueError(f'Empty person ROI: {box}')
    roi = frame[lo[1]:hi[1], lo[0]:hi[0]]
    th, tw = shape
    scale = min(tw / roi.shape[1], th / roi.shape[0])
    rw, rh = max(1, round(roi.shape[1] * scale)), max(1, round(roi.shape[0] * scale))
    resized = cv2.resize(roi, (rw, rh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.full((th, tw, 3), 114, np.uint8)
    left, top = (tw - rw) // 2, (th - rh) // 2
    canvas[top:top+rh, left:left+rw] = resized
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), np.concatenate([lo, hi])


def appearance(frame: np.ndarray, box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(int)
    crop = frame[y1+(y2-y1)//5:y1+4*(y2-y1)//5, x1:x2]
    if not crop.size:
        return np.zeros(96, np.float32)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256]).ravel()
    return np.sqrt(hist / max(float(hist.sum()), 1))


def person_boxes(detections: np.ndarray, confidence=.35) -> np.ndarray:
    rows = detections[(detections[:, 5] == 0) & (detections[:, 4] >= confidence)]
    rows = rows[((rows[:, 2:4] - rows[:, :2]) > [8, 20]).all(1)]
    if not len(rows):
        return rows
    xywh = rows[:, :4].copy()
    xywh[:, 2:] -= xywh[:, :2]
    keep = cv2.dnn.NMSBoxes(xywh.tolist(), rows[:, 4].tolist(), 0., .75)
    return rows[np.asarray(keep, dtype=int).reshape(-1)]


def associate_poses(boxes: np.ndarray, points: np.ndarray, scores: np.ndarray) -> list:
    """One-to-one body/pose assignment, independent of detector output order."""
    result = [None] * len(boxes)
    if not len(boxes) or not len(points):
        return result
    costs = np.full((len(boxes), len(points)), 1e3, np.float32)
    for i, box in enumerate(boxes):
        for j, (xy, cf) in enumerate(zip(points, scores)):
            good = cf > .3
            inside = ((xy >= box[:2]) & (xy <= box[2:4])).all(1) & good
            fraction = inside.sum() / max(good.sum(), 1)
            if good.sum() < 5 or fraction < .65:
                continue
            center = xy[good].mean(0)
            delta = np.linalg.norm((center - (box[:2]+box[2:4])/2) / np.maximum(box[2:4]-box[:2], 1))
            costs[i, j] = 1-fraction + .2*delta
    rr, cc = linear_sum_assignment(costs)
    for i, j in zip(rr, cc):
        if costs[i, j] < .65:
            result[i] = np.column_stack([points[j], scores[j]]).astype(np.float32)
    return result


@dataclass
class Sample:
    timestamp: float
    frame_index: int
    crop: np.ndarray
    crop_box: np.ndarray
    pose: np.ndarray
    luma: float | None
    person_box: np.ndarray | None = None
    binding_valid: bool = False
    frame_shape: tuple | None = None


@dataclass
class Track:
    tid: int
    box: np.ndarray
    hist: np.ndarray
    last_seen: float
    score: float
    pose: np.ndarray | None = None
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4, np.float32))
    history: deque = field(default_factory=lambda: deque(maxlen=120))
    prediction: dict | None = None
    last_inference: float = -1e9

    def append(self, frame: np.ndarray, timestamp: float, index: int, luma: float | None,
               binding_valid: bool = False) -> None:
        crop, crop_box = crop_person(frame, self.box)
        pose = np.zeros((17, 3), np.float32) if self.pose is None else self.pose.copy()
        pose[:, :2] /= np.array(frame.shape[1::-1])
        self.history.append(Sample(timestamp, index, crop, crop_box, pose, luma,
                                   self.box.copy(), binding_valid, frame.shape[:2]))
        while len(self.history) > 1 and timestamp - self.history[0].timestamp > 1.1:
            self.history.popleft()

    def window(self, fps: float) -> list | None:
        if len(self.history) < 8:
            return None
        history = list(self.history)
        if history[-1].timestamp - history[0].timestamp < 1. - 1.5/fps:
            return None
        timestamps = np.array([s.timestamp for s in history])
        targets = np.linspace(max(timestamps[0], timestamps[-1]-1.), timestamps[-1], 32)
        indices = np.abs(timestamps[:, None] - targets).argmin(0)
        if np.max(np.abs(timestamps[indices]-targets)) > .12:
            return None  # do not bridge a long detection gap with repeated crops
        return [history[i] for i in indices]


class PersonTracker:
    """Short-term location/appearance/pose association; not identity recognition."""
    def __init__(self, max_gap=.4):
        self.tracks: list[Track] = []
        self.next_id = 1
        self.max_gap = max_gap

    def update(self, boxes: np.ndarray, poses: list, frame: np.ndarray, timestamp: float) -> list[Track]:
        self.tracks = [t for t in self.tracks if timestamp-t.last_seen <= self.max_gap]
        histograms = [appearance(frame, box[:4]) for box in boxes]
        costs = np.full((len(self.tracks), len(boxes)), 1e3, np.float32)
        for i, track in enumerate(self.tracks):
            predicted = track.box + track.velocity * min(timestamp-track.last_seen, .15)
            size = max(float(np.linalg.norm(predicted[2:]-predicted[:2])), 20)
            for j, box in enumerate(boxes):
                overlap = iou(predicted, box[:4])
                distance = float(np.linalg.norm((predicted[:2]+predicted[2:]-box[:2]-box[2:4])/2)/size)
                if overlap < .05 and distance > .4:
                    continue
                body = max(0., 1.-float(track.hist @ histograms[j]))
                if body > .65:
                    continue
                cost = .55*(1-overlap)+.25*min(distance, 1)+.20*body
                if track.pose is not None and poses[j] is not None:
                    good = (track.pose[:, 2] > .4) & (poses[j][:, 2] > .4)
                    if good.sum() >= 5:
                        delta = np.linalg.norm(track.pose[good, :2]-poses[j][good, :2], axis=1).mean()/size
                        cost += .08*min(float(delta), 1.)
                costs[i, j] = cost
        matches = {}
        if costs.size:
            rr, cc = linear_sum_assignment(costs)
            matches = {int(j): self.tracks[i] for i, j in zip(rr, cc) if costs[i, j] < .7}
        current = []
        for j, box in enumerate(boxes):
            track = matches.get(j)
            if track is None:
                track = Track(self.next_id, box[:4].copy(), histograms[j], timestamp, float(box[4]))
                self.next_id += 1
                self.tracks.append(track)
            else:
                gap = timestamp-track.last_seen
                if gap > .2:
                    track.history.clear()
                    track.prediction = None
                    track.last_inference = -1e9
                velocity = (box[:4]-track.box)/max(gap, 1e-3)
                track.velocity = .5*track.velocity+.5*velocity
                track.box = box[:4].copy()
                track.hist = histograms[j]
                track.last_seen = timestamp
                track.score = float(box[4])
            track.pose = poses[j]
            current.append(track)
        return current  # never draw a disappeared person's stale box or action


def fuse_person(rgb: np.ndarray, skeleton: np.ndarray | None, lumas: list,
                alpha=0.972, threshold=DEFAULT_ACTION_THRESHOLD, lum_gate=35., no_pose='soft') -> dict:
    blocked = gate_luma(lumas, lum_gate)
    if blocked is not None:
        fused, mode = rgb.copy(), 'rgb_low_luma'
    elif alpha >= .999 or (skeleton is None and no_pose == 'rgb'):
        fused, mode = rgb.copy(), 'rgb_no_pose'
    elif skeleton is None and no_pose == 'other':
        fused, mode = np.eye(len(rgb), dtype=rgb.dtype)[BEHAVIORS.index('other')], 'other_no_pose'
    elif skeleton is not None:
        fused, mode = alpha*rgb+(1-alpha)*skeleton, 'rgb_skeleton'
    else:
        fallback = np.zeros_like(rgb)
        fallback[BEHAVIORS.index('other')] = 1
        fused, mode = alpha*rgb+(1-alpha)*fallback, 'rgb_soft_no_pose'
    decision = reject_probs(fused, BEHAVIORS, threshold)
    index = int(decision.argmax())
    return dict(label=BEHAVIORS[index], confidence=float(decision[index]), mode=mode,
                threshold=float(threshold),
                rgb_probs=rgb.tolist(), skeleton_probs=None if skeleton is None else skeleton.tolist(),
                fused_probs=fused.tolist(), decision_scores=decision.tolist(),
                raw_luma=None if not any(v is not None for v in lumas) else float(np.mean([v for v in lumas if v is not None])))


@lru_cache(maxsize=512)
def text_patch(text: str, color: tuple, size: int) -> np.ndarray:
    from PIL import Image, ImageDraw
    font = _cjk_font(size)
    if font is None:
        raise RuntimeError('Bundled CJK font is required for this Chinese demo')
    bounds = font.getbbox(text)
    tile = Image.new('RGB', (bounds[2]-bounds[0]+12, bounds[3]-bounds[1]+10), (24, 24, 24))
    ImageDraw.Draw(tile).text((6-bounds[0], 5-bounds[1]), text, font=font, fill=color[::-1])
    return np.asarray(tile)[:, :, ::-1].copy()


def place_labels(labels: list, width: int, height: int) -> list:
    """Place below each box first; stagger crowded labels and preserve their connectors."""
    placed, result = [], []
    for anchor, patch, color in sorted(labels, key=lambda item: item[0][1]):
        ph, pw = patch.shape[:2]
        base_x = int(np.clip(anchor[0]-pw/2, 0, max(0, width-pw)))
        base_y = int(np.clip(anchor[1]+3, 0, max(0, height-ph)))
        xs = [base_x, max(0, base_x-pw-6), min(width-pw, base_x+pw+6)]
        ys = list(range(base_y, max(base_y+1, height-ph+1), ph+4))
        ys += list(range(max(0, base_y-ph-4), -1, -(ph+4)))
        selected = (base_x, base_y)
        found = False
        for y in ys:
            for x in xs:
                rect = (x, y, x+pw, y+ph)
                if not any(x < r[2]+2 and x+pw+2 > r[0] and y < r[3]+2 and y+ph+2 > r[1] for r in placed):
                    selected, found = (x, y), True
                    break
            if found:
                break
        x, y = selected
        placed.append((x,y,x+pw,y+ph))
        result.append((anchor, patch, color, x, y))
    return result


def render_people(frame: np.ndarray, tracks: list[Track], model_shape: tuple, timestamp: float,
                  pose_overlay: bool = True, object_boxes: tuple | list = ()) -> np.ndarray:
    output = frame.copy()
    h, w = output.shape[:2]
    scale = np.array([w/model_shape[1], h/model_shape[0]])
    labels = []
    for track in tracks:
        prediction = track.prediction
        if prediction is not None and timestamp-track.last_inference > 1.:
            prediction = None
        display_label = None if prediction is None else prediction['label']
        if prediction is not None and display_label != 'other':
            threshold = prediction.get('threshold', DEFAULT_ACTION_THRESHOLD)
            if not np.isfinite(prediction['confidence']) or prediction['confidence'] < threshold:
                display_label = 'other'
        color = (70, 220, 80) if display_label=='other' else (40, 190, 255) if prediction else (190, 190, 190)
        box = np.rint(track.box*np.tile(scale, 2)).astype(int)
        cv2.rectangle(output, tuple(box[:2]), tuple(box[2:]), color, 2, cv2.LINE_AA)
        if pose_overlay and track.pose is not None and prediction and prediction['mode']=='rgb_skeleton':
            pts = np.rint(track.pose[:, :2]*scale).astype(int)
            visible = track.pose[:, 2] > .3
            for a, b in EDGES:
                if visible[a] and visible[b]:
                    cv2.line(output, tuple(pts[a]), tuple(pts[b]), color, 1, cv2.LINE_AA)
        if prediction is None:
            label = '识别中...'
        elif display_label == 'other':
            label = BEHAVIOR_ZH['other']
        else:
            label = f"{BEHAVIOR_ZH[display_label]} {prediction['confidence']*100:.0f}%"
        patch = text_patch(f'T{track.tid:02d} {label}', color, max(15, min(25, round(w/68))))
        labels.append(((int((box[0]+box[2])/2), int(box[3])), patch, color))
    from .object_overlay import OBJECT_NAMES
    for item in object_boxes:
        box = np.rint(np.asarray(item['bbox'])*[w, h, w, h]).astype(int)
        color = (255, 220, 20)
        cv2.rectangle(output, tuple(box[:2]), tuple(box[2:]), color, 2, cv2.LINE_AA)
        name = OBJECT_NAMES[item['class_id']]
        suffix = ' ~' if item['followed'] else ''
        patch = text_patch(f"T{item['track_id']:02d} {name} {item['confidence']*100:.0f}%{suffix}",
                           color, max(13, min(21, round(w/80))))
        labels.append(((int((box[0]+box[2])/2), int(box[3])), patch, color))
    placed = place_labels(labels, w, h)
    for anchor, patch, color, x, y in placed:
        endpoint = (int(np.clip(anchor[0], x+3, x+patch.shape[1]-3)), y)
        cv2.line(output, anchor, endpoint, color, 1, cv2.LINE_AA)
    for anchor, patch, color, x, y in placed:
        ph, pw = patch.shape[:2]
        output[y:y+ph, x:x+pw] = patch
    return output
