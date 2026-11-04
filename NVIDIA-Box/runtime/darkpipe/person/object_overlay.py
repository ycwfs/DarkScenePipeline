"""Visual-only, short-lived alignment of asynchronous object detections.

Coordinates always originate in a real detector sample. For later frames, a
forward/backward checked local optical flow must succeed; otherwise hide the box.
This never changes model scores, prior decisions, person tracking or clip triggers.
"""
from collections import Counter, OrderedDict

import cv2
import numpy as np

from .object_prior import allows_object_prior

OBJECT_NAMES = ('bottle', 'cup', 'phone', 'camera', 'paper', 'item', 'exhibit')


class ObjectOverlay:
    def __init__(self, max_age: float = .6):
        self.max_age = max_age
        self.history = OrderedDict()
        self.epoch = None
        self.last_stamp = -1e9
        self.stats = Counter()

    @staticmethod
    def follow(previous: np.ndarray, current: np.ndarray, box: np.ndarray) -> np.ndarray | None:
        """Translate a small object only when its local features agree in both directions."""
        h, w = previous.shape
        lo = np.maximum(0, np.floor(box[:2])).astype(int)
        hi = np.minimum([w, h], np.ceil(box[2:])).astype(int)
        if np.any(hi-lo < 3):
            return None
        corners = cv2.goodFeaturesToTrack(previous[lo[1]:hi[1], lo[0]:hi[0]],
            maxCorners=24, qualityLevel=.04, minDistance=2, blockSize=3)
        if corners is None or len(corners) < 3:
            return None
        points = corners + lo.astype(np.float32)
        options = dict(winSize=(15, 15), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, .03))
        forward, valid, errors = cv2.calcOpticalFlowPyrLK(previous, current, points, None, **options)
        if forward is None:
            return None
        backward, reverse, _ = cv2.calcOpticalFlowPyrLK(current, previous, forward, None, **options)
        if backward is None:
            return None
        good = ((valid.ravel() == 1) & (reverse.ravel() == 1) & (errors.ravel() < 30) &
                (np.linalg.norm(backward-points, axis=2).ravel() < 1.5))
        shifts = (forward-points).reshape(-1, 2)[good]
        if len(shifts) < 3:
            return None
        delta = np.median(shifts, axis=0)
        coherent = np.linalg.norm(shifts-delta, axis=1) < 2.5
        if coherent.sum() < 3 or coherent.mean() < .6 or np.linalg.norm(delta) > .08*max(w, h):
            return None
        moved = box + np.tile(np.median(shifts[coherent], axis=0), 2)
        moved = np.clip(moved, 0, [w, h, w, h])
        return moved if np.all(moved[2:]-moved[:2] >= 2) else None

    def update(self, frame: np.ndarray, tracks: list, record: dict, evidence_rows: tuple | list = ()) -> list:
        stamp, index, epoch = record['timestamp'], record['frame'], record.get('capture_epoch', 0)
        h, w = frame.shape[:2]
        ratio = min(1., 720/max(h, w))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if ratio < 1:
            gray = cv2.resize(gray, (round(w*ratio), round(h*ratio)), interpolation=cv2.INTER_AREA)
        if (epoch != self.epoch or stamp <= self.last_stamp or stamp-self.last_stamp > .5 or
                (self.history and next(reversed(self.history.values()))[1].shape != gray.shape)):
            self.history.clear()
        self.epoch, self.last_stamp = epoch, stamp
        self.history[index] = (stamp, gray)
        while self.history and (stamp-next(iter(self.history.values()))[0] > self.max_age or
                len(self.history) > 64 or len(self.history)*gray.nbytes > 24*1024*1024):
            self.history.popitem(last=False)
        self.stats['max_history_bytes'] = max(self.stats['max_history_bytes'], len(self.history)*gray.nbytes)
        output = []
        gh, gw = gray.shape
        for track in tracks:
            row = track.prediction
            key = record.get('keys', {}).get(track.tid)
            candidates = [r for r in evidence_rows if key ==
                (r.get('session'), r.get('track_id'), r.get('history_generation')) and
                r['history_generation'] >> 32 == epoch and abs(stamp-r['timestamp']) <= 1.2]
            if (row and key == (row.get('session'), row.get('track_id'), row.get('history_generation')) and
                    row['history_generation'] >> 32 == epoch and 0 <= stamp-row['timestamp'] <= 1.):
                candidates.append(row)
            if not candidates:
                continue
            samples = [(s, r['object_prior']) for r in candidates
                       if allows_object_prior(record.get('raw_luma'), r.get('object_prior', {}).get('lum_gate', 35.))
                       for s in r.get('object_prior', {}).get('samples', [])
                       if s.get('available') and s.get('binding_valid') and s.get('timestamp') is not None
                       and 0 <= stamp-s['timestamp'] <= self.max_age]
            if not samples:
                continue
            # A newer empty observation clears older boxes instead of leaving a stale trail.
            sample, prior = max(samples, key=lambda pair: pair[0]['timestamp'])
            source = self.history.get(sample['frame'])
            if source is None or abs(source[0]-sample['timestamp']) > 1e-5:
                continue
            sh, sw = sample['source_shape'][:2]
            for detection in sample['detections']:
                det = np.asarray(detection, dtype=float)
                if not np.isfinite(det).all() or det[4] < prior.get('object_min_conf', .581):
                    continue
                cls = int(det[5])
                if cls not in range(len(OBJECT_NAMES)):
                    continue
                box = det[:4]*np.array([gw/sw, gh/sh, gw/sw, gh/sh])
                followed = sample['frame'] != index
                if followed:
                    box = self.follow(source[1], gray, box)
                    if box is None:
                        self.stats['alignment_hidden'] += 1
                        continue
                box = np.clip(box / [gw, gh, gw, gh], 0, 1)
                if np.any(box[2:] <= box[:2]):
                    continue
                output.append(dict(bbox=box.tolist(), class_id=cls, confidence=float(det[4]),
                    track_id=track.tid, sample_frame=sample['frame'], sample_timestamp=sample['timestamp'],
                    age=stamp-sample['timestamp'], followed=followed))
        self.stats['frames'] += 1
        self.stats['frames_with_objects'] += bool(output)
        self.stats['boxes'] += len(output)
        self.stats['followed_boxes'] += sum(b['followed'] for b in output)
        return output
