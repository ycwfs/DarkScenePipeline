"""Draw action-compatible pose observations without adding pose-model inference.

Owned by the output thread. Sparse recognition observations are transported with their
reference frame, then followed by forward/backward optical flow on clean enhanced frames.
Uncertain, expired, future or absent observations disappear instead of sticking to a person
who has moved away. The classifier never consumes the rendered result.
"""
import cv2
import numpy as np

# Kept here so importing the renderer does not import torch or the skeleton model.
EDGES = ((15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
         (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
         (1, 3), (2, 4), (3, 5), (4, 6))


def tracking_gray(frame):
    """Bound CPU tracking cost; pose coordinates remain normalized to the whole frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if max(h, w) > 640:
        scale = 640 / max(h, w)
        gray = cv2.resize(gray, (round(w * scale), round(h * scale)),
                          interpolation=cv2.INTER_AREA)
    return gray


class ActionPoseOverlay:
    def __init__(self, max_age=1.0):
        self.max_age = float(max_age)
        self._event = None
        self._gray = None
        self._points = None
        self._valid = None
        self.last_drawn = 0

    def _clear(self):
        self._gray = self._points = self._valid = None

    def draw(self, output, enhanced_frame, event, timestamp):
        """Return a new rendered frame when drawing, otherwise the original output.

        `enhanced_frame` must be the clean, pre-SR image at `timestamp`; `output` can
        have different dimensions. Never track over our own overlay or the label bar.
        """
        self.last_drawn = 0
        actors = getattr(event, 'actor_poses', None)
        reference = getattr(event, 'pose_reference', None)
        if (event is None or not getattr(event, 'skeleton_active', False) or not actors
                or reference is None or str(event.label).lower() == 'other'
                or not 0 <= timestamp - event.timestamp <= self.max_age):
            self._clear()
            self._event = None
            return output
        gray = tracking_gray(enhanced_frame)
        fresh = event is not self._event
        if fresh:
            poses = np.asarray([a['keypoints'] for a in actors], dtype=np.float32)
            if poses.ndim != 3 or poses.shape[1:] != (17, 3) or gray.shape != reference.shape:
                self._clear()
                return output
            self._event = event
            self._gray = reference
            self._points = poses[:, :, :2].copy()
            self._points *= np.array([gray.shape[1], gray.shape[0]], dtype=np.float32)
            self._valid = ((poses[:, :, 2] > .3) & np.isfinite(poses).all(axis=2)
                           & (poses[:, :, :2] >= 0).all(axis=2)
                           & (poses[:, :, :2] < 1).all(axis=2))
        if self._gray is None or self._gray.shape != gray.shape:
            self._clear()
            return output
        # The event frame already has exact pose coordinates; only subsequent frames need flow.
        if not (fresh and abs(timestamp - event.timestamp) < 1e-6):
            indices = np.flatnonzero(self._valid.ravel())
            if len(indices):
                old = self._points.reshape(-1, 2)[indices].reshape(-1, 1, 2)
                params = dict(winSize=(21, 21), maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, .03))
                try:
                    moved, status, _ = cv2.calcOpticalFlowPyrLK(self._gray, gray, old, None, **params)
                    back, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, self._gray, moved, None, **params)
                except cv2.error:
                    self._clear()
                    return output
                if moved is None or back is None:
                    self._clear()
                    return output
                good = (status.ravel().astype(bool) & back_status.ravel().astype(bool)
                        & np.isfinite(moved).all(axis=(1, 2))
                        & (np.linalg.norm(back - old, axis=2).ravel() <= 1.5))
                self._points.reshape(-1, 2)[indices] = moved.reshape(-1, 2)
                self._valid.ravel()[indices] &= good
        self._gray = gray
        h, w = output.shape[:2]
        gh, gw = gray.shape
        self._valid &= ((self._points[:, :, 0] >= 0) & (self._points[:, :, 0] < gw)
                        & (self._points[:, :, 1] >= 0) & (self._points[:, :, 1] < gh))
        rendered = None
        scale = np.array([w / gw, h / gh], np.float32)
        thickness = max(2, round(w / 600))
        for points, visible in zip(self._points, self._valid):
            if visible.sum() < 5:
                continue
            if rendered is None:
                rendered = output.copy()
            self.last_drawn += 1
            coords = np.rint(np.nan_to_num(points * scale)).astype(int)
            for a, b in EDGES:
                if visible[a] and visible[b]:
                    cv2.line(rendered, tuple(coords[a]), tuple(coords[b]), (40, 230, 60),
                             thickness, cv2.LINE_AA)
            for point in coords[visible]:
                cv2.circle(rendered, tuple(point), thickness + 1, (0, 210, 255), -1, cv2.LINE_AA)
        return output if rendered is None else rendered
