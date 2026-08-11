"""Cap the working resolution before anything expensive touches the frame.

Enhancement cost tracks pixel count, and a modern camera hands over far more pixels than
recognition can use: the recogniser resizes to 224x224 internally no matter what it is given,
so everything above a moderate working size is spent on the picture alone. Measured on one
RTX 3090, a 1080p live source ran the whole pipeline at ~3 fps -- and at that rate a 1 s
recognition window holds only ~3 distinct frames resampled up to 32, which is a much thinner
motion sample than the model was trained on. Capping the long side to 1280 quarters the work.

This is deliberately a stage rather than a step inside the capture loop: placed first, it
applies to offline and serve alike, and the frames that reach the recogniser, the SR backend,
the label bar, the clips and the streams are then all the same size, with nothing needing to
know a downscale happened.
"""
import cv2

from .base import FrameStage


class DownscaleStage(FrameStage):
    """Shrink frames so the long side is at most `max_side`. Never upscales."""

    name = "downscale"

    def __init__(self, max_side: int):
        self.max_side = int(max_side)
        self._logged = False

    def load(self, device: str) -> None:
        pass

    def __call__(self, frames: list) -> list:
        if not frames or self.max_side <= 0:
            return frames
        h, w = frames[0].shape[:2]
        if max(h, w) <= self.max_side:
            return frames
        s = self.max_side / max(h, w)
        # Even dimensions: H.264 rejects odd ones, and this is the last place the size is
        # chosen freely (see darkpipe/render.py for the bar's half of the same constraint).
        nw, nh = (int(round(w * s)) // 2) * 2, (int(round(h * s)) // 2) * 2
        if not self._logged:
            self._logged = True
            print(f"[downscale] {w}x{h} -> {nw}x{nh} (proc_max_side={self.max_side})")
        # INTER_AREA is the correct filter for shrinking; INTER_LINEAR aliases, and aliasing
        # on a dark noisy frame is exactly the kind of high-frequency detail the enhancer
        # would then amplify.
        return [cv2.resize(f, (nw, nh), interpolation=cv2.INTER_AREA) for f in frames]
