"""Denoising for enhanced frames. Runs after recognition -- it only changes the picture.

Enhancement amplifies whatever noise the sensor produced; on real deployment footage the
enhanced frame measured 1.09-1.87 flat-block noise against 1.08-1.72 at the input. The
enhancer is already suppressing most of it (64-76% less than plain brightening at the same
gain), so what is left here is the residual.

Placement is after enhancement, and that is measured rather than assumed: denoising the dark
input first removed 59-65% against 73-75% for a post-pass, at similar or worse edge
retention. A dark frame has too little signal for NLM's patch-similarity test to be reliable,
and a pre-pass largely duplicates what the enhancer already does while whatever survives is
still amplified. Doing both ends reaches 92-100% but costs twice as much and drops edges to
62-79%. See compare/results/denoise/.

Modes, measured at 720p on real footage (noise removed / edge retention / cost):

    fast          bilateral d5     35% / 84% /   5 ms   -- affordable in the live path
    quality       NLM h3 win7      51% / 85% / 119 ms
    quality_high  NLM h3 win15     75% / 81% / 376 ms   -- visibly plastic, see the sheets

Cost is set by NLM's search window, not by its filter strength. `quality_high` scores best
on the metric and still looks over-smoothed, which is why the numbers alone were not enough
to pick a default.
"""
import cv2

from .base import FrameStage

MODES = ("off", "fast", "quality", "quality_high")


def denoise_frame(bgr, mode):
    """One frame, one mode. Shared with the clip writer, which denoises off-thread."""
    if mode == "fast":
        return cv2.bilateralFilter(bgr, 5, 40, 40)
    if mode == "quality":
        return cv2.fastNlMeansDenoisingColored(bgr, None, 3, 3, 7, 7)
    if mode == "quality_high":
        return cv2.fastNlMeansDenoisingColored(bgr, None, 3, 3, 7, 15)
    return bgr


class DenoiseStage(FrameStage):
    name = "denoise"
    post_recognition = True

    def __init__(self, mode: str):
        self.mode = mode
        self.name = f"denoise:{mode}"

    def load(self, device: str) -> None:
        pass

    def __call__(self, frames: list) -> list:
        if self.mode == "off":
            return frames
        return [denoise_frame(f, self.mode) for f in frames]
