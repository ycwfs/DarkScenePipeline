"""Restore colour strength after enhancement, without changing what colour anything is.

Low-light enhancement recovers brightness but leaves the result desaturated: measured mean
chroma 5.9 on Retinexformer output whose source frames plainly contain a blue chair, orange
shelf items and a teal desk reflection. The colours are there, just weak.

Scaling (a, b) about neutral in Lab is the operation that fixes exactly that and nothing
else. Hue is atan2(b - 128, a - 128); multiplying both by the same factor leaves that angle
untouched, so this can only make existing colour stronger, never different. Measured over
six frames: chroma 5.9 -> 15.2 at factor 2.6 while hue deviation from the source frames'
own colours stayed at 5.4 degrees (baseline 5.1).

The alternative that was tried first -- running the frames through a colourisation model
(richzhang/colorization, ECCV16) -- is not comparable and not suitable: that model consumes
only the L channel, so it discards the colour the enhancer produced and invents a new one
from luminance. Same measurement: 66.3 degrees of hue deviation, i.e. a different colour
family, visibly a uniform sepia wash over a scene whose real colours were blue and orange.
See compare/results/colorization/.

Runs after recognition (post_recognition = True): the behaviour head was trained on
un-boosted enhancer output, and this only exists to make the picture legible to people.
"""
import cv2
import numpy as np

from .base import FrameStage


class SaturationStage(FrameStage):
    """Scale chroma by `factor` in Lab. factor <= 1 is a no-op path (1.0 = unchanged)."""

    name = "saturate"
    post_recognition = True

    def __init__(self, factor: float):
        self.factor = float(factor)
        self.name = f"saturate:x{self.factor:g}"

    def load(self, device: str) -> None:
        pass

    def __call__(self, frames: list) -> list:
        if self.factor == 1.0:
            return frames
        out = []
        for f in frames:
            lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32)
            # Clipping is what bounds the hue guarantee: a pixel whose scaled a or b runs
            # past the 8-bit range does shift hue slightly. It only affects already-vivid
            # pixels, and the measured deviation at 2.6x was 0.3 degrees.
            lab[..., 1] = np.clip((lab[..., 1] - 128.0) * self.factor + 128.0, 0, 255)
            lab[..., 2] = np.clip((lab[..., 2] - 128.0) * self.factor + 128.0, 0, 255)
            out.append(cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR))
        return out
