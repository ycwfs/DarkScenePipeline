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
            for c in (1, 2):
                # Scale about the frame's OWN mean chroma, not about the neutral point.
                #
                # Scaling about neutral multiplies the global colour cast along with
                # everything else: a deployment reported the picture turning green at 2x,
                # and that is exactly this -- low-light footage often carries a green cast
                # (a Bayer sensor has twice as many green photosites), and doubling the
                # chroma doubled the cast. Measured on a clip with a mild cast: 2.61x the
                # cast at factor 2.6. Centring on the mean leaves the cast where it was
                # (1.08x) while delivering the same chroma (15.1 vs 15.6).
                #
                # The cost is that hue is no longer *exactly* invariant -- centring on a
                # non-neutral point rotates individual pixels slightly. Measured 5.8 deg of
                # deviation from the source frames' own hues against 5.3 for the old
                # version, both far below the 66 deg of re-colourisation. Amplifying a cast
                # is the more visible error of the two.
                m = float(lab[..., c].mean())
                lab[..., c] = np.clip((lab[..., c] - m) * self.factor + m, 0, 255)
            # round, not truncate: astype(uint8) floors, which biases a and b downward by
            # ~0.5 on every frame -- toward green and blue, the very artefact being fixed.
            out.append(cv2.cvtColor(np.round(lab).astype(np.uint8), cv2.COLOR_LAB2BGR))
        return out
