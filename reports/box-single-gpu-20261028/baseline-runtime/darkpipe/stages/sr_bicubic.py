"""Super-resolution stage: bicubic x2 — the classical interpolation baseline.

Shipping an interpolation alongside two neural backends needs a justification, and on this
footage the measurement is the justification (compare/results/SR_REPORT.md, same protocol
for all three, in-domain enhanced dark frames):

    bicubic      42.92 dB / 0.9903      0.3 ms/frame
    lightsr_x2   39.46 dB / 0.9851    108.0 ms/frame
    catanet_x2   39.56 dB / 0.9852     67.1 ms/frame

Bicubic wins on distortion metrics AND is ~200-3000x cheaper, because the input is 240p
enhanced dark video: there is little true high-frequency detail for a network to recover
after a /2, and both networks spend their capacity hallucinating texture onto amplified
sensor noise. What they buy over bicubic is perceptual sharpness, not fidelity.

(Those three ms/frame are the quality protocol's 160x120 -> 320x240; at the sizes the pipeline
runs, SR_REPORT.md section 3 measures bicubic at 0.6-2.0 ms/frame against 170-1010 ms.)

The consequence for the spec is the point of this stage: it is the only --sr setting that
keeps the pipeline inside `offline >= 15 fps` and `real-time <= 1 s`, because it adds
~1-2 ms/frame instead of ~800. End to end, in a paired run where both halves saw the same
machine (150-frame clips, RTX 3090, `retinexformer + behavior`), turning it on costs 4% at
320x240 (29.2 -> 28.0 fps) and 6% at 640x480 (17.4 -> 16.3): the resize itself is ~1 ms and
the doubled encode runs on the writer thread. On genuinely high-resolution sensor input the
quality ranking would likely invert, and the neural backends stay available for that.

No weights, no CUDA, no batching: cv2.INTER_CUBIC on the CPU, so this is also the only SR
backend that works with --device cpu.
"""
import cv2

from .base import FrameStage


class BicubicStage(FrameStage):
    name = "sr:bicubic_x2"
    post_recognition = True

    def __init__(self, scale: int = 2):
        # The only scale-generic backend: cv2.resize takes any factor, so x3/x4 need no
        # weights and no extra checkpoint — unlike the two neural stages.
        self.scale = scale
        self.name = f"sr:bicubic_x{scale}"

    def load(self, device: str) -> None:
        pass

    def __call__(self, frames: list) -> list:
        s = self.scale
        return [cv2.resize(f, (f.shape[1] * s, f.shape[0] * s),
                           interpolation=cv2.INTER_CUBIC) for f in frames]
