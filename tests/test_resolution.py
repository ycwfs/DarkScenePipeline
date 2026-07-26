"""Arbitrary input resolution: no stage requires a particular frame size or aspect ratio.

Every GPU stage has an architectural size constraint (retinexformer 4, cidnet 8, lightSR 16,
CATANet its per-block patch/group sizes), and every one of them is handled the same way --
reflect-pad up to the multiple, crop the output back -- so what comes out is exactly the input
size, or exactly `--sr-scale` times it. The recognizer is size-agnostic by construction: it
resizes the short side and center-crops to 224 before the network sees anything.

The end-to-end half of this needs CUDA and the two shipped checkpoints; it is skipped without
them, which is why the pad/crop invariant is also asserted on its own.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CKPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ckpts")

# Deliberately awkward: odd, prime, extreme aspect ratios, smaller than one recognizer crop,
# and larger than the pad multiple of every stage.
SIZES = [(483, 641), (197, 353), (98, 130), (41, 640), (721, 1281), (240, 320)]


@pytest.mark.parametrize("h,w", SIZES)
@pytest.mark.parametrize("multiple", [4, 8, 16])
def test_reflect_pad_round_trips_any_size(h, w, multiple):
    import torch
    from darkpipe.utils import reflect_pad_to
    x, (oh, ow) = reflect_pad_to(torch.rand(1, 3, h, w), multiple)
    assert (oh, ow) == (h, w)                       # the crop target is the original size
    assert x.shape[2] % multiple == 0 and x.shape[3] % multiple == 0
    assert x.shape[2] - h < multiple and x.shape[3] - w < multiple


def _cuda_or_skip():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    for f in ("NTIRE.pth", "videomamba_t_behavior_32f.pth"):
        if not os.path.exists(os.path.join(CKPTS, f)):
            pytest.skip(f"missing {f}")


def test_recommended_pipeline_accepts_any_resolution():
    """retinexformer -> bicubic -> behavior at six odd sizes, one scale each."""
    _cuda_or_skip()
    from darkpipe.config import PipelineConfig, validate
    from darkpipe.stages import build_stages

    for i, (h, w) in enumerate(SIZES):
        scale = (2, 3, 4)[i % 3]
        cfg = validate(PipelineConfig(input="x.mp4", enhance="retinexformer", sr="bicubic",
                                      sr_scale=scale, recognize="behavior", ckpt_dir=CKPTS))
        stages, reco = build_stages(cfg)
        for st in stages:
            st.load("cuda:0")
        reco.load("cuda:0")
        frames = [np.random.randint(0, 255, (h, w, 3), np.uint8) for _ in range(reco.window)]

        enhanced = stages[0](frames)
        assert enhanced[0].shape == (h, w, 3), (h, w, enhanced[0].shape)
        out = stages[1](enhanced)
        assert out[0].shape == (h * scale, w * scale, 3), (h, w, scale, out[0].shape)

        # Recognition taps the enhanced (pre-SR) frames; a full window must produce an event.
        ev = [reco.push(f, n, n / 25.0) for n, f in enumerate(enhanced)]
        assert any(e is not None for e in ev), (h, w)
        for st in stages:
            st.close()
        reco.close()
