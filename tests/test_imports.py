"""Every vendored arch imports and builds with its documented kwargs (no ckpts needed)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_retinexformer_builds():
    from darkpipe.vendor.retinexformer_arch import RetinexFormer
    net = RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1, 2, 2])
    n = sum(p.numel() for p in net.parameters())
    assert 1.4e6 < n < 1.8e6


def test_cidnet_builds():
    from darkpipe.vendor.cidnet import CIDNet
    net = CIDNet()
    n = sum(p.numel() for p in net.parameters())
    assert 1.8e6 < n < 2.2e6


def test_lightsr_builds():
    from darkpipe.stages.sr_lightsr import LIGHTSR_KWARGS
    from darkpipe.vendor.mambairv2light_arch import MambaIRv2Light
    net = MambaIRv2Light(**LIGHTSR_KWARGS)
    n = sum(p.numel() for p in net.parameters())
    assert 0.6e6 < n < 1.0e6


def test_bicubic_is_an_exact_2x_and_needs_nothing():
    """The one SR backend with no weights and no CUDA — so the test runs it for real."""
    import numpy as np
    from darkpipe.stages.sr_bicubic import BicubicStage
    st = BicubicStage()
    st.load("cpu")
    frames = [np.random.randint(0, 255, (h, w, 3), np.uint8) for h, w in [(31, 37), (240, 320)]]
    out = st(frames)
    assert [o.shape for o in out] == [(62, 74, 3), (480, 640, 3)]
    assert all(o.dtype == np.uint8 for o in out)


def test_catanet_builds():
    from darkpipe.vendor.catanet_arch import CATANet
    net = CATANet(upscale=2)
    n = sum(p.numel() for p in net.parameters())
    assert 0.4e6 < n < 0.6e6


def test_catanet_handles_odd_sizes():
    """No reflect-pad wrapper around this stage, so the arch must self-pad to an exact 2x.

    Sizes chosen to be coprime with every per-block patch size (16/20/24/28) and group size.
    """
    import torch
    from darkpipe.vendor.catanet_arch import CATANet
    net = CATANet(upscale=2).eval()
    for (h, w) in [(31, 37), (64, 64), (45, 90)]:
        with torch.no_grad():
            y = net(torch.rand(1, 3, h, w))
        assert y.shape[-2:] == (h * 2, w * 2), (h, w, y.shape)


def test_videomamba_builds():
    from darkpipe.vendor.videomamba import videomamba_tiny
    net = videomamba_tiny(num_classes=11, num_frames=32, img_size=224)
    n = sum(p.numel() for p in net.parameters())
    assert 6e6 < n < 8e6


def test_realrestorer_package_imports():
    import darkpipe.vendor.realrestorer.scheduler  # light module, no weights
