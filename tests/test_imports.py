"""Every vendored arch imports and builds with its documented kwargs (no ckpts needed)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_retinexformer_builds():
    from darkpipe.vendor.retinexformer_arch import RetinexFormer
    net = RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1, 2, 2])
    n = sum(p.numel() for p in net.parameters())
    assert 1.4e6 < n < 1.8e6


def test_lightsr_builds():
    from darkpipe.stages.sr_lightsr import LIGHTSR_KWARGS
    from darkpipe.vendor.mambairv2light_arch import MambaIRv2Light
    net = MambaIRv2Light(**LIGHTSR_KWARGS)
    n = sum(p.numel() for p in net.parameters())
    assert 0.6e6 < n < 1.0e6


def test_videomamba_builds():
    from darkpipe.vendor.videomamba import videomamba_tiny
    net = videomamba_tiny(num_classes=11, num_frames=32, img_size=224)
    n = sum(p.numel() for p in net.parameters())
    assert 6e6 < n < 8e6


def test_realrestorer_package_imports():
    import darkpipe.vendor.realrestorer.scheduler  # light module, no weights
