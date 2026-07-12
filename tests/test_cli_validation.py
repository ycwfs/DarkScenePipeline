"""CLI/config validation matrix (CPU-only, no checkpoints needed for rejection paths)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from darkpipe.config import PipelineConfig, validate

CKPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ckpts")


def test_realrestorer_serve_rejected():
    with pytest.raises(SystemExit, match="offline-only"):
        validate(PipelineConfig(input="x.mp4", mode="serve", enhance="realrestorer"))


def test_cpu_sr_rejected():
    with pytest.raises(SystemExit, match="CUDA"):
        validate(PipelineConfig(input="x.mp4", device="cpu"))


def test_missing_ckpt_rejected():
    with pytest.raises(SystemExit, match="missing checkpoint"):
        validate(PipelineConfig(input="x.mp4", ckpt_dir="/nonexistent"))


def test_all_off_passthrough_warns():
    cfg = validate(PipelineConfig(input="x.mp4", enhance="off", sr="off", recognize="off"))
    assert any("passthrough" in w for w in cfg.warnings)


def test_stride_gt_window_rejected():
    with pytest.raises(SystemExit, match="reco-stride"):
        validate(PipelineConfig(input="x.mp4", enhance="off", sr="off",
                                recognize="videomamba", reco_stride=64, ckpt_dir=CKPTS))


def test_default_output_derived():
    cfg = validate(PipelineConfig(input="/tmp/clip.mp4", enhance="off", sr="off",
                                  recognize="off"))
    assert cfg.output == "clip_out.mp4"


def test_serve_output_warns():
    cfg = validate(PipelineConfig(input="rtsp://cam", mode="serve", enhance="off",
                                  sr="off", recognize="off", output="x.mp4"))
    assert any("--record" in w for w in cfg.warnings)
