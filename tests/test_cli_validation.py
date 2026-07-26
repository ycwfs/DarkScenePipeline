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


def test_bicubic_sr_runs_on_cpu_without_weights():
    """bicubic_x2 is the spec-compliant SR path: no CUDA, no checkpoint, no slow-path warning."""
    cfg = validate(PipelineConfig(input="x.mp4", device="cpu", enhance="off",
                                  sr="bicubic_x2", recognize="off", ckpt_dir="/nonexistent"))
    assert not any("below the 10 fps floor" in w for w in cfg.warnings)


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


def test_missing_reco_ckpt_rejected():
    with pytest.raises(SystemExit, match="--reco-ckpt"):
        validate(PipelineConfig(input="x.mp4", enhance="off", sr="off",
                                recognize="behavior", reco_ckpt="/nonexistent/head.pth"))


def test_reco_ckpt_replaces_the_default_lookup(tmp_path):
    """A supplied head must satisfy the checkpoint check even with an empty ckpt_dir."""
    p = tmp_path / "head.pth"
    p.write_bytes(b"weights")
    cfg = validate(PipelineConfig(input="x.mp4", enhance="off", sr="off",
                                  recognize="behavior", reco_ckpt=str(p),
                                  ckpt_dir="/nonexistent"))
    assert cfg.reco_ckpt == str(p)


def test_reco_ckpt_ignored_for_xclip_warns():
    cfg = validate(PipelineConfig(input="x.mp4", enhance="off", sr="off", recognize="xclip",
                                  ckpt_dir=CKPTS, reco_ckpt="/tmp/head.pth"))
    assert any("--reco-ckpt ignored" in w for w in cfg.warnings)


def test_serve_output_warns():
    cfg = validate(PipelineConfig(input="rtsp://cam", mode="serve", enhance="off",
                                  sr="off", recognize="off", output="x.mp4"))
    assert any("--record" in w for w in cfg.warnings)


def test_gpus_serve_rejected():
    with pytest.raises(SystemExit, match="offline-only"):
        validate(PipelineConfig(input="x.mp4", mode="serve", enhance="off", sr="off",
                                recognize="off", gpus="0,1"))


def test_gpus_rejects_repeats_and_junk():
    with pytest.raises(SystemExit, match="repeated"):
        validate(PipelineConfig(input="x.mp4", enhance="off", sr="off", recognize="off",
                                gpus="0,1,1"))
    with pytest.raises(SystemExit, match="comma-separated"):
        validate(PipelineConfig(input="x.mp4", enhance="off", sr="off", recognize="off",
                                gpus="cuda:0,cuda:1"))


def test_gpus_realrestorer_rejected():
    with pytest.raises(SystemExit, match="whole video"):
        validate(PipelineConfig(input="x.mp4", enhance="realrestorer", sr="off",
                                recognize="off", gpus="0,1"))


def test_segments_tile_the_video_exactly():
    """Every frame lands in exactly one segment, and short clips are not over-split."""
    from darkpipe.shard import _segments

    for n in (300, 1000, 7, 4001):
        for k in (1, 2, 3, 6):
            segs = _segments(n, k, min_seg_frames=120)
            assert sum(ln for _, ln in segs) == n
            assert [s for s, _ in segs] == [0] + [s + ln for s, ln in segs][:-1]
            assert len(segs) <= max(1, min(k, n // 120))
