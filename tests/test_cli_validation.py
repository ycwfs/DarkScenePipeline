"""CLI/config validation matrix (CPU-only, no checkpoints needed for rejection paths)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from darkpipe import config
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


def test_sr_scale_defaults_to_2_and_names_the_stage():
    cfg = validate(PipelineConfig(input="x.mp4", device="cpu", enhance="off",
                                  sr="bicubic", recognize="off", ckpt_dir="/nonexistent"))
    assert (cfg.sr, cfg.sr_scale, cfg.sr_name()) == ("bicubic", 2, "bicubic_x2")


def test_sr_scale_3_and_4_accepted():
    for s in (3, 4):
        cfg = validate(PipelineConfig(input="x.mp4", device="cpu", enhance="off", sr="bicubic",
                                      sr_scale=s, recognize="off", ckpt_dir="/nonexistent"))
        assert cfg.sr_name() == f"bicubic_x{s}"


def test_sr_scale_out_of_range_rejected():
    with pytest.raises(SystemExit, match="--sr-scale must be one of"):
        validate(PipelineConfig(input="x.mp4", device="cpu", enhance="off", sr="bicubic",
                                sr_scale=8, recognize="off", ckpt_dir="/nonexistent"))


def test_legacy_x2_alias_conflicting_with_sr_scale_rejected():
    """`--sr bicubic_x2 --sr-scale 4` is contradictory: the alias pins the factor."""
    with pytest.raises(SystemExit, match="pins x2"):
        validate(PipelineConfig(input="x.mp4", device="cpu", enhance="off", sr="bicubic_x2",
                                sr_scale=4, recognize="off", ckpt_dir="/nonexistent"))


def test_sr_scale_ignored_when_sr_off():
    cfg = validate(PipelineConfig(input="x.mp4", enhance="off", sr="off", sr_scale=4,
                                  recognize="off"))
    assert cfg.sr_scale is None and any("ignored: --sr is off" in w for w in cfg.warnings)


def test_neural_sr_wants_the_checkpoint_for_that_scale():
    """Weights are per-factor: x3 must ask for mambairv2_lightSR_x3.pth, not the x2 file."""
    with pytest.raises(SystemExit, match=r"mambairv2_lightSR_x3\.pth"):
        validate(PipelineConfig(input="x.mp4", enhance="off", sr="lightsr", sr_scale=3,
                                recognize="off", ckpt_dir="/nonexistent"))


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


@pytest.fixture
def n_gpus(monkeypatch):
    """Pin the visible GPU count, so --gpus assertions do not depend on the test machine."""
    def _set(n):
        monkeypatch.setattr(config, "_cuda_device_count", lambda: n)
    return _set


def test_gpus_is_accepted_in_serve_and_deals_frames_round_robin(n_gpus):
    """serve fans out by dealing arriving frames, not by splitting a file into ranges.

    Offline shards by frame range, which needs future frames to exist; a live stream has
    none, so serve used to reject the flag outright. Round-robin dealing needs no future
    frames, so the flag now works in both modes with a different mechanism behind it.
    """
    from darkpipe.server import serve_devices

    n_gpus(4)
    cfg = validate(PipelineConfig(input="rtsp://cam", mode="serve", enhance="off", sr="off",
                                  recognize="off", gpus="2,3"))
    assert serve_devices(cfg) == ["cuda:2", "cuda:3"]


def test_gpus_with_one_id_warns_and_defers_to_device(n_gpus):
    """One id shards nothing offline and deals nothing in serve, so it must not look used.

    The trap is `--gpus 3 --device cuda:0` quietly running on cuda:0. Warn rather than die
    -- the value is harmless -- but do not silently honour it in serve only, which would
    make the same spelling pick different cards in the two modes.
    """
    from darkpipe.server import serve_devices

    n_gpus(4)
    cfg = validate(PipelineConfig(input="rtsp://cam", mode="serve", enhance="off", sr="off",
                                  recognize="off", gpus="3", device="cuda:0"))
    assert serve_devices(cfg) == ["cuda:0"]


def test_gpus_beyond_what_is_visible_degrades_instead_of_crashing(n_gpus, capsys):
    """Fewer cards than asked for must fall back, not die on `invalid device ordinal`.

    This is the platform case: the serve operator declares metadata.gpu.count 2 and defaults
    gpu_ids to "0,1", so a scheduler that hands over one card leaves the default pointing at
    a device that does not exist. Without this the failure is a stage load blowing up ~15 s
    in with a CUDA ordinal error; with it, the run drops to the single-GPU path that was
    measured and shipped.
    """
    from darkpipe.server import serve_devices

    n_gpus(1)
    cfg = validate(PipelineConfig(input="rtsp://cam", mode="serve", enhance="off", sr="off",
                                  recognize="off", gpus="0,1", device="cuda:0"))
    assert serve_devices(cfg) == ["cuda:0"], "must fall back to one device"
    assert "only 1 visible" in capsys.readouterr().out, "the degradation must be visible"


def test_gpus_survives_a_partial_grant(n_gpus):
    """Three asked for, two granted: keep both, do not collapse all the way to one."""
    from darkpipe.server import serve_devices

    n_gpus(2)
    cfg = validate(PipelineConfig(input="rtsp://cam", mode="serve", enhance="off", sr="off",
                                  recognize="off", gpus="0,1,2", device="cuda:0"))
    assert serve_devices(cfg) == ["cuda:0", "cuda:1"]


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


def _fake_shard_run(monkeypatch, tmp_path, env_before=None):
    """Run run_offline_sharded with the workers faked out; -> the env they were spawned with."""
    import darkpipe.shard as shard

    seen = {}

    class FakeProc:
        returncode = 0

        def communicate(self):
            return ("", None)

    def fake_popen(cmd, **kw):
        seen["cmd"], seen["env"] = cmd, kw.get("env")
        return FakeProc()

    monkeypatch.setattr(shard, "_frame_count", lambda src, mx: (600, 25.0))
    monkeypatch.setattr(shard, "_concat", lambda parts, out, fps: 600)
    monkeypatch.setattr(shard.subprocess, "Popen", fake_popen)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    if env_before is not None:
        monkeypatch.setenv("PYTHONPATH", env_before)
    cfg = PipelineConfig(input="in.mp4", output=str(tmp_path / "out.mp4"), mode="offline",
                         enhance="off", sr="off", recognize="off", events_json="")
    shard.run_offline_sharded(cfg, ["0", "1"])
    return seen


def test_shard_workers_are_told_where_darkpipe_lives(monkeypatch, tmp_path):
    """The workers are `python -m darkpipe.cli`, and a subprocess inherits the environment but
    not sys.path. In the repo that gap is invisible -- an editable install puts darkpipe on
    every interpreter's path -- so it only surfaced in the operator container, where darkpipe
    merely sits next to main.py in the unpacked zip: both workers died instantly with
    "No module named darkpipe.cli", and only on a 2-GPU host, since one GPU skips this path."""
    seen = _fake_shard_run(monkeypatch, tmp_path)
    import darkpipe

    pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(darkpipe.__file__)))
    assert seen["env"] is not None, "workers inherit os.environ verbatim, PYTHONPATH included"
    assert seen["env"]["PYTHONPATH"].split(os.pathsep)[0] == pkg_parent
    assert seen["cmd"][1:3] == ["-m", "darkpipe.cli"]


def test_an_existing_pythonpath_is_kept(monkeypatch, tmp_path):
    """Prepend, do not replace: the deployment may be pointing at vendored deps of its own."""
    seen = _fake_shard_run(monkeypatch, tmp_path, env_before="/opt/site-stuff")
    assert seen["env"]["PYTHONPATH"].split(os.pathsep)[-1] == "/opt/site-stuff"
