"""Live-output format selection, and the failure that must happen at startup.

Every format except mjpeg is muxed by an ffmpeg subprocess. The bad outcome is not "flv is
unavailable" -- it is a service that starts happily, reports healthy, serves the demo page,
and only fails when someone finally opens the FLV URL during a demonstration. So a missing
binary or a misspelled format has to stop the process at startup, naming what is wrong.
"""
import pytest

from darkpipe.config import PipelineConfig, validate
from darkpipe.streams import FORMATS, ffmpeg_path, parse_formats


def test_parse_formats_accepts_a_list_and_normalises_it():
    assert parse_formats("mjpeg, FLV ") == ["mjpeg", "flv"]
    assert parse_formats("") == ["mjpeg"]
    for f in FORMATS:
        assert parse_formats(f) == [f]


def test_parse_formats_rejects_unknown_names():
    """A typo must not silently degrade to "just mjpeg" -- that ships a demo missing a format."""
    with pytest.raises(ValueError) as e:
        parse_formats("mjpeg,rtmp")
    assert "rtmp" in str(e.value)


def serve_cfg(**kw):
    return PipelineConfig(mode="serve", input="rtsp://x", enhance="off", sr="off",
                          recognize="off", **kw)


def test_mjpeg_only_never_needs_ffmpeg():
    cfg = validate(serve_cfg(stream_formats="mjpeg"))
    assert cfg.stream_formats == "mjpeg"


@pytest.mark.parametrize("formats", ["mjpeg,flv", "mjpeg,hls", "hls"])
def test_ffmpeg_backed_formats_fail_fast_when_the_binary_is_missing(formats, monkeypatch):
    monkeypatch.setattr("darkpipe.streams.ffmpeg_path", lambda: None)
    with pytest.raises(SystemExit) as e:
        validate(serve_cfg(stream_formats=formats))
    msg = str(e.value)
    assert "ffmpeg" in msg and "Dockerfile" in msg, \
        f"the error has to say what is missing and where it comes from, got: {msg}"


def test_rtmp_push_also_requires_ffmpeg(monkeypatch):
    monkeypatch.setattr("darkpipe.streams.ffmpeg_path", lambda: None)
    with pytest.raises(SystemExit):
        validate(serve_cfg(stream_formats="mjpeg", rtmp_push_url="rtmp://host/live/k"))


def test_bad_format_name_is_reported_as_a_config_error():
    with pytest.raises(SystemExit) as e:
        validate(serve_cfg(stream_formats="mjpeg,webm"))
    assert "webm" in str(e.value)


def test_offline_mode_ignores_stream_formats():
    """The batch operator has no server; a stale value must not block it."""
    cfg = validate(PipelineConfig(mode="offline", input="x.mp4", output="/tmp/o.mp4",
                                  enhance="off", sr="off", recognize="off",
                                  stream_formats="mjpeg,flv"))
    assert cfg.mode == "offline"


@pytest.mark.skipif(ffmpeg_path() is None, reason="no ffmpeg on this host")
def test_ffmpeg_build_has_what_the_formats_need():
    """Ubuntu's ffmpeg is full-featured, but a slimmed build would be missing exactly these."""
    import subprocess
    enc = subprocess.run([ffmpeg_path(), "-hide_banner", "-encoders"],
                         capture_output=True, text=True).stdout
    mux = subprocess.run([ffmpeg_path(), "-hide_banner", "-muxers"],
                         capture_output=True, text=True).stdout
    assert "libx264" in enc
    assert " flv" in mux and " hls" in mux


@pytest.mark.parametrize("h,w", [(2160, 3840), (960, 1280), (1080, 1920), (480, 640), (720, 1280)])
def test_label_bar_keeps_an_even_frame_even(h, w):
    """H.264 rejects odd dimensions at encoder init, so the whole output never starts.

    Reported from production: a 1080p source upscaled x2 became 3840x2160, the bar added
    round(0.08*2160)=173, and libx264 refused 3840x2333. MJPEG does not care and cv2's mp4
    writer silently rounds down, so only the FLV/HLS/RTSP outputs actually broke -- and the
    640x480 test resolution happened to land on 528, even, which is why it went unnoticed.

    An odd-sized *source* is not this function's problem to solve (it would have to alter the
    picture); the crop filter in streams.LOW_LATENCY handles that.
    """
    import numpy as np

    from darkpipe.render import append_label_bar
    out = append_label_bar(np.zeros((h, w, 3), np.uint8), None)
    assert out.shape[0] % 2 == 0, f"{w}x{h} -> height {out.shape[0]} is odd"
    assert out.shape[0] > h, "the bar still has to be there"


def test_h264_outputs_force_even_dimensions():
    """The belt to render.py's braces: any odd source still has to encode."""
    from darkpipe.streams import LOW_LATENCY
    assert "-vf" in LOW_LATENCY
    assert "trunc(iw/2)*2:trunc(ih/2)*2" in LOW_LATENCY[LOW_LATENCY.index("-vf") + 1]


@pytest.mark.parametrize("w,h,cap,want", [
    (1920, 1080, 1280, (1280, 720)),      # the reported case: 1080p -> 720p
    (3840, 2160, 1280, (1280, 720)),
    (640, 480, 1280, (640, 480)),         # already small: untouched, never upscaled
    (1080, 1920, 1280, (720, 1280)),      # portrait: the LONG side is what is capped
    (1001, 777, 500, (500, 388)),         # odd arithmetic still lands on even dimensions
])
def test_downscale_caps_the_long_side(w, h, cap, want):
    import numpy as np

    from darkpipe.stages.downscale import DownscaleStage
    out = DownscaleStage(cap)([np.zeros((h, w, 3), np.uint8)])[0]
    assert (out.shape[1], out.shape[0]) == want
    assert out.shape[0] % 2 == 0 and out.shape[1] % 2 == 0, "H.264 needs even dimensions"


def test_downscale_is_wired_before_enhancement_and_not_dropped():
    """Both loops used to select stages by name prefix, which would have skipped this one."""
    from darkpipe.config import PipelineConfig
    from darkpipe.stages import build_stages
    cfg = PipelineConfig(input="x", enhance="off", sr="bicubic", sr_scale=2,
                         recognize="off", proc_max_side=1280)
    stages, _ = build_stages(cfg)
    names = [s.name for s in stages]
    assert names[0] == "downscale", f"downscale must run first, got {names}"
    srs = [s for s in stages if s.name.startswith("sr")]
    pre = [s for s in stages if s not in srs]
    assert any(s.name == "downscale" for s in pre), "would be silently skipped at runtime"


def test_downscale_absent_when_not_requested():
    from darkpipe.config import PipelineConfig
    from darkpipe.stages import build_stages
    stages, _ = build_stages(PipelineConfig(input="x", enhance="off", sr="off",
                                            recognize="off", proc_max_side=0))
    assert [s.name for s in stages] == []


@pytest.mark.parametrize("value,capped", [
    ("4M", True), ("8M", True), ("500k", True),
    ("", False), ("0", False), ("off", False), ("none", False), (" OFF ", False),
])
def test_bitrate_cap_can_be_turned_off_through_the_manifest(value, capped):
    """Turning the cap off has to be expressible by a field that cannot be empty.

    The cap was added on a misdiagnosis -- the green frames and dropped connection it was
    meant to explain were a network fault -- so leaving it un-disableable would silently
    hold every deployment to a softened picture. The platform makes a defaulted parameter
    mandatory and non-empty, so an empty string is not something the UI can send.
    """
    from darkpipe.streams import rate_limit
    assert bool(rate_limit(value)) is capped
