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
