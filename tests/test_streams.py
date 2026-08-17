"""Live-output format selection, and the failure that must happen at startup.

Every format except mjpeg is muxed by an ffmpeg subprocess. The bad outcome is not "flv is
unavailable" -- it is a service that starts happily, reports healthy, serves the demo page,
and only fails when someone finally opens the FLV URL during a demonstration. So a missing
binary or a misspelled format has to stop the process at startup, naming what is wrong.
"""
import io
import time

import pytest

from darkpipe import streams
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


def _probs(**kw):
    """Build a BEHAVIORS-shaped distribution from named entries."""
    import numpy as np

    from darkpipe.constants import BEHAVIORS
    p = np.zeros(len(BEHAVIORS), np.float32)
    for k, v in kw.items():
        p[BEHAVIORS.index(k)] = v
    return p


@pytest.mark.parametrize("tau,top,conf,expect", [
    (0.0,  "throw", 0.31, "throw"),    # off: the weak guess is reported as-is
    (0.5,  "throw", 0.31, "other"),    # below tau -> other
    (0.5,  "throw", 0.83, "throw"),    # above tau -> kept
    (0.5,  "throw", 0.50, "throw"),    # exactly tau counts as clearing it
    (0.95, "fall",  0.83, "other"),    # tau above 0.5 must still make `other` the argmax
])
def test_confidence_threshold_demotes_weak_guesses_to_other(tau, top, conf, expect):
    """The reported label is what everything downstream keys on.

    Demoting to `other` (rather than filtering separately) is what makes one threshold cover
    all three consumers at once: the bar shows 其他, no clip is cut because `other` is in
    clip_skip_labels, and the event log still records the window happened.
    """
    import numpy as np

    from darkpipe.constants import BEHAVIORS
    from darkpipe.stages.recognize import reject_probs
    # `other` deliberately small: the named guess is the model's own argmax, so the
    # threshold is the only thing that can change the answer.
    p = _probs(**{top: conf, "other": 0.1})
    out = reject_probs(p, BEHAVIORS, tau)
    assert BEHAVIORS[int(np.argmax(out))] == expect


def test_threshold_leaves_arid_heads_alone():
    """The ARID label set has no `other`, so there is nothing to demote to."""
    import numpy as np

    from darkpipe.constants import CLASSES
    from darkpipe.stages.recognize import reject_probs
    p = np.zeros(len(CLASSES), np.float32); p[CLASSES.index("Jump")] = 0.2
    assert np.array_equal(reject_probs(p, CLASSES, 0.9), p)


def test_threshold_is_idempotent():
    """xclip applies it inside _infer and the base push() applies it again."""
    from darkpipe.constants import BEHAVIORS
    from darkpipe.stages.recognize import reject_probs
    once = reject_probs(_probs(throw=0.31, other=0.1), BEHAVIORS, 0.5)
    assert (reject_probs(once, BEHAVIORS, 0.5) == once).all()


@pytest.mark.parametrize("factor", [1.4, 2.0, 2.6, 0.6])
def test_saturation_changes_chroma_but_not_hue(factor):
    """Scaling (a,b) in Lab moves chroma and leaves hue essentially where it was.

    That is what separates this from the colourisation model that was tried first, which
    scored 66 degrees of hue deviation because it discards the input's colour and invents
    one from luminance. Centring on the frame mean (rather than on neutral, so that a
    global cast is not amplified) costs about half a degree of exactness; the bound here is
    loose enough to cover that and tight enough to catch a real hue change.
    """
    import cv2
    import numpy as np

    from darkpipe.stages.saturation import SaturationStage
    # Chroma like a real enhanced frame (~6-20), not random RGB noise (~43). At 43 a 2.6x
    # boost clips the 8-bit range, and clipping is the one thing that does move hue -- so
    # noise would be testing the clip path, not the property.
    rng = np.random.default_rng(0)
    base = np.full((64, 64, 3), 110, np.int16)
    img = np.clip(base + rng.integers(-22, 22, (64, 64, 3)), 0, 255).astype(np.uint8)
    out = SaturationStage(factor)([img])[0]

    def lab(x):
        return cv2.cvtColor(x, cv2.COLOR_BGR2LAB).astype(np.float32)

    a, b = lab(img), lab(out)
    ca = np.hypot(a[..., 1] - 128, a[..., 2] - 128)
    cb_all = np.hypot(b[..., 1] - 128, b[..., 2] - 128)
    # Both sides must retain real chroma: at 0.6x the result is nearly neutral, where the
    # hue angle of an 8-bit value is dominated by quantisation rather than by the operation.
    m = (ca > 8) & (cb_all > 8)
    ha = np.arctan2(a[..., 2] - 128, a[..., 1] - 128)[m]
    hb = np.arctan2(b[..., 2] - 128, b[..., 1] - 128)[m]
    dev = np.abs(np.degrees(np.arctan2(np.sin(hb - ha), np.cos(hb - ha)))).mean()
    assert dev < 4.0, f"hue moved {dev:.1f} deg; scaling a/b must leave it ~alone"
    cb = np.hypot(b[..., 1] - 128, b[..., 2] - 128)
    assert bool(cb[m].mean() > ca[m].mean()) == (factor > 1.0)


def test_saturation_runs_after_recognition_not_before():
    """Recognition must see the enhancer's output as the head was trained on it."""
    from darkpipe.config import PipelineConfig
    from darkpipe.stages import build_stages
    stages, _ = build_stages(PipelineConfig(
        input="x", enhance="off", sr="bicubic", sr_scale=2, recognize="off",
        color_saturation=2.2))
    sat = [s for s in stages if s.name.startswith("saturate")]
    assert sat and sat[0].post_recognition, "boosting colour before recognition shifts the " \
                                            "input distribution away from training"


def test_saturation_absent_at_1x():
    from darkpipe.config import PipelineConfig
    from darkpipe.stages import build_stages
    stages, _ = build_stages(PipelineConfig(input="x", enhance="off", sr="off",
                                            recognize="off", color_saturation=1.0))
    assert not [s for s in stages if "saturate" in s.name]


@pytest.mark.parametrize("bad", [0.0, -1.0, 6.0])
def test_saturation_out_of_range_is_rejected(bad):
    from darkpipe.config import PipelineConfig, validate
    with pytest.raises(SystemExit):
        validate(PipelineConfig(input="x.mp4", output="/tmp/o.mp4", enhance="off", sr="off",
                                recognize="off", color_saturation=bad))


def test_label_bar_fps_is_the_configured_stream_rate():
    """The burnt-in number is the viewer's frame rate, not the GPU's.

    They are different quantities: the feeder resamples onto a fixed max_stream_fps cadence,
    repeating the last frame when the pipeline runs slower, so the stream really is delivered
    at that rate. The processing rate is still reported -- as fps_proc in /health, where it is
    a diagnostic rather than a number burnt into a monitoring wall.
    """
    import inspect

    from darkpipe import server
    src = inspect.getsource(server.process_loop)
    assert "max_stream_fps:g} fps" in src, "label bar should show the configured rate"
    assert "fps_proc:.1f} fps" not in src, "measured rate must not be burnt into the picture"


@pytest.mark.parametrize("factor", [2.0, 2.6])
def test_saturation_does_not_amplify_a_colour_cast(factor):
    """Regression: a deployment reported the picture turning green at 2x.

    Low-light footage often carries a green cast (a Bayer sensor has twice as many green
    photosites as red or blue). Scaling chroma about the neutral point multiplies that cast
    along with everything else, so 2x colour was also 2x green. Centring on the frame's own
    mean boosts the same amount without touching the average colour.
    """
    import cv2
    import numpy as np

    from darkpipe.stages.saturation import SaturationStage
    img = np.full((80, 80, 3), 120, np.uint8)
    img[:, :30] = (150, 110, 105)
    img[:, 55:] = (90, 130, 190)
    img[..., 1] = np.clip(img[..., 1].astype(np.int16) + 14, 0, 255)   # green cast

    def stats(x):
        lab = cv2.cvtColor(x, cv2.COLOR_BGR2LAB).astype(np.float32)
        return (lab[..., 1].mean() - 128, lab[..., 2].mean() - 128,
                float(np.hypot(lab[..., 1] - 128, lab[..., 2] - 128).mean()))

    a0, b0, c0 = stats(img)
    a1, b1, c1 = stats(SaturationStage(factor)([img])[0])
    assert c1 > c0 * 1.5, f"chroma barely moved: {c0:.1f} -> {c1:.1f}"
    for name, before, after in (("a", a0, a1), ("b", b0, b1)):
        if abs(before) > 1.0:
            assert abs(after) < abs(before) * 1.35, \
                f"mean {name} cast grew {before:+.2f} -> {after:+.2f} (x{after/before:.2f})"


@pytest.mark.parametrize("given,effective", [
    (0.0, 0.0), (0.6, 0.6), (1.0, 1.0),
    (60, 0.6), (95, 0.95), (50, 0.5),      # "60" means 60%, the way people say it
])
def test_threshold_accepts_percentages(given, effective):
    """A threshold typed as 60 must mean 60%, not 6000%.

    Users say "threshold 60" and the field is called a threshold. Read literally, 60 makes
    it impossible for any window to clear the bar, so every detection silently becomes
    `other` and the model looks broken. A probability cannot exceed 1, so values above 1 are
    unambiguous.
    """
    from darkpipe.config import PipelineConfig, validate
    cfg = validate(PipelineConfig(input="x.mp4", output="/tmp/o.mp4", enhance="off",
                                  sr="off", recognize="behavior", reco_min_conf=given))
    assert cfg.reco_min_conf == pytest.approx(effective)


@pytest.mark.parametrize("bad", [-0.1, 101, 1000])
def test_threshold_out_of_range_is_rejected(bad):
    from darkpipe.config import PipelineConfig, validate
    with pytest.raises(SystemExit):
        validate(PipelineConfig(input="x.mp4", output="/tmp/o.mp4", enhance="off", sr="off",
                                recognize="behavior", reco_min_conf=bad))


def test_no_named_label_can_survive_below_the_threshold():
    """The reported symptom: "投掷物品 40%" while the threshold was 60.

    Sweeps the whole distribution space rather than a few cases -- whatever the shape, a
    named behaviour must never be the reported label unless it cleared tau.
    """
    import numpy as np

    from darkpipe.constants import BEHAVIORS
    from darkpipe.stages.recognize import reject_probs
    rng = np.random.default_rng(0)
    tau = 0.6
    for _ in range(2000):
        p = rng.random(len(BEHAVIORS)).astype(np.float32)
        p /= p.sum()
        out = reject_probs(p.copy(), BEHAVIORS, tau)
        top = BEHAVIORS[int(np.argmax(out))]
        if top != "other":
            assert p[int(np.argmax(out))] >= tau, \
                f"reported {top} at {p[int(np.argmax(out))]:.2f}, below tau {tau}"


def test_reader_survives_a_transient_decode_failure(tmp_path, monkeypatch):
    """Regression: a 602-frame clip silently produced a 536-frame video.

    One failed read used to end the whole stream. It is not proof of EOF -- the same file,
    code and container read to the end on the next run. The damage is that the run still
    reports success: the summary quotes its own short count, so 11% of the footage vanished
    with nothing to indicate it.
    """
    import numpy as np

    from darkpipe.media import VideoReader

    class FlakyCap:
        def __init__(self, total, fail_at):
            self.i, self.total, self.fail_at, self.failed = 0, total, fail_at, False

        def get(self, prop):
            return float(self.total) if int(prop) == 7 else 25.0

        def read(self):
            if self.i == self.fail_at and not self.failed:
                self.failed = True
                return False, None            # one transient miss, mid-stream
            if self.i >= self.total:
                return False, None
            self.i += 1
            return True, np.zeros((4, 4, 3), np.uint8)

        def release(self):
            pass

    monkeypatch.setattr("darkpipe.media.open_capture", lambda src: FlakyCap(100, 40))
    r = VideoReader("fake.mp4", chunk=4)
    assert sum(len(c) for c in r.chunks()) == 100, "a transient miss truncated the stream"
    assert r.recovered == 1


def test_reader_stops_at_a_real_end(monkeypatch):
    """The retry must not turn a finished file into an infinite loop."""
    import numpy as np

    from darkpipe.media import VideoReader

    class Cap:
        def __init__(self):
            self.i = 0

        def get(self, prop):
            return 10.0 if int(prop) == 7 else 25.0

        def read(self):
            if self.i >= 10:
                return False, None
            self.i += 1
            return True, np.zeros((4, 4, 3), np.uint8)

        def release(self):
            pass

    monkeypatch.setattr("darkpipe.media.open_capture", lambda src: Cap())
    r = VideoReader("fake.mp4", chunk=3)
    assert sum(len(c) for c in r.chunks()) == 10


@pytest.mark.parametrize("mode,expect_change", [
    ("off", False), ("fast", True), ("quality", True), ("quality_high", True),
])
def test_denoise_modes_apply(mode, expect_change):
    import numpy as np

    from darkpipe.stages.denoise import DenoiseStage
    rng = np.random.default_rng(0)
    img = np.clip(np.full((64, 64, 3), 110, np.int16)
                  + rng.integers(-25, 25, (64, 64, 3)), 0, 255).astype(np.uint8)
    out = DenoiseStage(mode)([img])[0]
    assert (not np.array_equal(out, img)) is expect_change
    assert out.shape == img.shape


def test_denoise_runs_after_recognition_and_before_saturation():
    """Order matters twice over.

    After recognition, because the behaviour head was trained on un-denoised enhancer output.
    Before saturation, because boosting chroma amplifies chroma noise -- removing it first is
    both cheaper and cleaner.
    """
    from darkpipe.config import PipelineConfig
    from darkpipe.stages import build_stages
    stages, _ = build_stages(PipelineConfig(
        input="x", enhance="off", sr="bicubic", sr_scale=2, recognize="off",
        denoise="fast", color_saturation=2.0))
    names = [s.name for s in stages]
    dn = next(i for i, n in enumerate(names) if n.startswith("denoise"))
    sat = next(i for i, n in enumerate(names) if n.startswith("saturate"))
    assert stages[dn].post_recognition, "denoising before recognition shifts the input distribution"
    assert dn < sat, f"denoise must precede saturation, got {names}"


def test_clip_denoise_is_independent_of_the_stream():
    """The live stream and the clips share one frame, so clips need their own knob.

    Anything strong enough to matter (NLM, 119-376 ms) cannot go in the shared path without
    spending the whole latency budget -- but the clip writer thread is already off it.
    """
    import numpy as np

    from darkpipe.clips import ClipRecorder
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        rec = ClipRecorder(d, denoise="fast")
        # Needs real noise: a bilateral filter on a perfectly flat image correctly returns
        # it unchanged, so a clean test image would pass whether or not it ran.
        rng = np.random.default_rng(0)
        img = np.clip(np.full((32, 32, 3), 100, np.int16)
                      + rng.integers(-25, 25, (32, 32, 3)), 0, 255).astype(np.uint8)
        assert not np.array_equal(rec._clean(img), img), "clip denoise did not apply"
        plain = ClipRecorder(d, denoise="off")
        assert np.array_equal(plain._clean(img), img)
        rec.close(); plain.close()


@pytest.mark.parametrize("bad", ["nlm", "high", "true", "1"])
def test_denoise_rejects_unknown_modes(bad):
    from darkpipe.config import PipelineConfig, validate
    with pytest.raises(SystemExit):
        validate(PipelineConfig(input="x.mp4", output="/tmp/o.mp4", enhance="off",
                                sr="off", recognize="off", denoise=bad))


# --- push telemetry ---------------------------------------------------------------------
#
# The deployment pushes RTSP to a platform address and the platform re-serves it as FLV to a
# browser. When that browser stalls, the only question worth answering is which side of the
# push dropped it -- and `[process]` cannot answer it, because the JPEG slot is latest-wins
# and the process loop never blocks on an output. These counters are the evidence.

class _FakeStdin:
    def __init__(self, block=0.0):
        self.block, self.written = block, []

    def write(self, b):
        time.sleep(self.block)
        self.written.append(b)

    def flush(self):
        pass

    def close(self):
        pass


class _FakeProc:
    """Stands in for ffmpeg: never exits, and its stdin blocks for a configurable time."""

    def __init__(self, block=0.0, err=b""):
        self.stdin = _FakeStdin(block)
        self.stderr = io.BytesIO(err)
        self.stdout = None

    def poll(self):
        return None

    def terminate(self):
        pass

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


class _FixedSlot:
    """A LatestSlot whose sequence never advances -- i.e. a stalled pipeline."""

    def __init__(self, item=b"\xff\xd8jpeg\xff\xd9", seq=1):
        self.item, self.seq = item, seq

    def get(self):
        return self.item, self.seq


@pytest.fixture
def fake_ffmpeg(monkeypatch):
    def install(block=0.0, err=b""):
        proc = _FakeProc(block, err)
        monkeypatch.setattr(streams, "ffmpeg_path", lambda: "/bin/true")
        monkeypatch.setattr(streams.subprocess, "Popen", lambda *a, **k: proc)
        return proc
    return install


def test_snap_measures_time_blocked_writing_to_ffmpeg(fake_ffmpeg):
    """Back-pressure from the media server surfaces in stdin.write and nowhere else.

    A 1080p JPEG is several hundred KB against a 64 KB pipe, so the write already waits on
    ffmpeg; when the server downstream stops draining, that wait is the whole signal.
    """
    fake_ffmpeg(block=0.05)
    out = streams.FFmpegOut(_FixedSlot(), [], fps=50.0)
    time.sleep(0.6)
    c = out.snap()
    out.close()
    assert c["frames"] >= 3, "feeder should have written several frames"
    assert c["blocked"] > 0.1, f"blocked time not accumulated: {c}"
    assert c["worst_ms"] >= 40, f"worst write not recorded: {c}"


def test_snap_counts_repeated_frames_when_the_pipeline_lags(fake_ffmpeg):
    """The cadence is fixed, so a slower pipeline is pushed as duplicates, not as fewer fps."""
    fake_ffmpeg()
    out = streams.FFmpegOut(_FixedSlot(), [], fps=50.0)
    time.sleep(0.3)
    c = out.snap()
    out.close()
    assert c["dup"] >= c["frames"] - 1 > 0, f"repeats not counted: {c}"


def test_the_cadence_debt_a_long_write_leaves_is_dropped_not_repaid(fake_ffmpeg):
    """Debt beyond one step is forgiven, so the stream cannot fall behind wall clock.

    Repaying it would write the backlog back to back at sleep(0) while ffmpeg keeps stamping
    frames 1/fps apart, so the timeline drifts behind real time without bound -- deployment
    measured `lag` climbing monotonically to 30 s over ~11 min at 30 fps requested / 28.8
    sustainable. Dropping it instead pins the stream to the live edge.
    """
    fake_ffmpeg(block=0.05)             # 50 ms writes against a 10 ms cadence
    out = streams.FFmpegOut(_FixedSlot(), [], fps=100.0)
    time.sleep(0.5)
    c = out.snap()
    out.close()
    # Bounded by one step (10 ms) plus the write that overran it (50 ms), never accumulating.
    assert c["lag"] < 0.1, f"debt was repaid instead of dropped: {c}"
    # And the shortfall is reported rather than silently absorbed: writes take 5x the step,
    # so ~80% of the timeline has to go.
    assert c["skipped"] > 0.2, f"dropped timeline not reported: {c}"


def test_a_feeder_that_keeps_up_drops_nothing(fake_ffmpeg):
    """The re-anchor must not fire on a healthy push, or `dropped` stops meaning anything."""
    fake_ffmpeg(block=0.001)            # 1 ms writes against a 20 ms cadence
    out = streams.FFmpegOut(_FixedSlot(), [], fps=50.0)
    time.sleep(0.4)
    c = out.snap()
    out.close()
    assert c["skipped"] == 0.0, f"healthy push dropped timeline: {c}"
    assert c["frames"] > 5, f"nothing was written at all: {c}"


def test_snap_stall_grows_while_nothing_is_written(fake_ffmpeg):
    """The reason the telemetry line is printed by someone else.

    A feeder wedged inside stdin.write emits nothing at all, so `stall` -- time since the
    last successful write -- is the only counter that moves when the push is hung.
    """
    fake_ffmpeg()

    class _Empty:
        def get(self):
            return None, 0

    out = streams.FFmpegOut(_Empty(), [], fps=50.0)
    time.sleep(0.2)
    c = out.snap()
    out.close()
    assert c["frames"] == 0 and c["stall"] == 0.0, f"nothing was ever written: {c}"


def test_repeated_ffmpeg_warnings_are_collapsed(fake_ffmpeg, capsys):
    """ffmpeg emits some warnings once per frame; at 15 fps that buries the operator log."""
    fake_ffmpeg(err=b"same\nsame\nsame\ndifferent\n")
    out = streams.FFmpegOut(_FixedSlot(), [], fps=50.0)
    time.sleep(0.2)
    out.close()
    printed = capsys.readouterr().out
    assert printed.count("[ffmpeg] same") == 1, "duplicates should print once"
    assert "上一条重复 2 次" in printed, "the suppressed count must still be reported"
    assert "[ffmpeg] different" in printed, "a changed message must get through"


def test_push_is_louder_than_the_other_outputs(fake_ffmpeg):
    """The push leaves the container, so its failures are the unreproducible ones.

    At -loglevel error ffmpeg says nothing about a connection that degrades without dying,
    which is the case the deployment actually hit.
    """
    fake_ffmpeg()
    push = streams.rtmp_push(_FixedSlot(), 15.0, "rtsp://example/live")
    cmd = push._cmd
    push.close()
    assert cmd[cmd.index("-loglevel") + 1] == "warning"
    assert cmd[cmd.index("-rtsp_transport") + 1] == "tcp", "UDP silently goes nowhere via NAT"
    assert cmd[cmd.index("-f", cmd.index("pipe:0")) + 1] == "rtsp"
