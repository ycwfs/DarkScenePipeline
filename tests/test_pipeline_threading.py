"""The offline runner decodes and encodes on worker threads; frames must survive it.

Decode -> GPU -> encode are three threads joined by bounded queues. The property that
matters is that nothing is dropped, duplicated, or reordered, so this runs the real
`run_offline` in passthrough mode (all stages off, no label bar) over a clip of frames
stamped with their own index, and asserts the output is the input, in order.
"""
import glob
import json
import os
import re
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

N_FRAMES, W, H = 77, 64, 48          # not a multiple of the 32-frame chunk, on purpose


def _stamp(i):
    """A frame whose pixels encode its index, so reordering is detectable."""
    f = np.zeros((H, W, 3), np.uint8)
    f[:, :, 0] = i % 256
    f[:, :, 1] = (i * 7) % 256
    f[:, :, 2] = (i * 13) % 256
    return f


@pytest.fixture
def clip(tmp_path):
    p = str(tmp_path / "in.mp4")
    # FFV1 in .avi would be lossless, but mp4v is what the pipeline writes; a flat-colour
    # frame survives it exactly, which is why the stamp is per-channel constant.
    vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))
    for i in range(N_FRAMES):
        vw.write(_stamp(i))
    vw.release()
    return p


def _read(path):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out


def test_offline_preserves_frame_order_and_count(clip, tmp_path):
    from darkpipe.config import PipelineConfig
    from darkpipe.pipeline import run_offline

    out = str(tmp_path / "out.mp4")
    run_offline(PipelineConfig(input=clip, output=out, enhance="off", sr="off",
                               recognize="off", device="cpu", no_label_bar=True))

    src, dst = _read(clip), _read(out)
    assert len(dst) == len(src) == N_FRAMES
    # mp4v is lossy (median round-trip error 2.0 / max 3.0 on these stamps), so asserting
    # pixel equality would be testing the codec. The property under test is ORDER: every
    # output frame's nearest neighbour among the source frames must be itself.
    s = np.stack([f.reshape(-1).astype(np.int16) for f in src])
    for i, b in enumerate(dst):
        d = np.abs(s - b.reshape(-1).astype(np.int16)).mean(1)
        assert int(d.argmin()) == i, f"output frame {i} matches input frame {d.argmin()}"


def test_max_frames_is_respected(clip, tmp_path):
    from darkpipe.config import PipelineConfig
    from darkpipe.pipeline import run_offline

    out = str(tmp_path / "out40.mp4")
    run_offline(PipelineConfig(input=clip, output=out, enhance="off", sr="off",
                               recognize="off", device="cpu", no_label_bar=True,
                               max_frames=40))
    assert len(_read(out)) == 40


def test_stage_failure_aborts_instead_of_hanging(clip, tmp_path, monkeypatch):
    """A stage that raises mid-clip must abort the run, not deadlock it.

    The main loop stops consuming the moment `process_chunk` raises, while the decode thread
    is still blocked on `in_q.put()` against a bounded queue -- so `dec.join()` used to wait
    forever and a CUDA OOM under `--sr-chunk 8` presented as a 17-minute hang at 0% GPU
    instead of a traceback. Asserted with a join timeout because the failure mode is a hang,
    which no ordinary assertion can catch.
    """
    import threading

    from darkpipe import pipeline
    from darkpipe.config import PipelineConfig

    class Boom:
        name, whole_video = "enhance:boom", False

        def __init__(self):
            self.n = 0

        def load(self, device):
            pass

        def __call__(self, frames):
            self.n += 1
            if self.n == 2:          # fail early, with the decoder still filling the queue
                raise RuntimeError("boom")
            return frames

        def close(self):
            pass

    monkeypatch.setattr(pipeline, "build_stages", lambda cfg: ([Boom()], None))
    caught = []

    def go():
        try:
            pipeline.run_offline(PipelineConfig(
                input=clip, output=str(tmp_path / "boom.mp4"), enhance="off", sr="off",
                recognize="off", device="cpu", no_label_bar=True, enhance_chunk=8))
        except BaseException as e:          # noqa: BLE001 - the point is that it arrives
            caught.append(e)

    t = threading.Thread(target=go, daemon=True)
    t.start()
    t.join(timeout=30)
    assert not t.is_alive(), "run_offline deadlocked after a stage error"
    assert caught and "boom" in str(caught[0])


def test_decode_error_surfaces_on_the_main_thread(tmp_path):
    """A reader that raises must not be swallowed by the worker thread."""
    from darkpipe.config import PipelineConfig
    from darkpipe.pipeline import run_offline

    with pytest.raises(Exception):
        run_offline(PipelineConfig(input=str(tmp_path / "nope.mp4"),
                                   output=str(tmp_path / "o.mp4"), enhance="off", sr="off",
                                   recognize="off", device="cpu", no_label_bar=True))


def test_offline_clip_recording_uses_time_range_names_and_respects_skip_labels(
        clip, tmp_path, monkeypatch):
    """clip_dir + a recognizer, end to end: `run_offline` must feed its own ClipRecorder
    from the same (frame, event, ts) the output video is built from, name the result by
    source-timeline position (see naming="range" in darkpipe/clips.py), and still honour
    clip_skip_labels -- `other` must not itself become a clip, nor keep one open forever.
    """
    from darkpipe import pipeline
    from darkpipe.config import PipelineConfig

    class FakeEvent:
        def __init__(self, label, confidence=0.9):
            self.label, self.confidence = label, confidence

        def to_dict(self):
            return {"label": self.label, "confidence": self.confidence}

    class FakeRecognizer:
        name, window, stride = "fake", 1, 1

        def load(self, device):
            pass

        def close(self):
            pass

        def push(self, frame, idx, ts):
            # first 1s (30 of 77 frames @ 30fps) "Drinking water", the rest "Other".
            return FakeEvent("Drinking water" if idx < 30 else "Other")

    monkeypatch.setattr(pipeline, "build_stages", lambda cfg: ([], FakeRecognizer()))
    clip_dir = str(tmp_path / "clips")
    cfg = PipelineConfig(input=clip, output=str(tmp_path / "out.mp4"), enhance="off", sr="off",
                         recognize="behavior", device="cpu", no_label_bar=True,
                         clip_dir=clip_dir, clip_pre_sec=0.1, clip_post_sec=0.1,
                         clip_session="s1")
    events = pipeline.run_offline(cfg)
    assert events, "the fake recognizer must have produced events for the clipper to see"
    assert os.path.exists(cfg.output)

    got = sorted(glob.glob(os.path.join(clip_dir, "s1", "*.mp4")))
    assert len(got) == 1, f"expected exactly one clip (Other must not start its own): {got}"
    fname = os.path.basename(got[0])
    m = re.match(r"^(\d+)-(\d+)s-喝水\.mp4$", fname)
    assert m, f"unexpected clip filename: {fname}"
    start_s, end_s = int(m.group(1)), int(m.group(2))

    meta = json.load(open(os.path.splitext(got[0])[0] + ".json", encoding="utf-8"))
    assert meta["label"] == "Drinking water"
    assert meta["start_seconds"] is not None and meta["end_seconds"] is not None
    # The window must close shortly after the last qualifying event (idx 29, ~1s in) plus
    # clip_post_sec padding -- long before the "Other" tail would run the clip to EOF.
    assert start_s == 0 and end_s <= 2, f"clip ran past the drinking-water portion: {meta}"
    assert cfg.stats["clips"]["clips_saved"] == 1
