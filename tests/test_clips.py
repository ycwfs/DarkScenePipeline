"""ClipRecorder segmentation rules, driven by synthetic frames -- no GPU, no video decode.

The recorder decides *what* gets cut out of a live stream, and every one of its rules is
the kind that fails silently in production: a wrong skip list quietly fills the disk with
`other`, a missing merge rule turns one fall into a dozen overlapping files, a missing
pre-roll saves the aftermath instead of the behaviour. All of those still "work" -- they
just produce useless clips hours later on someone else's NFS mount.
"""
import glob
import json
import os

import numpy as np
import pytest

from darkpipe.clips import ClipRecorder, label_key


class FakeEvent:
    def __init__(self, label, confidence=0.9):
        self.label = label
        self.confidence = confidence

    def to_dict(self):
        return {"label": self.label, "confidence": self.confidence}


def frame():
    return np.zeros((64, 64, 3), np.uint8)


def drive(rec, script, fps=10.0):
    """script: [(label|None, repeat)] -> pushes frames at a steady 1/fps cadence."""
    t = 0.0
    for label, repeat in script:
        for _ in range(repeat):
            rec.push(frame(), FakeEvent(label) if label else None, t)
            t += 1.0 / fps
    return t


def clips_in(d):
    return sorted(glob.glob(os.path.join(d, "*", "*", "*.mp4")))


def test_other_never_starts_a_clip(tmp_path):
    rec = ClipRecorder(str(tmp_path), pre_sec=0.5, post_sec=0.5)
    drive(rec, [("Other", 30)])
    rec.close()
    assert clips_in(str(tmp_path)) == [], "`other` is the idle state; it must not be recorded"


def test_named_behavior_writes_one_clip_with_preroll(tmp_path):
    rec = ClipRecorder(str(tmp_path), pre_sec=1.0, post_sec=0.3)
    drive(rec, [(None, 20), ("Falling", 1), (None, 10)])   # 2 s of lead-in, then a trigger
    rec.close()
    got = clips_in(str(tmp_path))
    assert len(got) == 1
    assert "falling" in got[0], "clips are filed under a filesystem-safe label key"
    meta = json.load(open(os.path.splitext(got[0])[0] + ".json", encoding="utf-8"))
    # 1 s of pre-roll at 10 fps is ~10 frames, none of which exist if the recorder only
    # started writing at the trigger.
    assert meta["frames_written"] >= 8, f"pre-roll missing: {meta['frames_written']} frames"
    assert meta["label"] == "Falling"


def test_consecutive_events_merge_into_one_clip(tmp_path):
    rec = ClipRecorder(str(tmp_path), pre_sec=0.2, post_sec=1.0)
    # A recognizer fires every `stride` frames, so a sustained behaviour looks like this.
    drive(rec, [(None, 5)] + [("Fighting", 1), (None, 4)] * 6 + [(None, 20)])
    rec.close()
    assert len(clips_in(str(tmp_path))) == 1, \
        "six events during one continuous fight must not produce six overlapping files"


def test_gap_longer_than_post_sec_splits_clips(tmp_path):
    rec = ClipRecorder(str(tmp_path), pre_sec=0.2, post_sec=0.5)
    drive(rec, [("Waving", 1), (None, 20), ("Waving", 1), (None, 20)])   # 2 s of silence
    rec.close()
    assert len(clips_in(str(tmp_path))) == 2, "two separate incidents are two clips"


def test_max_sec_caps_a_never_ending_scene(tmp_path):
    rec = ClipRecorder(str(tmp_path), pre_sec=0.2, post_sec=5.0, max_sec=1.0)
    drive(rec, [("Chasing", 1), (None, 4)] * 10)      # 5 s of unbroken activity
    rec.close()
    got = clips_in(str(tmp_path))
    assert len(got) > 1, "a permanently active scene must not grow one unbounded file"
    for c in got:
        meta = json.load(open(os.path.splitext(c)[0] + ".json", encoding="utf-8"))
        assert meta["duration_seconds"] <= 1.5, f"{c} ran past clip_max_sec"


def test_min_confidence_filters_weak_events(tmp_path):
    rec = ClipRecorder(str(tmp_path), pre_sec=0.2, post_sec=0.3, min_confidence=0.8)
    t = 0.0
    for _ in range(20):
        rec.push(frame(), FakeEvent("Talking", confidence=0.4), t)
        t += 0.1
    rec.close()
    assert clips_in(str(tmp_path)) == []


def test_on_saved_receives_local_paths(tmp_path):
    seen = []
    rec = ClipRecorder(str(tmp_path), pre_sec=0.2, post_sec=0.3,
                       on_saved=lambda files, rel, meta: seen.append((files, rel)))
    drive(rec, [("Drinking water", 1), (None, 10)])
    rec.close()
    assert len(seen) == 1
    files, rel = seen[0]
    assert rel == "drinking_water"
    assert [os.path.splitext(f)[1] for f in files] == [".mp4", ".json"]
    assert all(os.path.exists(f) for f in files)


def test_on_saved_failure_does_not_kill_the_recorder(tmp_path):
    """A dead HDFS must not stop a service from recording locally."""
    def boom(files, rel, meta):
        raise RuntimeError("hdfs is down")

    rec = ClipRecorder(str(tmp_path), pre_sec=0.2, post_sec=0.3, on_saved=boom)
    drive(rec, [("Falling", 1), (None, 10), ("Falling", 1), (None, 10)])
    rec.close()
    assert len(clips_in(str(tmp_path))) == 2
    assert rec.saved == 2


def test_startup_stall_does_not_wreck_the_fps_estimate(tmp_path):
    """Regression: the first clip of a live run played back at 0.54x speed.

    CUDA warmup makes the first few processed frames arrive seconds apart. Those gaps flush
    the (time-bounded) pre-roll to a single frame, and estimating the rate from that snapshot
    fell back to a hardcoded 15 fps while the pipeline was really running at 28 -- so the
    clip's mp4 header said 15 and it played at half speed. Measured on a real run before the
    fix; this reproduces the shape of it.
    """
    rec = ClipRecorder(str(tmp_path), pre_sec=2.0, post_sec=0.5)
    t = 0.0
    # Warmup gaps longer than pre_sec, so each one flushes the pre-roll back to one frame
    # and none of them counts as a frame interval -- at the trigger there is nothing at all
    # to estimate from. This is the exact state the first live clip started in.
    for _ in range(3):
        rec.push(frame(), None, t)
        t += 3.0
    rec.push(frame(), FakeEvent("Falling"), t)          # fires immediately after the stall
    t += 1 / 28.0
    for _ in range(60):                                 # then the pipeline settles at 28 fps
        rec.push(frame(), None, t)
        t += 1 / 28.0
    rec.close()
    meta = json.load(open(os.path.splitext(clips_in(str(tmp_path))[0])[0] + ".json",
                          encoding="utf-8"))
    assert 25 <= meta["fps"] <= 30, f"header fps {meta['fps']} should track the real 28"
    assert 0.85 <= meta["playback_speed"] <= 1.15, \
        f"clip plays at {meta['playback_speed']}x real time"


def test_only_finished_clips_appear_in_the_output_directory(tmp_path):
    """out_dir is normally an NFS share someone browses, so it gets whole files only.

    Encoding straight into it would leave a half-written mp4 growing in the folder for the
    length of every incident (up to clip_max_sec), and would put NFS latency on the writer
    thread, where a stall costs dropped frames. Clips are staged locally and moved when done.
    """
    stage = tmp_path / "stage"
    out = tmp_path / "out"
    rec = ClipRecorder(str(out), pre_sec=0.2, post_sec=5.0, max_sec=60,
                       stage_dir=str(stage))
    t = 0.0
    rec.push(frame(), FakeEvent("Falling"), t)
    for _ in range(40):                       # mid-recording: nothing must be visible yet
        t += 0.05
        rec.push(frame(), None, t)
    import time as _t
    _t.sleep(0.2)                             # let the writer thread catch up
    assert clips_in(str(out)) == [], "a partially written clip leaked into the browse directory"
    rec.close()
    assert len(clips_in(str(out))) == 1, "the finished clip should be moved into place"
    left = [f for f in os.listdir(stage)] if stage.exists() else []
    assert left == [], f"staging directory not cleaned up: {left}"


@pytest.mark.parametrize("rate_pattern", [
    [30.0] * 80,                       # steady
    [30.0] * 25 + [20.0] * 55,         # sags once encoding starts -- the measured live case
    [30.0] * 25 + [8.0] * 55,          # sags hard (720p, or an upload hogging the box)
    [10.0] * 40 + [30.0] * 40,         # speeds up
])
def test_clip_duration_matches_wall_clock_whatever_the_rate_does(tmp_path, rate_pattern):
    """A 12 s incident must produce a 12 s video, at any capture rate.

    Estimating a header fps cannot deliver this: every estimate is taken from a prefix, and
    the rate sags later (measured 30 fps sampled vs 21.6 sustained -> 1.41x playback). The
    writer therefore resamples onto a fixed grid, which makes the duration right by
    construction rather than by prediction.
    """
    rec = ClipRecorder(str(tmp_path), pre_sec=0.5, post_sec=0.4, max_sec=600)
    t = 0.0
    rec.push(frame(), FakeEvent("Falling"), t)
    for i, fps in enumerate(rate_pattern):
        t += 1.0 / fps
        rec.push(frame(), FakeEvent("Falling") if i % 10 == 0 else None, t)
    rec.close()
    meta = json.load(open(os.path.splitext(clips_in(str(tmp_path))[0])[0] + ".json",
                          encoding="utf-8"))
    played = meta["frames_written"] / meta["fps"]
    assert abs(played - meta["duration_seconds"]) <= 0.35, (
        f"clip covers {meta['duration_seconds']:.2f}s but plays for {played:.2f}s "
        f"({meta['playback_speed']}x)")


def test_event_log_records_every_event_including_other(tmp_path):
    """The clip sidecars only cover clipped behaviours; the log is the complete record.

    That matters when `/events` is unreachable -- the spec gives an operator no way to declare
    a published port, so a deployment without one has no live event stream at all, and this
    file becomes the only place `other` (or anything in clip_skip_labels) is written down.
    """
    from darkpipe.clips import EventLog
    log = EventLog(str(tmp_path / "sub" / "events.jsonl"))
    for label in ("Other", "Falling", "Other", "Waving"):
        log.write(FakeEvent(label, confidence=0.7))
    log.close()
    lines = [json.loads(l) for l in
             open(tmp_path / "sub" / "events.jsonl", encoding="utf-8") if l.strip()]
    assert [r["label"] for r in lines] == ["Other", "Falling", "Other", "Waving"]
    assert all("wall_time" in r for r in lines), "a line without a clock is hard to correlate"
    assert log.stats()["events_written"] == 4


def test_event_log_appends_rather_than_truncates(tmp_path):
    """A restart within the same session must not wipe what is already recorded."""
    from darkpipe.clips import EventLog
    p = str(tmp_path / "events.jsonl")
    first = EventLog(p)
    first.write(FakeEvent("Falling"))
    first.close()
    second = EventLog(p)
    second.write(FakeEvent("Waving"))
    second.close()
    assert len(open(p, encoding="utf-8").read().strip().splitlines()) == 2


def test_event_log_write_never_blocks_the_caller(tmp_path):
    """Overflow drops and counts, it does not stall the recognition loop."""
    from darkpipe.clips import EventLog
    log = EventLog(str(tmp_path / "e.jsonl"), queue_size=16)
    for _ in range(5000):
        log.write(FakeEvent("Falling"))       # far faster than the writer can drain
    log.close()
    assert log.written + log.dropped == 5000, "every event is either written or counted as lost"


@pytest.mark.parametrize("label,key", [
    ("Picking up object", "picking_up_object"), ("Other", "other"),
    ("Drinking water", "drinking_water"), ("shake_hands", "shake_hands"), ("", "unknown"),
])
def test_label_key_is_filesystem_safe(label, key):
    assert label_key(label) == key


def test_skip_list_matches_regardless_of_spelling(tmp_path):
    """The manifest's skip list is raw labels; events carry display names."""
    rec = ClipRecorder(str(tmp_path), pre_sec=0.2, post_sec=0.3, skip_labels=["other"])
    drive(rec, [("Other", 20)])                       # display spelling, not "other"
    rec.close()
    assert clips_in(str(tmp_path)) == []
