"""The serving loop runs enhance / recognize / output concurrently; frames must survive it.

The four stages used to run back to back on one thread, so throughput was their sum and a
~300 ms window inference stalled the whole pipeline. They now run as two threads -- GPU
(enhance + recognize) and CPU (SR + label bar + JPEG). Two things have to hold for that to
be a win rather than a bug:

  - the two sides really do overlap, otherwise the split bought nothing;
  - frames still reach the recognizer in capture order, exactly once, and an event still
    reaches the clip recorder attached to the frame it was computed from.

The source is a latest-wins slot, so *dropping* frames is by design when the GPU falls
behind. Reordering and duplication are not. The feed here waits for each frame to be picked
up before publishing the next, which removes drops from the picture without serialising the
pipeline -- the read happens before enhancement, so the producer is released while the
enhanced frame is still in flight.
"""
import os
import sys
import threading
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from darkpipe import server  # noqa: E402
from darkpipe.config import PipelineConfig  # noqa: E402

W, H = 64, 48
ENH_MS, SR_MS = 25, 25          # equal, so serial is exactly 2x concurrent


def _stamp(i):
    """A frame whose pixels encode its index, so reordering is detectable."""
    f = np.zeros((H, W, 3), np.uint8)
    f[:, :, 0] = i % 256
    return f


def _index_of(frame):
    return int(frame[0, 0, 0])


class _Sleeper:
    """A stage that costs a fixed wall-clock time. `time.sleep` releases the GIL, which is
    the same reason the real stages (torch CUDA calls, OpenCV) overlap across threads."""

    def __init__(self, ms, post_recognition=False):
        self.ms, self.post_recognition = ms, post_recognition
        self.closed = False

    def load(self, device):
        pass

    def __call__(self, frames):
        time.sleep(self.ms / 1000)
        return frames

    def close(self):
        self.closed = True


class _Recorder:
    """Stands in for the window recognizer: records what it saw, optionally stalling."""

    def __init__(self, stall_every=0, stall_ms=0):
        self.seen, self.stamps = [], []
        self.stall_every, self.stall_ms = stall_every, stall_ms
        self.n = 0
        self.closed = False

    def load(self, device):
        pass

    def push(self, frame, idx, ts):
        self.n += 1
        if self.stall_every and self.n % self.stall_every == 0:
            time.sleep(self.stall_ms / 1000)
        self.seen.append(_index_of(frame))
        self.stamps.append(ts)
        return None

    def close(self):
        self.closed = True


class _ReadTrackingSlot(server.LatestSlot):
    """A latest-wins slot that signals when a *new* frame has been taken by the consumer.

    Lets the feed publish exactly one frame per consumer read, so nothing is dropped and the
    order assertions below are about the pipeline rather than about test timing.
    """

    def __init__(self):
        super().__init__()
        self.read = threading.Event()
        self._last_read = 0

    def get(self):
        item, seq = super().get()
        if seq != self._last_read:
            self._last_read = seq
            self.read.set()
        return item, seq


def _state(monkeypatch, stages, recognizer, **cfg_kw):
    """A live ServerState whose stages are fakes.

    `stages` is either a list -- one shared set, which is all the single-GPU tests need --
    or a zero-arg factory, called once per device. process_loop asks build_stages for a set
    per device so each card gets its own model instance; the factory form is what lets the
    multi-GPU tests tell those instances apart.
    """
    cfg = PipelineConfig(mode="serve", input="rtsp://x", enhance="off", sr="off",
                         recognize="off", device="cpu", **cfg_kw)
    build = stages if callable(stages) else (lambda: stages)
    monkeypatch.setattr(server, "build_stages", lambda c: (build(), recognizer))
    st = server.ServerState(cfg)
    st.raw = _ReadTrackingSlot()
    return st


def _wait(pred, timeout=30.0):
    deadline = time.time() + timeout
    while not pred() and time.time() < deadline:
        time.sleep(0.005)
    return pred()


def _run(st, n_frames, timeout=30.0):
    """Feed n_frames through a live process_loop, one per consumer read; return elapsed."""
    t = threading.Thread(target=server.process_loop, args=(st,), daemon=True)
    t.start()
    t0 = time.time()
    for i in range(n_frames):
        st.raw.read.clear()
        st.raw.put((_stamp(i), time.time()))
        assert st.raw.read.wait(timeout), f"pipeline stopped consuming at frame {i}"
    # The last frames are still in flight; wait for them to come out the far end.
    _wait(lambda: st.frames_proc >= n_frames, timeout)
    elapsed = time.time() - t0
    st.stop.set()
    t.join(timeout=10)
    assert not t.is_alive(), "process_loop did not shut down"
    return elapsed


def test_recognizer_sees_every_frame_once_and_in_order(monkeypatch):
    rec = _Recorder()
    st = _state(monkeypatch, [_Sleeper(5), _Sleeper(5, post_recognition=True)], rec,
                no_label_bar=True)
    _run(st, 25)

    assert st.frames_proc == 25
    assert rec.seen == list(range(25)), "recognition must see capture order, exactly once"
    assert rec.stamps == sorted(rec.stamps), "timestamps must be monotonic"


def test_stages_overlap_instead_of_summing(monkeypatch):
    """The load-bearing property: throughput is the max of the two sides, not the sum.

    Enhance (GPU side) and SR (CPU side) each cost the same fixed time, so a serial loop
    needs 2x per frame and a working split needs 1x. The bound is set at 1.4x so the test
    fails on a regression to serial (2x) without tripping on scheduler noise.
    """
    n = 24
    st = _state(monkeypatch, [_Sleeper(ENH_MS), _Sleeper(SR_MS, post_recognition=True)],
                _Recorder(), no_label_bar=True)
    elapsed = _run(st, n)

    serial = n * (ENH_MS + SR_MS) / 1000
    assert st.frames_proc == n
    assert elapsed < serial * 0.7, (
        f"{elapsed:.2f}s for {n} frames; serial would be ~{serial:.2f}s -- stages are not "
        f"overlapping")


def test_recognition_stalls_still_overlap_the_output_side(monkeypatch):
    """A window inference blocks the GPU thread, but the CPU thread keeps draining.

    Recognition deliberately shares the GPU thread rather than getting one of its own -- two
    threads issuing CUDA work on one device time-slice against each other, which measured
    worse on both throughput and tail latency (see process_loop's docstring). So a stall does
    hold up the frames behind it; what must NOT happen is the encode time being added to it,
    which is what the old serial loop did.
    """
    n, stall_ms, every = 24, 150, 4
    rec = _Recorder(stall_every=every, stall_ms=stall_ms)
    st = _state(monkeypatch, [_Sleeper(ENH_MS), _Sleeper(SR_MS, post_recognition=True)],
                rec, no_label_bar=True)
    elapsed = _run(st, n)

    # Serial pays enhance + stalls + SR/encode per frame; the split hides the SR/encode side
    # behind the GPU side, so it should come in near enhance + stalls alone.
    serial = (n * (ENH_MS + SR_MS) + (n // every) * stall_ms) / 1000
    gpu_only = (n * ENH_MS + (n // every) * stall_ms) / 1000
    assert st.frames_proc == n
    assert elapsed < (serial + gpu_only) / 2, (
        f"{elapsed:.2f}s for {n} frames; serial ~{serial:.2f}s, GPU side alone "
        f"~{gpu_only:.2f}s -- the output side is not overlapping")
    assert rec.seen == list(range(n)), "every frame must still reach the recognizer, in order"


def test_a_failing_stage_aborts_instead_of_hanging(monkeypatch):
    """A dead worker must take the pipeline down, not leave it wedged on a full queue.

    The failure mode a bounded queue creates is a hang, not an exception: the enhance thread
    blocks forever on a queue nobody drains, and the service keeps reporting healthy while
    the stream is frozen. So this asserts on the join, not on a raised error.
    """
    class _Boom(_Sleeper):
        def __call__(self, frames):
            raise RuntimeError("stage exploded")

    st = _state(monkeypatch, [_Sleeper(2), _Boom(0, post_recognition=True)], _Recorder(),
                no_label_bar=True)
    t = threading.Thread(target=server.process_loop, args=(st,), daemon=True)
    t.start()
    for _ in range(4):
        st.raw.put((_stamp(0), time.time()))
        time.sleep(0.02)
    t.join(timeout=10)
    assert not t.is_alive(), "a failing output stage wedged the pipeline"


def test_an_event_reaches_the_clipper_with_its_own_frame(monkeypatch):
    """The event rides the queue with the frame it was computed from, not a side channel.

    The clip recorder uses the event to decide where an incident starts, so pairing it with
    a later frame would shift every clip's trigger point. Passing them together is what keeps
    the split indistinguishable from the serial loop here.
    """
    fires_on = {7, 15}

    class _Firing(_Recorder):
        def push(self, frame, idx, ts):
            super().push(frame, idx, ts)
            return f"ev{_index_of(frame)}" if _index_of(frame) in fires_on else None

    class _Clipper:
        def __init__(self):
            self.pairs = []

        def push(self, frame, ev, ts):
            if ev:
                self.pairs.append((ev, _index_of(frame)))

    st = _state(monkeypatch, [_Sleeper(3), _Sleeper(3, post_recognition=True)], _Firing(),
                no_label_bar=True)
    st.clipper = _Clipper()
    _run(st, 20)

    assert st.clipper.pairs == [("ev7", 7), ("ev15", 15)]


def test_stages_are_closed_after_shutdown(monkeypatch):
    """Workers are joined before close(), so nothing calls into a closed stage."""
    enh, sr, rec = _Sleeper(2), _Sleeper(2, post_recognition=True), _Recorder()
    st = _state(monkeypatch, [enh, sr], rec, no_label_bar=True)
    _run(st, 5)

    assert enh.closed and sr.closed and rec.closed


# --- more than one GPU -------------------------------------------------------------------
#
# `--gpus 0,1` deals arriving frames round-robin across the devices: a dealer thread, one
# enhance thread per device, and device 0's thread doing recognition and putting the stream
# back in order. None of that logic is device-specific -- the stages here are the same CPU
# fakes as above, and `gpus` only decides how many sets get built and how many threads run --
# so the ordering, pairing and failure properties are testable without two real cards.
#
# The property that makes the whole design safe is ordering: frames arrive at the recognizer
# and the clip recorder in capture order even though they were enhanced on different cards
# and can finish out of order. It comes from strict rotation (`next_out % n_gpu` names the
# owner of the next frame), so a bug in the rotation shows up here as reordering.


def _gpu_state(monkeypatch, n_gpu, recognizer, enh_ms=8, sr_ms=2, **kw):
    """N devices, each with its own enhance stage; one shared post-recognition SR stage."""
    made = []

    def factory():
        # A fresh enhance stage per call; process_loop keeps set i for device i. The SR stage
        # is rebuilt too, but only the first set's is ever loaded and used.
        enh = _Sleeper(enh_ms)
        made.append(enh)
        return [enh, _Sleeper(sr_ms, post_recognition=True)]

    st = _state(monkeypatch, factory, recognizer, gpus=",".join(str(i) for i in range(n_gpu)),
                no_label_bar=True, **kw)
    return st, made


@pytest.mark.parametrize("n_gpu", [2, 3])
def test_frames_survive_the_round_robin_in_order_and_exactly_once(monkeypatch, n_gpu):
    """The load-bearing property: dealing across cards must not reorder or duplicate.

    Each frame is enhanced on a different card and they can finish in any order, so this is
    the assertion that the reordering-by-modulo actually reorders. A rotation bug here shows
    up as a shuffled or short `seen`, not as a crash.
    """
    rec = _Recorder()
    st, made = _gpu_state(monkeypatch, n_gpu, rec)
    _run(st, 30)

    assert len(made) == n_gpu, f"expected one enhance instance per device, built {len(made)}"
    assert st.frames_proc == 30
    assert rec.seen == list(range(30)), "round-robin must not reorder or drop frames"
    assert rec.stamps == sorted(rec.stamps), "timestamps must stay monotonic"


def test_uneven_device_speeds_still_come_out_in_order(monkeypatch):
    """Ordering must come from the rotation, not from the devices happening to be matched.

    If it came from timing, this passes on a quiet machine and reorders on a busy one. Here
    device 1 is deliberately 4x slower than device 0, so frame N+1 finishes well after frame
    N+2 was already enhanced -- the blocking get on done_q[k] is what has to absorb that.
    """
    speeds, made = [4, 16], []

    def factory():
        enh = _Sleeper(speeds[len(made)])
        made.append(enh)
        return [enh, _Sleeper(1, post_recognition=True)]

    rec = _Recorder()
    st = _state(monkeypatch, factory, rec, gpus="0,1", no_label_bar=True)
    _run(st, 24)

    assert rec.seen == list(range(24)), "a slow device must delay frames, not reorder them"


def test_devices_overlap_instead_of_summing(monkeypatch):
    """Two cards must halve the enhance cost, or the extra card bought nothing.

    Enhance dominates here (40 ms vs 2 ms of output work) precisely so the GPU side is the
    bottleneck -- with the equal costs the single-GPU tests use, the CPU side would cap
    throughput and hide the second card entirely.
    """
    n, enh_ms = 24, 40
    st, _ = _gpu_state(monkeypatch, 2, _Recorder(), enh_ms=enh_ms, sr_ms=2)
    elapsed = _run(st, n)

    one_card = n * enh_ms / 1000
    assert st.frames_proc == n
    assert elapsed < one_card * 0.7, (
        f"{elapsed:.2f}s for {n} frames; one card would be ~{one_card:.2f}s -- the second "
        f"device is not overlapping")


def test_events_still_pair_with_their_own_frame_across_devices(monkeypatch):
    """The event must ride with its frame even when that frame came off another card.

    Recognition runs on device 0's thread for every frame, including the ones device 1
    enhanced, so this is what catches an event being attached to whatever frame happened to
    be next rather than the one it was computed from.
    """
    fires_on = {5, 6, 13}

    class _Firing(_Recorder):
        def push(self, frame, idx, ts):
            super().push(frame, idx, ts)
            return f"ev{_index_of(frame)}" if _index_of(frame) in fires_on else None

    class _Clipper:
        def __init__(self):
            self.pairs = []

        def push(self, frame, ev, ts):
            if ev:
                self.pairs.append((ev, _index_of(frame)))

    st, _ = _gpu_state(monkeypatch, 2, _Firing())
    st.clipper = _Clipper()
    _run(st, 20)

    assert st.clipper.pairs == [("ev5", 5), ("ev6", 6), ("ev13", 13)]


def test_a_dead_secondary_device_aborts_instead_of_hanging(monkeypatch):
    """A card failing must take the pipeline down, the same as any other stage.

    This is the failure the extra threads newly make possible: device 1 dies while device 0
    is healthy, so frames keep flowing on half the rotation and the sequencer waits forever
    for a done_q[1] that will never be filled. Half a working pipeline reporting healthy is
    worse than a dead one, so it must not be reachable.
    """
    made = []

    class _Boom(_Sleeper):
        def __call__(self, frames):
            raise RuntimeError("device 1 fell over")

    def factory():
        enh = _Sleeper(2) if not made else _Boom(2)
        made.append(enh)
        return [enh, _Sleeper(2, post_recognition=True)]

    st = _state(monkeypatch, factory, _Recorder(), gpus="0,1", no_label_bar=True)
    t = threading.Thread(target=server.process_loop, args=(st,), daemon=True)
    t.start()
    for i in range(8):
        st.raw.put((_stamp(i), time.time()))
        time.sleep(0.02)
    t.join(timeout=10)
    assert not t.is_alive(), "a dead second device wedged the pipeline"


def test_every_device_stage_is_closed_after_shutdown(monkeypatch):
    """Each card's own instance has to be released, not just device 0's.

    Missing this leaks a full set of weights per extra card, which on a shared box is the
    kind of thing the next tenant discovers rather than you.
    """
    st, made = _gpu_state(monkeypatch, 3, _Recorder())
    _run(st, 8)

    assert len(made) == 3
    assert all(s.closed for s in made), "every device's enhance stage must be closed"


@pytest.mark.parametrize("no_label_bar", [True, False])
def test_jpeg_slot_is_populated(monkeypatch, no_label_bar):
    """The encoded output is what every stream format is fed from, label bar or not."""
    st = _state(monkeypatch, [_Sleeper(2), _Sleeper(2, post_recognition=True)], _Recorder(),
                no_label_bar=no_label_bar)
    _run(st, 5)

    jpg, seq = st.jpeg.get()
    assert seq > 0 and jpg[:2] == b"\xff\xd8", "slot should hold a JPEG"
