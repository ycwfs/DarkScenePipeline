"""Regressions from the October 19 platform recordings; no model/GPU required."""
import importlib
import importlib.util
import json
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
OP = Path(os.environ.get('DARKPIPE_V2_OPERATOR_DIR', ROOT / 'platform/dark-demo-v2'))


@pytest.fixture(scope='module')
def package():
    import sys
    name = '_darkpipe_v2_feedback'
    spec = importlib.util.spec_from_file_location(
        name, OP / 'darkpipe/__init__.py', submodule_search_locations=[str(OP / 'darkpipe')])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return name


def wait_for(predicate, seconds=3):
    deadline = time.monotonic() + seconds
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError('background operation did not complete')
        time.sleep(.005)


@pytest.mark.parametrize('writer_fails', [False, True])
def test_stop_with_unopened_clip_is_overload_but_writer_failure_stays_fatal(
        package, monkeypatch, tmp_path, writer_fails):
    clips = importlib.import_module(package + '.clips')
    Event = importlib.import_module(package + '.stages.base').RecognitionEvent
    entered, resume = threading.Event(), threading.Event()
    monkeypatch.setattr(clips, 'OPEN_AFTER', 1)
    rec = clips.ClipRecorder(str(tmp_path / 'clips'), pre_sec=0, post_sec=10,
                             queue_frames=4, stage_dir=str(tmp_path / 'stage'))
    errors = []

    def clean(frame):
        entered.set()
        assert resume.wait(10)
        if writer_fails:
            raise OSError('simulated storage failure')
        return frame

    def close():
        try:
            rec.close()
        except clips.ClipWriteError as exc:
            errors.append(str(exc))

    rec._clean = clean
    frame = np.zeros((48, 64, 3), np.uint8)
    try:
        rec.push(frame, Event(0, 0, 'Drinking water', .95), 0)
        rec.push(frame, None, .01)
        assert entered.wait(5)
        for ts in (.02, .03, .04):
            rec.push(frame, None, ts)
        for ts in (.1, .2):
            rec.push(frame, Event(1, ts, 'Waving', .95), ts)
        assert rec._pending_start is not None and rec._active is None
        closer = threading.Thread(target=close)
        closer.start()
        wait_for(lambda: rec._closing)
        resume.set()
        closer.join(15)
        assert not closer.is_alive()
        assert rec.abandoned == 1 and rec._pending_start is None
        if writer_fails:
            assert errors and rec.stats()['write_errors'] > 0
        else:
            assert not errors
            stats = rec.stats()
            assert stats['clips_saved'] == 1
            assert stats['clips_failed'] == stats['clips_forced'] == stats['clips_unfinished'] == 0
            assert stats['write_errors'] == 0 and stats['write_error'] is None
            rec.close()  # idempotent after the warning
    finally:
        resume.set()
        close()


@pytest.mark.parametrize('force_full', [False, True])
def test_congested_writer_keeps_end_and_next_clip_survives(package, monkeypatch, tmp_path,
                                                        force_full):
    clips = importlib.import_module(package + '.clips')
    base = importlib.import_module(package + '.stages.base')
    blocked, release = threading.Event(), threading.Event()

    class Writer:
        def __init__(self, path, fps):
            Path(path).write_bytes(b'encoded-test-data')
            self.count = 0

        def write(self, frame):
            self.count += 1

        def close(self):
            pass

    monkeypatch.setattr(clips, 'VideoWriter', Writer)
    monkeypatch.setattr(clips, 'OPEN_AFTER', 1)
    rec = clips.ClipRecorder(str(tmp_path / 'clips'), pre_sec=0, post_sec=.45,
                             queue_frames=4, stage_dir=str(tmp_path / 'stage'))

    def clean(frame):
        blocked.set()
        assert release.wait(5)
        return frame

    rec._clean = clean
    frame = np.zeros((32, 32, 3), np.uint8)
    event = base.RecognitionEvent(0, 0, 'Drinking water', .9)
    try:
        rec.push(frame, event, 0)
        rec.push(frame, None, .1)
        assert blocked.wait(2)
        began = time.monotonic()
        for ts in (.2, .3, .4):
            rec.push(frame, None, ts)
        if force_full:
            # Also exercise fallback retention for a queue filled before reservation applied.
            rec._q.put_nowait(('frame', frame, .45))
        rec.push(frame, None, .5)
        if force_full:
            # Expire the warning deadline as well as the old stream-time deadline.
            # A slow but healthy writer must retain its end message in either clock domain.
            msg, _, name, report = rec._pending_end
            rec._pending_end = (msg, time.monotonic() - 1, name, report)
        rec.push(frame, None, clips.END_RETRY_SEC + 1)
        assert time.monotonic() - began < .2, 'live producer waited for the writer'
        assert rec.abandoned == 0, 'a busy writer lost its mandatory close message'
        assert rec._q.qsize() <= 4
        release.set()
        wait_for(lambda: not rec._q.full())
        rec.push(frame, None, clips.END_RETRY_SEC + 1.1)
        wait_for(lambda: rec.saved == 1)
        next_event = base.RecognitionEvent(1, 12, 'Waving', .9)
        rec.push(frame, next_event, 12)
        rec.push(frame, None, 12.5)
        rec.close()
        stats = rec.stats()
        assert stats['clips_saved'] == 2
        assert stats['clips_forced'] == stats['clips_failed'] == stats['write_errors'] == 0
        assert stats['frames_dropped'] > 0, 'test did not create frame backpressure'
        assert {json.loads(p.read_text())['label'] for p in Path(rec.root).rglob('*.json')} == {
            'Drinking water', 'Waving'}
    finally:
        release.set()
        try:
            rec.close()
        except clips.ClipWriteError:
            pass


def test_async_event_arriving_during_jpeg_is_not_paired_with_previous_label(package, monkeypatch):
    server = importlib.import_module(package + '.server')
    config = importlib.import_module(package + '.config')
    base = importlib.import_module(package + '.stages.base')
    release_event = threading.Event()
    event = base.RecognitionEvent(1, .1, 'Drinking water', .9, model='behavior')
    cfg = config.PipelineConfig(mode='serve', recognize='behavior', no_label_bar=False)
    state = server.ServerState(cfg)
    calls, received = [], []
    pose_times = []
    overlay = importlib.import_module(package + '.pose_overlay')

    class PoseDrawer:
        def draw(self, output, frame, ev, timestamp):
            pose_times.append(timestamp)
            return output

    monkeypatch.setattr(overlay, 'ActionPoseOverlay', PoseDrawer)

    class Rec:
        sent = False

        def load(self, device):
            pass

        def warmup(self):
            pass

        def timings(self):
            return {}

        def push(self, frame, idx, ts, raw_frame=None):
            if self.sent:
                return None
            assert release_event.wait(3)
            self.sent = True
            return event

        def close(self):
            pass

    def render(frame, displayed, extra=''):
        calls.append(displayed)
        frame = frame.copy()
        frame[0, 0, 0] = 200 if displayed is event else 10
        return frame

    def encode(extension, frame, options):
        if not release_event.is_set():
            release_event.set()
            wait_for(lambda: state.last_event is event)
            # Give the recognizer time to publish the same event to the clip queue.
            time.sleep(.02)
        return True, np.array([1, 2, 3], np.uint8)

    def record(frame, received_event, ts):
        if received_event is not None:
            received.append((int(frame[0, 0, 0]), received_event))
            state.request_stop()

    state.clipper = SimpleNamespace(push=record, stats=lambda: {})
    monkeypatch.setattr(server, 'serve_devices', lambda cfg: ['cuda:0', 'cuda:1'])
    monkeypatch.setattr(server, 'bind_thread_device', lambda device: None)
    monkeypatch.setattr(server, 'build_stages', lambda cfg: ([], Rec()))
    monkeypatch.setattr(server, 'append_label_bar', render)
    monkeypatch.setattr(server.cv2, 'imencode', encode)

    def feed():
        while not state.stop.is_set():
            state.raw.put((np.zeros((32, 32, 3), np.uint8), time.time()))
            state.stop.wait(.01)

    feeder = threading.Thread(target=feed, daemon=True)
    processing = threading.Thread(target=server.process_loop, args=(state,), daemon=True)
    feeder.start()
    processing.start()
    try:
        processing.join(5)
        assert not processing.is_alive()
        assert state.processing_error is None
        assert received == [(200, event)], 'clip trigger and burnt-in label described different events'
        assert pose_times and all(0 <= ts < 10 for ts in pose_times), 'overlay needs stream-relative time'
    finally:
        state.request_stop()
        release_event.set()
        processing.join(6)
        feeder.join(1)
