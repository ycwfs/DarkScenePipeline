"""Exercise the delivered clip protocol across repeated congestion/switch/stop cycles.

Uses real H.264 writers with small frames and an injected processing pause. This tests
ordering and resource release, not full-size denoising throughput or model accuracy.
"""
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
OP = Path(os.environ.get('DARKPIPE_V2_OPERATOR_DIR', ROOT / 'platform/dark-demo-v2'))


def wait_for(predicate, seconds=10):
    deadline = time.monotonic() + seconds
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError('recorder did not make expected progress')
        time.sleep(.005)


@pytest.mark.parametrize('queue_limit', [4, 8, 64])
def test_repeated_congestion_switch_and_stop(monkeypatch, tmp_path, queue_limit):
    name = f'_v2_clip_sequence_{queue_limit}'
    spec = importlib.util.spec_from_file_location(
        name, OP / 'darkpipe/__init__.py', submodule_search_locations=[str(OP / 'darkpipe')])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[name] = pkg
    spec.loader.exec_module(pkg)
    clips = importlib.import_module(name + '.clips')
    Event = importlib.import_module(name + '.stages.base').RecognitionEvent
    # Open immediately so the injected pause acts on an active real encoder.
    monkeypatch.setattr(clips, 'OPEN_AFTER', 1)
    rec = clips.ClipRecorder(str(tmp_path), pre_sec=0, post_sec=.45, denoise='off',
                             queue_frames=queue_limit, session='clips',
                             stage_dir=str(tmp_path / 'stage'))
    entered, resume = threading.Event(), threading.Event()
    producer_times, video_records, close_errors = [], [], []
    expected, switches = 0, 0
    frame = np.zeros((48, 64, 3), np.uint8)

    def paused_clean(image):
        entered.set()
        if not resume.wait(15):
            raise TimeoutError('test did not resume the paused encoder')
        return image

    rec._clean = paused_clean

    def push(ts, label=None):
        before = time.monotonic()
        ev = Event(int(ts * 100), ts, label, .95) if label else None
        rec.push(frame, ev, ts)
        producer_times.append(time.monotonic() - before)

    try:
        for cycle in range(21):
            # The preceding cycle is fully delivered; every cycle fills a fresh queue.
            wait_for(lambda: rec._q.unfinished_tasks == 0)
            entered.clear()
            resume.clear()
            ts = cycle * 20.0
            push(ts, 'Drinking water')
            push(ts + .01)
            assert entered.wait(10)
            for j in range(queue_limit + 2):
                push(ts + .02 + j * .001)
            assert rec._q.qsize() == queue_limit - 1
            dropped = rec.dropped_frames
            assert dropped > 0

            if cycle == 20:
                # Stop with an active clip while the encoder and its bounded queue are busy.
                def close():
                    try:
                        rec.close()
                    except BaseException as exc:
                        close_errors.append(str(exc))
                closer = threading.Thread(target=close)
                closer.start()
                wait_for(lambda: rec._closing)
                assert closer.is_alive()
                resume.set()
                closer.join(15)
                assert not closer.is_alive() and not close_errors
                expected += 1
            elif cycle % 2:
                # Both confirmations arrive while the first encoder is paused. The new start
                # must wait behind the previous end even when both are issued in one push.
                push(ts + .3, 'Waving')
                push(ts + .4, 'Waving')
                assert rec._pending_start is not None
                resume.set()
                deadline = time.monotonic() + 10
                while rec._pending_start is not None:
                    assert time.monotonic() < deadline
                    push(ts + .41)
                    time.sleep(.005)
                assert rec._active is not None and rec._active['key'] == 'waving'
                push(ts + 1.0)
                expected += 2
                switches += 1
                wait_for(lambda: rec.saved == expected)
            else:
                push(ts + .6)
                assert rec._q.qsize() == queue_limit  # end occupies its reserved slot
                resume.set()
                expected += 1
                wait_for(lambda: rec.saved == expected)
            assert rec.forced_close == 0 and rec.stats()['write_errors'] == 0

        stats = rec.stats()
        assert expected == stats['clips_saved'] == 31
        assert all(stats[k] == 0 for k in (
            'clips_abandoned', 'clips_forced', 'clips_unfinished', 'clips_failed', 'write_errors'))
        assert not stats['writer_alive']
        assert max(producer_times) < .5
        videos = sorted(Path(rec.root).rglob('*.mp4'))
        assert len(videos) == expected
        for video in videos:
            meta = json.loads(video.with_suffix('.json').read_text())
            result = subprocess.run(
                ['ffmpeg', '-v', 'error', '-i', str(video), '-f', 'null', '-'],
                capture_output=True, text=True, timeout=15)
            assert result.returncode == 0, result.stderr
            assert meta['frames_written'] > 0
            video_records.append({'file': video.name, 'label': meta['label'],
                                  'closed_because': meta['closed_because'],
                                  'frames': meta['frames_written']})
        assert sum(r['closed_because'].startswith('行为切换') for r in video_records) == switches
        assert sum(r['closed_because'] == '服务停止' for r in video_records) == 1
        result = {'queue_limit': queue_limit, 'cycles': 21, 'switches': switches,
                  'stats': stats, 'all_videos_decoded': True,
                  'push_max_ms': max(producer_times) * 1000, 'videos': video_records,
                  'scope': 'Real H.264, 64x48 synthetic frames, injected writer pauses; '
                           'not a model or full-size quality throughput test.'}
        (tmp_path / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        resume.set()
        rec.close()
