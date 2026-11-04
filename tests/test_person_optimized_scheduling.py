"""Cache collision, stale result and overload regressions for the independent ROI runtime."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'reports/person-optimization-20261023/scripts'))
from action_engine import SlotCache
from scheduling import Mailbox, ResultStore


def task(tid, timestamp, generation=0):
    return SimpleNamespace(key=('session', tid, generation), timestamp=timestamp)


def test_cache_keeps_all_inputs_in_current_batch_when_evicting():
    cache = SlotCache(3)
    cache.reserve(['old', 'keep', 'last'])
    indices, missing = cache.reserve(['new', 'keep', 'new', 'last'])
    assert 'old' not in cache.entries
    assert indices[0] == indices[2]
    assert len(set(indices)) == 3
    assert [key for key, _ in missing] == ['new']


def test_cache_cannot_alias_different_people_or_track_generations():
    cache = SlotCache(4)
    keys = [('s', 1, 0, 12), ('s', 2, 0, 12), ('s', 1, 1, 12)]
    indices, _ = cache.reserve(keys)
    assert len(set(indices)) == 3
    with pytest.raises(ValueError):
        cache.reserve(list(range(5)))


def test_online_coalescing_keeps_other_people_and_fair_queue_position():
    box = Mailbox(capacity=2, coalesce=True)
    box.put(task(1, 1.))
    box.put(task(2, 1.))
    box.put(task(1, 1.5))
    box.put(task(1, .5))
    out = box.take(2, wait=False)
    assert [(r.key[1], r.timestamp) for r in out] == [(1, 1.5), (2, 1.)]
    assert box.stats['coalesced'] == 1
    assert box.stats['outdated_submission'] == 1
    assert not box.idle()
    box.finished(2)
    assert box.idle()


def test_strict_comparison_mode_never_replaces_person_windows():
    box = Mailbox(coalesce=False)
    box.put(task(1, 1.))
    box.put(task(1, 1.5))
    assert [r.timestamp for r in box.take(2)] == [1., 1.5]


def test_late_results_cannot_overwrite_new_result_or_cross_generation():
    store = ResultStore()
    def row(t, g=0):
        return dict(session='s', track_id=1, history_generation=g, timestamp=t, label='wave')
    assert store.update(row(2))
    assert not store.update(row(1))
    assert store.for_track(('s', 1, 0), 2.5)['timestamp'] == 2
    assert store.for_track(('s', 1, 1), 2.5) is None
    assert store.for_track(('s', 1, 0), 3.01) is None
    assert store.for_track(('s', 1, 0), 1.5) is None


def test_stopped_worker_rejects_tasks_instead_of_hanging():
    box = Mailbox()
    box.stop()
    with pytest.raises(RuntimeError, match='stopped'):
        box.put(task(1, 1))


def test_latest_source_preserves_original_index_without_backpressuring_producer(tmp_path):
    import cv2
    import numpy as np
    from live_replay import LatestDecoder
    path = tmp_path/'source.mp4'
    writer = cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),40.,(32,32))
    assert writer.isOpened()
    for i in range(5):
        writer.write(np.full((32,32,3),i*40,np.uint8))
    writer.release()
    decoder = LatestDecoder(path)
    try:
        assert decoder.done.wait(5)
        value = decoder.read()
        assert value[0] == 4  # newest actual source index, not fabricated frame zero
        assert decoder.count == 1 and decoder.dropped == 4
        assert decoder.read() is None
    finally:
        decoder.close()
