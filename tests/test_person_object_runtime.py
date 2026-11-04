"""Exact actor/crop binding and veto propagation at the production action boundary."""
from collections import Counter, OrderedDict
import pickle
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'platform/dark-demo-v2'))
from darkpipe.person.action_engine import ActionTask
from darkpipe.person.core import Sample, fuse_person
from darkpipe.person.object_runtime import EvidenceFrame, FrameHistory, PersonObjectPrior, regions


def fixture(label='drink', luma=80., valid=True, generation=0):
    frame = np.arange(100*200*3, dtype=np.uint8).reshape(100, 200, 3)
    box = np.array([40, 20, 100, 80.])
    pose = np.zeros((17, 3), np.float32)
    samples = tuple(Sample(i/31, i, frame, box, pose, luma, box, valid) for i in range(32))
    task = ActionTask((1, 7, generation), samples, tuple(box), .98, 0., 0.,
        tuple(EvidenceFrame(i, frame, (300, 600, 3)) for i in (0, 8, 16, 23, 31)))
    rgb = np.zeros(10, np.float32)
    from darkpipe.constants import BEHAVIORS
    rgb[BEHAVIORS.index(label)] = .99
    rgb[9] = .01
    return task, fuse_person(rgb, rgb, [luma])


def prior(detect):
    obj = PersonObjectPrior.__new__(PersonObjectPrior)
    obj.threshold, obj.lum_gate = .581, 35.
    obj.stats, obj.cache = Counter(), OrderedDict()
    obj.detector = SimpleNamespace(_frames_to_dets=detect)
    return obj


def test_five_frames_one_detection_supports_and_source_coordinates_are_recovered():
    images = []
    def detect(frames):
        images.extend([f.copy() for f in frames])
        return [np.array([[1, 2, 5, 6, .7, 0]]) if len(images) == 3 else np.empty((0, 6))]
    obj = prior(detect)
    task, row = fixture()
    obj.apply(task, row)
    assert len(images) == 5 and row['label'] == 'drink'
    assert row['object_prior']['reason'] == 'single_frame_support'
    assert row['object_prior']['evidence'][0]['frame'] == 16
    # Source rect [66,6,354,294] maps exactly to processed [22,2,118,98].
    assert np.array_equal(images[0], task.prior_frames[0].pixels[2:98, 22:118])
    assert row['object_prior']['evidence'][0]['bbox'] == pytest.approx([69, 12, 81, 24])


@pytest.mark.parametrize('luma,valid,label', [(34.999, True, 'drink'), (None, True, 'drink'),
    (80., False, 'drink'), (80., True, 'wave')])
def test_unavailable_or_unaffected_windows_never_run_detector(luma, valid, label):
    def forbidden(frames): raise AssertionError('Detector should not run')
    task, row = fixture(label, luma, valid)
    prior(forbidden).apply(task, row)
    assert row['label'] == label and not row['object_prior']['applied']


def test_empty_available_evidence_vetoes_before_event_publication_and_clip_trigger(tmp_path):
    from darkpipe.person.service import PersonEvent
    from darkpipe.person.clips import PersonClips
    import time
    task, row = fixture()
    before = row['fused_probs'][:]
    prior(lambda frames: [np.empty((0, 6))]).apply(task, row)
    row.update(session=1, track_id=7, history_generation=0, timestamp=3., frame_index=30)
    assert row['action_label'] == 'drink' and row['label'] == 'other'
    assert row['fused_probs'] == before and np.argmax(row['decision_scores']) == 9
    assert PersonEvent(row).to_dict()['label'] == 'other'
    recorder = PersonClips(tmp_path, time.monotonic()+1000, queue_frames=100)
    recorder.event(row)
    for i in range(60):
        recorder.frame(np.zeros((90, 160, 3), np.uint8),
                       dict(frame=i, timestamp=i/10, capture_epoch=0))
    assert not recorder.close()['clips']


def test_model_failure_propagates_instead_of_silently_removing_prior():
    def broken(frames): raise RuntimeError('ORT failure')
    task, row = fixture()
    with pytest.raises(RuntimeError, match='ORT failure'):
        prior(broken).apply(task, row)


def test_cache_reuses_pixels_but_never_crosses_track_generations():
    obj = prior(lambda frames: [np.empty((0, 6))])
    obj.apply(*fixture())
    obj.apply(*fixture())
    assert obj.stats['inferred_crops'] == obj.stats['reused_crops'] == 5
    obj.apply(*fixture(generation=1))
    assert obj.stats['inferred_crops'] == 10


def test_frame_history_is_bounded_and_submitted_task_survives_eviction_and_ipc():
    task, _ = fixture()
    history = FrameHistory(max_frames=5, max_bytes=5*task.prior_frames[0].pixels.nbytes)
    for frame in task.prior_frames:
        history.put(frame.index, frame.index/31, frame.pixels, frame.source_shape)
    assert len(history.select(task.window)) == 5
    copied = pickle.loads(pickle.dumps(task))
    history.put(100, 3., None, (300, 600, 3))
    assert history.bytes == 0 and not history.frames
    assert np.array_equal(copied.prior_frames[-1].pixels, task.prior_frames[-1].pixels)
    assert copied.window[-1].binding_valid
    history = FrameHistory(max_frames=2, max_bytes=task.prior_frames[0].pixels.nbytes)
    for frame in task.prior_frames:
        history.put(frame.index, frame.index/31, frame.pixels, frame.source_shape)
    assert len(history.frames) == 1


def test_pose_local_region_uses_original_resolution_padding_and_action_points():
    box = np.array([100, 100, 300, 500.])
    pose = np.zeros((17, 3))
    pose[0] = [150, 130, .8]
    pose[9] = [180, 180, .9]
    pose[13] = [200, 400, .9]
    drink = regions(box, pose, (1080, 1920), 'drink')
    pickup = regions(box, pose, (1080, 1920), 'pick_up')
    assert drink[0].tolist() == pickup[0].tolist() == [40, 0, 360, 620]
    assert drink[1].tolist() == [102, 82, 228, 228]
    assert pickup[1].tolist() == [132, 132, 248, 448]
