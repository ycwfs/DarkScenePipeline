"""Object drawing must not change decisions or attach stale evidence to another frame/person."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'platform/dark-demo-v2'))
from darkpipe.person.object_overlay import ObjectOverlay
from darkpipe.person.core import render_people


def scene():
    rng = np.random.default_rng(7)
    image = np.zeros((100, 200, 3), np.uint8)
    patch = rng.integers(40, 240, (25, 25, 1), dtype=np.uint8)
    image[30:55, 60:85] = patch
    row = dict(session=1, track_id=7, history_generation=0, timestamp=1., frame_index=30,
        label='other', confidence=1., mode='rgb_skeleton', object_prior=dict(
            lum_gate=35., object_min_conf=.581, samples=[dict(frame=30, timestamp=1.,
                source_shape=[300, 600], available=True, binding_valid=True,
                detections=[[180, 90, 255, 165, .8, 2]])]))
    track = SimpleNamespace(tid=7, prediction=row, last_inference=1.,
                            box=np.array([20., 10., 130., 95.]), pose=None)
    record = dict(frame=30, timestamp=1., capture_epoch=0, raw_luma=80., keys={7:(1,7,0)})
    return image, track, record


def test_exact_source_coordinates_scaled_and_non_support_object_on_other_is_visible():
    image, track, record = scene()
    frozen = copy.deepcopy(track.prediction)
    boxes = ObjectOverlay().update(image, [track], record)
    assert len(boxes) == 1 and boxes[0]['class_id'] == 2
    assert boxes[0]['bbox'] == pytest.approx([.3, .3, .425, .55])
    assert not boxes[0]['followed']
    shown = render_people(cv2.resize(image, (400,200)), [], image.shape, 1., object_boxes=boxes)
    assert np.array_equal(shown[60,120], [255,220,20])
    assert frozen == track.prediction


def test_async_local_motion_aligns_to_new_frame_and_expires():
    image, track, record = scene()
    overlay = ObjectOverlay()
    assert not overlay.update(image, [], record)
    moved = cv2.warpAffine(image, np.float32([[1,0,6],[0,1,3]]), (200,100))
    boxes = overlay.update(moved, [track], dict(record, frame=31, timestamp=1.1))
    assert len(boxes) == 1 and boxes[0]['followed']
    assert boxes[0]['bbox'] == pytest.approx([.33, .33, .455, .58], abs=.01)
    assert not overlay.update(moved, [track], dict(record, frame=50, timestamp=1.61))


@pytest.mark.parametrize('change', ['epoch','generation','person','future','dark','unknown','empty','unavailable'])
def test_no_wrong_person_epoch_future_low_luma_or_superseded_boxes(change):
    image, track, record = scene()
    if change == 'epoch': record['capture_epoch'] = 1
    if change == 'generation': record['keys'][7] = (1,7,1)
    if change == 'person': track.prediction['track_id'] = 8
    if change == 'future': track.prediction['timestamp'] = 1.1
    if change == 'dark': record['raw_luma'] = 34.
    if change == 'unknown': record['raw_luma'] = None
    if change == 'unavailable': track.prediction['object_prior']['samples'][0]['available'] = False
    if change == 'empty':
        track.prediction['object_prior']['samples'].append(dict(frame=30,timestamp=1.01,
            available=True,binding_valid=True,detections=[],source_shape=[300,600]))
        record['timestamp'] = 1.01
    assert not ObjectOverlay().update(image, [track], record)


def test_no_texture_or_missing_frame_hides_rather_than_painting_old_coordinates():
    image, track, record = scene()
    overlay = ObjectOverlay()
    overlay.update(np.zeros_like(image), [], record)
    assert not overlay.update(image, [track], dict(record,frame=31,timestamp=1.1))
    assert not ObjectOverlay().update(image,[track],dict(record,frame=31,timestamp=1.1))


def test_history_is_bounded_and_reset_at_reconnect():
    overlay = ObjectOverlay()
    image, _, record = scene()
    for i in range(1000):
        overlay.update(image, [], dict(record, frame=i, timestamp=i/1000))
    assert len(overlay.history) <= 64
    overlay.update(image, [], dict(record, frame=1000, timestamp=1., capture_epoch=1))
    assert len(overlay.history) == 1


def test_delayed_file_evidence_uses_original_sample_without_future_action_label():
    image, track, record = scene()
    future = copy.deepcopy(track.prediction)
    future['timestamp'] = 1.5
    track.prediction = None
    assert not ObjectOverlay().update(image, [track], record)
    boxes = ObjectOverlay().update(image, [track], record, [future])
    assert len(boxes) == 1 and not boxes[0]['followed']
    assert track.prediction is None
    assert not ObjectOverlay().update(image, [track], dict(record, capture_epoch=1), [future])


def test_file_output_lookahead_flushes_every_frame_in_order(tmp_path):
    from darkpipe.person.service import Output
    image, track, record = scene()
    row = copy.deepcopy(track.prediction)
    track.prediction = None
    engine = SimpleNamespace(realtime=False, picture=lambda frame:frame,
        cfg=SimpleNamespace(pose_overlay=False), overlay_evidence=lambda:[row])
    observed=[]
    output=Output(engine,lambda shown,clean,rec:observed.append(rec))
    for i in range(30,70):
        output.put((image,[copy.copy(track)],dict(record,frame=i,timestamp=i/30)))
    output.close()
    assert [r['frame']for r in observed] == list(range(30,70))
    assert observed[0]['object_overlays'][0]['sample_frame'] == 30


def test_live_output_switch_hides_skeleton_without_changing_action_label():
    from darkpipe.person.service import Output

    frame = np.zeros((100, 200, 3), np.uint8)
    keypoints = np.column_stack((np.linspace(.2, .8, 17), np.linspace(.2, .8, 17),
                                 np.full(17, .9))).astype(np.float32)
    prediction = dict(label='drink', confidence=.94, threshold=.925,
                      timestamp=1., mode='rgb_skeleton')
    track = SimpleNamespace(tid=7, prediction=prediction, last_inference=1.,
        box=np.array([20., 10., 180., 90.]), pose=keypoints)
    engine = SimpleNamespace(realtime=True, cfg=SimpleNamespace(pose_overlay=False),
        picture=lambda image:image, overlay_evidence=lambda:[])
    painted = []
    output = Output(engine, lambda shown, clean, record:painted.append(shown))
    output.put((frame, [track], dict(timestamp=1., frame=0, model_shape=frame.shape)))
    output.close()

    hidden = render_people(frame, [track], frame.shape, 1., pose_overlay=False)
    visible = render_people(frame, [track], frame.shape, 1., pose_overlay=True)
    assert np.array_equal(painted[0], hidden)
    assert not np.array_equal(painted[0], visible)


def test_recording_repaints_object_from_capture_time_event(tmp_path, monkeypatch):
    import time
    from darkpipe.person import clips
    image, track, record = scene()
    row = copy.deepcopy(track.prediction)
    row.update(label='drink', confidence=.99)
    shown = []
    def render(frame, tracks, shape, stamp, pose, boxes):
        shown.append((stamp, pose, copy.deepcopy(boxes)))
        return frame
    monkeypatch.setattr(clips, 'render_people', render)
    recorder = clips.PersonClips(tmp_path, time.monotonic()+100, offline=True, pose_overlay=False)
    recorder.event(row)
    for i in range(20, 40):
        t = copy.copy(track)
        t.prediction = None
        recorder.frame(image, dict(record, frame=i, timestamp=i/30,
            tracks=[t], model_shape=image.shape))
    result = recorder.close()
    assert result['clips_saved'] == 1
    assert not next(boxes for stamp, _, boxes in shown if stamp == 29/30)
    assert len(next(boxes for stamp, _, boxes in shown if stamp == 1.)) == 1
    assert shown and all(not pose for _, pose, _ in shown)
    assert result['object_overlay']['frames_with_objects'] > 0
