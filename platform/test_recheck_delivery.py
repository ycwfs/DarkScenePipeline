"""Confirmed recovery must survive live rendering and real H.264 clip delivery."""
import copy
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.environ.get('DARKPIPE_TEST_RUNTIME',
                                str(Path(__file__).parent / 'dark-demo-v2')))
from darkpipe.constants import BEHAVIOR_DISPLAY, BEHAVIOR_ZH
from darkpipe.person import core
from darkpipe.person.recheck import CandidateRecheck
from darkpipe.person.service import Output, PersonEvent, make_clips


def recovered_row(action, mechanism):
    key = ('delivery', 1, 0)
    state = CandidateRecheck(True, .85, .925)
    if mechanism == 'continuation':
        state.observe_high(key, action, .5)
    else:
        assert state.update(key=key, action=action, confidence=.90,
                            t_start=0., t_end=.5, run_speed=.4) is None
    decision = state.update(key=key, action=action, confidence=.90,
                            t_start=.5, t_end=1., run_speed=.4)
    assert decision is not None and decision.mechanism == mechanism
    return dict(session=key[0], track_id=1, history_generation=0,
                timestamp=1., frame_index=10, label=decision.action,
                confidence=decision.confidence, threshold=.925, mode='rgb_skeleton',
                recheck=dict(recovered=True, mechanism=mechanism, lower=.85, upper=.925))


def track(row):
    return core.Track(row['track_id'], np.array([80., 30., 180., 140.]),
                      np.zeros(96), row['timestamp'], .99,
                      prediction=copy.deepcopy(row), last_inference=row['timestamp'])


@pytest.mark.parametrize('action,mechanism', [
    ('wave', 'continuation'), ('drink', 'continuation'), ('chase', 'running_dual')])
def test_recovered_label_matches_accepted_action_pixels(action, mechanism):
    row = recovered_row(action, mechanism)
    image = np.full((200, 360, 3), 50, np.uint8)
    actual = core.render_people(image, [track(row)], image.shape, 1.)
    reference = dict(row, threshold=.85, recheck={'recovered': False})
    expected = core.render_people(image, [track(reference)], image.shape, 1.)
    assert np.array_equal(actual, expected)
    assert row['confidence'] == .90  # Do not inflate the probability to bypass the gate.


@pytest.mark.parametrize('recheck,confidence,label', [
    ({}, .90, 'wave'), ({'recovered': False}, .90, 'wave'),
    ({'recovered': True}, float('nan'), 'wave'),
    ({'recovered': True}, .90, 'other')])
def test_unconfirmed_invalid_and_other_stay_other(monkeypatch, recheck, confidence, label):
    row = dict(recovered_row('wave', 'continuation'),
               recheck=recheck, confidence=confidence, label=label)
    seen = []
    original = core.text_patch
    def capture(text, *args):
        seen.append(text)
        return original(text, *args)
    monkeypatch.setattr(core, 'text_patch', capture)
    image = np.full((200, 360, 3), 50, np.uint8)
    core.render_people(image, [track(row)], image.shape, 1.)
    assert seen == ['T01 其他']


@pytest.mark.parametrize('action,mechanism', [('wave', 'continuation'), ('chase', 'running_dual')])
@pytest.mark.parametrize('save', [False, True])
def test_live_event_and_saved_clip_share_confirmed_label(tmp_path, monkeypatch, action, mechanism, save):
    row = recovered_row(action, mechanism)
    cfg = SimpleNamespace(clip_dir=str(tmp_path) if save else '', clip_session='session',
                          candidate_recheck_enabled=True, candidate_recheck_lower=.85,
                          reco_min_conf=.925, clip_min_conf=0., clip_pre_sec=.2,
                          clip_post_sec=.2, clip_max_sec=15., clip_denoise='off',
                          clip_skip_labels='other', pose_overlay=True)
    recorder = make_clips(cfg, time.monotonic(), offline=True)
    assert (recorder is not None) == save
    published = PersonEvent(row).to_dict()
    assert published['label'] == action and published['confidence'] == .90
    assert published['recheck']['recovered'] is True
    if recorder:
        recorder.event(published)
        # Rejected low-confidence candidate must never create another clip.
        recorder.event(dict(published, track_id=2, label='other',
                            recheck={'recovered': False}))
    labels = []
    original = core.text_patch
    def capture(text, *args):
        labels.append(text)
        return original(text, *args)
    monkeypatch.setattr(core, 'text_patch', capture)
    images = []
    def callback(shown, clean, record):
        images.append(shown)
        if recorder:
            recorder.frame(clean, record)
    engine = SimpleNamespace(realtime=True, cfg=cfg, picture=lambda x: x.copy())
    output = Output(engine, callback)
    try:
        for i in range(21):
            current = track(row)
            if i < 10:
                current.prediction = None
            record = dict(timestamp=i/10, frame=i, capture_epoch=0,
                          tracks=[copy.deepcopy(current)],
                          keys={1: ('delivery', 1, 0)}, model_shape=(200, 360, 3))
            output.put((np.full((200, 360, 3), 50, np.uint8), [current], record))
    finally:
        output.close()
        result = recorder.close() if recorder else None
    assert len(images) == 21
    assert f'T01 {BEHAVIOR_ZH[action]} 90%' in labels
    assert 'T01 其他' not in labels
    if result is None:
        assert not list(tmp_path.rglob('*.mp4'))
        return
    assert len(result['clips']) == 1
    assert result['clips_failed'] == result['write_errors'] == result['clips_unfinished'] == 0
    clip = result['clips'][0]
    assert clip['label'] == BEHAVIOR_DISPLAY[action]
    assert clip['person_id'] == 1
    assert clip['confidence'] == pytest.approx(.90)
    sidecar = json.loads(Path(clip['video']).with_suffix('.json').read_text())
    assert sidecar['label'] == clip['label'] and sidecar['context_complete']
    cap = cv2.VideoCapture(clip['video'])
    decoded = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        decoded.append(frame)
    cap.release()
    assert len(decoded) == clip['frames_written']
    # H.264 must actually contain the named-action label, not merely correct metadata.
    expected = core.render_people(np.full((200, 360, 3), 50, np.uint8),
                                  [track(dict(row, threshold=.85))], (200, 360, 3), 1.)
    assert min(np.abs(frame[140:190].astype(float)-expected[140:190]).mean()
               for frame in decoded) < 5.
