"""The live overlay keeps a brief named-action grace period without changing events."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'platform/dark-demo-v2'))
from darkpipe.person.core import Track, effective_display_prediction


def prediction(label, timestamp):
    return dict(label=label, confidence=.94, timestamp=timestamp, mode='rgb_skeleton')


def test_other_window_keeps_recent_named_action_only_until_display_deadline():
    track = Track(1, np.array([0., 0., 20., 40.]), np.zeros(96, np.float32), 1., .9)
    track.display_prediction = prediction('drink', 1.0)
    track.display_hold_until = 1.8

    track.prediction = prediction('other', 1.2)
    track.last_inference = 1.2
    shown = effective_display_prediction(track, 1.4)
    assert shown['label'] == 'drink'
    assert shown['display_held'] is True

    expired = effective_display_prediction(track, 1.81)
    assert expired['label'] == 'other'
    assert 'display_held' not in expired


def test_deliberate_prior_veto_is_not_hidden_by_display_hold():
    track = Track(1, np.array([0., 0., 20., 40.]), np.zeros(96, np.float32), 1., .9)
    track.display_prediction = prediction('drink', 1.0)
    track.display_hold_until = 1.8
    track.prediction = dict(prediction('other', 1.2), display_hold_allowed=False)
    track.last_inference = 1.2

    shown = effective_display_prediction(track, 1.4)
    assert shown['label'] == 'other'


def test_held_action_marker_is_a_single_asterisk(monkeypatch):
    import darkpipe.person.core as core

    track = Track(1, np.array([0., 0., 20., 40.]), np.zeros(96, np.float32), 1., .9)
    track.display_prediction = prediction('drink', 1.0)
    track.display_hold_until = 1.8
    track.prediction = prediction('other', 1.2)
    track.last_inference = 1.2
    labels = []
    monkeypatch.setattr(core, 'text_patch', lambda text, color, size: labels.append(text) or
                        np.zeros((1, 1, 3), np.uint8))

    core.render_people(np.zeros((40, 20, 3), np.uint8), [track], (40, 20), 1.4)

    assert labels == ['T01 喝水 94% *']
    assert '[hold]' not in labels[0]
