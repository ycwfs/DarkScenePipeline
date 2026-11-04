"""Dark or unobservable windows must never be rejected by object evidence."""
import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1]/'platform/dark-demo-v2/darkpipe/person/object_prior.py'
spec = importlib.util.spec_from_file_location('candidate_person_object_prior', PATH)
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)


def sample(boxes=(), luma=80., valid=True):
    return dict(frame=20, binding_valid=valid, person_roi=[0, 0, 200, 300],
                raw_luma=luma, detections=list(boxes))


@pytest.mark.parametrize('luma', [0., 14., 34.999, None, float('nan')])
def test_dark_and_unknown_windows_bypass_even_empty_detection(luma):
    result = prior.filter_object_evidence('drink', [sample()], luma)
    assert result['label'] == 'drink' and not result['applied']
    assert not prior.allows_object_prior(luma)


def test_gate_boundary_and_disabled_gate():
    assert prior.filter_object_evidence('drink', [sample()], 35.)['label'] == 'other'
    assert prior.filter_object_evidence('drink', [sample()], 14., lum_gate=0)['label'] == 'other'


def test_one_frame_is_enough_but_object_threshold_still_applies():
    samples = [sample(), sample([[10, 10, 20, 20, .6, 0]]), sample()]
    assert prior.filter_object_evidence('drink', samples, 80.)['label'] == 'drink'
    assert prior.filter_object_evidence('drink', samples, 80., .7)['label'] == 'other'


def test_dark_or_ambiguous_sample_cannot_support_another_person():
    samples = [sample([[10, 10, 20, 20, .99, 0]], luma=10.),
               sample([[10, 10, 20, 20, .99, 0]], valid=False), sample()]
    assert prior.filter_object_evidence('drink', samples, 80.)['label'] == 'other'
    result = prior.filter_object_evidence('drink', samples[:2], 80.)
    assert result['label'] == 'drink' and not result['applied']


def test_class_and_roi_binding_and_unaffected_action():
    boxes = [[10, 10, 20, 20, .99, 2], [300, 10, 320, 20, .99, 0]]
    assert prior.filter_object_evidence('drink', [sample(boxes)], 80.)['label'] == 'other'
    assert prior.filter_object_evidence('pick_up', [sample(boxes)], 80.)['label'] == 'pick_up'
    assert prior.filter_object_evidence('wave', [sample()], 80.)['label'] == 'wave'
