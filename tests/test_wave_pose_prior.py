"""Physical observation counts and veto propagation for the deployed wave prior."""
from pathlib import Path
from types import SimpleNamespace
import copy
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'platform/dark-demo-v2'))
from darkpipe.person.wave_prior import apply_wave_prior, wave_evidence
from darkpipe.constants import BEHAVIORS


def sample(index, left=.3, right=.8, confidence=.9, luma=80.):
    points = np.full((17, 3), [.5, .8, .9])
    points[5] = [.4, .5, .9]
    points[6] = [.6, .6, .9]
    points[9] = [.3, left, confidence]
    points[10] = [.7, right, confidence]
    return SimpleNamespace(frame_index=index, timestamp=index/25, pose=points, luma=luma)


def test_repeated_sampling_cannot_invent_three_observations():
    one = sample(0)
    assert not wave_evidence([one]*32, 80.)['supported']
    assert wave_evidence([sample(i) for i in range(3)], 80.)['supported']


def test_same_wrist_must_meet_count_and_be_above_both_shoulders():
    assert not wave_evidence([sample(0),sample(1),sample(2,left=.8,right=.3)],80.)['supported']
    assert not wave_evidence([sample(i,left=.55)for i in range(10)],80.)['supported']
    assert not wave_evidence([sample(i,left=.5)for i in range(10)],80.)['supported']
    assert wave_evidence([sample(i,left=.8,right=.3)for i in range(3)],80.)['support_wrist']=='right'


@pytest.mark.parametrize('confidence',[.49, float('nan'), float('inf')])
def test_unreliable_wrists_never_support(confidence):
    assert not wave_evidence([sample(i,confidence=confidence)for i in range(5)],80.)['supported']


def test_both_shoulders_and_valid_coordinates_are_required():
    rows=[sample(i)for i in range(5)]
    for s in rows:s.pose[6,2]=.2
    assert not wave_evidence(rows,80.)['supported']
    for s in rows:s.pose[6,2]=.9;s.pose[9,1]=-.2
    assert not wave_evidence(rows,80.)['supported']


@pytest.mark.parametrize('luma',[0.,34.999,None,float('nan')])
def test_dark_unknown_windows_abstain(luma):
    r=wave_evidence([],luma)
    assert r['supported'] and not r['applied']


def test_bright_missing_pose_rejects_and_dark_sample_cannot_support():
    assert not wave_evidence([],35.)['supported']
    assert not wave_evidence([sample(i,luma=10.)for i in range(8)],80.)['supported']
    assert wave_evidence([sample(i,luma=None)for i in range(3)],None,lum_gate=0)['supported']
    assert not wave_evidence([],80.,pose_enabled=False)['applied']


def test_optional_contiguity_does_not_bridge_missing_observations():
    rows=[sample(0),sample(1,left=.8),sample(2),sample(3,left=.8),sample(4)]
    assert wave_evidence(rows,80.)['supported']
    assert not wave_evidence(rows,80.,consecutive=True)['supported']
    assert not wave_evidence([sample(0),sample(10),sample(20)],80.,consecutive=True)['supported']


def test_veto_retains_model_probabilities_and_changes_published_label():
    probs=[.99]+[.001]*9
    row=dict(label='wave',confidence=.99,raw_luma=80.,decision_scores=probs[:],fused_probs=probs[:])
    apply_wave_prior(SimpleNamespace(window=[]),row)
    assert row['label']=='other' and row['action_label']=='wave'
    assert row['fused_probs']==probs and row['action_decision_scores']==probs
    assert BEHAVIORS[np.argmax(row['decision_scores'])]=='other'
    from darkpipe.person.service import PersonEvent
    assert PersonEvent(row).to_dict()['label']=='other'


@pytest.mark.parametrize('label',['drink','pick_up','other','chase'])
def test_non_wave_results_are_unchanged(label):
    row=dict(label=label,confidence=.96,raw_luma=80.,decision_scores=[.1]*10)
    before=copy.deepcopy(row)
    apply_wave_prior(SimpleNamespace(window=[]),row)
    assert {k:row[k]for k in before}==before and not row['wave_prior']['applied']
