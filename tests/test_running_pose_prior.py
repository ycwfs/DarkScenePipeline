import copy
from types import SimpleNamespace as S
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'platform/dark-demo-v2'))
from darkpipe.person.running_prior import speed_evidence, apply_running_prior


def sequence(speed=1., height=100, shape=(404,720), step=.04, count=26):
    result=[]
    for i in range(count):
        t=i*step;shift=t*height*speed;box=np.array([50+shift,50,90+shift,50+height])
        xy=np.column_stack([np.linspace(55,85,17)+shift,np.linspace(55,45+height,17)])
        p=np.column_stack([xy/[shape[1],shape[0]],np.ones(17)])
        result.append(S(frame_index=i,timestamp=t,pose=p,person_box=box,frame_shape=shape,luma=100))
    return result


def test_known_speed_and_boundary():
    s=sequence(.8);e=speed_evidence(s,100,.8);assert e['speed']==pytest.approx(.8)
    assert speed_evidence(s,100,.79)['supported']
    assert not speed_evidence(s,100,.81)['supported']


def test_scale_and_aspect_invariance():
    s=sequence(.7);first=speed_evidence(s,100,.6)['speed']
    for sample in s:
        sample.person_box*=2;sample.frame_shape=(808,1440)
    assert speed_evidence(s,100,.6)['speed']==pytest.approx(first)


def test_time_units():
    assert speed_evidence(sequence(.6,step=.02,count=51),100,.5)['speed']==pytest.approx(.6)


def test_duplicate_samples_not_extra_evidence():
    s=sequence();indices=np.linspace(0,25,32).round().astype(int)
    e=speed_evidence([s[i]for i in indices],100,.6)
    assert e['unique_frames']==26
    assert e['speed']==pytest.approx(1.)


@pytest.mark.parametrize('luma',[10,None,float('nan')])
def test_dark_unknown_bypass(luma):
    e=speed_evidence(sequence(),luma,2.);assert not e['applied']and e['supported']


def test_missing_pose_bright_rejects():
    s=sequence()
    for x in s:x.pose[:]=0
    e=speed_evidence(s,100,.5,missing_policy='reject');assert not e['supported']and e['speed']is None
    e=speed_evidence(s,100,.5);assert e['supported']and not e['applied']and e['reason']=='insufficient_pose_bypass'


def test_gap_not_bridged():
    s=sequence()[:3]+sequence()[20:]
    e=speed_evidence(s,100,.5,missing_policy='reject');assert not e['supported']


def test_zero_disables_and_disabled_branch():
    assert speed_evidence(sequence(0),100,0)['supported']
    assert not speed_evidence(sequence(),100,1,pose_enabled=False)['applied']


@pytest.mark.parametrize('threshold',[-1,float('nan'),float('inf')])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError):speed_evidence(sequence(),100,threshold)


def test_does_not_modify_probabilities_or_other_actions():
    r=dict(label='chase',confidence=.95,raw_luma=100,rgb_probs=[.1,.9],fused_probs=[.1,.9],decision_scores=[.1,.9])
    old=copy.deepcopy(r);apply_running_prior(S(window=sequence(.1)),r,1.)
    assert r['label']=='other'and r['rgb_probs']==old['rgb_probs']and r['fused_probs']==old['fused_probs']
    assert r['action_label']=='chase'
    r['label']='drink';apply_running_prior(S(window=sequence()),r,1.);assert r['label']=='drink'


def test_reversed_timestamp_rejected():
    with pytest.raises(ValueError):speed_evidence(list(reversed(sequence())),100,1.)
