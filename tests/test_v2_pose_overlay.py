"""Pose evidence, drawing alignment and isolation from recognition (no GPU required)."""
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
OP = Path(os.environ.get('DARKPIPE_V2_OPERATOR_DIR', ROOT / 'platform/dark-demo-v2'))


@pytest.fixture(scope='module')
def package():
    name = '_v2_pose_preview'
    spec = importlib.util.spec_from_file_location(name, OP / 'darkpipe/__init__.py',
                                                submodule_search_locations=[str(OP / 'darkpipe')])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return name


def pose():
    return np.column_stack((np.linspace(.3, .6, 17), np.linspace(.25, .7, 17),
                            np.full(17, .9))).astype(np.float32)


def observation(package, frame, timestamp=0):
    Event = importlib.import_module(package + '.stages.base').RecognitionEvent
    overlay = importlib.import_module(package + '.pose_overlay')
    return Event(0, timestamp, 'Waving', .9, skeleton_active=True,
                 actor_poses=[dict(slot=0, label='wave', confidence=.9, keypoints=pose().tolist())],
                 pose_reference=overlay.tracking_gray(frame))


def test_only_current_supported_actors_are_selected(package):
    skel = importlib.import_module(package + '.stages.skeleton_aux')
    aux = skel.SkeletonAux('unused', 'cpu', .7)
    sk = np.tile(pose()[None, None], (2, 3, 1, 1))
    probs = np.zeros((3, 10), np.float32)
    probs[0, 0] = probs[2, 0] = probs[1, 9] = 1
    sk[-1, 2, :, 2] = 0  # This slot existed earlier but has left the current frame.
    aux.last_analysis = dict(sk=sk, slot_p=probs, presence=np.ones(3))
    assert [p['slot'] for p in aux.action_poses('wave')] == [0]
    assert aux.action_poses('other') == aux.action_poses('drink') == []


def test_skeleton_metadata_does_not_change_weighted_prediction(package, monkeypatch):
    import torch
    skel = importlib.import_module(package + '.stages.skeleton_aux')
    aux = skel.SkeletonAux('unused', 'cpu', .7)
    sk = np.tile(pose()[None, None], (2, 2, 1, 1))
    sk[:, 1, :, 2] = .6
    monkeypatch.setattr(aux, '_fill_slots', lambda frames, count: sk)
    logits = torch.tensor([[3., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [1., 4., 0., 0., 0., 0., 0., 0., 0., 0.]])
    calls = []
    def net(x):
        calls.append(1)
        return logits[len(calls)-1:len(calls)]
    aux.net = net
    p, presence = aux.probs([None, None])
    expected = (.6 * logits[0] + .4 * logits[1]).softmax(0).numpy()
    np.testing.assert_allclose(p, expected, atol=1e-7)
    assert len(calls) == 2 and presence == pytest.approx(.9)
    assert len(aux.action_poses('wave')) == 1


def test_duplicate_pose_is_hidden_but_distinct_person_is_kept(package):
    skel = importlib.import_module(package + '.stages.skeleton_aux')
    aux = skel.SkeletonAux('unused', 'cpu', .7)
    sk = np.tile(pose()[None, None], (2, 3, 1, 1))
    sk[:, 1, :, :2] += .001
    sk[:, 2, :, 0] += .3
    probs = np.zeros((3, 10), np.float32); probs[:, 0] = 1
    aux.last_analysis = dict(sk=sk, slot_p=probs, presence=np.ones(3))
    assert [p['slot'] for p in aux.action_poses('wave')] == [0, 2]
    assert aux.last_analysis['sk'].shape[1] == 3  # fusion inputs remain unchanged


def test_overlay_scales_without_mutating_model_input_or_serializing_image(package):
    overlay = importlib.import_module(package + '.pose_overlay')
    frame = np.zeros((100, 160, 3), np.uint8)
    event = observation(package, frame)
    original = frame.copy()
    output = np.zeros((200, 320, 3), np.uint8)
    drawer = overlay.ActionPoseOverlay()
    result = drawer.draw(output, frame, event, 0)
    assert drawer.last_drawn == 1 and result is not output
    assert result[50, 96].any()  # first keypoint at x=.3*w, y=.25*h
    assert not output.any() and np.array_equal(frame, original)
    serialized = event.to_dict()
    assert 'pose_reference' not in serialized and serialized['skeleton_active']
    json.dumps(serialized)


@pytest.mark.parametrize('condition', ['other', 'inactive', 'expired', 'future', 'no_actor'])
def test_unsupported_or_stale_overlay_is_cleared(package, condition):
    overlay = importlib.import_module(package + '.pose_overlay')
    frame = np.zeros((100, 160, 3), np.uint8)
    event = observation(package, frame)
    ts = 0
    if condition == 'other': event.label = 'Other'
    if condition == 'inactive': event.skeleton_active = False
    if condition == 'expired': ts = 1.1
    if condition == 'future': ts = -.1
    if condition == 'no_actor': event.actor_poses = []
    drawer = overlay.ActionPoseOverlay()
    assert drawer.draw(frame, frame, event, ts) is frame
    assert drawer.last_drawn == 0


def test_pose_follows_translation_instead_of_freezing(package):
    overlay = importlib.import_module(package + '.pose_overlay')
    rng = np.random.default_rng(42)
    gray = cv2.GaussianBlur(rng.integers(0, 255, (100, 160), dtype=np.uint8), (5, 5), 1)
    frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    event = observation(package, frame)
    drawer = overlay.ActionPoseOverlay()
    drawer.draw(frame, frame, event, 0)
    before = drawer._points.copy()
    moved = cv2.warpAffine(frame, np.float32([[1, 0, 4], [0, 1, 3]]), (160, 100))
    drawer.draw(moved, moved, event, .04)
    assert drawer.last_drawn == 1 and drawer._valid.sum() >= 10
    np.testing.assert_allclose((drawer._points - before)[drawer._valid],
                               np.tile([4, 3], (drawer._valid.sum(), 1)), atol=.5)


def test_gate_disables_pose_and_does_not_reuse_previous_actors(package, monkeypatch):
    reco = importlib.import_module(package + '.stages.recognize')
    model = reco.BehaviorRecognizer('unused', reject_tau=.75, stride=1)
    model.window = 2
    p = np.zeros(10, np.float32); p[0] = .9; p[9] = .1
    monkeypatch.setattr(model, '_preprocess', lambda f: np.zeros((1, 1, 3), np.float32))
    monkeypatch.setattr(model, '_infer', lambda arr: p.copy())
    model.skel = SimpleNamespace(probs=lambda raw: (p, .9), fuse=lambda pr, ps, pres: pr,
                                action_poses=lambda label: [dict(slot=0, keypoints=pose().tolist())])
    frame = np.full((100, 160, 3), 120, np.uint8)
    assert model.push(frame, 0, 0, raw_frame=frame) is None
    ev = model.push(frame, 1, .04, raw_frame=frame)
    assert ev.skeleton_active and ev.actor_poses
    ev = model.push(frame, 2, .08, raw_frame=np.zeros_like(frame))
    assert ev.label == 'Waving' and not ev.skeleton_active and not ev.actor_poses


def test_offline_chunk_uses_each_frames_event_not_future_event(package, monkeypatch, tmp_path):
    pipe = importlib.import_module(package + '.pipeline')
    Config = importlib.import_module(package + '.config').PipelineConfig
    Event = importlib.import_module(package + '.stages.base').RecognitionEvent
    seen = []
    class Reader:
        n_frames, fps = 3, 10.
        def __init__(self, *args, **kwargs): pass
        def chunks(self): yield [np.zeros((32, 32, 3), np.uint8) for _ in range(3)]
    class Writer:
        count = 0
        def __init__(self, *args, **kwargs): pass
        def write(self, frame): self.count += 1
        def close(self): pass
    class Rec:
        name, window, stride = 'fake', 2, 1
        def load(self, device): pass
        def close(self): pass
        def push(self, frame, idx, ts, **kw):
            return Event(idx, ts, 'Waving', .9) if idx == 1 else None
    monkeypatch.setattr(pipe, 'VideoReader', Reader)
    monkeypatch.setattr(pipe, 'VideoWriter', Writer)
    monkeypatch.setattr(pipe, 'build_stages', lambda cfg: ([], Rec()))
    def label(frame, event):
        seen.append(None if event is None else event.frame_index)
        return frame
    monkeypatch.setattr(pipe, 'append_label_bar', label)
    pipe.run_offline(Config(input='unused', output=str(tmp_path/'out.mp4'), device='cpu'))
    assert seen == [None, 1, 1]
