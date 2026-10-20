"""CPU contract checks against the current operator; no CUDA allocations."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import os
import sys

import cv2
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
BOX = Path(os.environ.get('NVIDIA_BOX_ROOT', ROOT / 'NVIDIA-Box'))
sys.path.insert(0, str(BOX))
sys.path.insert(0, str(ROOT / 'platform/dark-demo-v2'))
import behavior_infer as box
import lowlight_infer as enhance
import skeleton_aux as skeleton
import video_io
from darkpipe.constants import RECO_CFG, BEHAVIORS
from darkpipe.stages import recognize as deployed
from darkpipe.stages.downscale import DownscaleStage
from darkpipe.stages import skeleton_aux as deployed_skeleton


@pytest.mark.parametrize('shape', [(1080, 1920), (471, 839), (224, 400)])
@pytest.mark.parametrize('fov', [None, (224, 400)])
def test_preprocess_matches_deployment(shape, fov):
    frame = np.random.default_rng(42).integers(0, 256, (*shape, 3), dtype=np.uint8)
    cfg = dict(RECO_CFG['behavior'])
    if fov:
        cfg['fov'] = fov
    assert np.array_equal(box.preprocess_frame(frame, fov), deployed.preprocess_frame(frame, cfg))
    assert box.frame_luma(frame) == deployed.frame_luma(frame)


@pytest.mark.parametrize('values,gate', [([74.9] * 32, 75), ([75.] * 32, 75),
                                       ([135.] * 32, 75), ([0.] * 32, 0), ([], 75), ([None], 75)])
def test_gate_matches_deployment(values, gate):
    assert box.gate_luma(values, gate) == deployed.gate_luma(values, gate)


@pytest.mark.parametrize('mode', ['soft', 'other', 'rgb'])
@pytest.mark.parametrize('presence', [0., .4999, .5, .9])
def test_fusion_and_rejection_match_deployment(mode, presence):
    a = skeleton.SkeletonAux(str(BOX / 'weights'), 'cpu', .7, no_person=mode)
    b = deployed_skeleton.SkeletonAux(str(BOX / 'weights'), 'cpu', .7, no_person=mode)
    rng = np.random.default_rng(10)
    for _ in range(10):
        pr, ps = rng.dirichlet(np.full(10, .1), 2).astype(np.float32)
        actual = a.fuse(pr, ps, presence)
        expected = b.fuse(pr, ps, presence)
        assert np.array_equal(actual, expected)
        idx, result = box.reject_probs(actual)
        deployed_result = deployed.reject_probs(expected, BEHAVIORS, .65)
        assert np.array_equal(result, deployed_result)
        assert idx == int(deployed_result.argmax())


def test_pose_probe_warms_before_timing(monkeypatch):
    calls = []
    class Pose:
        def __init__(self, **kw):
            pass
        def __call__(self, frame):
            calls.append(frame.shape)
            return [], []
    monkeypatch.setitem(sys.modules, 'rtmlib', SimpleNamespace(RTMO=Pose))
    aux = skeleton.SkeletonAux(str(BOX / 'weights'), 'cpu', .7)
    assert aux.load() == ''
    assert len(calls) == 6  # three untimed calls and three timed calls, same as operator


def test_skeleton_real_head_matches_deployment():
    # Fixed deterministic pose output isolates ST-GCN + normalization + person merge.
    torch.set_num_threads(2)
    kp = np.random.default_rng(12).uniform([50, 30], [380, 220], (3, 17, 2)).astype(np.float32)
    conf = np.stack([np.full(17, p, np.float32) for p in [.92, .73, .1]])
    state = torch.load(BOX / 'weights/stgcn_behavior_32f_realactall_full.pth', map_location='cpu')['model']
    instances = []
    for module in [skeleton, deployed_skeleton]:
        aux = module.SkeletonAux(str(BOX / 'weights'), 'cpu', .7)
        aux.pose = lambda frame: (kp, conf)
        aux.net = module.STGCN(num_classes=10).eval()
        aux.net.load_state_dict(state)
        instances.append(aux)
    frames = [np.zeros((270, 480, 3), np.uint8)] * 32
    a, b = [aux.probs(frames) for aux in instances]
    assert np.array_equal(a[0], b[0])
    assert a[1] == b[1]


def test_recognition_cpu_rejected():
    with pytest.raises(SystemExit, match='只支持 CUDA'):
        box.BehaviorRecognizer(device='cpu')


def test_raw_window_alignment_required():
    rec = box.BehaviorRecognizer.__new__(box.BehaviorRecognizer)
    frame = np.zeros((20, 30, 3), np.uint8)
    with pytest.raises(ValueError, match='等长同序'):
        rec.predict([frame] * 32, raw_frames_bgr=[frame])
    with pytest.raises(ValueError, match='不能为空'):
        rec.predict([])


def test_recognition_cli_routes_device_and_720(monkeypatch):
    calls = {}
    class Enhancer:
        def __init__(self, weights, device):
            calls['enhance_device'] = device
        def enhance(self, frames):
            return frames
    class Recognizer:
        def __init__(self, weights, device, **kwargs):
            calls['recognize_device'] = device
            self.device, self.aux, self.input_pipeline = device, None, 'fullfov224x400'
            self.net = SimpleNamespace(parameters=lambda: [])
        def predict(self, frames, **kwargs):
            calls['shape'] = frames[0].shape
            return dict(label='other', label_zh='其他', label_reco='other', label_zh_reco='其他',
                        confidence=1., confidence_reco=1., probs_final={'other': 1.})
    monkeypatch.setattr(enhance, 'LowLightEnhancer', Enhancer)
    monkeypatch.setattr(box, 'BehaviorRecognizer', Recognizer)
    monkeypatch.setattr(box, 'read_image', lambda p: np.zeros((1080, 1920, 3), np.uint8))
    monkeypatch.setattr(sys, 'argv', ['behavior_infer.py', '--input', 'frame.png', '--enhance', '--device', 'cuda:2'])
    box.main()
    assert calls == {'enhance_device': 'cuda:2', 'recognize_device': 'cuda:2', 'shape': (404, 720, 3)}


def test_enhance_cli_defaults_to_720(monkeypatch):
    calls = {}
    monkeypatch.setattr(enhance, 'LowLightEnhancer', lambda *a: SimpleNamespace(device='cpu', fp16=False))
    def run(model, src, dst, proc):
        calls['proc'] = proc
    monkeypatch.setattr(enhance, 'run_one', run)
    monkeypatch.setattr(sys, 'argv', ['lowlight_infer.py', '--input', 'frame.png'])
    enhance.main()
    assert calls['proc'] == 720


def test_explicit_video_output_is_file_and_decodes(tmp_path):
    output = tmp_path / 'nested' / 'enhanced.mp4'
    args = SimpleNamespace(output=str(output), _srcs=['demo.mp4'])
    destination = enhance.decide_output('demo.mp4', args)
    assert destination == str(output) and not output.is_dir()
    frames = [np.full((48, 64, 3), i * 15, np.uint8) for i in range(5)]
    video_io.encode_video(frames, destination, 25)
    decoded, fps = video_io.read_video(destination)
    assert len(decoded) == len(frames) and decoded[0].shape == frames[0].shape
    assert fps == 25
