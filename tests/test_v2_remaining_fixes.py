"""Remaining v2 audit fixes: real files/subprocesses, CPU substitutes only for models."""
from collections import defaultdict
from dataclasses import asdict
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
OP = Path(os.environ.get('DARKPIPE_V2_OPERATOR_DIR', ROOT / 'platform/dark-demo-v2'))


@pytest.fixture(scope='module')
def pkg():
    name = '_darkpipe_v2_remaining'
    spec = importlib.util.spec_from_file_location(name, OP / 'darkpipe/__init__.py',
                                                submodule_search_locations=[str(OP / 'darkpipe')])
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    return name


def mod(pkg, name):
    return importlib.import_module(f'{pkg}.{name}')


class Event:
    label, confidence = 'Waving', 0.9

    def to_dict(self):
        return dict(label=self.label, confidence=self.confidence)


def recorder(clips, path):
    return clips.ClipRecorder(str(path / 'clips'), pre_sec=0, post_sec=100, max_sec=100,
                             denoise='off', realtime=False, stage_dir=str(path / 'stage'))


def feed(rec, count=10):
    for i in range(count):
        rec.push(np.full((32, 32, 3), i, np.uint8), Event() if i == 0 else None, i / 10)


class Writer:
    def __init__(self, path, fps):
        Path(path).write_bytes(b'staged-data')
        self.count = 0

    def write(self, frame):
        self.count += 1

    def close(self):
        pass


def test_slow_progress_drains_and_metadata_survives_move(pkg, monkeypatch, tmp_path):
    clips = mod(pkg, 'clips')
    monkeypatch.setattr(clips, 'VideoWriter', Writer)
    monkeypatch.setattr(clips, 'CLOSE_STALL_SEC', 0.15)
    rec = recorder(clips, tmp_path)

    def slow(frame):
        time.sleep(0.035)
        return frame

    rec._clean = slow
    feed(rec, 12)
    t0 = time.monotonic()
    rec.close()
    assert time.monotonic() - t0 > 0.3  # total > stall limit, still making progress
    rec.close()  # success is idempotent
    stats = rec.stats()
    assert stats['clips_saved'] == 1 and not stats['writer_alive']
    assert stats['clips_unfinished'] == stats['clips_failed'] == stats['write_errors'] == 0
    sidecar = next(Path(rec.root).rglob('*.json'))
    meta = json.loads(sidecar.read_text())
    assert not Path(meta['video']).is_absolute()
    assert (sidecar.parent / meta['video']).is_file()


def test_stalled_writer_reports_unfinished_and_keeps_staged_data(pkg, monkeypatch, tmp_path):
    clips = mod(pkg, 'clips')
    monkeypatch.setattr(clips, 'VideoWriter', Writer)
    monkeypatch.setattr(clips, 'CLOSE_STALL_SEC', 0.15)
    rec = recorder(clips, tmp_path)
    release = threading.Event()
    rec._clean = lambda frame: (release.wait(3), frame)[1]
    feed(rec)
    try:
        with pytest.raises(clips.ClipWriteError, match='无进展'):
            rec.close()
        stats = rec.stats()
        assert stats['writer_alive'] and stats['clips_unfinished'] == 1
        assert stats['clips_saved'] == 0 and stats['write_errors'] > 0
        assert list((tmp_path / 'stage').glob('*.mp4'))
    finally:
        release.set()
        rec._thread.join(2)
    assert not rec._thread.is_alive() and rec.stats()['clips_failed'] == 1
    with pytest.raises(clips.ClipWriteError):
        rec.close()


@pytest.mark.parametrize('fault', ['clean', 'write', 'close', 'move'])
def test_clip_errors_propagate_without_success_marker(pkg, monkeypatch, tmp_path, fault):
    clips = mod(pkg, 'clips')

    def fail(*a, **kw):
        raise OSError('injected ' + fault)

    class BrokenWriter(Writer):
        pass

    if fault in ('write', 'close'):
        setattr(BrokenWriter, fault, fail)
    monkeypatch.setattr(clips, 'VideoWriter', BrokenWriter)
    if fault == 'move':
        monkeypatch.setattr(clips.shutil, 'move', fail)
    rec = recorder(clips, tmp_path)
    if fault == 'clean':
        rec._clean = fail
    feed(rec)
    with pytest.raises(clips.ClipWriteError, match='injected'):
        rec.close()
    assert rec.saved == 0 and not rec._thread.is_alive()
    assert rec.stats()['clips_failed'] == 1 and rec.stats()['write_errors'] > 0
    assert not list(Path(rec.root).rglob('*.json'))
    with pytest.raises(clips.ClipWriteError):
        rec.push(np.zeros((32, 32, 3), np.uint8), Event(), 2)


@pytest.mark.parametrize('device,expected', [('cuda:2', 2), ('cuda', 0), ('cpu', None)])
def test_detector_provider_uses_requested_device(pkg, monkeypatch, tmp_path, device, expected):
    prior = mod(pkg, 'stages.object_prior')
    (tmp_path / prior.YOLO_ONNX).touch()
    seen = []

    def session(path, providers):
        seen.append(providers)
        return SimpleNamespace(get_providers=lambda: ['CUDAExecutionProvider', 'CPUExecutionProvider'])

    monkeypatch.setitem(sys.modules, 'onnxruntime', SimpleNamespace(InferenceSession=session))
    detector = prior.YoloOnnxDetector(str(tmp_path), device)
    monkeypatch.setattr(detector, '_frames_to_dets', lambda frames: [])
    assert detector.load() == ''
    assert seen == [(['CPUExecutionProvider'] if expected is None else
                     [('CUDAExecutionProvider', {'device_id': expected})])]


def test_object_threshold_filters_boxes_without_changing_person_detector(pkg):
    prior = mod(pkg, 'stages.object_prior')
    rows = np.array([[10, 10, 20, 20, .30, 0],
                     [30, 30, 40, 40, .40, 39],
                     [50, 50, 60, 60, .60, 41]], dtype=np.float32)
    session = SimpleNamespace(get_inputs=lambda: [SimpleNamespace(name='images')],
                              run=lambda *args: [rows[None].copy()])
    person = prior.YoloOnnxDetector('.', 'cpu')
    gate = prior.PriorGate('.', 'cpu', min_conf=.35)
    stricter = prior.PriorGate('.', 'cpu', min_conf=.50)
    image = np.zeros((640, 640, 3), np.uint8)
    outputs = []
    for detector in (person, gate.det, stricter.det):
        detector.sess = session
        outputs.append(detector._frames_to_dets([image])[0][:, 5].tolist())
    assert outputs == [[0., 39., 41.], [39., 41.], [41.]]
    assert person.min_conf == .25  # Shared ONNX wrapper must keep person filtering independent.


@pytest.mark.parametrize('value', [-.01, 1.01, float('nan'), float('inf')])
def test_object_threshold_rejects_invalid_config_and_direct_detector(pkg, value):
    config = mod(pkg, 'config')
    cfg = config.PipelineConfig(input='probe', enhance='off', sr='off', recognize='off',
                                object_min_conf=value)
    with pytest.raises(SystemExit, match='object_min_conf'):
        config.validate(cfg)
    with pytest.raises(ValueError, match='min_conf'):
        mod(pkg, 'stages.object_prior').YoloOnnxDetector('.', 'cpu', min_conf=value)


@pytest.mark.parametrize('fault', ['create', 'fallback', 'warmup', 'timing', 'runtime', 'gate_warmup'])
def test_detector_errors_fail_soft(pkg, monkeypatch, tmp_path, fault):
    prior = mod(pkg, 'stages.object_prior')
    (tmp_path / prior.YOLO_ONNX).touch()

    def session(*a, **kw):
        if fault == 'create':
            raise RuntimeError('session failed')
        return SimpleNamespace(get_providers=lambda: (['CPUExecutionProvider'] if fault == 'fallback'
                                                      else ['CUDAExecutionProvider']))

    monkeypatch.setitem(sys.modules, 'onnxruntime', SimpleNamespace(InferenceSession=session))
    gate = prior.PriorGate(str(tmp_path), 'cuda:1')
    calls = []

    def infer(frames):
        calls.append(1)
        if ((fault == 'warmup' and len(calls) == 1) or
                (fault == 'timing' and len(calls) == 3) or
                (fault in ('runtime', 'gate_warmup') and len(calls) > 5)):
            raise RuntimeError('inference failed')
        return []

    monkeypatch.setattr(gate.det, '_frames_to_dets', infer)
    reason = gate.load()
    if fault in ('runtime', 'gate_warmup'):
        assert reason == ''
        if fault == 'gate_warmup':
            gate.warmup()
        else:
            assert gate.det([np.zeros((10, 10, 3))]) is None
    else:
        assert reason
    assert gate.det.sess is None
    before = len(calls)
    assert gate.det([]) is None and len(calls) == before
    assert gate.check('drink', None, None, [])[0]  # malformed evidence also fails open


def test_shard_config_preserves_all_decision_and_picture_fields(pkg, tmp_path):
    config = mod(pkg, 'config')
    shard = mod(pkg, 'shard')
    cfg = config.PipelineConfig(input='sample.mp4', reco_min_conf=.77, skel_fusion_alpha=.8,
                                skel_no_person='rgb', fuse_mode='max_prior', lum_gate=81.125,
                                act_tau=.71, skel_tau=.69, proc_max_side=720, denoise='fast',
                                color_saturation=1.4, sr='bicubic', sr_scale=3)
    result = shard.worker_config(cfg, '2', 160, 140, '/tmp/out.mp4', '/tmp/events.json')
    overridden = {'input', 'ckpt_dir', 'mode', 'device', 'gpus', 'start_frame', 'max_frames',
                  'output', 'events_json', 'warnings', 'stats'}
    for key, value in asdict(cfg).items():
        if key not in overridden:
            assert result[key] == value, key
    assert result['device'] == 'cuda:2' and result['start_frame'] == 160


def make_video(path, n):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 20, (64, 48))
    assert writer.isOpened()
    for i in range(n):
        writer.write(np.full((48, 64, 3), i % 200, np.uint8))
    writer.release()


def test_real_multishard_worker_and_merge(pkg, tmp_path):
    config, shard = mod(pkg, 'config'), mod(pkg, 'shard')
    src = tmp_path / 'source.mp4'
    make_video(src, 320)
    cfg = config.PipelineConfig(input=str(src), output=str(tmp_path / 'out.mp4'), device='cpu',
                                enhance='off', sr='off', recognize='off', start_frame=10,
                                max_frames=300, events_json=str(tmp_path / 'events.json'),
                                proc_max_side=32)
    events = shard.run_offline_sharded(cfg, ['0', '1'])
    assert events == [] and cfg.stats['gpus'] == 2
    assert cfg.stats['frames_written'] == cfg.stats['frames'] == 300
    cap = cv2.VideoCapture(cfg.output)
    means = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        assert frame.shape[:2] == (24, 32)  # proc_max_side reached both workers
        means.append(frame.mean())
    cap.release()
    assert len(means) == 300
    # Boundary/order and start offset survive seeking, worker encoding, and concatenation.
    assert abs(means[0] - 7) < 7 and abs(means[149] - 156) < 7
    assert abs(means[150] - 157) < 7 and abs(means[-1] - 106) < 7
    assert json.loads(Path(cfg.events_json).read_text()) == []
    assert not list(tmp_path.glob('.shard_*'))


def test_shard_failure_has_no_partial_output(pkg, tmp_path):
    config, shard = mod(pkg, 'config'), mod(pkg, 'shard')
    src = tmp_path / 'source.mp4'
    make_video(src, 240)
    cfg = config.PipelineConfig(input=str(src), output=str(tmp_path / 'out.mp4'), device='cpu',
                                enhance='retinexformer', sr='off', recognize='off',
                                ckpt_dir=str(tmp_path / 'missing'))
    with pytest.raises(RuntimeError, match='segment.*exited'):
        shard.run_offline_sharded(cfg, ['0', '1'])
    assert not Path(cfg.output).exists() and not list(tmp_path.glob('.shard_*'))


def test_concat_refuses_truncated_segment(pkg, tmp_path):
    shard = mod(pkg, 'shard')
    part = tmp_path / 'part.mp4'
    make_video(part, 5)
    with pytest.raises(RuntimeError, match='expected 10 frames, decoded 5'):
        shard._concat([str(part)], str(tmp_path / 'merged.mp4'), 20, [10])


@pytest.fixture
def evaluation(pkg, monkeypatch):
    # Load the research tools against this exact deployed package, even when root tests
    # have already imported the root-level darkpipe. Restore aliases after each test.
    for suffix in ('', '.constants', '.stages', '.stages.downscale',
                   '.stages.enhance_retinexformer', '.stages.object_prior',
                   '.stages.recognize', '.stages.skeleton_aux'):
        monkeypatch.setitem(sys.modules, 'darkpipe' + suffix,
                            importlib.import_module(pkg + suffix))
    loaded = []
    original_path = sys.path[:]
    for name in ('prior_eval', 'lum_sweep'):
        spec = importlib.util.spec_from_file_location('_test_' + name,
            ROOT / 'compare/behavior/skeleton' / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        loaded.append(module)
    sys.path[:] = original_path
    return loaded


@pytest.mark.parametrize('presence', [0.0, .49999997, .5, .50000006, 1.0])
@pytest.mark.parametrize('policy', ['soft', 'other', 'rgb'])
def test_eval_alpha_matches_production_and_json_roundtrip(pkg, evaluation, presence, policy):
    ev, sweep = evaluation
    skel = mod(pkg, 'stages.skeleton_aux')
    recognize = mod(pkg, 'stages.recognize')
    rgb = np.array([.99] + [0.] * 8 + [.01], np.float32)
    ps = np.array([.4, .1] + [0.] * 7 + [.5], np.float32)
    expected = skel.SkeletonAux.fuse(SimpleNamespace(alpha=.7, no_person=policy), rgb, ps, presence)
    assert np.array_equal(expected, ev.alpha_fuse(rgb, ps, .7, presence, policy))
    stack = SimpleNamespace(labels=ev.LABELS, lum_gate=75., act_tau=.65, skel_tau=.65,
                            prior=None, t_parts=defaultdict(float))
    info = dict(p=None, pres=presence, sk=None, valid=np.zeros(6, bool))
    row = ev.decision_record(stack, rgb, info, ps, presence, [], [75.000001], [.7, .8], policy)
    row = json.loads(json.dumps(row))
    assert np.array_equal(sweep.alpha_outcome(row, 75, 'a0.70'),
                          recognize.reject_probs(expected, ev.LABELS, .65))
    if presence == 0 and policy == 'soft':
        assert sweep.fired(sweep.alpha_outcome(row, 75, 'a0.70'))


@pytest.mark.parametrize('lum', [74.9999999, 75., 75.0000001])
@pytest.mark.parametrize('prob', [.6499999, .65, .6500001, .8189])
def test_eval_gate_reject_ties_and_ungated_replay(pkg, evaluation, lum, prob):
    ev, sweep = evaluation
    recognize = mod(pkg, 'stages.recognize')
    rgb = np.array([prob] + [0.] * 8 + [1 - prob], np.float32)
    info = dict(p=None, pres=0., sk=None, valid=np.zeros(6, bool))
    stack = SimpleNamespace(labels=ev.LABELS, lum_gate=75., act_tau=.65, skel_tau=.65,
                            prior=SimpleNamespace(check=lambda *a: (False, 'test veto')),
                            t_parts=defaultdict(float))
    row = json.loads(json.dumps(ev.decision_record(stack, rgb, info, rgb, 0., [], [lum], [.7, .8])))
    expected, _ = recognize.BehaviorRecognizer._decide_max_prior(stack, rgb, info, [], [lum])
    expected = recognize.reject_probs(expected, ev.LABELS, .65)
    assert np.array_equal(sweep.outcome(row, 75), expected)
    assert np.array_equal(np.asarray(row['mp'], np.float32), expected)
    if prob > .65:
        assert sweep.outcome(row, 0).argmax() == ev.OTHER  # +1e-6 veto survives JSON
        assert sweep.outcome(row, 80).argmax() == 0
    assert row['lum'] == lum


def test_sweep_rejects_old_rounded_json(evaluation):
    _, sweep = evaluation
    with pytest.raises(ValueError, match='重新导出'):
        sweep.outcome(dict(rgb=[.99] + [0.] * 9, lum=75., mp=[1.] + [0.] * 9), 75)


@pytest.mark.parametrize('batch', [False, True])
def test_offline_failure_keeps_workdir_and_never_delivers(tmp_path, batch):
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    src = inputs / 'first.mp4'
    make_video(src, 12)
    if batch:
        make_video(inputs / 'second.mp4', 12)
    session = tmp_path / 'session.json'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    proc = subprocess.run([sys.executable, str(ROOT / 'tests/v2_offline_probe.py'),
        '--op-dir', str(OP), '--input', str(inputs if batch else src), '--out', str(session)],
        env=env, capture_output=True, text=True, timeout=30)
    assert proc.returncode != 0, proc.stdout + proc.stderr
    payload = json.loads(session.read_text())
    assert payload['status'] == 'failed'
    assert payload['clips']['write_errors'] > 0
    recovery = Path(payload['recovery_dir'])
    assert recovery.is_dir() and list(recovery.rglob('*.mp4'))
    assert not list((tmp_path / 'delivered').rglob('*.mp4'))
    if batch:
        assert len(payload['items']) == 1 and payload['items'][0]['status'] == 'failed'
    # Test-created recovery data only; production deliberately retains it for the operator.
    import shutil
    shutil.rmtree(recovery)


def test_service_clip_close_error_fails_session(tmp_path):
    import socket
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    proc = subprocess.run([sys.executable, str(ROOT / 'tests/v2_serve_probe.py'),
        '--op-dir', str(OP), '--port', str(port), '--work-dir', str(tmp_path),
        # Allow startup to complete under concurrent CPU/encoding load; the intended
        # fault is clip cleanup, not Uvicorn cancelling an unfinished lifespan startup.
        '--seconds', '3', '--fault', 'clip_close'],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), capture_output=True, text=True, timeout=30)
    assert proc.returncode != 0, proc.stdout + proc.stderr
    session = json.loads((tmp_path / 'session.json').read_text())
    assert session['status'] == 'failed'
    assert session['clips']['write_errors'] > 0
    assert 'probe clip close failure' in proc.stdout + proc.stderr


def test_real_shards_rebase_recognition_events(pkg, monkeypatch, tmp_path):
    config, shard = mod(pkg, 'config'), mod(pkg, 'shard')
    src = tmp_path / 'source.mp4'
    make_video(src, 300)
    hook = tmp_path / 'cpu-model'
    hook.mkdir()
    (hook / 'sitecustomize.py').write_text('''
from darkpipe import config, pipeline
from darkpipe.stages.base import RecognitionEvent
config.validate = lambda cfg: cfg
class Recognizer:
    name, window, stride = 'cpu-test', 32, 16
    def load(self, device): pass
    def close(self): pass
    def push(self, frame, idx, ts, **kw):
        if idx >= 31 and (idx - 31) % 16 == 0:
            return RecognitionEvent(idx, ts, 'wave', .9)
pipeline.build_stages = lambda cfg: ([], Recognizer())
''')
    monkeypatch.setenv('PYTHONPATH', str(hook))
    cfg = config.PipelineConfig(input=str(src), output=str(tmp_path / 'out.mp4'),
                                events_json=str(tmp_path / 'events.json'), device='cpu',
                                enhance='off', sr='off', recognize='behavior', no_label_bar=True)
    events = shard.run_offline_sharded(cfg, ['0', '1'])
    expected = list(range(31, 150, 16)) + list(range(181, 300, 16))
    assert [e['frame_index'] for e in events] == expected
    assert [e['timestamp'] for e in events] == pytest.approx([idx / 20 for idx in expected])
    assert json.loads(Path(cfg.events_json).read_text()) == events
    assert cfg.stats['events'] == len(expected)


def test_partial_worker_launch_terminates_started_child(pkg, monkeypatch, tmp_path):
    config, shard = mod(pkg, 'config'), mod(pkg, 'shard')
    monkeypatch.setattr(shard, '_frame_count', lambda path: (300, 20))
    state = {'started': False, 'terminated': False, 'waited': False}

    class Child:
        def poll(self):
            return 0 if state['terminated'] else None

        def terminate(self):
            state['terminated'] = True

        def wait(self, timeout):
            state['waited'] = True
            return 0

    def spawn(*a, **kw):
        if state['started']:
            raise OSError('injected spawn failure')
        state['started'] = True
        return Child()

    monkeypatch.setattr(shard.subprocess, 'Popen', spawn)
    cfg = config.PipelineConfig(input='unused', output=str(tmp_path / 'out.mp4'), device='cpu')
    with pytest.raises(OSError, match='spawn failure'):
        shard.run_offline_sharded(cfg, ['0', '1'])
    assert state['terminated'] and state['waited']
    assert not list(tmp_path.glob('.shard_*'))
