"""Native deployment wrapper boundaries; the pipeline itself uses the v2 regressions."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BOX = Path(os.environ.get('NVIDIA_BOX_ROOT', ROOT / 'NVIDIA-Box'))
spec = importlib.util.spec_from_file_location('_box_native_launcher', BOX / 'run.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def parsed(argv):
    actual = launcher.prepare_argv(argv)
    return dict(zip(actual[::2], actual[1::2]))


def test_defaults_keep_algorithm_but_use_local_paths():
    args = parsed(['--video_path', 'rtsp://camera/live'])
    assert args['--ckpt_dir'] == str(BOX.resolve() / 'weights')
    assert args['--extra_hosts'] == args['--hdfs_output_dir'] == ''
    assert args['--gpu_ids'] == '0'
    assert args['--reco_min_conf'] == '0.925'
    assert args['--lum_gate'] == '35.0'
    assert args['--person_mode'] == 'true'
    assert args['--proc_max_side'] == '720'
    assert args['--pose_overlay'] == 'true'
    assert args['--sr'] == 'off'
    assert Path(args['--session_json']).parent == BOX.resolve() / 'outputs/sessions'


def test_relative_paths_follow_callers_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = parsed(['--video_path', '视频.mp4', '--output-dir', '结果 目录',
                   '--ckpt_dir', '模型', '--session_json', '状态.json'])
    assert args['--video_path'] == '视频.mp4'
    assert args['--clip_dir'] == str(tmp_path / '结果 目录')
    assert args['--ckpt_dir'] == str(tmp_path / '模型')
    assert args['--session_json'] == str(tmp_path / '状态.json')
    assert '--output-dir' not in args


def test_sessions_are_unique():
    one = parsed(['--video_path', 'a.mp4'])
    two = parsed(['--video_path', 'a.mp4'])
    assert one['--session_json'] != two['--session_json']


@pytest.mark.parametrize('flag', ['--output-dir', '--clip_dir', '--session_json'])
def test_empty_destination_is_rejected(flag):
    with pytest.raises(SystemExit) as exc:
        parsed(['--video_path', 'a.mp4', flag, ''])
    assert exc.value.code == 2


def test_explicit_platform_options_survive():
    args = parsed(['--video_path', 'rtsp://camera/live', '--gpu_ids', '0,1',
                   '--stream_formats', 'mjpeg,flv,hls', '--run_seconds', '30',
                   '--pose_overlay', 'false', '--reco_min_conf', '0.8'])
    assert args['--gpu_ids'] == '0,1'
    assert args['--stream_formats'] == 'mjpeg,flv,hls'
    assert args['--run_seconds'] == '30.0'
    assert args['--pose_overlay'] == 'false'
    assert args['--reco_min_conf'] == '0.8'


def test_environment_keeps_visible_gpus_and_removes_foreign_import_path(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '4')
    monkeypatch.setenv('PYTHONPATH', '/unrelated/repo')
    monkeypatch.setenv('LD_LIBRARY_PATH', '/custom/cuda')
    env = launcher.runtime_environment()
    assert env['CUDA_VISIBLE_DEVICES'] == '4'
    assert 'PYTHONPATH' not in env
    assert env['LD_LIBRARY_PATH'].endswith('/custom/cuda')
    assert 'PYTHONPATH' in os.environ


def test_missing_input_writes_failed_session_and_exits(tmp_path):
    dest = tmp_path / 'results'
    result = subprocess.run([sys.executable, str(BOX / 'run.py'), '--video_path',
                             str(tmp_path / 'missing.mp4'), '--output-dir', str(dest)],
                            cwd=tmp_path, capture_output=True, text=True, timeout=20, check=False)
    assert result.returncode != 0
    sessions = list((dest / 'sessions').glob('*.json'))
    assert len(sessions) == 1
    payload = json.loads(sessions[0].read_text())
    assert payload['status'] == 'failed'
    assert 'missing.mp4' in payload['error']


def test_runtime_sources_have_only_reviewed_box_overrides():
    import hashlib
    overrides = json.loads((BOX/'runtime-overrides.json').read_text())
    for path in (BOX / 'runtime').rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts:
            rel = path.relative_to(BOX/'runtime').as_posix()
            base = ROOT/'platform/dark-demo-v2'/rel
            if rel in overrides['added']:
                assert not base.exists()
            elif rel in overrides['modified']:
                assert hashlib.sha256(base.read_bytes()).hexdigest() == overrides['modified'][rel]['platform_sha256']
            else:
                assert path.read_bytes() == base.read_bytes()


def test_box_options_are_local_and_explicit_output_upscale_survives(monkeypatch):
    args = parsed(['--video_path', 'a.mp4', '--box-profile', 'reference',
                   '--box-backend', 'tensorrt', '--sr', 'bicubic', '--sr_scale', '2'])
    assert '--box_profile' not in args and '--box_backend' not in args
    assert os.environ['DARKPIPE_BOX_PROFILE'] == 'reference'
    assert os.environ['DARKPIPE_BOX_BACKEND'] == 'tensorrt'
    assert args['--sr'] == 'bicubic' and args['--sr_scale'] == '2'
