#!/usr/bin/env python3
"""Verify the delivered files and native production runtime, including real CUDA forwards."""
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from run import ROOT, runtime_environment


def verify_files() -> str:
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    for name, record in manifest['files'].items():
        path = ROOT / name
        if not path.is_file():
            raise RuntimeError(f'缺少文件: {name}')
        if path.stat().st_size != record['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise RuntimeError(f'文件被修改或损坏: {name}')
    return f"{manifest['version']}: {len(manifest['files'])} 份文件 SHA256 一致"


def check_system() -> str:
    assert sys.version_info[:2] == (3, 10), '需要 Python 3.10'
    assert platform.system() == 'Linux' and platform.machine() in ('x86_64', 'aarch64'), '需要 Linux x86_64 或 Jetson ARM64'
    for name in ('ffmpeg', 'ffprobe', 'gcc'):
        assert shutil.which(name), f'缺少 {name}'
    for flag, names in (('-encoders', ('libx264',)), ('-muxers', ('flv', 'hls'))):
        result = subprocess.run(['ffmpeg', '-hide_banner', flag], capture_output=True, text=True, check=True)
        assert all(name in result.stdout for name in names), f'FFmpeg 缺少 {names}'
    return f'{platform.machine()}, Python {platform.python_version()}, FFmpeg H.264/FLV/HLS'


def check_dependencies() -> str:
    from importlib.metadata import PackageNotFoundError, version

    import causal_conv1d  # noqa: F401
    import cv2
    import fastapi
    import mamba_ssm  # noqa: F401
    import torch
    import uvicorn  # noqa: F401
    assert torch.cuda.is_available(), 'torch 检测不到 CUDA'
    sys.path.insert(0, str(ROOT / 'runtime'))
    from darkpipe.box_onnx import backend
    if backend() == 'tensorrt':
        import tensorrt
        assert int(tensorrt.__version__.split('.')[0]) >= 10, '需要 TensorRT 10+'
    else:
        import onnxruntime as ort
        assert 'CUDAExecutionProvider' in ort.get_available_providers(), 'ORT 未提供 CUDA EP'
        try:
            version('onnxruntime')
        except PackageNotFoundError:
            pass
        else:
            raise RuntimeError('环境同时装了 CPU onnxruntime，请移除并重新安装 onnxruntime-gpu')
    return f'torch {torch.__version__}, GPU {torch.cuda.get_device_name(0)}, cv2 {cv2.__version__}, fastapi {fastapi.__version__}'


def check_pipeline() -> str:
    sys.path.insert(0, str(ROOT / 'runtime'))
    import time
    import cv2
    import numpy as np
    import torch
    from darkpipe.config import PipelineConfig
    from darkpipe.person.runtime import PersonEngine
    from darkpipe.person.core import Sample, fuse_person
    from darkpipe.person.action_engine import ActionTask
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    cfg = PipelineConfig(person_mode=True, recognize='behavior', gpus='0',
        device='cuda:0', ckpt_dir=str(ROOT / 'weights'), proc_max_side=720,
        skel_fusion_alpha=.972, reco_min_conf=.925, lum_gate=35.)
    engine = None
    try:
        engine = PersonEngine(cfg, time.monotonic(), lambda row: None, realtime=False)
        dark = cv2.imread(str(ROOT / 'sample_data/dark_frame.png'))
        assert dark is not None
        enhanced, luma = engine.prepare(dark)
        assert enhanced.mean() > dark.mean(), '增强前向异常'
        engine.front(enhanced, True)  # Real new YOLO + RTMO inference, even if no person.
        crop = np.full((224, 400, 3), 128, np.uint8)
        pose = np.zeros((17, 3), np.float32)
        pose[:, 0] = np.linspace(.3, .7, 17)
        pose[:, 1] = np.linspace(.1, .9, 17)
        pose[:, 2] = .95
        rows = []
        for j, brightness in enumerate((20., 60.)):
            window = tuple(Sample(i/31, i, crop, np.array([0,0,400,224]), pose, brightness)
                           for i in range(32))
            task = ActionTask((1,j,0), window, (0,0,400,224), .95, time.monotonic(), time.monotonic()-1)
            rows.extend(engine.pool.engines[0].predict([task]))
        assert rows[0]['mode'] == 'rgb_low_luma'
        assert rows[1]['mode'] == 'rgb_skeleton' and rows[1]['pose_valid']
        assert all(np.isfinite(row['fused_probs']).all() for row in rows)
        detector = engine.pool.engines[0].object_prior
        assert detector is not None
        detections = detector.detector._frames_to_dets([enhanced])[0]
        assert detections.shape[1] == 6 and np.isfinite(detections).all()
        return '实际交付的 Retinexformer / YOLO26m / RTMO-s / ROI VideoMamba / ST-GCN / 七类物体YOLO26m 前向通过；门35分支通过（合成骨架只作接口检查）'
    finally:
        if engine is not None:
            engine.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--files-only', action='store_true', help='仅校验交付文件哈希，无需 GPU 依赖')
    parser.add_argument('--prepared-env', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.files_only and not args.prepared_env:
        os.execve(sys.executable, [sys.executable, '-u', str(Path(__file__).resolve()),
                                  *sys.argv[1:], '--prepared-env'], runtime_environment())
    checks = [('交付完整性', verify_files)]
    if not args.files_only:
        checks += [('系统与编码器', check_system), ('CUDA 与服务依赖', check_dependencies),
                   ('平台完整模型链', check_pipeline)]
    failures = []
    for name, check in checks:
        try:
            print(f'[OK] {name}: {check()}', flush=True)
        except Exception as exc:  # noqa: BLE001 -- report each independent environment check
            failures.append(name)
            print(f'[FAIL] {name}: {type(exc).__name__}: {exc}', flush=True)
    if failures:
        raise SystemExit(f'自检失败: {failures}')
    print('全部通过。完整离线/流式流程请使用 bash run.sh --video_path ...')


if __name__ == '__main__':
    main()
