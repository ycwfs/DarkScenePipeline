import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import statistics
import sys
import time

import cv2
import numpy as np
import torch

PKG = Path('/opt/darkpipe/op')
sys.path.insert(0, str(PKG))
from darkpipe.constants import CKPT_FILES
from darkpipe.stages.recognize import BehaviorRecognizer
from darkpipe.stages.object_prior import YoloOnnxDetector

torch.set_num_threads(4)
cv2.setNumThreads(2)
out = {'versions': {n: md.version(n) for n in (
    'torch', 'torchvision', 'numpy', 'opencv-python-headless', 'timm', 'rtmlib',
    'onnxruntime-gpu', 'mamba-ssm', 'causal-conv1d', 'triton')},
    'cuda_available': torch.cuda.is_available(), 'cuda_count': torch.cuda.device_count(),
    'library_path': os.environ.get('LD_LIBRARY_PATH'),
    'weights': {}}
assert out['cuda_available']
for p in sorted((PKG / 'ckpts').iterdir()):
    out['weights'][p.name] = {'md5': hashlib.md5(p.read_bytes()).hexdigest(),
                              'bytes': p.stat().st_size}
path = PKG / 'ckpts' / CKPT_FILES['behavior']
meta = torch.load(path, map_location='cpu')
out['rgb_metadata'] = {k: str(v)[:1500] for k, v in meta.items()
                       if k not in ('model', 'optimizer', 'scaler', 'scheduler')}
r = BehaviorRecognizer(str(PKG / 'ckpts'), span_sec=1.0, reject_tau=0.65)
r.load('cuda:1')
assert r.skel is not None, 'skeleton unexpectedly disabled'
assert r.fov == (224, 400), r.fov
out['rgb_fov'] = r.fov
out['pose_probe_ms'] = r.skel.pose_ms
out['pose_object_attributes'] = list(vars(r.skel.pose))
for k, v in vars(r.skel.pose).items():
    if hasattr(v, 'get_providers'):
        out['pose_providers'] = v.get_providers()
        out['pose_provider_options'] = v.get_provider_options()
assert 'CUDAExecutionProvider' in out.get('pose_providers', [])
r.warmup()
torch.cuda.set_device(1)
arr = np.zeros((32, 224, 400, 3), np.float32)
times = []
for _ in range(5):
    t = time.perf_counter()
    p = r._infer(arr)
    times.append((time.perf_counter() - t) * 1000)
assert p.shape == (10,) and np.isfinite(p).all()
out['rgb_forward_ms'] = times
out['rgb_forward_median_ms'] = statistics.median(times)
with torch.cuda.device(1):
    detector = YoloOnnxDetector(str(PKG / 'ckpts'), 'cuda:1')
    out['optional_detector_load_reason'] = detector.load()
    if detector.sess is not None:
        out['optional_detector_provider_options'] = detector.sess.get_provider_options()
        out['optional_detector_probe_ms'] = detector.det_ms
r.close()
Path('/audit/image-probe.json').write_text(json.dumps(out, ensure_ascii=False, indent=2))
print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
