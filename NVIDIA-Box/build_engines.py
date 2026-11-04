#!/usr/bin/env python3
"""Build device-local TensorRT engines before loading the full model chain."""
import gc
import os
from pathlib import Path
import sys

from run import ROOT, runtime_environment

if __name__ == '__main__':
    if '--prepared-env' not in sys.argv:
        os.execve(sys.executable, [sys.executable, '-u', str(Path(__file__).resolve()), '--prepared-env'],
                  runtime_environment())
    sys.path.insert(0, str(ROOT/'runtime'))
    from darkpipe.box_onnx import TensorRTSession
    import torch
    for name in ('yolo26m_person_20261024.onnx', 'rtmo-s_person_20261024_box.onnx',
                 'yolo26m_objects_20261026.onnx'):
        session = TensorRTSession(str(ROOT/'weights'/name), 'cuda:0')
        del session
        gc.collect()
        torch.cuda.empty_cache()
    print('本机 TensorRT engine 构建完成；请运行 check_env.py 和 benchmark_box.py 验证完整流程。')
