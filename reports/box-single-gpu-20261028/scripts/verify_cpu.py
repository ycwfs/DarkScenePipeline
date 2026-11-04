import importlib
import os
from pathlib import Path
import pkgutil
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,os.environ.get('BOX_RUNTIME',str(ROOT/'NVIDIA-Box/runtime')))
import darkpipe.person
for module in pkgutil.walk_packages(darkpipe.person.__path__, darkpipe.person.__name__+'.'):
    importlib.import_module(module.name)
import pytest
files=['test_nvidia_box_native.py','test_box_optimization.py',
       'test_person_object_runtime.py','test_person_object_overlay.py',
       'test_person_release.py','test_wave_pose_prior.py','test_running_pose_prior.py',
       'test_person_stream_clips.py','test_person_optimized_scheduling.py']
raise SystemExit(pytest.main(['-q',*[str(ROOT/'tests'/f)for f in files],
    '--basetemp',str(ROOT/'reports/box-single-gpu-20261028/cpu-tmp'),'-p','no:cacheprovider']))
