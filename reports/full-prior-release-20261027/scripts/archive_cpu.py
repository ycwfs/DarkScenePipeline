"""Execute integration regressions against the independently extracted archive in-image."""
from pathlib import Path
import sys, pkgutil, importlib
sys.path.insert(0,'/test-tools')
import pytest

sys.path.insert(0,'/app')
import darkpipe.person.object_runtime
import darkpipe.person
for module in pkgutil.walk_packages(darkpipe.person.__path__, darkpipe.person.__name__+'.'):
    importlib.import_module(module.name)
assert Path(darkpipe.person.object_runtime.__file__).is_relative_to('/app')
raise SystemExit(pytest.main(['-q','/repo/tests/test_person_object_runtime.py', '/repo/tests/test_person_object_overlay.py',
    '/repo/tests/test_person_release.py','/repo/tests/test_wave_pose_prior.py','/repo/tests/test_running_pose_prior.py','/repo/tests/test_person_stream_clips.py','-k','not assignment_matches_optimal_cost',
    '--basetemp','/tmp/darkpipe-object-tests','-p','no:cacheprovider']))
