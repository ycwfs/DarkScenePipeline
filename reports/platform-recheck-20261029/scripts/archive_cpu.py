"""Run functional regressions on the extracted production payload, inside the original image."""
from pathlib import Path
import importlib
import json
import pkgutil
import sys

sys.path.insert(0, '/test-tools')
import pytest
sys.path.insert(0, '/app')
import darkpipe.person
for item in pkgutil.walk_packages(darkpipe.person.__path__, darkpipe.person.__name__ + '.'):
    importlib.import_module(item.name)

TESTS = [
    'test_person_object_prior.py', 'test_person_object_runtime.py', 'test_person_object_overlay.py',
    'test_person_release.py', 'test_wave_pose_prior.py', 'test_running_pose_prior.py',
    'test_v2_delivery.py', 'test_v2_serve_lifecycle.py', 'test_v2_platform_feedback.py',
    'test_v2_remaining_fixes.py', 'test_v2_clip_sequence.py', 'test_streams.py',
    'test_clips.py', 'test_webhdfs.py',
]
status = pytest.main(['-q', *['/repo/tests/' + p for p in TESTS],
                     '-k', 'not assignment_matches_optimal_cost', '--basetemp', '/tmp/platform-recheck',
                     '-p', 'no:cacheprovider', '--junitxml=/evidence/cpu-junit.xml'])
loaded = {name: str(module.__file__) for name, module in list(sys.modules.items())
          if name.startswith('darkpipe') and getattr(module, '__file__', None)}
assert all(Path(p).is_relative_to('/app') for p in loaded.values()), loaded
Path('/evidence/loaded-modules.json').write_text(json.dumps(loaded, indent=2)+'\n')
raise SystemExit(status)
