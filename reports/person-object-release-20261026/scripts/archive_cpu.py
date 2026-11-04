"""Execute integration regressions against the independently extracted archive in-image."""
from pathlib import Path
import sys
sys.path.insert(0,'/evidence/test-tools')
import pytest

sys.path.insert(0,'/app')
import darkpipe.person.object_runtime
assert Path(darkpipe.person.object_runtime.__file__).is_relative_to('/app')
raise SystemExit(pytest.main(['-q','/repo/tests/test_person_object_runtime.py',
    '/repo/tests/test_person_release.py','-k','not assignment_matches_optimal_cost',
    '--basetemp','/tmp/darkpipe-object-tests','-p','no:cacheprovider']))
