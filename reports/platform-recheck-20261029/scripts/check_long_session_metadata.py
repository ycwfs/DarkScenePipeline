"""CPU-only lifecycle probe of actual packaged scheduling; no model or video simulation claim."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

OUT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OUT/'release-extracted/dark-demo-v2'))
from darkpipe.person.scheduling import ActionPool

pool = ActionPool.__new__(ActionPool)
pool.errors = []
pool.devices = ['cuda:2']
pool.assignment, pool.assignment_next = {}, 0
pool.mailboxes = [SimpleNamespace(put=lambda task: None)]
for track_id in range(50000):
    pool.submit(SimpleNamespace(key=(1, track_id, 0)))
at_churn = len(pool.assignment)
for _ in range(10000):
    pool.submit(SimpleNamespace(key=(1, 49999, 0)))
retained = len(pool.assignment)
result = dict(
    observation='ActionPool.submit retains per-person/generation ownership until process termination',
    model_execution=False, real_source='release-extracted/dark-demo-v2/darkpipe/person/scheduling.py',
    distinct_track_generations=50000, retained_after_churn=at_churn,
    repeated_calls_using_only_last_identity=10000, retained_after_only_one_identity=retained,
    shallow_mapping_and_tuple_bytes=sys.getsizeof(pool.assignment)+sum(sys.getsizeof(k) for k in pool.assignment),
    counts_are_synthetic_not_elapsed_hours=True,
    other_static_findings='PersonEngine last_seen/generations/last_submitted reset on input epoch changes, but do not prune departed track IDs during an uninterrupted epoch.',
    limitation='This demonstrates accumulating small metadata, not a measured production OOM or the cause of observed RSS growth; ROI/model caches have separate bounds.',
    production_modified=False,
)
assert at_churn == retained == 50000
(OUT/'long-session-metadata.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
