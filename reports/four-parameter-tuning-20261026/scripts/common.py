"""Fixed-model four-parameter calibration; artifacts stay outside production."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
PREV = ROOT / 'reports/threshold-tuning-20261026'
CACHE = ROOT / 'reports/person-release-20261024'
LOW = ROOT / 'reports/luma-gate15-20261024'
OP = ROOT / 'platform/dark-demo-v2'
sys.path.insert(0, str(OP))
from darkpipe.constants import BEHAVIORS
from darkpipe.person.core import fuse_person
from darkpipe.person.object_prior import filter_object_evidence

GATES = list(range(15, 76, 5))
ALPHAS = [i / 100 for i in range(50, 100)]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def prepare():
    assert not (OUT / 'protocol.json').exists(), 'Protocol already frozen'
    previous = json.loads((PREV / 'protocol.json').read_text())
    affected = []
    for s in previous['sources']:
        if s['scope'] != 'onsite':
            continue
        data = json.loads((ROOT / s['cache']).read_text())
        if any(f['luma'] is not None and 15 <= f['luma'] < 75 for f in data['frames']):
            path = LOW / 'data' / s['id'] / 'result.json'
            assert path.exists()
            affected.append(dict(id=s['id'], cache=str(path.relative_to(ROOT)), sha256=digest(path)))
    protocol = dict(
        baseline=dict(alpha=.95, gate=30, action=.85, object=.35),
        previous_candidate=dict(alpha=.95, gate=30, action=.905, object=.358),
        grid=dict(alpha=ALPHAS, gate=GATES, action=[i/100 for i in range(70, 100)],
                  object=[i/100 for i in range(1, 91)],
                  fine='After coarse lock: alpha +/- .01 step .001, action/object +/- .01 step .001, gate kept on measured 5-unit grid.'),
        fixed='Six model hashes, preprocessing, person ROI, tracker, 32-frame sampling, soft fallback, one compatible object in one of five same-person crops; no smoothing/training/new rule.',
        selection=dict(inputs='Old onsite val + online val + binary user feedback on new9. Train and historical test diagnostic only.',
                       protection='Each supported old-val named-class TP >= baseline; old-val other FP, online-val FP, new known drink FP and unreviewed drink triggers no worse than baseline. Preserve the true new drink at action stage (object stage may reject).',
                       objective='Minimize 0.5*(old-val other FP/162 + new known drink FP/526); ties maximize old-val supported macro F1, old-val correct positives, new true drink; then normalized distance to original four settings.',
                       alternatives='Also retain best feasible true-drink option and best fixed-action .85 option.',
                       caveat='New9 are feedback calibration, old historical test repeatedly inspected; no untouched generalization set.'),
        inference='Decode/enhance affected old sources, independent production trackers per gate, reuse detector/ungated pose by identical model hashes. Reuse RGB only for exact crop SHA; rerun gate-specific ST-GCN. Audit every raw new frame before reusing all new action caches across gates. Object scores >=.001 completed for union of reachable drink/pickup candidates.',
        coverage='All544 sources stay in inventory. Missing tracks/windows not correct; unmatched windows unknown. New binary not_drink never treated as multiclass other.',
        models=previous['models'], sources=previous['sources'], affected=affected,
        files={str(p.relative_to(ROOT)):digest(p) for p in [PREV/'protocol.json', PREV/'windows.json', CACHE/'evaluated-windows.json', LOW/'protocol.json', LOW/'label-review.json', OP/'darkpipe/person/core.py', OP/'darkpipe/person/object_prior.py', Path(__file__)]})
    save(OUT / 'protocol.json', protocol)
    print('FROZEN', len(affected), 'affected sources;', len(ALPHAS)*len(GATES)*30*90, 'coarse settings', flush=True)


if __name__ == '__main__':
    prepare()
