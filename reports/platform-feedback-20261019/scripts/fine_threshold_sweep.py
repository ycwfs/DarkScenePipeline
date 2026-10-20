"""Replay deployed rejection on full-precision scores and actual logged online events."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--op-dir', type=Path, required=True)
ap.add_argument('--evidence', type=Path, required=True)
args = ap.parse_args()
sys.path.insert(0, str(args.op_dir))
from darkpipe.constants import BEHAVIORS
from darkpipe.stages.recognize import reject_probs

root = args.evidence
window_path = root / 'native-720/native-windows.jsonl'
windows = [json.loads(s) for s in window_path.read_text().splitlines()]
assert Counter(r['suite'] for r in windows) == dict(A=177, B=365, C=18, P=460)
events = []
for path in sorted((root / 'input').glob('*/events.jsonl')):
    for line in path.read_text().splitlines():
        row = json.loads(line)
        # Whitelist event fields; never read session endpoints or credentials.
        events.append({**{k: row[k] for k in ('label', 'confidence', 'wall_time')},
                       'session': path.parent.name})
assert len(events) == 11593
actions = [r for r in events if r['label'] != 'Other']
assert len(actions) == 72
coarse = json.loads((root / 'native-720/native-summary.json').read_text())['thresholds']
rows, details = [], {}

for tau in [0.65] + [i / 100 for i in range(70, 81)]:
    kept = [r for r in actions if r['confidence'] >= tau]
    boundary = [r for r in actions if abs(r['confidence'] - tau) <= .00005 + 1e-12]
    result = {'reco_min_conf': tau, 'new_false_events': len(kept),
              'new_false_drink': sum(r['label'] == 'Drinking water' for r in kept),
              'new_false_wave': sum(r['label'] == 'Waving' for r in kept),
              'reduction_vs_065_percent': round(100 * (72 - len(kept)) / 72, 2),
              'rounding_boundary_events': len(boundary)}
    sub = {'online_remaining': kept, 'online_rounding_boundaries': boundary, 'suites': {}}
    for suite in 'ABCP':
        selected = [r for r in windows if r['suite'] == suite]
        correct, fired, missed = 0, [], []
        for r in selected:
            original = np.array(r['decision_before_reject'], dtype=np.float32)
            rejected = reject_probs(original, BEHAVIORS, tau)
            predicted = BEHAVIORS[int(rejected.argmax())]
            correct += predicted == r['expect']
            item = {'name': r['name'], 'expect': r['expect'], 'predicted': predicted,
                    'before_label': BEHAVIORS[int(original.argmax())],
                    'before_confidence': float(original.max())}
            if predicted != 'other':
                fired.append(item)
            if predicted != r['expect']:
                missed.append(item)
        stats = {'windows': len(selected), 'correct': correct, 'fired': len(fired),
                 'incorrect': missed, 'fired_windows': fired}
        sub['suites'][suite] = stats
        result[f'{suite}_correct'] = correct
        result[f'{suite}_fired'] = len(fired)
        if str(tau) in coarse:
            for key in ('windows', 'correct', 'fired'):
                assert stats[key] == coarse[str(tau)][suite][key], (tau, suite, key)
    details[f'{tau:.2f}'] = sub
    rows.append(result)

report = {'method': 'Stored full-precision 720-pixel native scores through the shipped '
                    'reject_probs; online rounded event scores filtered at each higher threshold.',
          'threshold_step': .01, 'online_assumed_all_other': True,
          'native_windows': len(windows), 'online_events': len(events),
          'native_windows_sha256': hashlib.sha256(window_path.read_bytes()).hexdigest(),
          'operator_recognize_sha256': hashlib.sha256(
              (args.op_dir / 'darkpipe/stages/recognize.py').read_bytes()).hexdigest(),
          'rows': rows, 'details': details}
out = root / 'fine-threshold'
out.mkdir(exist_ok=True)
(out / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
with (out / 'thresholds.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
for r in rows:
    print(f"{r['reco_min_conf']:.2f}: new={r['new_false_events']} "
          f"drink={r['new_false_drink']} wave={r['new_false_wave']} "
          f"A={r['A_correct']}/177 B={r['B_fired']}/365 "
          f"C={r['C_correct']}/18 P={r['P_fired']}/460 "
          f"rounding_boundary={r['rounding_boundary_events']}", flush=True)
