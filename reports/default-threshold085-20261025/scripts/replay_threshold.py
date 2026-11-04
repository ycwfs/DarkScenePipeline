"""Replay the production rejection rule on unchanged full-precision onsite predictions."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'reports/manypeople-inference-20261025'
sys.path.insert(0, str(ROOT / 'platform/dark-demo-v2'))
from darkpipe.constants import BEHAVIORS
from darkpipe.stages.recognize import reject_probs


def main():
    totals = {str(t): Counter() for t in (.75, .85)}
    records, selections = [], []
    original_labels_match = 0
    for src in json.loads((BASE / 'input-manifest.json').read_text()):
        number = src['number']
        event_path = BASE / f'events/{number:02d}_events.json'
        events = json.loads(event_path.read_text())
        counts = {str(t): Counter() for t in (.75, .85)}
        for event in events:
            probs = np.asarray(event['fused_probs'], dtype=np.float32)
            for threshold in (.75, .85):
                decision = reject_probs(probs, BEHAVIORS, threshold)
                label = BEHAVIORS[int(decision.argmax())]
                counts[str(threshold)][label] += 1
                if threshold == .75:
                    assert label == event['label'], (number, event['frame_index'])
                    assert abs(float(decision.max()) - event['confidence']) < 1e-7
                    original_labels_match += 1
        for key in totals:
            totals[key].update(counts[key])
        records.append(dict(number=number, source=src['absolute_path'], events=len(events),
                            events_sha256=hashlib.sha256(event_path.read_bytes()).hexdigest(),
                            threshold_counts=counts))
        # Keep representative high-confidence interaction candidates, not truth labels.
        for action in ('drink', 'pick_up'):
            rows = sorted((e for e in events if e['label'] == action),
                          key=lambda e: e['confidence'], reverse=True)
            selected = []
            for event in rows:
                if any(abs(event['timestamp']-x['timestamp']) < 10 for x in selected):
                    continue
                if any(np.linalg.norm(np.asarray(event['bbox'])-x['bbox']) < 45 for x in selected):
                    continue
                selected.append(event)
                selections.append(dict(number=number, source=src['absolute_path'],
                                       source_sha256=src['sha256'], source_width=src['width'],
                                       source_height=src['height'], event=event,
                                       annotation_status='unreviewed model prediction; not ground truth'))
                if len(selected) == 2:
                    break
    result = dict(original_threshold=.75, requested_threshold=.85,
                  probability_source='unchanged full-precision fused_probs',
                  original_event_decisions_matched=original_labels_match,
                  counts=totals, videos=records,
                  caveat='Window trigger counts only; no person/action ground truth. '
                         'Not false-positive counts, action episodes, or regenerated clip counts.')
    (OUT / 'threshold-replay.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    review = ROOT / 'reports/object-prior-finetune-20261025'
    review.mkdir(exist_ok=True)
    (review / 'review-candidates.json').write_text(json.dumps(selections, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(matched=original_labels_match, counts=totals,
                         review_candidates=len(selections)), ensure_ascii=False))


if __name__ == '__main__':
    main()
