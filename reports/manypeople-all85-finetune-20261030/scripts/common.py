"""Immutable inputs and atomic artifacts for human-reviewed crowded-scene adaptation."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'reports/manypeople-action-review-20261029'
OLD = ROOT / 'reports/person-finetune-20261023'
CURRENT = ROOT / 'reports/person-release-20261024'
OP = ROOT / 'platform/dark-demo-v2'
LABELS = ['wave', 'throw', 'chase', 'fall', 'fight', 'talk', 'drink', 'pick_up', 'shake_hands', 'other']
GATES = [30., 35., 40., 50.]


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def scored_indices(rows):
    return [i for i, r in enumerate(rows) if r['label_kind'] != 'pseudo'
            and (r['scope'] == 'new' or r['split'] in ('val', 'test')
                 and r.get('current_index') is not None)]


def score_role(row):
    if row['scope'] == 'old':
        return 'old_' + row['split']
    if row['label_kind'] == 'pseudo':
        return 'unused_pseudo'
    if row['label'] != 'other':
        return 'new_positive_training_fit'
    return 'new_other_training_fit' if row['train_selected'] else 'new_other_untrained_same_sources'
