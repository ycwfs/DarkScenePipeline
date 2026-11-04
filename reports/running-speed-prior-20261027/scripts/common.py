from pathlib import Path
import hashlib
import json
import sys

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
OP = ROOT/'platform/dark-demo-v2'
WAVE = ROOT/'reports/wave-pose-prior-20261027'
sys.path[:0] = [str(OUT/'implementation'), str(WAVE/'implementation'), str(OP)]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()
