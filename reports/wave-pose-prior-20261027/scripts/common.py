from pathlib import Path
import hashlib
import json
import sys

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
OP = ROOT/'platform/dark-demo-v2'
sys.path[:0] = [str(OUT/'implementation'), str(OP)]


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()
