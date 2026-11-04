"""Frozen paths, original COCO classes and explicitly approved project semantics."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT / 'reports/object-hardneg-finetune-20261027'
COCO = Path('/ssd2/coco/coco17')
BASE = ROOT.parent / 'yolo26m-detect.pt'
MAPPING = {0: 39, 1: 41, 2: 67, 3: 67, 4: 73, 5: None, 6: None, 7: 67}


def save(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
