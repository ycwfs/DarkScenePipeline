"""Inventory user-labelled negatives and freeze session-level holdouts before training."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import tarfile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'reports/feedback-finetune-20261019'
ARCHIVE = ROOT / 'realdata-10-09/darkpipe.tar.gz'
BASE = ROOT.parent / 'compare/behavior/cache/realact_all_rf_v1_all_pn/train'
HOLDOUT = {'20261015_163555_77', '20261019_092255_83'}


def windows(n: int, fps: float) -> list:
    width, step = max(2, round(fps)), max(1, round(fps * .5))
    return [('whole', 0, n)] + [(f't{i / fps:.3f}', i, i + width)
                               for i in range(0, n - width + 1, step)]


def main() -> None:
    cv2.setNumThreads(2)
    data = OUT / 'data'
    data.mkdir(exist_ok=True)
    records, clips, seen = [], [], set()
    with tarfile.open(ARCHIVE) as archive:
        for member in sorted(archive.getmembers(), key=lambda x: x.name):
            if not member.name.endswith('.mp4'):
                continue
            rel = Path(member.name)
            assert member.isfile() and not rel.is_absolute() and '..' not in rel.parts
            raw = archive.extractfile(member).read()
            sha = hashlib.sha256(raw).hexdigest()
            assert sha not in seen, f'Duplicate video: {rel}'
            seen.add(sha)
            path = data / 'videos' / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            cap = cv2.VideoCapture(str(path))
            assert cap.isOpened(), path
            n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
            width, height = int(cap.get(3)), int(cap.get(4))
            heights = [h for h in range(height - 150, height) if h % 2 == 0
                       and h + max(48, round(.08 * h)) + (max(48, round(.08 * h)) & 1) == height]
            assert len(heights) == 1, (path, height, heights)
            content_h = heights[0]
            frames = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = frame[:content_h]
                frame = cv2.resize(frame, (400, 224), interpolation=cv2.INTER_AREA)
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            assert len(frames) == n and n >= 2 and fps > 0, path
            split = 'holdout' if rel.parts[0] in HOLDOUT else 'train'
            clip = dict(source=rel.as_posix(), session=rel.parts[0], split=split,
                        sha256=sha, frames=n, fps=fps, width=width, height=height,
                        content_height=content_h, removed_bar=height-content_h,
                        label='other', provenance='user-labelled processed platform clip')
            clips.append(clip)
            for tag, lo, hi in windows(n, fps):
                inds = np.linspace(lo, hi - 1, 32).round().astype(int)
                name = f'{rel.parts[0]}_{path.stem}_{tag}.npy'
                dest = data / 'windows' / split / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                np.save(dest, np.stack([frames[i] for i in inds]))
                records.append(dict(path=str(dest), split=split, source=rel.as_posix(),
                                    session=rel.parts[0], tag=tag, lo=lo, hi=hi, label='other'))
    old = sorted(BASE.glob('*/*.npy'))
    assert len(old) == 3220 and len(clips) == 32
    manifest = dict(archive_sha256=hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
                    base_cache=str(BASE), base_count=len(old),
                    base_counts=dict(Counter(p.parent.name for p in old)),
                    clips=clips, windows=records, heldout_sessions=sorted(HOLDOUT),
                    clip_counts=dict(Counter(c['split'] for c in clips)),
                    window_counts=dict(Counter(r['split'] for r in records)),
                    preprocessing='Remove measured label bar; no second enhancement; whole-frame RGB 400x224',
                    note='ABCP are existing training-domain regression; only these held-out sessions are new to this fine-tune. Same-camera and processed-video limitations remain.')
    (OUT/'data-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('clips','windows')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
