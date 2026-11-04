"""Inventory original videos without assigning video labels to individual people."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / 'compare'
PREFIXES = {
    '喝水': 'drink', '抛物': 'throw', '挥手': 'wave', '捡东西': 'pick_up',
    '摔倒': 'fall', '追逐': 'chase', 'other': 'other',
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def inspect(item: dict) -> dict:
    path = ROOT / item['source']
    sha = digest(path)
    expected = item.get('expected_sha256')
    if expected is not None and expected != sha:
        raise ValueError(f'Feedback source differs from the previous manifest: {path}')
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f'Cannot open video: {path}')
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width, height = int(cap.get(3)), int(cap.get(4))
    finally:
        cap.release()
    if min(frames, fps, width, height) <= 0:
        raise ValueError(f'Invalid video metadata: {path}')
    return dict(item, sha256=sha, bytes=path.stat().st_size, frames=frames,
                fps=fps, width=width, height=height, seconds=frames/fps,
                validation='container metadata only; full decode pending')


def main() -> None:
    cv2.setNumThreads(0)
    items = []
    for group in ['real-actiondata-all', 'real-actiondata-0909',
                  'real-actiondata', 'real-actiondata-cutdrink']:
        for path in sorted((ROOT / group).rglob('*.mp4')):
            labels = [label for prefix, label in PREFIXES.items() if path.stem.startswith(prefix)]
            if len(labels) != 1:
                raise ValueError(f'Video needs an explicit label mapping: {path}')
            items.append(dict(source=path.relative_to(ROOT).as_posix(), dataset=group,
                              video_label=labels[0], label_provenance='historical video label',
                              processed=False, sealed_holdout=False,
                              person_annotations='not yet reviewed'))
    for path in sorted((ROOT / 'online-negative-sample').rglob('*.mp4')):
        items.append(dict(source=path.relative_to(ROOT).as_posix(), dataset='online-negative-sample',
                          video_label='other', label_provenance='user-supplied negative; known ambiguities require review',
                          processed=True, sealed_holdout=False,
                          person_annotations='not yet reviewed'))
    feedback = json.loads((ROOT / 'reports/feedback-finetune-20261019/data-manifest.json').read_text())
    for clip in feedback['clips']:
        path = ROOT / 'reports/feedback-finetune-20261019/data/videos' / clip['source']
        items.append(dict(source=path.relative_to(ROOT).as_posix(), dataset='feedback-20261019',
                          video_label=clip['label'], label_provenance='user-supplied other clip',
                          processed=True, sealed_holdout=clip['split']=='holdout',
                          session=clip['session'], expected_sha256=clip['sha256'],
                          person_annotations='not yet reviewed'))
    with ThreadPoolExecutor(max_workers=4) as executor:
        records = list(executor.map(inspect, items))
    groups = defaultdict(list)
    for record in records:
        groups[record['sha256']].append(record)
    videos = []
    for sha, copies in groups.items():
        labels = {record['video_label'] for record in copies}
        if len(labels) != 1:
            raise ValueError(f'Conflicting labels for byte-identical videos: {copies}')
        videos.append(dict(copies[0], id=sha[:16], aliases=[r['source'] for r in copies],
                           datasets=sorted({r['dataset'] for r in copies}),
                           sealed_holdout=any(r['sealed_holdout'] for r in copies)))
    summary = dict(
        source_files=len(records), unique_videos=len(videos),
        duplicate_copies_removed=len(records)-len(videos),
        source_counts=dict(Counter(r['dataset'] for r in records)),
        unique_video_labels=dict(Counter(r['video_label'] for r in videos)),
        raw_unique_videos=sum(not r['processed'] for r in videos),
        processed_unique_videos=sum(r['processed'] for r in videos),
        sealed_holdout_videos=sum(r['sealed_holdout'] for r in videos),
        total_frames=sum(r['frames'] for r in videos),
        total_minutes=sum(r['seconds'] for r in videos)/60,
        unknown_person_labels=True,
        temporal_overlap_between_different_encodings='not checked; original/cut versions must share split',
    )
    public_available = {}
    for name, path in [
        ('hmdb', DATA/'behavior_data/hmdb'),
        ('k700', DATA/'behavior_data/k700/videos'),
        ('arid_train', DATA/'arid/train_clips'),
        ('arid_test', DATA/'arid/test_clips'),
    ]:
        public_available[name] = dict(path=str(path), exists=path.exists(),
                                      available_mp4_files=sum(1 for _ in path.rglob('*.mp4')),
                                      scope='available files, not confirmed previous training membership')
    caches = {}
    for name in ['realact_all_rf_v1_all', 'realact_all_rf_v1_all_pn',
                 'realact_all_rf_v1_all_pn2_20261019']:
        path = DATA / 'behavior/cache' / name / 'train'
        files = list(path.glob('*/*.npy'))
        caches[name] = dict(path=str(path), windows=len(files),
                            labels=dict(Counter(p.parent.name for p in files)))
    manifest = dict(summary=summary, videos=videos,
                    existing_training_caches=caches, public_data_available=public_available,
                    heldout_sessions=feedback['heldout_sessions'],
                    split_status='pending person annotation policy and data scope; no training started',
                    models_modified=False)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'source-inventory.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(summary=summary, existing_training_caches=caches,
                         public_data_available=public_available), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
