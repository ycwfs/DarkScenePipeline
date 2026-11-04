"""Build grouped native-pixel object data from explicitly reviewed boxes only."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import yaml

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]


def save(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def main() -> None:
    records = json.loads((OUT / 'reviewed-labels.json').read_text())['records']
    frozen = json.loads((OUT / 'source-inventory.json').read_text())['records']
    sources = {r['id']: r for r in frozen}
    names = json.loads((OUT / 'protocol.json').read_text())['classes']
    target = OUT / 'dataset'
    if target.exists():
        raise FileExistsError('Dataset already frozen; do not overwrite a running experiment.')
    manifest, excluded, group_splits = [], [], {}
    counts = Counter()
    # Pure exhibit near-duplicates are abundant in the single old feedback session.
    # Keep at most three per group and twelve per split/session prefix.
    for r in records:
        if not r['training_allowed']:
            continue
        src = sources[r['video_id']]
        assert src['split'] in ('train', 'val', 'test')
        assert r['split'] == src['split'] and r['group'] == src['source_group']
        old = group_splits.setdefault(r['group'], r['split'])
        assert old == r['split'], 'Source group leakage'
        classes = {b['class_id'] for b in r['boxes']}
        if classes == {6}:
            family = 'feedback-old' if 'online-negative-sample/' in r['source'] else r['group']
            key = (r['split'], family)
            limit = 12 if family == 'feedback-old' else 3
            if counts[key] >= limit:
                excluded.append({'id': r['id'], 'reason': 'pure exhibit repetition cap'})
                continue
            counts[key] += 1
        variants = [('view', r['image_view'])]
        # Gamma only reveals existing pixels, no generative enhancement.
        if r['split'] == 'train' and r.get('brightness_gamma', 1.) != 1.:
            variants.append(('raw', r['image_raw']))
        for variant, image_path in variants:
            im = cv2.imread(image_path)
            assert im is not None
            h, w = im.shape[:2]
            labels = []
            for box in r['boxes']:
                x1, y1, x2, y2 = box['bbox']
                coords = [(x1+x2)/2/w, (y1+y2)/2/h, (x2-x1)/w, (y2-y1)/h]
                assert all(0 <= x <= 1 for x in coords) and min(coords[2:]) > 0
                labels.append(str(box['class_id']) + ' ' + ' '.join(f'{x:.8f}' for x in coords))
            stem = f'{r["review_kind"]}_{r["tile_index"]:04d}_{r["id"]}_{variant}'
            image_dest = target / 'images' / r['split'] / (stem + '.png')
            label_dest = target / 'labels' / r['split'] / (stem + '.txt')
            image_dest.parent.mkdir(parents=True, exist_ok=True)
            label_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, image_dest)
            label_dest.write_text('\n'.join(labels) + ('\n' if labels else ''))
            manifest.append(dict(r, variant=variant, dataset_image=str(image_dest),
                dataset_label=str(label_dest), image_sha256=hashlib.sha256(image_dest.read_bytes()).hexdigest()))
    stats = {}
    for split in ('train', 'val', 'test'):
        rows = [r for r in manifest if r['split'] == split]
        stats[split] = dict(images=len(rows), unique_views=len({r['id'] for r in rows}),
            sources=len({r['video_id'] for r in rows}), groups=len({r['group'] for r in rows}),
            boxes=dict(Counter(b['class_id'] for r in rows for b in r['boxes'])),
            background_images=sum(not r['boxes'] for r in rows))
    # Exact duplicate pixels must not appear across splits either.
    hashes = {}
    for r in manifest:
        assert hashes.setdefault(r['image_sha256'], r['split']) == r['split']
    config = dict(path=str(target), train='images/train', val='images/val', test='images/test',
                  names={int(k): v for k, v in names.items()})
    (OUT / 'objects.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    save(OUT / 'dataset-manifest.json', dict(records=manifest, stats=stats, excluded=excluded,
        evaluation_variant='native-pixel crop; gamma 1 if sampled raw luma >=75, .5 if >=15, else .32',
        limitations=['Assistant-reviewed auxiliary GT', 'Single scene repeated across old source groups',
                      'Cup absent; camera has no heldout support; portable_object has minimal support']))
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == '__main__':
    main()
