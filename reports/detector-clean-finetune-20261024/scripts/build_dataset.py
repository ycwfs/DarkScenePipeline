"""Build a separate reviewed detector dataset; never modify historical labels."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from prepare_bright import ROOT, OUT, OLD, save, iou


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reviewed_bright() -> list:
    records = json.loads((OUT / 'bright-proposals.json').read_text())['records']
    removed = []
    for row in records:
        index = row['index']
        # These frame IDs were individually viewed; this is NOT a runtime mask.
        has_exhibit = index < 48 or 72 <= index < 84
        regions = [[48, 155, 140, 387]] if has_exhibit else []
        if index in (24, 25, 37, 47):
            regions.append([645, 99, 678, 177])  # Printed person on the luminous column.
        boxes = []
        for model, proposals in row['proposals'].items():
            for j, box in enumerate(proposals):
                bad = next((k for k, region in enumerate(regions) if iou(box[:4], region) >= .4), None)
                if bad is not None:
                    removed.append(dict(index=index, model=model, proposal=j, box=box,
                        reason='humanoid exhibit' if bad == 0 else 'printed person'))
                elif model == 'original':
                    boxes.append(box[:4])
        # Seated distant person, visible behind the sign; missed by both proposals.
        if index in (81, 82, 83):
            boxes.append([328., 29., 349., 58.])
        row['person_boxes'] = boxes
        row['known_nonperson_regions'] = regions
        row['review'] = 'Assistant visual review of all 96 images; auxiliary, not independent human GT'
        row['manual_additions'] = 1 if index in (81, 82, 83) else 0
        assert len(boxes) >= 1
    save(OUT / 'bright-reviewed-labels.json', dict(records=records, removed_proposals=removed,
        counts=dict(images=len(records), boxes=sum(len(r['person_boxes']) for r in records),
                    removals_by_model=dict(Counter(r['model'] for r in removed))),
        policy='Keep exhibit pixels as background. No runtime region suppression. Original model proposals are corrected labels; previous proposals are diagnostic only.'))
    return records


def materialize(source: Path, split: str, name: str, normalized_boxes: list) -> Path:
    image = OUT / 'dataset/images' / split / (name + '.png')
    label = OUT / 'dataset/labels' / split / (name + '.txt')
    image.parent.mkdir(parents=True, exist_ok=True)
    label.parent.mkdir(parents=True, exist_ok=True)
    assert source.is_file(), source
    if not image.exists():
        image.symlink_to(source)
    assert image.resolve() == source.resolve()
    lines = []
    for box in normalized_boxes:
        b = np.clip(np.array(box), 0, 1)
        assert np.all(b[2:] > b[:2]), b
        xywh = np.r_[(b[:2] + b[2:]) / 2, b[2:] - b[:2]]
        lines.append('0 ' + ' '.join(f'{x:.8f}' for x in xywh))
    label.write_text('\n'.join(lines) + '\n')
    return image


def main() -> None:
    old_manifest = OLD / 'manifest.json'
    original_hash = sha(old_manifest)
    dark = json.loads(old_manifest.read_text())['samples']
    excluded = [s for s in dark if s['split'] == 'train' and
                s['source'] == 'real-actiondata-all/other/other238.mp4']
    assert sorted(s['frame'] for s in excluded) == [267, 272, 277, 345, 350]
    dark = [s for s in dark if s not in excluded]
    bright = reviewed_bright()
    groups = {sp: {s['group'] for s in dark + bright if s['split'] == sp}
              for sp in ('train', 'val', 'test')}
    assert all(not (groups[a] & groups[b]) for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')])
    records = []
    lists = {'train': [], 'val': [], 'test': [], 'dark_val': [], 'bright_val': []}
    for s in dark:
        boxes = [np.array(p['bbox_xyxy']) / np.array([s['width'], s['height']] * 2) for p in s['people']]
        for view in (['raw', 'enhanced'] if s['split'] == 'train' else ['enhanced']):
            name = f'{s["video_id"]}_{s["frame"]:06d}_{view}'
            source = OLD / 'dataset/images' / s['split'] / (name + '.png')
            image = materialize(source, s['split'], name, boxes)
            lists[s['split']].append(str(image))
            if s['split'] == 'val':
                lists['dark_val'].append(str(image))
            records.append(dict(domain='dark', split=s['split'], source=s['source'], frame=s['frame'],
                group=s['group'], image=str(image), view=view, repeat=0, boxes=len(boxes)))
    for s in bright:
        boxes = [np.array(b) / np.array([s['width'], s['height']] * 2) for b in s['person_boxes']]
        for view in (['raw', 'enhanced'] if s['split'] == 'train' else ['enhanced']):
            for repeat in range(8 if s['split'] == 'train' else 1):
                name = f'bright_{s["index"]:03d}_{view}_r{repeat}'
                source = OUT / 'data/bright' / f'{s["index"]:03d}' / (view + '.png')
                image = materialize(source, s['split'], name, boxes)
                lists[s['split']].append(str(image))
                if s['split'] == 'val':
                    lists['bright_val'].append(str(image))
                records.append(dict(domain='bright', split=s['split'], source=s['source'], frame=s['frame'],
                    group=s['group'], image=str(image), view=view, repeat=repeat, boxes=len(boxes)))
    for name, paths in lists.items():
        (OUT / f'{name}.txt').write_text('\n'.join(paths) + '\n')
    data = yaml.safe_load((OLD / 'detector.yaml').read_text())
    data.update(path=str(OUT / 'dataset'), train=str(OUT / 'train.txt'), val=str(OUT / 'val.txt'), test=str(OUT / 'test.txt'))
    (OUT / 'detector.yaml').write_text(yaml.safe_dump(data, allow_unicode=True))
    for domain in ('dark', 'bright'):
        (OUT / f'detector-{domain}-val.yaml').write_text(yaml.safe_dump(
            dict(data, val=str(OUT / f'{domain}_val.txt')), allow_unicode=True))
    counts = {sp: dict(dark_source_frames=sum(s['split'] == sp for s in dark),
        bright_source_frames=sum(s['split'] == sp for s in bright),
        unique_views=sum(r['split'] == sp and r['repeat'] == 0 for r in records),
        weighted_training_or_eval_entries=len(lists[sp]),
        unique_box_instances=sum(r['boxes'] for r in records if r['split'] == sp and r['repeat'] == 0))
        for sp in ('train', 'val', 'test')}
    save(OUT / 'dataset-manifest.json', dict(records=records, counts=counts, excluded_dark_samples=excluded))
    assert sha(old_manifest) == original_hash
    protocol = dict(task='Person detector retraining with exhibit/person-image proposals removed and mixed bright/dark data',
        counts=counts, removed_dark_frames=5, bright_training_repeat=8,
        labels='Previously accepted filtered auxiliary dark boxes + assistant reviewed bright boxes; not independent human ground truth',
        split='Frozen source/group train, val, test; group intersections checked empty',
        initialization=str(ROOT.parent / 'yolo26m-detect.pt'), seed=24, epochs=12,
        selection='Highest equal-weight mean of dark/bright validation mAP50-95; explicit nms=False one-to-one head; earliest tie',
        comparison='Original detector, previous dark-only candidate, new validation-selected candidate; fixed RTMO candidate',
        thresholds=dict(detector=.35, pose=.45, joint=.3, IoU=.5),
        smoothing=False, runtime_exhibit_filter=False, frozen_models=['Retinexformer', 'RTMO', 'VideoMamba', 'ST-GCN'],
        limitation='Bright test is an already-inspected regression set; mixed-data change is not a pure annotation-removal ablation; one seed only',
        production_modified=False, historical_manifest_sha256=original_hash,
        input_hashes={str(p.relative_to(ROOT)): sha(p) for p in [old_manifest, OUT / 'bright-proposals.json', OUT / 'bright-reviewed-labels.json', ROOT / 'reports/person-finetune-20261023/splits.json']})
    save(OUT / 'protocol.json', protocol)
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    main()
