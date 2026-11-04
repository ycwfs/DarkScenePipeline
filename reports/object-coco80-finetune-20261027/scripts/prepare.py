"""COCO replay plus site labels; never relabel cameras as literal COCO ground truth."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import yaml

from common import BASE, COCO, MAPPING, OUT, PREVIOUS, ROOT, save, sha


def materialize(image, split, stem, boxes, *, extension=None):
    image = image.resolve()
    assert image.is_file(), image
    suffix = extension or image.suffix
    dst = OUT / 'dataset/images' / split / (stem + suffix)
    label = OUT / 'dataset/labels' / split / (stem + '.txt')
    dst.parent.mkdir(parents=True, exist_ok=True)
    label.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        dst.symlink_to(image)
    assert dst.resolve() == image
    label.write_text('\n'.join(f'{int(c)} ' + ' '.join(f'{float(v):.8f}' for v in xywh)
                               for c, xywh in boxes) + ('\n' if boxes else ''))
    return str(dst)


def choose(images, labels, size, minimum, rng):
    ids = sorted(images)
    rng.shuffle(ids)
    pool = defaultdict(list)
    for iid in ids:
        for c in {c for c, _ in labels[iid]}:
            pool[c].append(iid)
    chosen, counts = set(), Counter()
    for c in sorted(range(80), key=lambda c: len(pool[c])):
        for iid in pool[c]:
            if counts[c] >= minimum:
                break
            if iid not in chosen:
                chosen.add(iid)
                counts.update(cc for cc, _ in labels[iid])
    assert len(chosen) <= size, (len(chosen), size)
    for iid in ids:
        if len(chosen) == size:
            break
        chosen.add(iid)
    assert len(chosen) == size
    return sorted(chosen)


def read_coco(split):
    p = COCO / 'annotations' / f'instances_{split}2017.json'
    data = json.loads(p.read_text())
    cats = sorted(data['categories'], key=lambda r: r['id'])
    ids = {r['id']: i for i, r in enumerate(cats)}
    images = {r['id']: r for r in data['images']}
    labels = defaultdict(list)
    for r in data['annotations']:
        if r.get('iscrowd'):
            continue
        im = images[r['image_id']]
        x, y, w, h = r['bbox']
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(im['width'], x+w), min(im['height'], y+h)
        if x2 <= x1 or y2 <= y1:
            continue
        labels[r['image_id']].append((ids[r['category_id']],
            [(x1+x2)/2/im['width'], (y1+y2)/2/im['height'],
             (x2-x1)/im['width'], (y2-y1)/im['height']]))
    return images, labels, [r['name'] for r in cats]


def overlap(a, b):
    inter = np.maximum(0, np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2])).prod()
    aa = np.maximum(0, a[2:]-a[:2]).prod()
    bb = np.maximum(0, b[2:]-b[:2]).prod()
    return float(inter/max(aa+bb-inter, 1e-6)), float(inter/max(aa, 1e-6))


def main():
    if (OUT / 'dataset-manifest.json').exists():
        raise FileExistsError('Frozen dataset already exists')
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    rng = random.Random(27)
    train_im, train_gt, names = read_coco('train')
    val_im, val_gt, val_names = read_coco('val')
    assert names == val_names and len(names) == 80
    save(OUT/'classes.json', names)
    train_ids = choose(train_im, train_gt, 2400, 30, rng)
    val_ids = choose(val_im, val_gt, 1000, 5, rng)
    test_ids = sorted(set(val_im)-set(val_ids))
    lists = defaultdict(list)
    coco_records = []
    for split, ids, images, labels, origin in [
        ('train', train_ids, train_im, train_gt, 'train'),
        ('val', val_ids, val_im, val_gt, 'val'),
        ('test', test_ids, val_im, val_gt, 'val')]:
        for iid in ids:
            path = COCO / f'{origin}2017' / images[iid]['file_name']
            if not path.is_file():
                path = Path('/ssd2/DATASETS/coco') / f'{origin}2017' / images[iid]['file_name']
            dest = materialize(path, split, f'coco_{iid:012}', labels[iid])
            lists[f'coco_{split}'].append(dest)
            coco_records.append(dict(id=iid, split=split, origin=origin, image=dest,
                counts=dict(Counter(c for c, _ in labels[iid]))))
    del train_im, train_gt, val_im, val_gt
    teacher = YOLO(str(BASE))
    assert list(teacher.names.values()) == names
    source = json.loads((PREVIOUS/'dataset-manifest.json').read_text())['records']
    known_splits = {str((ROOT / r['source']).resolve()): r['split'] for r in source}
    person_rows = json.loads((ROOT/'reports/detector-clean-finetune-20261024/dataset-manifest.json').read_text())['records']
    # Add accepted real-human boxes and background exhibits; source leakage is prohibited.
    extra = [r for r in person_rows if r['split'] == 'train' and r['repeat'] == 0
             and known_splits.get(str((ROOT/r['source']).resolve()), 'train') == 'train']
    rng.shuffle(extra)
    bright = [r for r in extra if r['domain'] == 'bright']
    dark = [r for r in extra if r['domain'] == 'dark'][:160]
    extra = bright + dark
    rows = []
    for r in source:
        rows.append(dict(r, kind='objects', original_boxes=r['boxes']))
    for i, r in enumerate(extra):
        frame = cv2.imread(r['image']); h, w = frame.shape[:2]
        label = r['image'].replace('/images/', '/labels/').rsplit('.', 1)[0]+'.txt'
        boxes = []
        for line in open(label):
            if not line.strip():
                continue
            c, x, y, bw, bh = map(float, line.split())
            boxes.append(dict(class_id=0, bbox=[(x-bw/2)*w, (y-bh/2)*h, (x+bw/2)*w, (y+bh/2)*h]))
        rows.append(dict(id=f'human_{i:04}', kind='human', split='train', group=r['group'],
            source=r['source'], frame=r['frame'], variant=r['view'], dataset_image=r['image'],
            original_boxes=boxes, domain=r['domain']))
    site_records = []
    for i, r in enumerate(rows):
        image = cv2.imread(r['dataset_image']); assert image is not None
        h, w = image.shape[:2]
        normalized, annotated, blocked = [], [], []
        for b in r['original_boxes']:
            c = b['class_id'] if r['kind']=='human' else MAPPING[b['class_id']]
            box = np.array(b['bbox'], float)
            if c is None:
                blocked.append(box)
                continue
            annotated.append(dict(class_id=c, bbox=box.tolist(), origin='reviewed_project_mapping'))
        pseudo = []
        if r['split'] == 'train':
            pred = teacher.predict(image, imgsz=640, conf=.35, device=0, verbose=False)[0].boxes
            for b, conf, cls in zip(pred.xyxy.cpu().numpy(), pred.conf.cpu().numpy(), pred.cls.cpu().numpy()):
                cls = int(cls)
                if r['kind']=='human' and cls==0:
                    continue  # Accepted real-human labels override teacher exhibit/person proposals.
                if any(overlap(b, bb)[1] >= .3 or overlap(b, bb)[0] >= .25 for bb in blocked):
                    continue
                if any(overlap(b, np.array(bb['bbox']))[0] >= .25 or
                       overlap(b, np.array(bb['bbox']))[1] >= .5 for bb in annotated):
                    continue
                pseudo.append(dict(class_id=cls, bbox=b.tolist(), confidence=float(conf),
                    origin='unreviewed_original80_teacher_retention_not_ground_truth'))
        for b in annotated+pseudo:
            box = np.clip(np.array(b['bbox']), 0, [w,h]*2)
            xywh = np.r_[(box[:2]+box[2:])/2, box[2:]-box[:2]] / np.array([w,h]*2)
            normalized.append((b['class_id'], xywh))
        stem = f'site_{i:04}'
        dest = materialize(Path(r['dataset_image']), r['split'], stem, normalized)
        lists[f'site_{r["split"]}'].append(dest)
        site_records.append(dict(r, image=dest, boxes=annotated, teacher_boxes=pseudo,
            background_regions=[b.tolist()for b in blocked], image_sha256=sha(Path(dest))))
        if i % 100 == 0:print('site', i, '/',len(rows),flush=True)
    groups, sources, pixels = {}, {}, {}
    for r in site_records:
        assert groups.setdefault(r['group'],r['split']) == r['split'], r['group']
        assert sources.setdefault(str((ROOT/r['source']).resolve()),r['split']) == r['split']
        assert pixels.setdefault(r['image_sha256'],r['split']) == r['split']
    lists['train'] = lists['coco_train'] + lists['site_train']*3
    for key, paths in lists.items():
        (OUT/f'{key}.txt').write_text('\n'.join(paths)+'\n')
    base = dict(path=str(OUT/'dataset'), train=str(OUT/'train.txt'),
        val=str(OUT/'coco_val.txt'), test=str(OUT/'coco_test.txt'), names=dict(enumerate(names)))
    (OUT/'objects.yaml').write_text(yaml.safe_dump(base, sort_keys=False))
    (OUT/'site.yaml').write_text(yaml.safe_dump(dict(base, val=str(OUT/'site_val.txt'),test=str(OUT/'site_test.txt')),sort_keys=False))
    stats = {s:dict(images=len([r for r in site_records if r['split']==s]),
        reviewed_boxes=dict(Counter(names[b['class_id']] for r in site_records if r['split']==s for b in r['boxes'])),
        pseudo_boxes=sum(len(r['teacher_boxes'])for r in site_records if r['split']==s))for s in ['train','val','test']}
    coco_counts = {s:dict(images=len(lists[f'coco_{s}']),classes=len(set(c for r in coco_records if r['split']==s for c in r['counts'])))for s in ['train','val','test']}
    save(OUT/'dataset-manifest.json',dict(site=site_records,coco=coco_records,stats=stats,coco_counts=coco_counts,
        train_entries=len(lists['train']),site_repeat=3,group_source_pixel_checks=True))
    save(OUT/'protocol.json',dict(initialization=str(BASE),initialization_sha256=sha(BASE),nc=80,names=names,
        mapping=MAPPING,approved_semantics='camera and microphone -> cell phone; paper -> book; robot/exhibit background; ambiguous portable items background',
        caveats=['Expanded project semantics are not literal COCO labels.','Teacher retention proposals in TRAIN only are weak labels, not reviewed truth.',
                 'Site val/test only score reviewed target classes; other classes are incompletely annotated.',
                 'COCO validation historically used by pretrained model development; retention regression, not novel generalization.',
                 'Site source groups previously inspected/calibrated; not blind evaluation.'],
        coco=coco_counts,site=stats,training=dict(seed=27,epochs=20,imgsz=640,batch=16,lr0=.0001,freeze=10,replay_images=2400,site_repeat=3),
        selection='Max reviewed site val mAP50-95 among epochs with COCO1000 val mAP50-95 >= original - .01; earliest tie. If none pass, retain original, report best candidate separately.',
        fixed_policy=dict(alpha=.972,lum_gate=35,action_threshold=.925,object_threshold=.581,
                          drink=[39,41],pick_up=[39,41,67,73,65]),production_updated=False))
    print(json.dumps(dict(site=stats,coco=coco_counts,train_entries=len(lists['train'])),indent=2),flush=True)


if __name__ == '__main__':
    main()
