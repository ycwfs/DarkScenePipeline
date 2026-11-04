"""Train on frozen source groups, select on validation, lock before test scoring."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'platform/dark-demo-v2'), str(ROOT/'reports/person-finetune-20261023/scripts')]
from darkpipe.constants import BEHAVIORS
from darkpipe.person.core import fuse_person
from darkpipe.stages.skeleton_aux import STGCN
from train import metrics
from assemble import assign_label


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dataset(split):
    protocol = json.loads((OUT/'protocol.json').read_text())
    reviews = json.loads((ROOT/'reports/luma-gate15-20261024/label-review.json').read_text())['records']
    reviews = {(r['video_id'], r['track_id']): r for r in reviews}
    rows, features, coverage = [], [], []
    for video in protocol['videos']:
        if video['split'] != split:
            continue
        data = OUT/'data'/video['id']
        doc = json.loads((data/'result.json').read_text())
        assert doc['protocol_sha256'] == sha(OUT/'protocol.json')
        sk = np.load(data/'skeleton-inputs.npz')['normalized']
        assert len(sk) == len(doc['events']['30'])
        additional = {}
        if any(key[0] == video['id'] for key in reviews):
            old = json.loads((ROOT/'reports/luma-gate15-20261024/data'/video['id']/'result.json').read_text())
            for row in old['events']['15']:
                review = reviews.get((video['id'], row['track_id']))
                if review:
                    additional[(row['crop_sha256'], tuple(row['source_frames']))] = assign_label(row['timestamps'], review['segments'])[0]
        known = 0
        for row, x in zip(doc['events']['30'], sk):
            extra = additional.get((row['crop_sha256'], tuple(row['source_frames'])))
            if extra is not None:
                row.update(truth=extra, truth_provenance='exact crop/frames matched to previous prediction-free visual review')
            row.update(video_id=video['id'], source=video['source'], group=video['group'], split=split,
                       people=len(doc['frames'][row['frame_index']]['boxes']))
            rows.append(row)
            features.append(x)
            known += row['truth'] is not None
        coverage.append(dict(video_id=video['id'], source=video['source'], split=split,
            frames=video['frames'], box_frames=sum(bool(f['boxes']) for f in doc['frames']),
            windows=len(sk), known=known, unknown=len(sk)-known))
    return rows, np.asarray(features, np.float32).reshape(-1, 3, 32, 17), coverage


def score(rows, labels):
    known = [i for i,r in enumerate(rows) if r['truth'] is not None]
    result = metrics([BEHAVIORS.index(rows[i]['truth']) for i in known], [labels[i] for i in known])
    positive = [i for i in known if rows[i]['truth'] != 'other']
    result.update(positive_count=len(positive), positive_correct=sum(BEHAVIORS[labels[i]]==rows[i]['truth'] for i in positive))
    return result


@torch.inference_mode()
def infer(net, features):
    net.eval()
    return np.concatenate([net(torch.from_numpy(features[i:i+128]).cuda()).softmax(1).cpu().numpy()
        for i in range(0, len(features), 128)]) if len(features) else np.zeros((0, 10), np.float32)


def labels_for(rows, skeleton, alpha, threshold, no_pose):
    rgb = np.asarray([r['rgb_probs'] for r in rows], np.float32)
    sk = np.array(skeleton, dtype=np.float32, copy=True)
    valid = np.asarray([r['pose_valid'] for r in rows], bool)
    blocked = np.asarray([r['raw_luma'] is not None and r['raw_luma'] < 30 for r in rows], bool)
    sk[~valid] = np.eye(10, dtype=np.float32)[9]
    fused = alpha*rgb+(1-alpha)*sk
    if no_pose == 'rgb':
        fused[~valid] = rgb[~valid]
    if no_pose == 'other':
        fused[~valid] = sk[~valid]
    if alpha >= .999:
        fused = rgb.copy()
    fused[blocked] = rgb[blocked]
    labels = fused.argmax(1)
    labels[(fused.max(1) < threshold) & (labels != 9)] = 9
    return labels, fused


def train():
    protocol = json.loads((OUT/'protocol.json').read_text())
    for video in protocol['videos']:
        if not (OUT/'data'/video['id']/'result.json').exists():
            raise RuntimeError('Extraction incomplete: '+video['id'])
    torch.set_num_threads(2)
    random.seed(24); np.random.seed(24); torch.manual_seed(24); torch.cuda.manual_seed_all(24)
    rows, x, coverage = dataset('train')
    val, vx, valcoverage = dataset('val')
    ids = [i for i,r in enumerate(rows) if r['pose_valid'] and r['truth'] is not None]
    vids = [i for i,r in enumerate(val) if r['pose_valid'] and r['truth'] is not None]
    labels = np.array([BEHAVIORS.index(rows[i]['truth']) for i in ids])
    counts = Counter((rows[i]['truth'], rows[i]['video_id']) for i in ids)
    sources = Counter(c for c,_ in counts)
    weights = [1/(sources[rows[i]['truth']]*counts[rows[i]['truth'], rows[i]['video_id']]) for i in ids]
    loader = DataLoader(TensorDataset(torch.from_numpy(x[ids]), torch.from_numpy(labels)), batch_size=64,
        sampler=WeightedRandomSampler(weights, len(ids), replacement=True), num_workers=0)
    net = STGCN(num_classes=10).cuda()
    old = ROOT/protocol['models']['skeleton']['path']
    state = torch.load(old, map_location='cpu', weights_only=False)
    net.load_state_dict(state.get('model', state))
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-3)
    history, best, best_epoch = [], None, 0
    info = dict(seed=24, train_windows=len(ids), val_windows=len(vids), train_labels=dict(Counter(rows[i]['truth'] for i in ids)),
                val_labels=dict(Counter(val[i]['truth'] for i in vids)), coverage=coverage+valcoverage,
                initialization_sha256=sha(old), protocol_sha256=sha(OUT/'protocol.json'), epochs=20,
                training='Only pose-valid reviewed train person/time windows, no test; normalized same as deployment; source/class balanced sampling; no pose smoothing',
                selection='Validation skeleton macro F1 on eligible reviewed windows, ties prefer fewer other false positives; epoch0 eligible')
    save(OUT/'training-protocol.json', info)
    for epoch in range(21):
        losses = []
        if epoch:
            net.train()
            for xx, yy in loader:
                xx, yy = xx.cuda(), yy.cuda()
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(net(xx), yy)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.)
                optimizer.step()
                losses.append(float(loss.detach()))
        probs = infer(net, vx)
        m = score([val[i] for i in vids], probs[vids].argmax(1))
        key = (m['macro_f1'], -m['other_false_positives'], m['top1'])
        row = dict(epoch=epoch, loss=float(np.mean(losses)) if losses else None, validation=m)
        history.append(row)
        if best is None or key > best:
            best, best_epoch = key, epoch
            torch.save(dict(model={k:v.detach().cpu() for k,v in net.state_dict().items()},
                epoch=epoch, training=info, validation=m), OUT/'models/stgcn_person_20261024.pth')
        save(OUT/'training-history.json', dict(selected_epoch=best_epoch, rows=history))
        print('[train]', epoch, 'loss', row['loss'], 'val', m['macro_f1'], 'other', m['other_false_positives'], 'best', best_epoch, flush=True)


def select():
    torch.set_num_threads(2)
    rows, x, coverage = dataset('val')
    net = STGCN(num_classes=10).cuda()
    candidate = OUT/'models/stgcn_person_20261024.pth'
    net.load_state_dict(torch.load(candidate, map_location='cpu', weights_only=False)['model'])
    new = infer(net, x)
    old = np.array([r['skeleton_probs'] if r['skeleton_probs'] is not None else np.eye(10)[9] for r in rows], np.float32)
    baseline = score(rows, labels_for(rows, old, .7, .75, 'soft')[0])
    grid, best, winner = [], None, None
    for model, probs in [('old', old), ('adapted', new)]:
        for no_pose in ['soft', 'rgb']:
            for ai in range(41):
                alpha = ai/40
                for ti in range(50, 96):
                    threshold = ti/100
                    labels, _ = labels_for(rows, probs, alpha, threshold, no_pose)
                    m = score(rows, labels)
                    accepted = m['other_false_positives'] <= baseline['other_false_positives']
                    config = dict(model=model, alpha=alpha, threshold=threshold, no_pose=no_pose, lum_gate=30.)
                    grid.append(dict(**config, macro_f1=m['macro_f1'], other_false_positives=m['other_false_positives'],
                        positive_correct=m['positive_correct'], correct=m['correct'], accepted=accepted))
                    key = (m['macro_f1'], -m['other_false_positives'], m['positive_correct'],
                           -abs(alpha-.7)-abs(threshold-.75), no_pose=='soft', model=='adapted')
                    if accepted and (best is None or key > best):
                        best, winner = key, dict(config=config, validation=m)
    if winner['config']['model'] == 'old':
        shutil.copy2(ROOT/'platform/dark-demo-v2/ckpts/stgcn_behavior_32f_realactall_full.pth', OUT/'models/stgcn_selected_20261024.pth')
    else:
        shutil.copy2(candidate, OUT/'models/stgcn_selected_20261024.pth')
    save(OUT/'parameter-grid.json', grid)
    save(OUT/'selection-lock.json', dict(**winner, baseline=baseline, coverage=coverage,
        protocol_sha256=sha(OUT/'protocol.json'), selected_checkpoint_sha256=sha(OUT/'models/stgcn_selected_20261024.pth'),
        criterion='Validation macro F1; other false positives <= baseline; tie fewer false positives, more true positive actions, nearest previous defaults',
        test_read_for_selection=False, candidate_count=len(grid)))
    print(json.dumps(winner, ensure_ascii=False, indent=2), flush=True)


def evaluate():
    lock = json.loads((OUT/'selection-lock.json').read_text())
    model = OUT/'models/stgcn_selected_20261024.pth'
    assert sha(model) == lock['selected_checkpoint_sha256']
    net = STGCN(num_classes=10).cuda()
    state = torch.load(model, map_location='cpu', weights_only=False)
    net.load_state_dict(state.get('model', state))
    outputs, scores, coverage = [], {}, []
    c = lock['config']
    for split in ('val', 'test', 'train'):
        rows, x, cov = dataset(split); coverage.extend(cov)
        probs = infer(net, x)
        before = np.asarray([BEHAVIORS.index(r['label']) for r in rows])
        after, fused = labels_for(rows, probs, c['alpha'], c['threshold'], c['no_pose'])
        rgb, _ = labels_for(rows, probs, 1., c['threshold'], 'rgb')
        scores[split] = dict(baseline=score(rows, before), selected=score(rows, after), rgb=score(rows, rgb))
        for i,r in enumerate(rows):
            # Call the packaged fusion path too, so vectorized tuning cannot drift silently.
            check = fuse_person(np.array(r['rgb_probs'], np.float32), probs[i] if r['pose_valid'] else None,
                [r['raw_luma']], alpha=c['alpha'], threshold=c['threshold'], lum_gate=30., no_pose=c['no_pose'])
            assert check['label'] == BEHAVIORS[after[i]]
            outputs.append(dict(**r, before=BEHAVIORS[before[i]], after=BEHAVIORS[after[i]],
                selected_skeleton_probs=probs[i].tolist() if r['pose_valid'] else None,
                selected_fused_probs=fused[i].tolist(), selected_confidence=check['confidence'], selected_mode=check['mode']))
    save(OUT/'evaluated-windows.json', outputs)
    save(OUT/'results.json', dict(selection_lock_sha256=sha(OUT/'selection-lock.json'), scores=scores, coverage=coverage,
        limits=['Single seed; validation searched many configurations.', 'Historical initializers saw many old source videos; test previously inspected.',
                'Only supported onsite classes measured, not all 9 action classes.', 'Undetected and unreviewed people excluded from classifier score, coverage reported separately.']))
    print(json.dumps(scores, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['train','select','evaluate'])
    globals()[parser.parse_args().stage]()
