"""Sweep recognition thresholds on the current 720-pixel deployed BehaviorRecognizer.push.

Only frames used by a window are retained/enhanced; enhancement is a per-frame operation.
This avoids enhancing unused frames without changing window sampling or decision rules.
"""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

PKG = Path('/opt/darkpipe/op')
REPO = Path('/ssd1/wfs/project/lowlight/DarkScenePipeline')
VAL = Path('/ssd1/wfs/project/lowlight/compare/behavior/cache/realact_rf_v10_full/val')
sys.path.insert(0, str(PKG))
from darkpipe.constants import BEHAVIORS, CKPT_FILES
from darkpipe.stages.downscale import DownscaleStage
from darkpipe.stages.enhance_retinexformer import RetinexformerStage
from darkpipe.stages.recognize import BehaviorRecognizer, reject_probs, frame_luma


def windows(n, fps):
    out = [('whole', 0, n)]
    w, step = max(2, round(fps)), max(1, round(fps * .5))
    out.extend((f't{i / fps:.1f}', i, i + w) for i in range(0, n - w + 1, step))
    return out


def summary(rows, threshold):
    result = {}
    for suite in 'ABCP':
        sub = [r for r in rows if r['suite'] == suite]
        cm = np.zeros((10, 10), np.int64)
        bad = []
        fires = []
        for r in sub:
            p = reject_probs(np.array(r['decision_before_reject'], np.float32),
                             BEHAVIORS, threshold)
            i, j = BEHAVIORS.index(r['expect']), int(p.argmax())
            cm[i, j] += 1
            if i != j:
                bad.append(r['name'])
            if j != 9 and float(p[j]) >= threshold:
                fires.append({'name': r['name'], 'label': BEHAVIORS[j],
                              'confidence': float(p[j]), 'raw_luma': r['raw_luma']})
        tp = np.diag(cm).astype(float)
        f1 = 2 * tp / np.maximum(cm.sum(0) + cm.sum(1), 1e-9)
        result[suite] = {'windows': len(sub), 'correct': int(tp.sum()),
                         'top1': float(tp.sum() / max(1, cm.sum())),
                         'macro_f1': float(f1[cm.sum(1) > 0].mean()) if len(sub) else None,
                         'fired': len(fires), 'fires': fires,
                         'incorrect': bad,
                         'below_gate': sum(r['raw_luma'] < 75 for r in sub)}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suites', default='ABCP')
    a = ap.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    plan = defaultdict(list)
    src = REPO / 'real-actiondata-0909'
    if 'A' in a.suites:
        for p in sorted(VAL.glob('*/*.npy')):
            stem, start = p.stem.rsplit('_', 1)
            path = src / (stem + '.mp4')
            if not path.exists():
                path = src / 'other' / (stem + '.mp4')
            plan[path].append(('A', p.stem, p.parent.name, int(start), int(start) + 32))
    groups = []
    if 'B' in a.suites:
        groups.extend(('B', p, 'other') for p in sorted((src / 'other').glob('other*.mp4')))
    if 'C' in a.suites:
        groups.extend(('C', src / (stem + '.mp4'), lab) for stem, lab in (
            ('挥手22', 'wave'), ('捡东西2', 'pick_up'), ('追逐16', 'chase'), ('喝水16', 'drink')))
    if 'P' in a.suites:
        groups.extend(('P', p, 'other') for p in sorted(
            (REPO / 'online-negative-sample/20261012_083305_78').glob('*/*.mp4')))
    for suite, path, expect in groups:
        cap = cv2.VideoCapture(str(path))
        assert cap.isOpened(), path
        n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS) or 25
        cap.release()
        for tag, lo, hi in windows(n, fps):
            plan[path].append((suite, f'{path.stem}@{tag}', expect, lo, hi))
    planned = Counter(row[0] for rows in plan.values() for row in rows)
    print('[plan]', dict(planned), 'videos', len(plan), flush=True)
    for k, count in {'A': 177, 'B': 365, 'C': 18, 'P': 460}.items():
        if k in a.suites:
            assert planned[k] == count, (k, planned[k], count)
    ds = DownscaleStage(720)
    enhancer = RetinexformerStage(str(PKG / 'ckpts'), chunk=1)
    enhancer.load('cuda:0')
    r = BehaviorRecognizer(str(PKG / 'ckpts'), span_sec=None, reject_tau=.65,
                           skel_fusion_alpha=.7, skel_no_person='soft',
                           fuse_mode='alpha', lum_gate=75)
    r.load('cuda:0')
    assert r.skel is not None, 'skeleton branch must be active for acceptance'
    r.warmup()
    captured = {}
    original_infer, original_fuse = r._infer, r.skel.fuse
    def infer(arr):
        p = original_infer(arr)
        captured['rgb'] = p.copy()
        return p
    def fuse(pr, ps, pres):
        captured['skeleton'] = ps.copy()
        captured['presence'] = pres
        out = original_fuse(pr, ps, pres)
        captured['fused'] = out.copy()
        return out
    r._infer, r.skel.fuse = infer, fuse
    rows = []
    started = time.perf_counter()
    darkest_luma = 255.0
    visual_dir = Path('/audit/visual')
    visual_dir.mkdir(exist_ok=True)
    output = Path('/audit/native-windows.jsonl')
    with output.open('w') as stream:
        for vidno, (path, items) in enumerate(plan.items(), 1):
            selected = sorted({int(i) for row in items for i in
                               np.linspace(row[3], row[4] - 1, 32).round().astype(int)})
            cap = cv2.VideoCapture(str(path))
            assert cap.isOpened(), path
            raw = {}
            want = set(selected)
            idx = 0
            while idx <= selected[-1]:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx in want:
                    raw[idx] = ds([frame])[0]
                idx += 1
            cap.release()
            assert len(raw) == len(selected), (path, len(raw), len(selected))
            reduced = [raw[i] for i in selected]
            enhanced_list = enhancer(reduced)
            enhanced = dict(zip(selected, enhanced_list))
            middle = selected[len(selected) // 2]
            luminance = frame_luma(raw[middle])
            if luminance < darkest_luma:
                darkest_luma = luminance
                cv2.imwrite(str(visual_dir / 'darkest-raw.jpg'), raw[middle])
                cv2.imwrite(str(visual_dir / 'darkest-enhanced.jpg'), enhanced[middle])
                (visual_dir / 'darkest-source.json').write_text(json.dumps(
                    {'video': str(path), 'frame_index': middle, 'raw_luma': luminance,
                     'enhanced_luma': frame_luma(enhanced[middle])}, ensure_ascii=False, indent=2))
            if path.stem in ('喝水16', '捡东西2', '挥手22', '追逐16'):
                cv2.imwrite(str(visual_dir / f'{path.stem}-raw.jpg'), raw[middle])
                cv2.imwrite(str(visual_dir / f'{path.stem}-enhanced.jpg'), enhanced[middle])
            for suite, name, expect, lo, hi in items:
                inds = np.linspace(lo, hi - 1, 32).round().astype(int)
                r.reset()
                captured.clear()
                t0 = time.perf_counter()
                for i, index in enumerate(inds):
                    ev = r.push(enhanced[int(index)], i, i / 32,
                                raw_frame=raw[int(index)])
                assert ev is not None
                before = captured.get('fused', captured['rgb'])
                p = reject_probs(before, BEHAVIORS, .65)
                label = BEHAVIORS[int(p.argmax())]
                assert ev.label == r.display(label), (name, ev.label, label)
                row = {'suite': suite, 'name': name, 'expect': expect, 'label': label,
                       'confidence': ev.confidence,
                       'raw_luma': float(np.mean([frame_luma(raw[int(i)]) for i in inds])),
                       'note': r.prior_note,
                       'rgb': captured['rgb'].tolist(),
                       'decision_before_reject': before.tolist(),
                       'presence': captured.get('presence'),
                       'window_seconds': time.perf_counter() - t0}
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                stream.flush()
            print(f'[progress] video {vidno}/{len(plan)} windows {len(rows)}/{sum(planned.values())}'
                  f' elapsed {time.perf_counter()-started:.0f}s {path.name}', flush=True)
            del enhanced, enhanced_list, reduced, raw
    report = {'package': str(PKG), 'checkpoint': CKPT_FILES['behavior'],
              'skeleton_active': True, 'fov': r.fov, 'alpha': .7,
              'no_person': 'soft', 'lum_gate': 75,
              'elapsed_seconds': time.perf_counter() - started,
              'proc_max_side': 720, 'enhance_chunk': 1,
              'thresholds': {str(t): summary(rows, t) for t in (.65, .7, .75, .8, .85, .89, .9)}}
    Path('/audit/native-summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print('[finished]', report['elapsed_seconds'], 'seconds', flush=True)
    r.close()
    enhancer.close()


if __name__ == '__main__':
    main()
