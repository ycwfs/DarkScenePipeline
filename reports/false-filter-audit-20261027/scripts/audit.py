"""Retrospective fixed-evidence audit; no production mutation or new tuning claim."""
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

os.environ.setdefault('MPLCONFIGDIR', '/tmp/darkpipe-false-filter-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
BASE = ROOT / 'reports/four-parameter-tuning-20261026'
sys.path.insert(0, str(ROOT / 'platform/dark-demo-v2'))
from darkpipe.person.object_prior import filter_object_evidence

LABELS = ['wave', 'throw', 'chase', 'fall', 'fight', 'talk', 'drink', 'pick_up', 'shake_hands', 'other']


def load(path: Path):
    return json.loads(path.read_text())


def save(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def reject(probs: np.ndarray) -> np.ndarray:
    pred = probs.argmax(1)
    pred[probs.max(1) < .925] = 9
    return pred


def carry_support(rows: list, initial: np.ndarray, before_objects: np.ndarray,
                  evidence: dict, sources: dict, seconds: float) -> tuple:
    """Only past measured detections; same source, track, generation and action.

    The candidate must still clear the unchanged action threshold. A different
    action, low/unknown luma or unavailable binding clears the evidence memory.
    Memory uses detection-frame time, never refreshes on a carried prediction,
    and never borrows from another actor or a later frame.
    """
    result = initial.copy()
    state = {}
    changes = []
    order = sorted(range(len(rows)), key=lambda i: (rows[i]['source_id'], rows[i]['timestamp'], rows[i]['event_index']))
    for i in order:
        r = rows[i]
        key = (r['source_id'], r['track_id'], r['history_generation'])
        label = LABELS[int(before_objects[i])]
        luma = r['raw_luma']
        if label not in ('drink', 'pick_up') or luma is None or not math.isfinite(luma) or luma < 35:
            state.pop(key, None)
            continue
        e = evidence[(r['id'], label)]
        decision = filter_object_evidence(label, e['samples'], luma, .581, 35.)
        assert LABELS[int(initial[i])] == decision['label'], r['id']
        if not decision['applied']:
            state.pop(key, None)
            continue
        previous = state.get(key)
        if previous is not None and previous['label'] != label:
            state.pop(key, None)
        if decision['evidence']:
            fps = float(Fraction(str(sources[r['source_id']]['fps'])))
            proof = max(d['frame'] / fps for d in decision['evidence'])
            assert proof <= r['timestamp'] + 1e-6
            state[key] = dict(label=label, proof_time=proof, from_window=r['id'])
        previous = state.get(key)
        if (initial[i] == 9 and previous is not None
                and 0 <= r['timestamp'] - previous['proof_time'] <= seconds + 1e-6):
            result[i] = before_objects[i]
            changes.append(dict(id=r['id'], truth=r['truth'], is_drink=r['is_drink'],
                                source=sources[r['source_id']]['source'], timestamp=r['timestamp'],
                                track_id=r['track_id'], **previous))
    return result, changes


def main() -> None:
    rows = load(ROOT / 'reports/threshold-tuning-20261026/windows.json')
    saved = load(BASE / 'predictions.json')
    assert [r['id'] for r in rows] == saved['ids']
    sources = {s['id']: s for s in load(BASE / 'protocol.json')['sources']}
    assert all(abs(r['timestamp'] - r['frame_index'] / sources[r['source_id']]['fps']) < 1e-4 for r in rows)
    arrays = np.load(BASE / 'inputs.npz')
    gi = arrays['gates'].tolist().index(35)
    probs = .972 * arrays['rgb'] + .028 * arrays['skeleton'][gi]
    dark = arrays['luma'] < 35
    probs[dark] = arrays['rgb'][dark]
    rgb = reject(arrays['rgb'])
    fusion = reject(probs)
    support = arrays['object_support'][gi]

    def objects(threshold: float) -> np.ndarray:
        pred = fusion.copy()
        for cls, column in [(6, 0), (7, 1)]:
            pred[(fusion == cls) & (support[:, column] < threshold)] = 9
        return pred

    formal = objects(.581)
    assert np.array_equal(formal, saved['predictions']['recommended'])
    wave_rows = load(ROOT / 'reports/wave-pose-prior-20261027/old-window-results.json')
    running_rows = load(ROOT / 'reports/running-speed-prior-20261027/old-paired-results.json')
    index = {r['id']: i for i, r in enumerate(rows)}
    wave, running = formal.copy(), formal.copy()
    for r in wave_rows:
        assert formal[index[r['id']]] == LABELS.index(r['baseline'])
        wave[index[r['id']]] = LABELS.index(r['variants']['3_total'])
    for r in running_rows:
        assert wave[index[r['id']]] == LABELS.index(r['baseline'])
        running[index[r['id']]] = LABELS.index(r['candidate'])
    old_mask = np.array([r['scope'] != 'new_multipeople' for r in rows])
    truth = np.array([LABELS.index(r['truth']) if r['truth'] is not None else -1 for r in rows])
    split = np.array([r['split'] for r in rows])
    new_false = np.array([r['is_drink'] is False for r in rows])
    new_true = np.array([r['is_drink'] is True for r in rows])

    def metrics(pred: np.ndarray, mask: np.ndarray) -> dict:
        known = mask & (truth >= 0)
        positive = known & (truth != 9)
        other = known & (truth == 9)
        classes = {}
        for c in sorted(set(truth[known])):
            tp = int((known & (truth == c) & (pred == c)).sum())
            fp = int((known & (truth != c) & (pred == c)).sum())
            fn = int((known & (truth == c) & (pred != c)).sum())
            classes[LABELS[c]] = dict(total=tp + fn, tp=tp, fp=fp, fn=fn,
                                     f1=2 * tp / max(1, 2 * tp + fp + fn))
        return dict(known=int(known.sum()), unknown=int((mask & (truth < 0)).sum()),
                    positive_total=int(positive.sum()), positive_correct=int((positive & (pred == truth)).sum()),
                    other_total=int(other.sum()), other_false=int((other & (pred != 9)).sum()),
                    macro_f1=float(np.mean([v['f1'] for v in classes.values()])), classes=classes)

    def scores(pred: np.ndarray) -> dict:
        return dict(old={s: metrics(pred, old_mask & (split == s)) for s in ['train', 'val', 'test']},
                    new_known_drink_fp=int((pred[new_false] == 6).sum()),
                    new_known_drink_tp=int((pred[new_true] == 6).sum()))

    stages = {name: scores(p) for name, p in [('rgb', rgb), ('fusion', fusion),
              ('objects_formal', formal), ('wave_candidate', wave), ('running_candidate', running)]}
    loss = []
    for i in np.flatnonzero(old_mask & (truth >= 0) & (truth != 9) & (rgb == truth) & (running != truth)):
        r = rows[i]
        stage = 'fusion' if fusion[i] != truth[i] else 'objects' if formal[i] != truth[i] else 'wave' if wave[i] != truth[i] else 'running'
        confidence = support[i, 0 if truth[i] == 6 else 1] if truth[i] in (6, 7) else None
        loss.append(dict(id=r['id'], source=sources[r['source_id']]['source'], split=r['split'],
                         timestamp=r['timestamp'], track_id=r['track_id'], truth=r['truth'], stage=stage,
                         rgb_score=float(arrays['rgb'][i, truth[i]]), fused_score=float(probs[i, truth[i]]),
                         pose_valid=bool(arrays['pose_valid'][gi, i]),
                         object_support=float(confidence) if confidence is not None and math.isfinite(confidence) else None))
    thresholds = [dict(threshold=t, **scores(objects(t))) for t in [.581, .55, .50, .35, .10, .01]]
    evidence = {}
    for path in (BASE / 'evidence').glob('*.json'):
        for record in load(path)['records']:
            evidence[(record['window_id'], record['label'])] = record
    memory = []
    for duration in [1., 2.]:
        pred, changes = carry_support(rows, formal, fusion, evidence, sources, duration)
        memory.append(dict(seconds=duration, **scores(pred), changes=changes))
    # Separate exact runtime binding from the historical nine-video approximation.
    runtime_memory = []
    for number in [1, 7, 9]:
        sid = f'new-{number:02}'
        directory = ROOT / 'reports/running-speed-prior-20261027/long-videos' / sid
        events = load(directory / 'events.json')
        source = load(directory / 'summary.json')['source']
        rr = [dict(r, id=f'{sid}_{i}', source_id=sid, event_index=i, truth=None, is_drink=None)
              for i, r in enumerate(events)]
        ee = {(r['id'], LABELS[int(np.argmax(r['fused_probs']))]): dict(samples=r['object_prior'].get('samples', []))
              for r in rr}
        original = np.array([LABELS.index(r['label']) for r in rr])
        pre = reject(np.array([r['fused_probs'] for r in rr], dtype=np.float32))
        source_meta = {sid: dict(fps=source['fps'], source=source['absolute_path'])}
        for duration in [1., 2.]:
            pred, changes = carry_support(rr, original, pre, ee, source_meta, duration)
            runtime_memory.append(dict(source=sid, seconds=duration, events=len(rr),
                                       changed_windows=len(changes), changes=changes))
    qa = load(ROOT / 'reports/algorithm-flow-20261027/editable-qa.json')
    model_file = ROOT / 'platform/dark-demo-v2/models.json'
    assert hashlib.sha256(model_file.read_bytes()).hexdigest() == qa['models_sha256']
    assert load(model_file)['defaults'] == qa['defaults']
    result = dict(scope='Retrospective cached evidence diagnosis; no new threshold selected, no deployed code/model/package changed.',
                  windows=len(rows), old_windows=int(old_mask.sum()), flow_defaults_match=True,
                  stages=stages, lost_vs_rgb=loss, object_threshold_diagnostics=thresholds, causal_memory_diagnostics=memory,
                  exact_runtime_memory=runtime_memory,
                  limits='New 9-video evidence uses historical approximate actor binding; only 527 original drink candidates have binary feedback. Old test is historical and already inspected. Memory is decision-only replay, not integrated real-time behavior.')
    save(OUT / 'analysis-results.json', result)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    names = ['RGB', '+ fusion', '+ objects', '+ wave', '+ running']
    tests = [s['old']['test'] for s in stages.values()]
    axes[0].bar(names, [s['positive_correct'] for s in tests], color='#457c9b')
    axes[0].set_ylabel('Correct action windows / 51')
    axes[1].bar(names, [s['other_false'] for s in tests], color='#ae644e')
    axes[1].set_ylabel('False actions on other / 163')
    for ax in axes:
        ax.tick_params(axis='x', rotation=20)
    fig.tight_layout()
    fig.savefig(OUT / 'figures/stage-effects.png', dpi=170)
    fig.savefig(OUT / 'figures/stage-effects.pdf')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    for r in thresholds:
        x = r['new_known_drink_fp']; y = r['old']['test']['positive_correct']
        ax.scatter(x, y)
        ax.annotate(str(r['threshold']), (x, y), xytext=(4, 6), textcoords='offset points')
    ax.set_xlabel('New reviewed false drink windows / 526')
    ax.set_ylabel('Old test correct action windows / 51')
    fig.tight_layout()
    fig.savefig(OUT / 'figures/object-tradeoff.png', dpi=170)
    fig.savefig(OUT / 'figures/object-tradeoff.pdf')
    plt.close(fig)
    print(json.dumps(dict(stages={k:v['old']['test'] for k,v in stages.items()},
        memory=[{k:v for k,v in r.items() if k!='changes'} | {'changed_windows':len(r['changes'])} for r in memory]), ensure_ascii=False))


if __name__ == '__main__':
    main()
