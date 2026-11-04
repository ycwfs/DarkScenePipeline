"""Reproducible descriptive analysis; source groups, never frames, are resampled."""
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from build_dataset import OUT, ROOT, save
from evaluate_actions import decision


VARIANTS = ('pretrained_full', 'pretrained_roi', 'finetuned_full', 'finetuned_roi')
NAMES = ('bottle', 'cup', 'phone', 'camera', 'paper', 'portable_object', 'robot_or_exhibit')
ACTION_KEYS = dict(drinking_water='drink', picking_up_object='pick_up', waving='wave',
                   chasing='chase', running='chase', falling='fall', throwing_object='throw')


def write_csv(path: Path, rows: list) -> None:
    assert rows
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_group_stats(groups: dict, repetitions=5000) -> dict:
    """Each group contains (numerator before, numerator after, shared denominator)."""
    a = np.asarray(list(groups.values()), dtype=float)
    assert len(a) and (a[:, 2] > 0).all()
    rng = np.random.default_rng(25)
    draws = a[rng.integers(len(a), size=(repetitions, len(a)))].sum(axis=1)
    rates = draws[:, :2] / draws[:, 2, None]
    delta = rates[:, 1] - rates[:, 0]
    total = a.sum(axis=0)
    return dict(groups=len(a), denominator=int(total[2]), before=float(total[0]/total[2]),
                after=float(total[1]/total[2]), difference=float((total[1]-total[0])/total[2]),
                before_ci95=np.quantile(rates[:, 0], [.025, .975]).tolist(),
                after_ci95=np.quantile(rates[:, 1], [.025, .975]).tolist(),
                difference_ci95=np.quantile(delta, [.025, .975]).tolist(),
                method='paired source-group percentile bootstrap, 5000 resamples, RNG seed25; descriptive only',
                inference_limit='same camera/scenes and assistant GT; not an independent-population interval')


def objects() -> tuple:
    data = json.loads((OUT/'object-evaluation.json').read_text())
    rows, stats = [], {}
    for split in ('val', 'test'):
        for name in VARIANTS:
            for c, label in enumerate(NAMES):
                total = data['summary'][split][name][str(c)]
                rows.append(dict(split=split, variant=name, class_id=c, label=label,
                    supported=not name.startswith('pretrained') or c in (0, 1, 2),
                    tp=total['tp'], fp=total['fp'], fn=total['fn'],
                    recall=total['recall'] if total['tp']+total['fn'] else None,
                    precision=total['precision'] if total['tp']+total['fp'] else None))
        for c in (0, 2, 4, 6):
            groups = defaultdict(lambda: np.zeros(3))
            for r in data['records']:
                if r['split'] != split:
                    continue
                a = r['variants']['finetuned_full']['counts'][str(c)]
                b = r['variants']['finetuned_roi']['counts'][str(c)]
                if a['tp']+a['fn']:
                    groups[r['group']] += [a['tp'], b['tp'], a['tp']+a['fn']]
            if groups:
                stats[f'{split}_{NAMES[c]}_finetuned_roi_minus_full'] = paired_group_stats(groups)
    write_csv(OUT/'object-counts.csv', rows)
    return data, stats


def actions() -> tuple:
    files = [json.loads(f.read_text()) for f in sorted((OUT/'action-evaluation').glob('*.json'))]
    counts, rows, stats = [], [], {}
    for d in files:
        for r in d['records']:
            rows.append(dict(source_id=d['job']['id'], split=d['job']['split'], group=d['job']['group'],
                event_index=r['event_index'], person_id=r['track_id'], history_generation=r['history_generation'],
                timestamp=r['timestamp'], truth=r['truth'], baseline=r['baseline'],
                conservative=r['conservative'], strict=r['strict'], reason=r['reason']))
        if d['job']['split'] == 'feedback':
            ss = [s for r in d['records'] for s in r['samples']]
            counts.append(dict(source_id=d['job']['id'], windows=len(d['records']),
                drink=sum(r['baseline']=='drink' for r in d['records']),
                pick_up=sum(r['baseline']=='pick_up' for r in d['records']),
                conservative_veto=sum(r['conservative']!=r['baseline'] for r in d['records']),
                strict_veto=sum(r['strict']!=r['baseline'] for r in d['records']),
                samples=len(ss), binding_valid=sum(s['binding_valid'] for s in ss),
                pose_available=sum(s['pose'] is not None for s in ss)))
    for split in ('val', 'test'):
        for truth in ('drink', 'pick_up', 'other'):
            for policy in ('conservative', 'strict'):
                groups = defaultdict(lambda: np.zeros(3))
                for r in rows:
                    if r['split'] == split and r['truth'] == truth:
                        groups[r['group']] += [r['baseline']==truth, r[policy]==truth, 1]
                if groups:
                    stats[f'{split}_{truth}_{policy}_correct_minus_baseline'] = paired_group_stats(groups)
    write_csv(OUT/'action-windows.csv', rows)
    write_csv(OUT/'feedback-counts.csv', counts)
    save(OUT/'action-summary.json', dict(feedback=counts, old_truth_breakdown={
        split: {str(k): v for k, v in Counter((r['truth'], r['baseline'], r['conservative'], r['strict'])
            for r in rows if r['split']==split).items()} for split in ('val', 'test')},
        old_windows=sum(r['split']!='feedback' for r in rows), new_windows=sum(r['split']=='feedback' for r in rows),
        interpretation='Feedback has no complete person-action truth: suppression counts, not corrected errors.'))
    return files, rows, counts, stats


def map_clips(files: list) -> dict:
    """Coverage of existing tau075 clips; this does not simulate the clip state machine."""
    base = ROOT/'reports/manypeople-inference-20261025'
    lookup = {d['job']['number']: {r['event_index']: r for r in d['records']}
              for d in files if d['job']['split']=='feedback'}
    output, summaries, event_totals = [], [], Counter()
    for number in range(1, 10):
        events = json.loads((base/'events'/f'{number:02d}_events.json').read_text())
        by_person = defaultdict(list)
        for i, e in enumerate(events):
            label = decision(e)['label']
            prior = lookup[number].get(i)
            assert (prior is not None) == (label in ('drink', 'pick_up'))
            if prior:
                assert prior['track_id']==e['track_id'] and prior['history_generation']==e['history_generation']
                assert prior['frame_index']==e['frame_index'] and prior['baseline']==label
            row = dict(index=i, timestamp=e['timestamp'], tau075=e['label'], tau085=label,
                conservative=prior['conservative'] if prior else label,
                strict=prior['strict'] if prior else label)
            by_person[(e['track_id'], e['history_generation'])].append(row)
            for policy in ('tau075', 'tau085', 'conservative', 'strict'):
                event_totals[f'{policy}:{row[policy]}'] += 1
        clip_files = sorted((base/'jobs'/f'{number:02d}'/'results').glob('**/*.json'))
        for path in clip_files:
            meta = json.loads(path.read_text())
            if 'action_key' not in meta or 'video' not in meta:
                continue
            action = ACTION_KEYS[meta['action_key']]
            person_events = by_person[(meta['person_id'], meta['history_generation'])]
            lo, hi = meta['start_source_time'], meta['end_source_time']
            covered = [e for e in person_events if lo-1e-6 <= e['timestamp'] <= hi+1e-6]
            first = meta['trigger_stream_time']
            last = meta['first_event']['last_action_source_time']
            trigger = [e for e in covered if first-1e-6 <= e['timestamp'] <= last+1e-6]
            video = path.parent/meta['video']
            assert video.exists() and any(e['tau075']==action for e in covered), str(path)
            row = dict(number=number, original_clip=str(video.relative_to(ROOT)),
                metadata=str(path.relative_to(ROOT)), person_id=meta['person_id'],
                history_generation=meta['history_generation'], action=action,
                start=lo, end=hi, first_trigger=first, last_trigger=last,
                context_complete=meta['context_complete'])
            for policy in ('tau075', 'tau085', 'conservative', 'strict'):
                row[f'{policy}_matching_windows_in_saved_range'] = sum(e[policy]==action for e in covered)
                row[f'{policy}_matching_windows_in_trigger_range'] = sum(e[policy]==action for e in trigger)
            output.append(row)
        nrows = [r for r in output if r['number']==number]
        summaries.append(dict(number=number, clips=len(nrows), **{
            f'{policy}_clips_with_matching_window': sum(r[f'{policy}_matching_windows_in_saved_range']>0 for r in nrows)
            for policy in ('tau075', 'tau085', 'conservative', 'strict')}))
    assert len(output)==454
    write_csv(OUT/'clip-coverage-454.csv', output)
    result = dict(rows=output, summary=summaries, event_counts=dict(event_totals),
        note='Existing tau075 files matched by source, person, generation and time. No new files were saved; '
             'coverage is NOT predicted new clip count, event-level accuracy or error correction.')
    save(OUT/'clip-coverage-454.json', result)
    return result


def figures(obj: dict, action_rows: list, feedback: list) -> None:
    folder = OUT/'figures'; folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = ['#9aa5b1', '#71a9ce', '#e7a25b', '#157f79']
    def save_figure(fig, name):
        fig.savefig(folder/f'{name}.png', dpi=180, bbox_inches='tight')
        fig.savefig(folder/f'{name}.pdf', bbox_inches='tight')
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    classes = [0, 2, 4, 6]
    for j, variant in enumerate(VARIANTS):
        xs, ys = [], []
        for i, c in enumerate(classes):
            if variant.startswith('pretrained') and c not in (0, 2):
                continue
            d = obj['summary']['test'][variant][str(c)]
            xs.append(i+(j-1.5)*.18); ys.append(d['recall'])
            ax.text(xs[-1], ys[-1]+.025, f"{d['tp']}/{d['tp']+d['fn']}", ha='center', fontsize=9)
        ax.bar(xs, ys, width=.17, color=colors[j], label=variant.replace('_', ' '))
    ax.set_xticks(range(4), ['Bottle', 'Phone', 'Paper*', 'Robot/exhibit*'])
    ax.set_ylim(0, 1.02); ax.set_ylabel('Recall at conf 0.35 / IoU 0.50')
    ax.set_title('Held-out objects: crop gains do not solve phone detection')
    ax.legend(ncol=2, loc='upper left'); ax.grid(axis='y', alpha=.2)
    fig.text(.13, -.02, '* No matching pretrained class. Assistant GT; one seed; no error bars.', fontsize=10)
    save_figure(fig, '01-object-test')
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for j, policy in enumerate(('baseline', 'conservative', 'strict')):
        vals=[]; den=[]
        for truth in ('drink', 'pick_up', 'other'):
            rs=[r for r in action_rows if r['split']=='test' and r['truth']==truth]
            vals.append(sum(r[policy]==truth for r in rs));den.append(len(rs))
        xx=np.arange(3)+(j-1)*.23
        ax.bar(xx, vals, width=.22, color=colors[[0,3,2][j]], label=policy)
        for x, n, d in zip(xx, vals, den):ax.text(x,n+.1,f'{n}/{d}',ha='center',fontsize=10)
    ax.set_xticks(range(3), ['True drink', 'True pick-up', 'Known other'])
    ax.set_ylabel('Correct selected person windows');ax.set_ylim(0,13)
    ax.set_title('Strict evidence causes missed true actions (old test subset)')
    ax.legend(); save_figure(fig,'02-action-retention')
    selection=json.loads((OUT/'selection.json').read_text())['rows']
    fig, ax=plt.subplots(figsize=(8.5,4))
    ax.plot(range(1,21),[r['score'] for r in selection],marker='o',color=colors[3])
    ax.axvline(17,linestyle='--',color=colors[2],label='Selected epoch 17')
    ax.set(xlabel='Epoch',ylabel='Old validation mAP50-95',title='Selection used deployed one-to-one head (seed 25)')
    ax.set_xticks([1,5,10,15,17,20]);ax.legend();save_figure(fig,'03-selection')
    fig, ax=plt.subplots(figsize=(9,4))
    x=np.arange(9)
    ax.bar(x-.18,[r['windows'] for r in feedback],width=.35,label='Baseline = conservative',color=colors[3])
    ax.bar(x+.18,[r['windows']-r['strict_veto'] for r in feedback],width=.35,label='Strict retained',color=colors[2])
    ax.set_xticks(x,[r['source_id'][-2:] for r in feedback])
    ax.set(xlabel='Feedback source',ylabel='Drink / pick-up windows',title='New footage: suppression is not verified error correction')
    ax.legend();save_figure(fig,'04-feedback')


def main() -> None:
    obj, obj_stats = objects()
    files, rows, counts, action_stats = actions()
    coverage = map_clips(files)
    save(OUT/'statistics.json', dict(objects=obj_stats, actions=action_stats,
        runs=1, seed=25, significance_tests='Not performed: too few independent scenes, assistant GT, selected action subset.',
        multiplicity='No confirmatory p-values or significance claims. All descriptive contrasts are listed.'))
    figures(obj, rows, counts)
    print(json.dumps(dict(feedback=counts, clip_summary=coverage['summary']), ensure_ascii=False, indent=2))


if __name__=='__main__':
    main()
