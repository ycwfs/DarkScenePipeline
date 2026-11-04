"""Validate human intervals, exclude ambiguity, freeze group splits before fitting."""
from collections import Counter, defaultdict
from fractions import Fraction
import re
from common import *


def overlap(a, b):
    lo = [max(a[i], b[i]) for i in (0, 1)]
    hi = [min(a[i], b[i]) for i in (2, 3)]
    inter = max(0, hi[0]-lo[0]) * max(0, hi[1]-lo[1])
    return inter / max(1e-9, (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)


def main():
    assert not (OUT/'protocol.json').exists(), 'Do not replace an existing frozen protocol'
    uploaded = REVIEW/'10-25_动作审核.json'
    payload = read(uploaded)
    assert payload['schema_version'] == 1 and payload['manifest_sha256'] == sha(REVIEW/'review-items.json')
    # No XCLIP observations, labels, exclusion intervals or model outputs are consulted.
    items = {r['id']: r for r in read(REVIEW/'review-items.json') if r['category'] == 'videomamba'}
    sources = {r['number']: r for r in read(REVIEW/'sources.json')}
    annotations, seen = [], set()
    for row in payload['annotations']:
        uid = row['review_id']
        assert uid in items and uid not in seen and uid.startswith('M')
        seen.add(uid)
        g = items[uid]
        assert row['review_status'] in ('confirmed', 'exclude', 'uncertain')
        a, b = float(row['human_start']), float(row['human_end'])
        assert 0 <= a < b <= sources[g['number']]['duration']+.001
        if row['review_status'] == 'confirmed':
            assert row['human_label'] in LABELS and row['entity_type'] == 'human'
        match = re.fullmatch(r'B(\d+):T(\d+):G(\d+)', g['identity'])
        assert match and int(match[1]) == g['number'] and g['baseline_events']
        assert g['representative_run'] == f'manypeople-inference-20261025/events/{g["number"]:02d}_events.json'
        baseline = read(ROOT/'reports'/g['representative_run'])
        assert all((baseline[i]['track_id'], baseline[i]['history_generation']) == (int(match[2]), int(match[3])) for i in g['baseline_events'])
        annotations.append(dict(row, number=g['number'], identity=g['identity'],
            track_id=int(match[2]), generation=int(match[3]), start=a, end=b))
    assert seen == set(items)
    by_person = defaultdict(list)
    excluded_tracks = set()
    for r in annotations:
        by_person[(r['number'], r['track_id'], r['generation'])].append(r)
        if r['review_status'] == 'exclude':
            excluded_tracks.add((r['number'], r['track_id']))
    conflicts = []
    for key, rows in by_person.items():
        for i, a in enumerate(rows):
            for b in rows[i+1:]:
                if a['review_status'] == b['review_status'] == 'confirmed' and a['human_label'] != b['human_label']:
                    lo, hi = max(a['start'], b['start']), min(a['end'], b['end'])
                    if lo < hi:
                        conflicts.append(dict(person=list(key), start=lo, end=hi, reviews=[a['review_id'], b['review_id']]))
    # Group assignment uses acquisition dates and label availability only, before scoring models.
    new_split = {1:'test', 2:'test', 3:'test', 4:'val', 6:'val', 8:'val', 5:'train', 7:'train', 9:'train'}
    groups = {1:'20261005_1300_multiview', 2:'20261005_1300_multiview', 3:'20261005_1300_multiview',
        4:'20261001', 6:'20261001', 8:'20261001', 5:'20261016_14xx_multiview', 7:'20261016_14xx_multiview', 9:'20261016_14xx_multiview'}
    new, excluded, weak_candidates = [], [], []
    for number, source in sources.items():
        fps = float(Fraction(source['fps']))
        events = read(ROOT/f'reports/manypeople-inference-20261025/events/{number:02d}_events.json')
        for index, event in enumerate(events):
            key = (number, event['track_id'], event['history_generation'])
            times = [f/fps for f in event['source_frames']]
            rows = by_person.get(key, [])
            uid = f'new-{number:02d}_{index}'
            reason = None
            if key[:2] in excluded_tracks:
                reason = 'excluded_entity_all_generations'
            elif any(c['person'] == list(key) and times[-1] >= c['start'] and times[0] <= c['end'] for c in conflicts):
                reason = 'user_confirmed_exclude_conflicting_window'
            elif any(r['review_status'] == 'uncertain' and times[-1] >= r['start']-1 and times[0] <= r['end']+1 for r in rows):
                reason = 'uncertain_interval_guard'
            if reason:
                excluded.append(dict(id=uid, reason=reason)); continue
            votes = Counter()
            for t in times:
                labels = {r['human_label'] for r in rows if r['review_status'] == 'confirmed' and r['start']-1e-6 <= t <= r['end']+1e-6}
                assert len(labels) <= 1, (uid, labels)
                votes.update(labels or ['unknown'])
            label = None
            actions = [c for c in LABELS[:-1] if votes[c] >= 16]
            if len(actions) == 1 and votes['unknown'] <= 6.4:
                label = actions[0]
            elif votes['other'] >= 28.8 and not any(votes[c] for c in LABELS[:-1]):
                label = 'other'
            base = dict(id=uid, scope='new', source_id=f'new-{number:02d}', number=number,
                source=source['absolute_path'], source_sha256=source['sha256'], fps=fps,
                group=groups[number], split=new_split[number], track_id=key[1], generation=key[2],
                source_frames=event['source_frames'], start=times[0], end=times[-1], bbox=event['bbox'],
                event_index=index, raw_luma=event['raw_luma'], legacy_rgb=event['rgb_probs'],
                legacy_skeleton=event['skeleton_probs'], path=str(OUT/'data'/f'new-{number:02d}'/(uid+'.npz')))
            if label is not None:
                refs = [r['review_id'] for r in rows if r['review_status']=='confirmed' and r['end'] >= times[0] and r['start'] <= times[-1]]
                new.append(dict(base, label=label, label_kind='human', weight=1., review_ids=refs))
            elif (event['label']=='other' and event['rgb_probs'][9] >= .90 and event['detector_score'] >= .5
                  and any(r['review_status']=='confirmed' for r in rows)
                  and not any(times[-1] >= r['start']-1 and times[0] <= r['end']+1 for r in rows)):
                weak_candidates.append(dict(base, label='other', label_kind='pseudo', weight=.25, review_ids=[]))
    # Weak labels restricted to human-confirmed identities, temporally thinned and downweighted.
    recent, counts, per_source = {}, Counter(), Counter()
    weak = []
    for r in weak_candidates:
        key = (r['number'], r['track_id'], r['generation'])
        if r['start']-recent.get(key, -1e9) >= 2 and counts[key] < 8 and per_source[r['number']] < 200:
            weak.append(r); counts[key] += 1; per_source[r['number']] += 1; recent[key] = r['end']
    new.extend(weak)
    current = read(CURRENT/'evaluated-windows.json')
    videos = {v['id']: v for v in read(CURRENT/'protocol.json')['videos']}
    old23 = read(OLD/'training-data.json')['records']
    exact = {(r['video_id'], r['crop_sha256'], tuple(r['source_frames'])):r for r in old23}
    old, duplicates = [], []
    current_by_time = defaultdict(list)
    for i, r in enumerate(current):
        if r['truth'] is None: continue
        v = videos[r['video_id']]
        uid = 'old-'+r['video_id']+'-'+str(i)
        same = exact.get((r['video_id'], r['crop_sha256'], tuple(r['source_frames'])))
        path = str(OLD/same['path']) if same and same['label']==r['truth'] else str(OUT/'data'/r['video_id']/(uid+'.npz'))
        old.append(dict(id=uid, scope='old', source_id=r['video_id'], source=str(ROOT/r['source']),
            source_sha256=v['sha256'], split=r['split'], group=r['group'], fps=v['fps'],
            processed=v['processed'], label=r['truth'], label_kind='historical_review', weight=1.,
            track_id=r['track_id'], generation=0, source_frames=r['source_frames'], start=r['timestamps'][0], end=r['timestamp'],
            bbox=r['bbox'], raw_luma=r['raw_luma'], path=path, current_index=i, crop_sha256=r['crop_sha256'],
            legacy_rgb=r['rgb_probs'], legacy_skeleton=r['skeleton_probs'], review_ids=[]))
        current_by_time[(r['video_id'], tuple(r['source_frames']))].append(r)
    for r in old23:
        peers = current_by_time.get((r['video_id'], tuple(r['source_frames'])), [])
        paired = [x for x in peers if sum(overlap(a,b) >= .8 for a,b in zip(r['crop_boxes'],x['crop_boxes'])) >= 26]
        if paired:
            assert all(x['split']==r['split'] for x in paired)
            duplicates.append(dict(id=r['id'], reason='same_actor_same_frames_current_frontend_preferred', labels=[x['truth'] for x in paired]))
            continue
        v=videos[r['video_id']]
        old.append(dict(id='legacy-'+r['id'], scope='old', source_id=r['video_id'], source=str(ROOT/r['source']),
            source_sha256=v['sha256'], split=r['split'], group=r['group'], fps=v['fps'], processed=r['processed'],
            label=r['label'], label_kind='historical_review', weight=1., track_id=r['track_id'], generation=0,
            source_frames=r['source_frames'], start=r['start'], end=r['end'], raw_luma=None if r['processed'] else sum(r['lumas'])/32,
            path=str(OLD/r['path']), current_index=None, legacy_record_id=r['id'], crop_sha256=r['crop_sha256'], review_ids=[]))
    records = old + new
    split_groups = defaultdict(set)
    split_sources = defaultdict(set)
    for r in records:
        split_groups[r['group']].add(r['split']); split_sources[r['source_sha256']].add(r['split'])
    assert all(len(x)==1 for x in split_groups.values()) and all(len(x)==1 for x in split_sources.values())
    summary = {scope:{s:dict(n=sum(r['scope']==scope and r['split']==s for r in records),
        labels=dict(Counter(r['label'] for r in records if r['scope']==scope and r['split']==s)),
        kinds=dict(Counter(r['label_kind'] for r in records if r['scope']==scope and r['split']==s)))
        for s in ('train','val','test')} for scope in ('old','new')}
    protocol = dict(user_request='Fine-tune VideoMamba and ST-GCN with human labels, predicted other and all previous ROI source groups. No XCLIP.',
        inputs={str(p):sha(p) for p in (uploaded, REVIEW/'review-items.json', OLD/'training-data.json', CURRENT/'evaluated-windows.json', OLD/'splits.json')},
        models={name:dict(path=str(OP/'ckpts'/file),sha256=sha(OP/'ckpts'/file)) for name,file in
            [('rgb','videomamba_t_behavior_32f_person_roi_20261023.pth'),('skeleton','stgcn_person_20261024.pth'),
             ('detector','yolo26m_person_20261024.onnx'),('pose','rtmo-s_person_20261024.onnx'),('enhancer','NTIRE.pth'),('object','yolo26m_objects_20261026.onnx')]},
        new_splits=new_split, new_groups=groups, conflict_policy='User approved excluding every intersecting window; original JSON preserved',
        excluded_policy='Excluded entity: entire track across generations. Uncertain intervals: +/-1 second. Conflicts: every overlapping input window.',
        weak_policy='Only baseline other with RGB-other >=.9, detector>=.5, same identity confirmed human, outside all reviewed intervals +/-1s. At least2s gap, <=8/identity, <=200/source. Loss weight.25. Never evaluation truth.',
        old_policy='Keep frozen old source splits. Prefer current frontend at same-person exact source-frame windows (>=26/32 boxes IoU>=.8); preserve legacy-only windows and all sources. Report canonical and legacy separately.',
        training=dict(seed=29,rgb_epochs=6,rgb_batch=8,backbone_lr=5e-6,head_lr=5e-5,
            skeleton_epochs=20,skeleton_lr=5e-5,distillation_old=.5,distillation_temperature=2.,
            sampling='Per class 70% old / 30% new when both exist; equal source mass, per-window inverse count. Weak labels loss*.25. Old replay + old-teacher KL.',
            checkpoint_selection='Validation only: old positive macro recall >= epoch0-.05 and new positive macro recall >=epoch0-.10; minimize equal-domain human-other FPR, then maximize supported macro F1. If infeasible retain epoch0. Individual branches at action .75; final full scheme grid separately.'),
        tuning=dict(alpha=[.7,.8,.9,.95,.972,.99,1.],gate=GATES,action=[.7,.75,.8,.85,.875,.9,.925,.95,.975],object=[.1,.25,.35,.5,.581,.65,.75],wave_frames=3,running_speed=.35,
            objective='Validation only, no pseudo truth. Preserve old supported-class positive recall within 5pp and new positives within10pp of production baseline; minimize equal-domain other FPR, then macro F1, then positive TP.',
            geometry='Fixed matched ROI/action windows; frame-level luma masks applied to skeleton and priors. Full frontend replay of selected configuration required before deployment claim.'),
        historical_test_limit='Old initializers/history saw sources; all new9 already influenced earlier calibration. Group-isolated this round, not a new blind generalization set.',
        xclip_used=False, production_modified=False, summary=summary)
    save(OUT/'annotations.json',dict(source_sha256=sha(uploaded),annotations=annotations,conflicts=conflicts))
    save(OUT/'excluded-windows.json',excluded)
    save(OUT/'old-deduplication.json',duplicates)
    save(OUT/'dataset-plan.json',dict(records=records,summary=summary))
    save(OUT/'protocol.json',protocol)
    print(json.dumps(dict(summary=summary,conflicts=conflicts,deduplicated=len(duplicates)),ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
