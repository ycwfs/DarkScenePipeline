"""Collect completed historical offline predictions; never turn predictions into labels."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
REPORTS = ROOT / 'reports'
BASE = REPORTS / 'manypeople-inference-20261025'
NAMES = dict(wave='挥手', throw='投掷', chase='跑动', talk='交谈', fight='打架',
             fall='跌倒', drink='喝水', pick_up='捡物', shake_hands='握手', other='其他')
ALIASES = dict(waving='wave', throwing='throw', chasing='chase', running='chase',
               drinking_water='drink', picking_up_object='pick_up', falling='fall',
               throwing_object='throw', falling_down='fall')
REPORT_NAMES = ['object-overlay-20261027', 'object-hardneg-finetune-20261027',
                'object-coco80-finetune-20261027', 'wave-pose-prior-20261027',
                'running-speed-prior-20261027', 'xclip-person-20261027',
                'full-prior-release-20261027', 'box-single-gpu-20261028',
                'platform-recheck-20261029', 'person-object-release-20261026']


def read(path: Path):
    return json.loads(path.read_text())


def save(name: str, data) -> None:
    (OUT/name).write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def write_csv(name: str, rows: list[dict]) -> None:
    with (OUT/name).open('w', encoding='utf-8-sig', newline='') as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def iou(a, b) -> float:
    iw = max(0., min(a[2], b[2])-max(a[0], b[0]))
    ih = max(0., min(a[3], b[3])-max(a[1], b[1]))
    inter = iw*ih
    return inter/max(1e-8, (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)


def source_spec(d, inputs):
    """Resolve original source and trim offset, not filename-based track identity."""
    if not isinstance(d, dict):
        return None
    for key in ('source', 'input'):
        if isinstance(d.get(key), dict):
            value = source_spec(d[key], inputs)
            if value:
                return value
    for row in inputs:
        for key in ('source', 'original', 'absolute_path', 'path'):
            if isinstance(d.get(key), str) and Path(d[key]).name == Path(row['absolute_path']).name:
                offset = float(d.get('source_offset', d.get('start', 0)))
                return row['number'], offset
    if 'number' in d and d.get('number') in range(1, 10) and ('archive' in d or 'original_video_sha256' in d):
        return int(d['number']), float(d.get('source_offset', 0))
    return None


def main() -> None:
    inputs = read(BASE/'input-manifest.json')
    base = {s['number']: read(BASE/'events'/f"{s['number']:02d}_events.json") for s in inputs}
    fps = {s['number']:float(Fraction(str(s['fps']))) for s in inputs}
    # The common frontend downsamples 1920x1080 to 720x404 (even height).
    by_time = {}
    by_track = {}
    for n, rows in base.items():
        by_time[n] = defaultdict(list)
        by_track[n] = defaultdict(list)
        for i, e in enumerate(rows):
            by_time[n][round(e['timestamp']*25)].append((i, e))
            key = f"B{n:02d}:T{e['track_id']}:G{e.get('history_generation',0)}"
            by_track[n][key].append(e)
    observations, runs, ignored = [], [], []
    all_run_tracks = {}

    def ingest(path: Path, n: int, offset: float, original=False):
        if path.suffix == '.jsonl':
            with path.open() as f:
                rows = [json.loads(line) for line in f if line.strip()]
        else:
            rows = read(path)
        if not isinstance(rows, list):
            ignored.append(dict(path=str(path.relative_to(ROOT)), reason='not event array'))
            return
        run = str(path.relative_to(REPORTS))
        category = 'xclip' if 'xclip-person' in run else 'videomamba'
        tracks = defaultdict(list)
        count = 0
        for i, e in enumerate(rows):
            if 'bbox' not in e or 'timestamp' not in e:
                continue
            t = round(float(e['timestamp'])+offset, 6)
            local_key = f"T{e['track_id']}:G{e.get('history_generation',0)}"
            tracks[local_key].append(dict(t=t, box=e['bbox'], label=e['label']))
            if e['label'] == 'other':
                continue
            assert e['label'] in NAMES, (path, e['label'])
            count += 1
            identity = f"{run}:{local_key}"
            matched = None
            if original:
                matched = (i, e, 1.)
            else:
                matches = []
                for frame in range(round(t*25)-7, round(t*25)+8):
                    for j, b in by_time[n].get(frame, []):
                        score = iou(e['bbox'], b['bbox'])
                        if score >= .75:
                            matches.append((score, -abs(b['timestamp']-t), j, b))
                matches.sort(key=lambda v: (v[0], v[1]), reverse=True)
                if matches:
                    first = matches[0]
                    key = (first[3]['track_id'], first[3].get('history_generation',0))
                    competing = [x for x in matches if (x[3]['track_id'],x[3].get('history_generation',0)) != key]
                    if not competing or first[0]-competing[0][0] >= .10:
                        matched = (first[2], first[3], first[0])
            if matched:
                b = matched[1]
                identity = f"B{n:02d}:T{b['track_id']}:G{b.get('history_generation',0)}"
            observations.append(dict(id=f'{run}#{i}', run=run, category=category, number=n,
                identity=identity, local_track=local_key, timestamp=t,
                window_start=round(offset+min(e.get('source_frames',[max(0,e.get('frame_index',0)-24)]))/fps[n],6),
                label=e['label'], confidence=float(e.get('confidence',0)), box=e['bbox'],
                original=original, base_event=matched[0] if matched else None,
                match_iou=matched[2] if matched else None,
                identity_link='exact_baseline' if original else ('geometric_candidate' if matched else 'unmatched_run_track')))
        all_run_tracks[run] = tracks
        runs.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path), number=n,
                         offset=offset, category=category, events=len(rows), non_other=count))

    for n in base:
        ingest(BASE/'events'/f'{n:02d}_events.json', n, 0, True)
    for name in REPORT_NAMES:
        for parent, dirs, files in os.walk(REPORTS/name):
            dirs[:] = [d for d in dirs if d not in ('operator','release-extracted','ckpts','models','data','cache','transport','before','test-tools')]
            p = Path(parent)
            for filename in sorted(files):
                if filename not in ('events.json','events.jsonl','baseline-events.jsonl','candidate-events.jsonl'):
                    continue
                event_path = p/filename
                if any(x.startswith(('live','fault','diagnostic')) or x.endswith('smoke') for x in p.relative_to(REPORTS/name).parts):
                    ignored.append(dict(path=str(event_path.relative_to(ROOT)),reason='stream/fault/partial diagnostic; not source-aligned offline inference'))
                    continue
                summary = p/'summary.json'
                spec = source_spec(read(summary), inputs) if summary.exists() else None
                if spec is None:
                    ignored.append(dict(path=str(event_path.relative_to(ROOT)),reason='not one of the nine sources or no completed source mapping'))
                    continue
                ingest(event_path, *spec)
    # Saved selected threshold replays use exact original event indices.
    for name in ('threshold-tuning-20261026','four-parameter-tuning-20261026'):
        path=REPORTS/name/'window-results.csv'
        for column in ('baseline','recommended'):
            run=f'{name}/window-results.csv:{column}'
            count=total=0
            with path.open(encoding='utf-8-sig') as f:
                for row in csv.DictReader(f):
                    if not row['source_id'].startswith('new-'): continue
                    total+=1
                    if row[column]=='other': continue
                    n=int(row['source_id'].split('-')[1]); i=int(row['id'].rsplit('_',1)[1]); e=base[n][i]
                    assert abs(float(row['timestamp'])-e['timestamp'])<1e-5 and int(row['track_id'])==e['track_id']
                    runbase=f'manypeople-inference-20261025/events/{n:02d}_events.json'
                    observations.append(dict(id=f'{run}#{row["id"]}',run=run, category='videomamba',number=n,
                        identity=f"B{n:02d}:T{e['track_id']}:G{e.get('history_generation',0)}",
                        local_track=f"T{e['track_id']}:G{e.get('history_generation',0)}",
                        timestamp=e['timestamp'],window_start=min(e['source_frames'])/fps[n],label=row[column],
                        confidence=float(row.get('top_score',row.get('score',e['confidence']))), box=e['bbox'],
                        original=False, base_event=i,match_iou=1.,identity_link='exact_cached_window',track_run=runbase))
                    count+=1
            runs.append(dict(path=str(path.relative_to(ROOT)),column=column,sha256=sha(path),number='all9',
                offset=0, category='videomamba', events=total,non_other=count))
    buckets=defaultdict(list)
    for e in observations:
        # Keep XCLIP supplemental review distinct from the production-family queue.
        buckets[(e['category'],e['number'],e['identity'],e['label'])].append(e)
    groups=[]
    for key, rows in buckets.items():
        rows.sort(key=lambda x:(x['timestamp'],not x['original'],x['id']))
        parts=[]
        for e in rows:
            if not parts or e['timestamp']-parts[-1][-1]['timestamp']>1.2+1e-6 or e['timestamp']-parts[-1][0]['timestamp']>10.0+1e-6:
                parts.append([])
            parts[-1].append(e)
        for part in parts:
            selected=min(part,key=lambda e:(not e['original'],e['run'],e['timestamp']))
            rep=sorted([e for e in part if e['run']==selected['run']],key=lambda e:e['timestamp'])
            trace=all_run_tracks[selected.get('track_run',selected['run'])][selected['local_track']]
            start=max(0.,min(e['window_start'] for e in part)-2)
            end=min(inputs[key[1]-1]['duration'],max(e['timestamp'] for e in part)+2)
            groups.append(dict(category=key[0],number=key[1],identity=key[2],label=key[3],label_zh=NAMES[key[3]],
                trigger_start=min(e['timestamp'] for e in part),trigger_end=max(e['timestamp'] for e in part),
                proposed_action_start=min(e['window_start'] for e in part),proposed_action_end=max(e['timestamp'] for e in part),
                preview_start=round(start,4),preview_end=round(end,4),confidence=max(e['confidence'] for e in part),
                provenance=sorted(set(e['run'] for e in part)),observation_ids=[e['id'] for e in part],
                baseline_events=sorted(set(e['base_event'] for e in part if e['base_event'] is not None)),
                has_initial_trigger=any(e['original'] for e in part),
                approximate_identity_link=any(e['identity_link']=='geometric_candidate' for e in part),
                representative_run=selected.get('track_run',selected['run']),representative_track=selected['local_track'],
                observations=[dict(t=e['timestamp'],box=e['box'],confidence=e['confidence']) for e in rep],
                trace=[e for e in trace if start-.6<=e['t']<=end+.6],original_clips=[],
                review_status='pending',human_label=None,training_eligible=False))
    groups.sort(key=lambda g:(g['category']=='xclip',g['number'],g['trigger_start'],g['identity'],g['label']))
    counters=Counter()
    for g in groups:
        prefix=('X' if g['category']=='xclip' else 'M')+f'{g["number"]:02d}'
        counters[prefix]+=1; g['id']=f'{prefix}-{counters[prefix]:03d}'
        g['video']=f'clips/{g["id"]}.mp4';g['poster']=f'posters/{g["id"]}.jpg'
    # Keep all original 454 saved files, even when their 15s continuation splits are merged.
    clips=[]
    (OUT/'original-clips').mkdir(exist_ok=True)
    verification=read(BASE/'verification.json')
    for item in verification['items']:
        n=item['number']
        for c in item['clips']:
            m=c['metadata']; label=ALIASES.get(m['label_key'],m['label_key'])
            identity=f"B{n:02d}:T{m['person_id']}:G{m['history_generation']}"
            matches=[g for g in groups if g['category']=='videomamba' and g['number']==n and g['identity']==identity
                     and g['label']==label and g['trigger_start']<=m['end_source_time']+.01 and g['trigger_end']>=m['start_source_time']-.01]
            assert matches,(n,c['path'])
            uid=f'C{n:02d}-{len([v for v in clips if v["number"]==n])+1:03d}'
            dest=OUT/'original-clips'/f'{uid}.mp4'
            if not dest.exists():os.link(c['path'],dest)
            row=dict(id=uid,number=n,label=label,label_zh=NAMES[label],identity=identity,
                     start=m['start_source_time'],end=m['end_source_time'],video=str(dest.relative_to(OUT)),
                     original_path=c['path'],review_ids=[g['id'] for g in matches],context_complete=m['context_complete'])
            clips.append(row)
            for g in matches:g['original_clips'].append(row['video'])
    assert len(clips)==454
    # A prediction is not GT. Other pool deliberately excludes all candidate intervals,
    # conservatively across people, including XCLIP-only proposals and a 1s margin.
    other=[];held=[]
    with (OUT/'base-windows.jsonl').open('w') as f:
        for n, rows in base.items():
            windows=[g for g in groups if g['number']==n]
            for i,e in enumerate(rows):
                uid=f'new-{n:02d}_{i}'
                overlaps=[g['id'] for g in windows if e['timestamp']>=g['proposed_action_start']-1 and min(e['source_frames'])/fps[n]<=g['proposed_action_end']+1]
                record=dict(id=uid,number=n,source_sha256=inputs[n-1]['sha256'],track_id=e['track_id'],
                    generation=e.get('history_generation',0),timestamp=e['timestamp'],source_frames=e['source_frames'],
                    bbox=e['bbox'],prediction=e['label'],raw_luma=e['raw_luma'],human_label=None,
                    training_eligible=False,review_ids=overlaps,
                    status='other_candidate_unreviewed' if e['label']=='other' and not overlaps else 'held_for_review')
                f.write(json.dumps(record,ensure_ascii=False)+'\n')
                compact={k:record[k] for k in ('id','number','track_id','generation','timestamp','prediction','status','training_eligible')}
                (other if record['status']=='other_candidate_unreviewed' else held).append(compact)
    save('sources.json',inputs);save('runs.json',runs);save('excluded-runs.json',ignored)
    save('observations.json',observations);save('review-items.json',groups);save('original-clips.json',clips)
    write_csv('original-clips.csv',[{**c,'review_ids':'|'.join(c['review_ids'])} for c in clips])
    write_csv('other-candidates.csv',other)
    write_csv('review.csv',[dict(review_id=g['id'],source_number=g['number'],prediction=g['label'],
        prediction_zh=g['label_zh'],identity=g['identity'],action_start=g['proposed_action_start'],
        action_end=g['proposed_action_end'],preview_start=g['preview_start'],preview_end=g['preview_end'],
        confidence=round(g['confidence'],6),scope=g['category'],video=g['video'],review_status='pending',
        human_label='',human_start='',human_end='',entity_type='',note='') for g in groups])
    summary=dict(sources=9,source_minutes=sum(s['duration'] for s in inputs)/60,
        historical_runs=len(runs),observations=len(observations),original_clips=len(clips),
        initial_events=47600,initial_non_other=sum(e['label']!='other' for rows in base.values() for e in rows),
        main_review=sum(g['category']=='videomamba' for g in groups),xclip_review=sum(g['category']=='xclip' for g in groups),
        main_with_initial_trigger=sum(g['category']=='videomamba' and g['has_initial_trigger'] for g in groups),
        by_label=dict(Counter(g['label_zh'] for g in groups if g['category']=='videomamba')),
        by_source={f'{n:02d}':dict(Counter(g['label_zh'] for g in groups if g['category']=='videomamba' and g['number']==n)) for n in base},
        other_candidates=len(other),held_windows=len(held),human_annotations=0,training_started=False,
        grouping='same source + conservative person association + same label; gap<=1.2s; trigger span<=10s',
        other_policy='unreviewed prediction pool only; exclude all-person candidate time intervals plus 1s; not GT',
        exclusions='No millions of unselected grid configurations, no interrupted diagnostics or concatenated live/fault reruns; see excluded-runs.json')
    save('summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
