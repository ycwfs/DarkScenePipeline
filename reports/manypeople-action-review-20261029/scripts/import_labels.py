"""Validate human review JSON/CSV and stage labels; never silently label the remainder."""
import argparse
import csv
import json
from pathlib import Path

from collect import OUT, NAMES, read, sha


def validate(path: Path) -> list[dict]:
    items={g['id']:g for g in read(OUT/'review-items.json')}
    sources={s['number']:s for s in read(OUT/'sources.json')}
    if path.suffix.lower()=='.csv':
        with path.open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    else:
        payload=read(path)
        if payload.get('schema_version')!=1 or payload.get('manifest_sha256')!=sha(OUT/'review-items.json'):
            raise ValueError('Review manifest/version mismatch')
        rows=payload['annotations']
    results=[];seen=set()
    for row in rows:
        if row.get('review_status','pending') in ('','pending'):continue
        uid=row.get('review_id');status=row.get('review_status')
        if uid not in items or uid in seen:raise ValueError(f'Unknown/duplicate review ID: {uid}')
        seen.add(uid);g=items[uid]
        if status not in ('confirmed','exclude','uncertain'):raise ValueError(f'{uid}: invalid status')
        label=row.get('human_label')
        if label in NAMES.values():label=next(k for k,v in NAMES.items()if v==label)
        if status=='confirmed' and label not in NAMES:raise ValueError(f'{uid}: invalid human label')
        if status!='confirmed':label=None
        start=float(g['proposed_action_start'] if row.get('human_start') in (None,'') else row['human_start'])
        end=float(g['proposed_action_end'] if row.get('human_end') in (None,'') else row['human_end'])
        if not (0<=start<end<=sources[g['number']]['duration']+.001):raise ValueError(f'{uid}: invalid source time interval')
        entity=row.get('entity_type') or ('human' if status=='confirmed' else 'unknown')
        if status=='confirmed' and entity!='human':raise ValueError(f'{uid}: confirmed action must target an actual human')
        results.append(dict(review_id=uid,review_status=status,human_label=label,human_start=start,human_end=end,
            entity_type=entity,note=row.get('note',''),source_number=g['number'],source=sources[g['number']]['absolute_path'],
            source_sha256=sources[g['number']]['sha256'],identity=g['identity'],category=g['category'],
            original_review_prediction=g['label'],requires_person_trace_verification=not g['identity'].startswith('B')))
    conflicts=[]
    for i,a in enumerate(results):
        if a['review_status']!='confirmed':continue
        for b in results[i+1:]:
            if b['review_status']!='confirmed' or a['source_number']!=b['source_number'] or a['identity']!=b['identity']:continue
            if a['human_label']!=b['human_label'] and min(a['human_end'],b['human_end'])>max(a['human_start'],b['human_start']):
                conflicts.append([a['review_id'],b['review_id']])
    if conflicts:raise ValueError(f'Conflicting labels on same person/time; resolve before import: {conflicts[:20]}')
    return results


def main():
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    labels=validate(a.input)
    if a.output.exists():raise FileExistsError('Choose a new output file; existing annotations are preserved')
    a.output.write_text(json.dumps(dict(schema_version=1,manifest_sha256=sha(OUT/'review-items.json'),
        annotation_source=str(a.input),annotation_source_sha256=sha(a.input),annotations=labels,
        training_ready=False,remaining_other_auto_labelled=False,
        next_step='Resolve review, verify person traces/entities, extract matched raw RGB/skeleton inputs and assign recording-group splits.'),ensure_ascii=False,indent=2)+'\n')
    print(f'Validated {len(labels)} human review records; remainder kept unlabelled')


if __name__=='__main__':main()
