"""Audit 0.85 drink/pick-up candidates against frozen per-window object crops."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'reports/manypeople-inference-20261025'
PRIOR=ROOT/'reports/object-prior-finetune-20261025'
OUT=ROOT/'reports/manypeople-interaction-audit-20261026'
sys.path.insert(0,str(PRIOR/'scripts'))
from evaluate_actions import decision

NAMES={0:'bottle',1:'cup',2:'phone',3:'camera',4:'paper',5:'portable_object',6:'robot_or_exhibit'}


def write_csv(path:Path,rows:list)->None:
    if not rows:return
    with path.open('w',newline='',encoding='utf-8-sig')as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main()->None:
    all_rows=[];source_counts=[];candidate_segments=[];selected=[];hashes=[]
    for number in range(1,10):
        path=BASE/'events'/f'{number:02d}_events.json';events=json.loads(path.read_text())
        frozen_path=PRIOR/'action-evaluation'/f'new-{number:02d}.json'
        frozen=json.loads(frozen_path.read_text()).get('records',[])
        # Previous object replay stored the selected 0.85 rows in source order and event index.
        by_event={r['event_index']:r for r in frozen}
        h=hashlib.sha256(path.read_bytes()).hexdigest();hashes.append(dict(path=str(path.relative_to(ROOT)),sha256=h))
        candidate_indices=[];counts=Counter();people=defaultdict(list)
        for idx,event in enumerate(events):
            pred=decision(event);label=pred['label']
            if label not in ('drink','pick_up'):continue
            rec=by_event.get(idx)
            assert rec and rec['track_id']==event['track_id'] and rec['history_generation']==event['history_generation']
            counts[label]+=1;candidate_indices.append(idx)
            conf=pred['confidence'];dets=[]
            for sample in rec['samples']:
                for box in sample['detections']:
                    if box[4]>=.35:
                        dets.append(dict(frame=sample['frame'],time=sample['timestamp'],class_id=int(box[5]),
                            class_name=NAMES[int(box[5])],confidence=float(box[4]),bbox=[float(x)for x in box[:4]],
                            binding_valid=sample['binding_valid']))
            compatible={0,1} if label=='drink' else {0,1,2,3,4,5}
            has_any=any(d['class_id']in compatible and d['binding_valid']for d in dets)
            compatible_rows=[d for d in dets if d['class_id']in compatible and d['binding_valid']]
            row=dict(source_number=number,event_index=idx,timestamp=float(event['timestamp']),
                frame_index=int(event['frame_index']),track_id=int(event['track_id']),
                history_generation=int(event['history_generation']),action=label,confidence=float(conf),
                bbox=[float(x)for x in event['bbox']],object_boxes=dets,
                single_roi_support=has_any,compatible_boxes=compatible_rows,
                conservative=rec['conservative'],strict=rec['strict'],
                held_clip_count=0,truth='unknown: no complete person-level annotations')
            all_rows.append(row);people[(row['track_id'],row['history_generation'])].append(row)
        for key,rows in people.items():
            rows.sort(key=lambda r:r['timestamp']);groups=[]
            for row in rows:
                if not groups or groups[-1][-1]['action']!=row['action'] or row['timestamp']-groups[-1][-1]['timestamp']>1.6:
                    groups.append([row])
                else:groups[-1].append(row)
            for gi,g in enumerate(groups):
                if len(g)<2 and g[0]['confidence']<.90:continue
                cls=Counter(b['class_name']for r in g for b in r['compatible_boxes'])
                segment=dict(source_number=number,track_id=key[0],history_generation=key[1],action=g[0]['action'],
                    start=g[0]['timestamp'],end=g[-1]['timestamp'],windows=len(g),
                    max_confidence=max(r['confidence']for r in g),
                    roi_supported_windows=sum(r['single_roi_support']for r in g),
                    object_classes=dict(cls),event_indices=[r['event_index']for r in g],
                    review_status='unreviewed candidate segment; inspect video before calling FP/TP')
                candidate_segments.append(segment)
                # Diverse review queue: strongest unsupported, supported, and pickup per source/action.
                buckets=defaultdict(list)
                for r in g:
                    buckets[(r['action'],r['single_roi_support'])].append(r)
                for (action,supported),candidates in buckets.items():
                    chosen=max(candidates,key=lambda r:r['confidence'])
                    selected.append((segment,chosen))
        source_counts.append(dict(source=number,events=len(events),drink=counts['drink'],pick_up=counts['pick_up'],
            candidate_segments=sum(s['source_number']==number for s in candidate_segments)))
    # Reconcile each sampled candidate with existing real saved clip metadata.
    clips=[]
    for f in (BASE/'jobs').glob('*/results/**/*.json'):
        try:m=json.loads(f.read_text())
        except (OSError,json.JSONDecodeError):continue
        if 'action_key' not in m or 'person_id' not in m:continue
        clips.append((m,f))
    for row in all_rows:
        for m,_ in clips:
            match=__import__('re').match(r'E(\d+)_',str(m.get('session','')))
            n=int(match.group(1))+1 if match else -1
            if n!=row['source_number'] or m['person_id']!=row['track_id'] or m['history_generation']!=row['history_generation']:continue
            if m['start_source_time']-1e-6<=row['timestamp']<=m['end_source_time']+1e-6 and m['action_key']==('drinking_water'if row['action']=='drink'else'pick_up_object'):
                row['held_clip_count']+=1
    out_rows=[]
    for seg,row in selected:
        out_rows.append(dict(source_number=row['source_number'],event_index=row['event_index'],timestamp=row['timestamp'],
            track_id=row['track_id'],action=row['action'],confidence=row['confidence'],single_roi_support=row['single_roi_support'],
            object_classes=';'.join(f"{k}:{v}"for k,v in Counter(b['class_name']for b in row['compatible_boxes']).items()),
            held_clip_count=row['held_clip_count'],bbox=json.dumps(row['bbox']),
            description='select representative from continuous candidate; video and crops added by render script'))
    OUT.mkdir(exist_ok=True,parents=True)
    for r in all_rows:
        for d in r['object_boxes']:d['bbox']=json.dumps(d['bbox'])
        r['object_boxes']=json.dumps(r['object_boxes'],ensure_ascii=False)
        r['compatible_boxes']=json.dumps(r['compatible_boxes'],ensure_ascii=False)
        r['bbox']=json.dumps(r['bbox'])
    write_csv(OUT/'all-0.85-windows.csv',all_rows);write_csv(OUT/'source-counts.csv',source_counts)
    write_csv(OUT/'candidate-segments.csv',candidate_segments);write_csv(OUT/'review-queue.csv',out_rows)
    (OUT/'selected-samples.json').write_text(json.dumps([dict(segment=s,event=r)for s,r in selected],ensure_ascii=False,indent=2)+'\n')
    summary=dict(threshold=.85,action_windows=len(all_rows),drink=sum(r['action']=='drink'for r in all_rows),
        pick_up=sum(r['action']=='pick_up'for r in all_rows),segments=len(candidate_segments),
        segments_with_object_support=sum(s['roi_supported_windows']>0for s in candidate_segments),
        windows_with_object_support=sum(r['single_roi_support']for r in all_rows),
        windows_without_object_support=sum(not r['single_roi_support']for r in all_rows),
        existing_related_clip_files=sum(r['held_clip_count']>0for r in all_rows),
        caveat='Predictions and detector boxes only. No complete person-level action truth. Five sampled frames per window; not dense video detection.',
        event_hashes=hashes,by_source=source_counts)
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
