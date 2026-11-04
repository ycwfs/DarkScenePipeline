"""Freeze audited window labels without converting unknown periods into negatives."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parents[1]
LABELS=['wave','throw','chase','fall','fight','talk','drink','pick_up','shake_hands','other']


def assign_label(timestamps: list, segments: list) -> tuple:
    counts=Counter()
    for timestamp in timestamps:
        values={s['label']for s in segments if s['start']-1e-6<=timestamp<=s['end']+1e-6}
        if len(values)>1:
            raise ValueError(f'Conflicting annotations at {timestamp}: {values}')
        counts.update(values or ['unknown'])
    n=len(timestamps)
    actions=[label for label in LABELS[:-1]if counts[label]>=n*.5]
    if len(actions)==1 and counts['unknown']<=n*.2:
        return actions[0],dict(counts)
    if counts['other']>=n*.9 and not any(counts[label]for label in LABELS[:-1]):
        return 'other',dict(counts)
    return None,dict(counts)


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--partial',action='store_true')
    parser.add_argument('--verify-pixels',action='store_true')
    parser.add_argument('--reuse-pixel-audit',type=Path)
    args=parser.parse_args()
    inventory=json.loads((OUT/'source-inventory.json').read_text())['videos']
    splits=json.loads((OUT/'splits.json').read_text())['records']
    protocol=json.loads((OUT/'protocol.json').read_text())
    prior_audit=None
    if args.reuse_pixel_audit:
        prior_audit=json.loads(args.reuse_pixel_audit.read_text())
        assert prior_audit['ready_for_training'] and prior_audit['audit']['pixel_hashes_verified']
        assert {r['video_id']for r in prior_audit['video_coverage']}=={r['id']for r in inventory}
        prior_audit_time=args.reuse_pixel_audit.stat().st_mtime_ns
    assert hashlib.sha256((OUT/'splits.json').read_bytes()).hexdigest()==protocol['split_sha256']
    by_id={row['video_id']:row for row in splits}
    assert len(by_id)==len(inventory)==535
    grouped=defaultdict(set)
    for row in splits:grouped[row['group']].add(row['split'])
    assert all(len(values)==1 for values in grouped.values())
    annotations=json.loads((OUT/'annotations.json').read_text())['records']
    annotated={(r['video_id'],r['track_id']):r for r in annotations}
    assert len(annotated)==len(annotations)
    for r in annotations:
        for s in r['segments']:
            assert s['label'] in LABELS and 0<=s['start']<s['end'],r
    records,excluded,coverage,missing=[],[],[],[]
    for item in inventory:
        path=OUT/'data'/item['id']/'manifest.json'
        if not path.exists():missing.append(item['source']);continue
        m=json.loads(path.read_text()); split=by_id[item['id']]
        if prior_audit is not None:
            assert path.stat().st_mtime_ns<=prior_audit_time,'Cache manifest changed after original audit'
        assert m['full_decode'] and len(m['frames'])==item['frames']
        assert m['video']['sha256']==item['sha256']
        assert hashlib.sha256((ROOT/item['source']).read_bytes()).hexdigest()==item['sha256']
        accepted=[]
        for w in m['windows']:
            ts=w['timestamps']; frames=w['source_frames']
            assert len(ts)==len(frames)==len(w['crop_boxes'])==len(w['lumas'])==32
            assert ts==sorted(ts) and frames==sorted(frames)
            assert abs(ts[-1]-w['end'])<1e-6 and abs(ts[0]-w['start'])<1e-6
            assert all(abs(t-f/item['fps'])<1e-6 for t,f in zip(ts,frames))
            assert all(any(t['track_id']==w['track_id']for t in m['frames'][f]['tracks'])for f in frames)
            assert w['skeleton_probs'] is None or (len(w['skeleton_probs'])==10 and abs(sum(w['skeleton_probs'])-1)<1e-5)
            reuse=prior_audit is not None and (OUT/w['path']).stat().st_mtime_ns<=prior_audit_time
            if args.verify_pixels and not reuse:
                with np.load(OUT/w['path'])as a:
                    rgb=a['rgb']
                    assert rgb.shape==(32,224,400,3) and rgb.dtype==np.uint8
                    assert a['pose'].shape==(32,17,3)
                    assert hashlib.sha256(rgb.tobytes()).hexdigest()==w['crop_sha256']
            annotation=annotated.get((item['id'],w['track_id']))
            counts={}
            if annotation:
                if annotation['status']!='reviewed':
                    label=None; reason=annotation['note']
                else:
                    label,counts=assign_label(ts,annotation['segments'])
                    reason='temporal transition or insufficient reviewed coverage'
                provenance=annotation.get('reviewer','assistant visual person/time review')
            elif item['video_label']=='other':
                label='other'; reason=''; provenance=item['label_provenance']
            else:
                label=None; reason='person track not reviewed'; provenance=None
            if label is None:
                excluded.append(dict(id=w['id'],video_id=item['id'],source=item['source'],
                    track_id=w['track_id'],start=w['start'],end=w['end'],reason=reason,coverage=counts))
                continue
            row={**w,'label':label,'person_label':label,'split':split['split'],'group':split['group'],
                 'source':item['source'],'processed':item['processed'],
                 'sealed_feedback_holdout':split['sealed_feedback_holdout'],
                 'annotation_provenance':provenance,'annotation_coverage':counts,
                 'entity_type':annotation.get('entity_type','person')if annotation else 'unreviewed_detection'}
            records.append(row);accepted.append(row)
        coverage.append(dict(video_id=item['id'],source=item['source'],video_label=item['video_label'],
            split=split['split'],group=split['group'],frames=item['frames'],
            frames_with_detection=sum(bool(f['tracks'])for f in m['frames']),
            windows=len(m['windows']),accepted=len(accepted),
            accepted_actions=sum(r['label']!='other'for r in accepted)))
    assert len({r['id']for r in records})==len(records)
    summary={s:dict(windows=sum(r['split']==s for r in records),
                   videos=len({r['video_id']for r in records if r['split']==s}),
                   groups=len({r['group']for r in records if r['split']==s}),
                   labels=dict(Counter(r['label']for r in records if r['split']==s)))for s in ['train','val','test']}
    result=dict(ready_for_training=not missing and args.verify_pixels,records=records,summary=summary,
        excluded_windows=excluded,video_coverage=coverage,missing=missing,
        audit=dict(pixel_hashes_verified=args.verify_pixels,
                   source_hashes_verified=True,no_cross_split_groups=True,positive_person_tracks_reviewed=True,
                   annotations_sha256=hashlib.sha256((OUT/'annotations.json').read_bytes()).hexdigest(),
                   splits_sha256=hashlib.sha256((OUT/'splits.json').read_bytes()).hexdigest(),
                   protocol_sha256=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()),
        limitations=['Other labels inherited from source unless explicitly visually reviewed.',
                    'No person detection means no classification window, not a true negative.',
                    'Historical videos were mostly seen by the pn2 initialization.'])
    if prior_audit is not None:
        result['audit']['reused_pixel_audit']=dict(path=str(args.reuse_pixel_audit),
            sha256=hashlib.sha256(args.reuse_pixel_audit.read_bytes()).hexdigest(),
            basis='Annotation-only amendment; cache manifests and tensors predate the successful original audit. Modified tensors rehashed.')
    print(json.dumps(dict(summary=summary,missing=len(missing),excluded=len(excluded),coverage=len(coverage)),ensure_ascii=False,indent=2),flush=True)
    if not args.partial:
        assert not missing,'Preprocessing incomplete'
        assert args.verify_pixels,'Final dataset requires pixel audit'
        assert not any(r['reason']=='person track not reviewed'for r in excluded)
        assert not (OUT/'training-history.json').exists(),'Do not change data after training'
        (OUT/'training-data.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        unresolved=[r for r in annotations if r['status']!='reviewed']
        (OUT/'annotation-uncertainties.json').write_text(json.dumps(unresolved,ensure_ascii=False,indent=2)+'\n')
    else:
        (OUT/'assembly-progress.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
