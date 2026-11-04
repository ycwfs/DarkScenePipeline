"""Exact stored same-track pose geometry; historical full action probabilities unchanged."""
from collections import Counter, defaultdict
import copy
import json
from types import SimpleNamespace

import numpy as np

from common import OUT, ROOT, save, digest
from wave_prior import apply_wave_prior
from darkpipe.person.core import fuse_person


def shape(v):
    h,w=v['height'],v['width']
    if v['processed']:
        possibilities=[x for x in range(h-180,h)if x%2==0 and x+max(48,round(.08*x))+(max(48,round(.08*x))&1)==h]
        assert len(possibilities)==1
        h=possibilities[0]
    if max(h,w)>720:
        scale=720/max(h,w)
        w,h=int(round(w*scale))//2*2,int(round(h*scale))//2*2
    return h,w


def main():
    prior=ROOT/'reports/four-parameter-tuning-20261026'
    rows=json.loads((ROOT/'reports/threshold-tuning-20261026/windows.json').read_text())
    saved=json.loads((prior/'predictions.json').read_text())
    assert [r['id']for r in rows]==saved['ids']
    pred=[saved['labels'][i]for i in saved['predictions']['recommended']]
    inputs=np.load(prior/'inputs.npz');gi=inputs['gates'].tolist().index(35)
    sources=json.loads((OUT/'protocol.json').read_text())['old_sources']
    lookup={v['id']:v for v in sources if v['scope']!='new_multipeople'}
    groups=defaultdict(list)
    for i,row in enumerate(rows):
        if row['source_id']in lookup:groups[row['source_id']].append(i)
    # Select real positive demo sources using prior truth and baseline only, before new-rule scores.
    demos=[];selected=set()
    positives={r['source_id']for r,p in zip(rows,pred)if r['truth']=='wave'and p=='wave'and r['source_id']in lookup}
    for split in ['val','test','train']:
        choices=sorted([lookup[k]for k in positives if lookup[k]['split']==split],key=lambda v:v['source'])
        for v in choices[:2]:
            selected.add(v['id']);demos.append(v)
    for v in sorted(lookup.values(),key=lambda v:v['source']):
        if v['source'].endswith('/挥手3.mp4')and v['id']not in selected:
            selected.add(v['id']);demos.append(v)
    save(OUT/'old-demo-selection.json',dict(basis='Frozen prior person/time truth and baseline wave predictions; not selected on new-rule outcomes.',sources=demos))
    records=[];stats=Counter()
    for sid,indices in groups.items():
        v=lookup[sid];cache=None
        affected=[i for i in indices if pred[i]=='wave']
        if affected:
            cache=json.loads((ROOT/v['cache']).read_text())
            assert isinstance(cache,dict)
            replay_file=prior/'replay'/f'{sid}.json'
            replay=json.loads(replay_file.read_text())if replay_file.exists()else None
            pose_file=ROOT/'reports/luma-gate15-20261024/data'/sid/'result.json'
            frame_data=json.loads(pose_file.read_text())['frames']if replay else cache['frames']
            tracks=replay['tracks']['35']if replay else [f['tracks']['30']for f in cache['frames']]
            events=replay['events']['35']if replay else cache['events']['30']
            h,w=shape(v)
        for i in indices:
            original=rows[i]
            result=dict(id=original['id'],source_id=sid,source=v['source'],scope=v['scope'],
                split=v['split'],group=v['group'],truth=original['truth'],baseline=pred[i],
                timestamp=original['timestamp'],track_id=original['track_id'],raw_luma=original['raw_luma'])
            result['variants']={f'{k}_{mode}':pred[i]for k in [1,3,5,8]for mode in ['total','consecutive']}
            if pred[i]=='wave':
                e=events[original['event_index']]
                assert e['source_frames']==original['source_frames']
                sk=inputs['skeleton'][gi,i]if inputs['pose_valid'][gi,i]else None
                row=fuse_person(inputs['rgb'][i],sk,[original['raw_luma']],alpha=.972,threshold=.925,lum_gate=35.)
                assert row['label']=='wave'
                samples=[]
                for fi in e['source_frames']:
                    meta=frame_data[fi]
                    actors=[t for t in tracks[fi]if t['track_id']==e['track_id']]
                    assert len(actors)==1,(sid,fi,e['track_id'])
                    found=[j for j,b in enumerate(meta['boxes'])if np.allclose(b[:4],actors[0]['bbox'],rtol=0,atol=1e-4)]
                    assert len(found)==1,(sid,fi,'exact detector/track binding')
                    pose=meta['poses'][found[0]]
                    points=np.zeros((17,3),np.float32)if pose is None else np.array(pose,np.float32)
                    points[:,:2]/=[w,h]
                    if meta['luma']is not None and meta['luma']<35:points[:]=0
                    samples.append(SimpleNamespace(frame_index=fi,timestamp=meta['timestamp'],pose=points,luma=meta['luma']))
                task=SimpleNamespace(window=samples)
                for k in [1,3,5,8]:
                    for mode in ['total','consecutive']:
                        alternative=copy.deepcopy(row)
                        apply_wave_prior(task,alternative,k,35.,consecutive=mode=='consecutive')
                        result['variants'][f'{k}_{mode}']=alternative['label']
                        if k==3 and mode=='total':result['evidence']=alternative['wave_prior']
                stats['exactly_bound_wave_windows']+=1
                stats['source_samples_checked']+=len(samples)
            records.append(result)
    summary={}
    variants=['baseline']+list(records[0]['variants'])
    for scope in ['onsite','online_supplement']:
        summary[scope]={}
        for split in ['val','test','train','all']:
            rr=[r for r in records if r['scope']==scope and (split=='all'or r['split']==split)]
            table={}
            for setting in variants:
                value=lambda r:r['baseline']if setting=='baseline'else r['variants'][setting]
                table[setting]=dict(windows=len(rr),true_wave_total=sum(r['truth']=='wave'for r in rr),
                    true_wave_retained=sum(r['truth']=='wave'and value(r)=='wave'for r in rr),
                    false_wave=sum(r['truth']is not None and r['truth']!='wave'and value(r)=='wave'for r in rr),
                    unknown_wave=sum(r['truth']is None and value(r)=='wave'for r in rr),
                    filtered_true_wave=sum(r['truth']=='wave'and r['baseline']=='wave'and value(r)!='wave'for r in rr))
            summary[scope][split]=table
    save(OUT/'old-window-results.json',records)
    save(OUT/'old-window-summary.json',dict(summary=summary,stats=dict(stats),
        source_probabilities_sha256=digest(prior/'inputs.npz'),
        caveat='Historical cached exact actor windows and current frozen probabilities; no new action forward, no new independent generalization test. Only preexisting wave outputs may change.'))
    print(json.dumps(dict(stats=dict(stats),onsite_val=summary['onsite']['val'],onsite_test=summary['onsite']['test']),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
