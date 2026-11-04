"""Source-group paired evidence; unknown actors never become background truth."""
from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.stats import binomtest

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(OUT/'scripts'))
from evaluate import save, sha, BEHAVIORS, assign_label
sys.path.insert(0,str(ROOT/'reports/person-finetune-20261023/scripts'))
from train import metrics


def bucket(luma: float|None) -> str:
    if luma is None:return 'raw_unknown'
    for hi,name in [(15,'0-15'),(30,'15-30'),(45,'30-45'),(60,'45-60'),(75,'60-75')]:
        if luma<hi:return name
    return '75+'


def score(rows: list,key: str) -> dict:
    known=[r for r in rows if r['truth']is not None]
    m=metrics([BEHAVIORS.index(r['truth'])for r in known],[BEHAVIORS.index(r[key])for r in known])
    m.update(positive_count=sum(r['truth']!='other'for r in known),
             positive_correct=sum(r['truth']!='other'and r[key]==r['truth']for r in known))
    return m


def cluster_stats(rows: list) -> dict:
    groups=defaultdict(list)
    for r in rows:
        if r['truth']is not None:groups[r['group']].append([r['before']==r['truth'],r['after']==r['truth']])
    if not groups:return dict(groups=0)
    pairs=np.array([np.mean(groups[g],axis=0)for g in sorted(groups)])
    delta=pairs[:,1]-pairs[:,0];rng=np.random.default_rng(2415)
    boot=delta[rng.integers(len(delta),size=(5000,len(delta)))].mean(1)
    wins=int((delta>1e-12).sum());losses=int((delta< -1e-12).sum())
    return dict(groups=len(groups),before_mean=float(pairs[:,0].mean()),after_mean=float(pairs[:,1].mean()),
        before_sd=float(pairs[:,0].std(ddof=1))if len(pairs)>1 else None,
        after_sd=float(pairs[:,1].std(ddof=1))if len(pairs)>1 else None,
        difference=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),
        wins=wins,losses=losses,ties=len(delta)-wins-losses,
        sign_p=float(binomtest(wins,wins+losses,.5).pvalue)if wins+losses else 1.,
        per_group=[dict(group=g,before=float(a),after=float(b),difference=float(b-a))for g,(a,b)in zip(sorted(groups),pairs)])


def main() -> None:
    protocol=json.loads((OUT/'protocol.json').read_text());fingerprint=sha(OUT/'protocol.json')
    reviews={}
    if (OUT/'label-review.json').exists():
        reviews={(r['video_id'],r['track_id']):r for r in json.loads((OUT/'label-review.json').read_text())['records']}
    paired=[];allrows=[];coverage=[];invariants=[];missing=[]
    for v in protocol['videos']:
        path=OUT/'data'/v['id']/'result.json'
        if not path.exists():missing.append(v['source']);continue
        d=json.loads(path.read_text());assert d['protocol_sha256']==fingerprint
        assert len(d['frames'])==v['frames']
        maps={}
        for gate in ['75','15']:
            bykey={}
            for e in d['events'][gate]:
                key=(tuple(e['source_frames']),e['crop_sha256'])
                assert key not in bykey,(v['source'],gate,key)
                assert np.isfinite(e['rgb_probs']).all()and abs(sum(e['rgb_probs'])-1)<.003
                assert len(e['source_frames'])==32
                e.update(video_id=v['id'],source=v['source'],group=v['group'],split=v['split'],brightness=bucket(e['raw_luma']))
                if gate=='15' and (v['id'],e['track_id'])in reviews:
                    r=reviews[(v['id'],e['track_id'])]
                    if r['status']=='reviewed':
                        label,counts=assign_label(e['timestamps'],r['segments'])
                        e.update(truth=label,truth_provenance='additional prediction-free visual track/time review',annotation_coverage=counts)
                bykey[key]=e
            maps[gate]=bykey
        common=set(maps['75'])&set(maps['15'])
        for key in sorted(common):
            a,b=maps['75'][key],maps['15'][key]
            if b['truth_provenance']=='additional prediction-free visual track/time review':
                a.update(truth=b['truth'],truth_provenance=b['truth_provenance'])
            assert a['truth']==b['truth']
            assert a['rgb_probs']==b['rgb_probs']
            assert a['raw_luma']==b['raw_luma']
            paired.append(dict(video_id=v['id'],source=v['source'],split=v['split'],group=v['group'],
                frame=a['frame_index'],track75=a['track_id'],track15=b['track_id'],crop_sha256=a['crop_sha256'],
                source_frames=a['source_frames'],truth=a['truth'],truth_provenance=a['truth_provenance'],
                raw_luma=a['raw_luma'],brightness=bucket(a['raw_luma']),before=a['label'],after=b['label'],
                mode75=a['mode'],mode15=b['mode'],confidence75=a['confidence'],confidence15=b['confidence'],
                rgb_label=a['rgb_label'],pose_valid15=b['pose_valid'],skeleton_probs15=b['skeleton_probs']))
        for gate in ['75','15']:
            allrows.extend(maps[gate].values())
            coverage.append(dict(video_id=v['id'],source=v['source'],split=v['split'],gate=int(gate),
                frames=len(d['frames']),frames_with_box=sum(bool(f['tracks'][gate])for f in d['frames']),
                windows=len(maps[gate]),known=sum(r['truth']is not None for r in maps[gate].values()),
                unknown=sum(r['truth']is None for r in maps[gate].values()),unpaired=len(set(maps[gate])-common),
                raw_mean_luma=v['raw_mean_luma']))
        ls=[f['luma']for f in d['frames']if f['luma']is not None]
        if not ls or max(ls)<15 or min(ls)>=75:
            assert set(maps['75'])==set(maps['15']),v['source']
            assert all(maps['75'][k]['fused_probs']==maps['15'][k]['fused_probs']for k in common),v['source']
            invariants.append(v['source'])
    subsets={}
    for split in ['val','test','train']:
        for band in ['all','15-75','0-15','15-30','30-45','45-60','60-75','75+','raw_unknown']:
            rows=[r for r in paired if r['split']==split and (band=='all'or r['brightness']==band or
                (band=='15-75'and r['raw_luma']is not None and 15<=r['raw_luma']<75))]
            subsets[split+'/'+band]=dict(paired=len(rows),unknown=sum(r['truth']is None for r in rows),
                gate75=score(rows,'before'),gate15=score(rows,'after'),
                modes15=dict(Counter(r['mode15']for r in rows)),
                changed=sum(r['before']!=r['after']for r in rows),
                fixed=sum(r['truth']is not None and r['before']!=r['truth']and r['after']==r['truth']for r in rows),
                broken=sum(r['truth']is not None and r['before']==r['truth']and r['after']!=r['truth']for r in rows),
                stats=cluster_stats(rows))
    # Two predeclared held-out split accuracy contrasts; exploratory bands have no significance claim.
    tests=[subsets[s+'/all']['stats']for s in ['val','test']if subsets[s+'/all']['stats']['groups']]
    previous=0.
    for i,s in enumerate(sorted(tests,key=lambda s:s['sign_p'])):
        previous=max(previous,min(1.,s['sign_p']*(len(tests)-i)));s['holm_p']=previous
    all_scores={split:{str(g):score([r for r in allrows if r['split']==split and r['gate']==g],'label')
        for g in (75,15)}for split in ('val','test','train')}
    result=dict(complete=not missing,protocol_sha256=fingerprint,
        additional_labels_sha256=sha(OUT/'label-review.json')if (OUT/'label-review.json').exists()else None,
        completed=len(protocol['videos'])-len(missing),
        expected=len(protocol['videos']),missing=missing,subsets=subsets,all_unpaired_inclusive=all_scores,
        unchanged_domain_sources=invariants,coverage=coverage,
        limitations=['One fixed checkpoint per model; no seed replication.',
        'Old person/time annotations transferred spatially; unmatched positive windows excluded, not background.',
        'Train replay cannot establish generalization; historical pretrained weights saw many old sources.',
        'Video decoding without real-time pacing; elapsed is not real-time performance acceptance.'])
    save(OUT/'results.json',result);save(OUT/'paired-windows.json',paired);save(OUT/'all-events.json',allrows)
    with (OUT/'paired-windows.csv').open('w')as f:
        fields=[k for k in paired[0]if k not in ('source_frames','skeleton_probs15')]
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(paired)
    for split in ['val','test','train']:
        r=subsets[split+'/15-75'];print(split,'15-75:',{k:r[k]for k in ('paired','unknown','changed','fixed','broken')},
            'correct',r['gate75']['correct'],r['gate15']['correct'],'/',r['gate75']['n'],flush=True)
    print('completed',result['completed'],'/',result['expected'],flush=True)


if __name__=='__main__':main()
