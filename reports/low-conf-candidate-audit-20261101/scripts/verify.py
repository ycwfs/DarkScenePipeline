"""Artifact, identity, causality and production-preservation checks."""
from collections import defaultdict
import numpy as np
from common import *
from evaluate import replay,METHODS

def main():
    rows=read(OUT/'replay-rows.json');out=read(OUT/'decisions.json');live,p=load();byid={r['id']:r for r in live}
    checks={};count=0
    for r in rows:
        for m in METHODS:
            d=out[r['id']][m]
            if not d['recovered']:continue
            count+=1
            assert r['baseline']=='other'and .65<r['score']<.925 and r['raw_label']!='other'and r['support']
            if 'anchor_time'in d:assert d['anchor_time']<=r['timestamp']and r['timestamp']-d['anchor_time']<=1.200001
            if 'supporting_windows'in d:
                for rid in d['supporting_windows']:
                    q=byid[rid];assert q['timestamp']<=r['timestamp']
                    assert(q['source_id'],q['track_id'],q['history_generation'])==(r['source_id'],r['track_id'],r['history_generation'])
            assert d['wait_s']>=0
    checks['accepted_rule_checks']=count
    # All prefixes for affected identities: future rows must not change old outputs.
    groups=defaultdict(list)
    for r in rows:groups[(r['source_id'],r['track_id'],r['history_generation'])].append(r)
    prefixes=0
    for rr in groups.values():
        if not any(out[r['id']][m]['recovered']for r in rr for m in METHODS):continue
        rr.sort(key=lambda r:r['timestamp'])
        for stop in range(1,len(rr)+1):
            replayed,_=replay(rr[:stop])
            assert replayed[rr[stop-1]['id']]==out[rr[stop-1]['id']]
            prefixes+=1
    checks['causal_prefixes']=prefixes
    # Exact feedback dataset labels are the sole source of crowded truth.
    dataset=read(ROOT/'reports/manypeople-finetune-20261029/dataset.json')['records']
    labels={r['id']:r for r in dataset if r['scope']=='new'and r['label_kind']=='human'}
    assert len(labels)==736
    for r in rows:
        if r['scope']!='new_multipeople':continue
        assert(r['truth']is not None)==(r['id']in labels)
        if r['id']in labels:assert r['truth']==labels[r['id']]['label']
    checks['crowded_human_windows']=736
    # Frozen 10/30 baseline was evaluated through the same production functions.
    archive=np.load(ROOT/'reports/manypeople-all85-finetune-20261030/final-predictions.npz')
    evaluated=read(ROOT/'reports/manypeople-all85-finetune-20261030/dataset.json')['records']
    current={r['id']:r for r in rows};comparison=[]
    for i,r in enumerate(evaluated):
        if r['scope']=='new'and r['label_kind']=='human':
            if int(archive['baseline'][i])<0:continue
            comparison.append(dict(id=r['id'],archived=LABELS[int(archive['baseline'][i])],replayed=current[r['id']]['baseline']))
    mismatches=[r for r in comparison if r['archived']!=r['replayed']]
    # Float differences/ROI versions must be shown, never silently replaced.
    checks['crowded_baseline_archived_comparison']=dict(checked=len(comparison),differences=mismatches)
    frozen=read(OUT/'input-hashes.json');different=[]
    for path,digest in frozen.items():
        if sha(ROOT/path)!=digest:different.append(path)
    assert not different,different
    checks['frozen_input_hashes_unchanged']=len(frozen)
    expected=[r for r in live if r['raw_label']!='other'and r['score']>.65]
    for r in expected:
        ss=samples(OUT/'features'/(r['id']+'.npz'))
        assert[s.frame_index for s in ss]==r['source_frames']
        assert np.allclose(ss[-1].person_box,r['bbox'],atol=.02,rtol=0)
    checks['exact_geometry_windows']=len(expected)
    assert all(r['exit_code']==0 for r in read(OUT/'frontend-jobs.json'))
    assert read(OUT/'objects-complete.json')['complete']
    checks['gpu_supplement_complete']=True
    save(OUT/'verification.json',dict(passed=True,checks=checks))
    print(checks)

if __name__=='__main__':main()
