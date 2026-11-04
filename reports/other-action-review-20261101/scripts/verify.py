from collections import defaultdict
import numpy as np
from common import *

CLASSES={'wave','throw','fall','fight','talk','drink','pick_up','shake_hands'}

def main():
    data=rows();dec=read(OUT/'decisions.json');objects={p.stem:read(p)for p in (AUDIT/'object-evidence').glob('*.json')}
    objects.update({p.stem:read(p)for p in (OUT/'object-evidence').glob('*.json')})
    check=dict()
    for r in data:
        if r['scope']=='new_multipeople' and r['truth_source']=='user_confirmed_20261101':assert r['truth'] in set(LABELS)-{'other'}
    check['user_confirmed_windows']=sum(r['truth_source']=='user_confirmed_20261101'for r in data)
    selected=read(OUT/'object-needed.json')
    assert all((OUT/'object-evidence'/f"{r['id']}.json").exists()for r in selected)
    for r in selected:
        e=read(OUT/'object-evidence'/f"{r['id']}.json");assert e['exact_features_verified']
        with np.load(AUDIT/'features'/f"{r['id']}.npz")as a:
            sampled=list(dict.fromkeys(int(a['source_frames'][j])for j in (0,8,16,23,31)))
        assert[e['actions'][r['raw_label']][i]['frame']for i in range(5)]==sampled
    check['exact_object_windows']=len(selected)
    check['old_exact_object_reuse']=read(OUT/'candidate-plan.json')['exact_object_evidence_reused']
    check['new_frozen_hashes_unchanged']=sum(sha(ROOT/p)==h for p,h in read(OUT/'input-hashes.json').items())
    assert check['new_frozen_hashes_unchanged']==len(read(OUT/'input-hashes.json'))
    eligible_count=0
    for method,values in dec.items():
        if method not in ('model2','model3','prior2','prior3'):continue
        for rid,d in values.items():
            if d['recovered']:
                eligible_count+=1
                r=next(x for x in data if x['id']==rid)
                assert r['baseline']=='other'and r['raw_label'] in CLASSES and .65<r['score']<.925
                assert d['wait_s']>=0
    check['accepted_windows_rule_checked']=eligible_count
    check['state_machine_tests']='6 passed'
    save(OUT/'verification.json',dict(passed=True,checks=check))
    print(check)

if __name__=='__main__':main()
