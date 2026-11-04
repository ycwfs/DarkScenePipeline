"""Score held-out labels only after the joint checkpoint+parameter lock exists."""
from collections import defaultdict
import itertools
import sys
from common import *
sys.path.insert(0,str(OP))
import numpy as np
from darkpipe.person.object_prior import filter_object_evidence
from evaluation_metrics import metrics


def main():
    lock=read(OUT/'selection-lock.json');assert not lock['test_used_for_selection']
    rows=read(OUT/'dataset.json')['records'];n=len(rows);inputs={}
    for branch,which in itertools.product(['rgb','skeleton'],['baseline','candidate','selected']):
        with np.load(OUT/'predictions'/f'{branch}-{which}.npz')as a:
            assert a['ids'].tolist()==[r['id']for r in rows];inputs[branch,which]=a['probs']
    cache=np.load(OUT/'skeleton-inputs.npz');valid=cache['valid'];wave=cache['wave_counts'];speed=cache['speed']
    luma=np.array([np.nan if r['raw_luma']is None else r['raw_luma']for r in rows]);lookup={r['id']:i for i,r in enumerate(rows)}
    support=np.full((len(GATES),n,2),np.inf,np.float32);covered=set()
    for path in (OUT/'object-evidence').glob('*.json'):
        for e in read(path)['records']:
            i=lookup[e['id']];covered.add(i)
            for gi,g in enumerate(GATES):
                for ci,label in enumerate(('drink','pick_up')):
                    result=filter_object_evidence(label,e['actions'][label],rows[i]['raw_luma'],.001,g)
                    if result['applied']:support[gi,i,ci]=max((x['confidence']for x in result['evidence']),default=0.)
    def ids(split,domain):return np.array([i for i,r in enumerate(rows)if r['split']==split and r['scope']==domain
        and r['label_kind']!='pseudo'and (domain=='new'or r.get('current_index')is not None)],int)
    groups={split:{d:ids(split,d)for d in ('old','new')}for split in ('val','test')}
    assert set(np.concatenate([v for x in groups.values()for v in x.values()]))<=covered
    def predict(c):
        gi=GATES.index(c['gate']);rgb=inputs['rgb',c['rgb']];sk=inputs['skeleton',c['skeleton']][gi].copy();sk[~valid[gi]]=np.eye(10,dtype=np.float32)[9]
        p=c['alpha']*rgb+(1-c['alpha'])*sk
        if c['alpha']>=.999:p=rgb.copy()
        else:p[luma<c['gate']]=rgb[luma<c['gate']]
        labels=p.argmax(1);labels[(labels!=9)&(p.max(1)<c['action'])]=9
        for ci,cls in enumerate((6,7)):labels[(labels==cls)&(support[gi,:,ci]<c['object'])]=9
        if c['alpha']<.999:
            labels[(labels==0)&(wave[gi]<3)]=9
            labels[(labels==2)&np.isfinite(speed[gi])&(speed[gi]<.35)]=9
        return labels
    selected=lock['config'];baseline=lock['baseline_config']
    configurations=dict(baseline=baseline,trained_same_defaults=dict(baseline,rgb='selected',skeleton='selected'),
        selected=selected,individual_branch_selections=dict(baseline,rgb='candidate',skeleton='candidate'),
        rgb_without_skeleton_with_object=dict(selected,alpha=1.))
    predictions={};results={}
    for name,c in configurations.items():
        p=predict(c);predictions[name]=p;result=dict(config=c)
        for split in ('val','test'):
            result['validation'if split=='val'else'test']={d:metrics([rows[i]for i in inds],p[inds])for d,inds in groups[split].items()}
        results[name]=result
    assert results['selected']['validation']==lock['validation'],'Joint lock did not reproduce'
    np.savez_compressed(OUT/'final-predictions.npz',**predictions);np.savez_compressed(OUT/'object-support.npz',scores=support)
    save(OUT/'results.json',dict(comparisons=results,selection_lock_sha256=sha(OUT/'selection-lock.json'),
        object_evidence_windows=len(covered),pseudo_labels_excluded_from_metrics=True,
        test_group_caveat='One new acquisition group held out this round; historical sources already inspected, no new blind generalization claim.'))
    print(json.dumps(results,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
