"""Compare held-out old regression and explicitly labelled crowded training fit."""
import sys
import itertools
from common import *
sys.path.insert(0,str(OP))
import numpy as np
from darkpipe.person.object_prior import filter_object_evidence
from evaluation_metrics import metrics


def main() -> None:
    lock=read(OUT/'selection-lock.json');assert not lock['test_used_for_selection']
    rows=read(OUT/'dataset.json')['records'];n=len(rows);scored=scored_indices(rows)
    inputs={}
    for branch,which in itertools.product(['rgb','skeleton'],['baseline','candidate','selected','last','trained']):
        with np.load(OUT/'predictions'/f'{branch}-{which}.npz') as cache:
            assert cache['ids'].tolist()==[r['id'] for r in rows]
            inputs[branch,which]=cache['probs']
    # Prior runs selected original RGB, so their all-row baseline cache is complete.
    for tag,folder in [('previous','manypeople-finetune-20261029'),('rebalanced','manypeople-positive-rebalance-20261030')]:
        path=ROOT/'reports'/folder
        assert read(path/'selection-lock.json')['epochs']['rgb']==0
        for branch in ['rgb','skeleton']:
            inputs[branch,tag]=np.load(path/'predictions'/f'{branch}-selected.npz')['probs']
    cache=np.load(OUT/'skeleton-inputs.npz');gi=GATES.index(35.)
    valid=cache['valid'][gi];wave=cache['wave_counts'][gi];speed=cache['speed'][gi]
    luma=np.array([np.nan if r['raw_luma'] is None else r['raw_luma'] for r in rows])
    lookup={r['id']:i for i,r in enumerate(rows)};support=np.full((n,2),np.inf,np.float32);covered=set()
    for path in (OUT/'object-evidence').glob('*.json'):
        for evidence in read(path)['records']:
            i=lookup[evidence['id']];covered.add(i)
            for ci,label in enumerate(['drink','pick_up']):
                filtered=filter_object_evidence(label,evidence['actions'][label],rows[i]['raw_luma'],.001,35.)
                if filtered['applied']:support[i,ci]=max((d['confidence'] for d in filtered['evidence']),default=0.)
    assert set(scored)<=covered
    for (branch,which),values in inputs.items():
        assert np.isfinite(values[scored] if branch=='rgb' else values[:,scored]).all(),(branch,which)
    roles=['old_val','old_test','new_positive_training_fit','new_other_training_fit','new_other_untrained_same_sources']
    groups={role:[i for i in scored if score_role(rows[i])==role] for role in roles}
    assert sum(map(len,groups.values()))==len(scored)==1148
    base=lock['baseline_config']
    configurations={name:dict(base,rgb=which,skeleton=which) for name,which in [
        ('baseline','baseline'),('previous_same_defaults','previous'),('rebalanced_same_defaults','rebalanced'),
        ('selected','selected'),('best_trained','trained'),('last_epoch','last'),('individual_branch_selections','candidate')]}
    def predict(config: dict) -> np.ndarray:
        rgb=inputs['rgb',config['rgb']];sk=inputs['skeleton',config['skeleton']][gi].copy()
        sk[~valid]=np.eye(10,dtype=np.float32)[9]
        p=config['alpha']*rgb+(1-config['alpha'])*sk;p[luma<35.]=rgb[luma<35.]
        labels=p.argmax(1);labels[(labels!=9)&(p.max(1)<config['action'])]=9
        for ci,cls in enumerate([6,7]):labels[(labels==cls)&(support[:,ci]<config['object'])]=9
        labels[(labels==0)&(wave<3)]=9
        labels[(labels==2)&np.isfinite(speed)&(speed<.35)]=9
        unscored=np.ones(n,dtype=bool);unscored[scored]=False;labels[unscored]=-1
        return labels
    predictions={};results={}
    for name,config in configurations.items():
        pred=predict(config);predictions[name]=pred
        results[name]=dict(config=config,groups={role:metrics([rows[i] for i in ids],pred[ids]) for role,ids in groups.items()})
    assert results['selected']['groups']['old_val']==lock['validation']['old']
    np.savez_compressed(OUT/'final-predictions.npz',**predictions)
    np.savez_compressed(OUT/'object-support.npz',scores=support)
    save(OUT/'results.json',dict(comparisons=results,groups={k:len(v) for k,v in groups.items()},
        selection_lock_sha256=sha(OUT/'selection-lock.json'),object_evidence_windows=len(covered),
        crowded_positive_evaluation='ALL 85 ARE TRAINING EXAMPLES, not held-out performance',
        crowded_other_evaluation='trained subset versus untrained windows from seen source groups, not independent negatives',
        old_evaluation='Source-isolated historical regression; old val used for checkpoint selection'))
    branch_results={}
    for branch in ['rgb','skeleton']:
        branch_results[branch]={}
        for which in ['baseline','previous','rebalanced','candidate','selected','trained','last']:
            probs=inputs[branch,which] if branch=='rgb' else inputs[branch,which][gi]
            branch_results[branch][which]={}
            for threshold in [.75,.925]:
                pred=probs.argmax(1);pred[(pred!=9)&(probs.max(1)<threshold)]=9
                branch_results[branch][which][str(threshold)]={}
                for role,ids in groups.items():
                    actual=[i for i in ids if branch=='rgb' or valid[i]]
                    branch_results[branch][which][str(threshold)][role]=metrics([rows[i] for i in actual],pred[actual])
    save(OUT/'branch-results.json',dict(priors_applied=False,skeleton_valid_only=True,comparisons=branch_results))
    print(json.dumps({name:{g:dict(fp=m['other_fp'],negative=m['other_n'],tp=m['positive_tp'],positive=m['positive_n'])
        for g,m in r['groups'].items()}for name,r in results.items()},ensure_ascii=False))


if __name__=='__main__':main()
