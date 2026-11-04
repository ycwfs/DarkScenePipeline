"""Freeze a validation-only joint model/parameter choice, then expose test scores."""
from collections import defaultdict
import itertools
import sys
from common import *
sys.path.insert(0,str(OP))
import numpy as np
from darkpipe.person.object_prior import filter_object_evidence
from evaluation_metrics import metrics


def main():
    assert not (OUT/'selection-lock.json').exists()
    protocol=read(OUT/'protocol.json');rows=read(OUT/'dataset.json')['records'];n=len(rows)
    inputs={}
    for branch,which in itertools.product(['rgb','skeleton'],['baseline','candidate']):
        p=OUT/'predictions'/f'{branch}-{which}.npz'
        with np.load(p)as a:
            assert a['ids'].tolist()==[r['id']for r in rows]
            inputs[branch,which]=a['probs']
    features=np.load(OUT/'skeleton-inputs.npz');valid=features['valid'];wave=features['wave_counts'];speed=features['speed']
    luma=np.array([np.nan if r['raw_luma']is None else r['raw_luma']for r in rows])
    supported=np.full((len(GATES),n,2),np.inf,np.float32);covered=set();index={r['id']:i for i,r in enumerate(rows)}
    for path in (OUT/'object-evidence').glob('*.json'):
        evidence=read(path)
        assert evidence['model_sha256']==protocol['models']['object']['sha256']
        for e in evidence['records']:
            i=index[e['id']];assert i not in covered;covered.add(i)
            for gi,g in enumerate(GATES):
                for ci,label in enumerate(('drink','pick_up')):
                    out=filter_object_evidence(label,e['actions'][label],rows[i]['raw_luma'],.001,g)
                    if out['applied']:supported[gi,i,ci]=max((d['confidence']for d in out['evidence']),default=0.)
    def indices(split,scope):
        return np.array([i for i,r in enumerate(rows) if r['split']==split and r['scope']==scope and r['label_kind']!='pseudo'
            and (scope=='new' or r.get('current_index')is not None)],int)
    val={d:indices('val',d)for d in ('old','new')};test={d:indices('test',d)for d in ('old','new')}
    needed=set(np.concatenate(list(val.values())+list(test.values())).tolist());assert needed<=covered
    np.savez_compressed(OUT/'object-support.npz',scores=supported)
    def predict(config):
        gi=GATES.index(config['gate']);rgb=inputs['rgb',config['rgb']];sk=inputs['skeleton',config['skeleton']][gi].copy()
        sk[~valid[gi]]=np.eye(10,dtype=np.float32)[9]
        alpha=config['alpha'];fused=alpha*rgb+(1-alpha)*sk
        blocked=luma<config['gate']
        if alpha>=.999:fused=rgb.copy()
        else:fused[blocked]=rgb[blocked]
        labels=fused.argmax(1);labels[(labels!=9)&(fused.max(1)<config['action'])]=9
        for ci,cls in enumerate((6,7)):
            labels[(labels==cls)&(supported[gi,:,ci]<config['object'])]=9
        # Existing wave/running rules are enabled only while the skeleton branch is enabled.
        if alpha<.999:
            labels[(labels==0)&(wave[gi]<3)]=9
            labels[(labels==2)&np.isfinite(speed[gi])&(speed[gi]<.35)]=9
        return labels
    def score_ids(pred, ids):return {d:metrics([rows[i]for i in inds],pred[inds])for d,inds in ids.items()}
    baseline_config=dict(rgb='baseline',skeleton='baseline',alpha=.972,gate=35.,action=.925,object=.581)
    baseline=score_ids(predict(baseline_config),val)
    best=None;winner=None;grid=[];params=dict(protocol['tuning'])
    amendment=read(OUT/'protocol-amendment-before-training.json')
    params.update(amendment.get('grid_extension',{}))
    for rgb_model,sk_model,alpha,gate,action,obj in itertools.product(['baseline','candidate'],['baseline','candidate'],
            params['alpha'],params['gate'],params['action'],params['object']):
        config=dict(rgb=rgb_model,skeleton=sk_model,alpha=alpha,gate=gate,action=action,object=obj)
        pred=predict(config);scores=score_ids(pred,val)
        accepted=(alpha<.999 and all(scores['old']['classes'][c]['recall']>=b['recall']-.05-1e-9 for c,b in baseline['old']['classes'].items()if c!='other' and b['n'])
            and scores['new']['positive_macro_recall']>=baseline['new']['positive_macro_recall']-.10-1e-9)
        fpr=(scores['old']['other_fpr']+scores['new']['other_fpr'])/2
        f1=(scores['old']['macro_f1']+scores['new']['macro_f1'])/2
        tp=scores['old']['positive_tp']+scores['new']['positive_tp']
        distance=abs(alpha-.972)+abs(gate-35)/100+abs(action-.925)+abs(obj-.581)
        key=(-fpr,f1,tp,-distance,rgb_model=='candidate',sk_model=='candidate')
        grid.append(dict(**config,accepted=accepted,other_fpr=fpr,macro_f1=f1,positive_tp=tp,
            old_fp=scores['old']['other_fp'],new_fp=scores['new']['other_fp']))
        if accepted and (best is None or key>best):best=key;winner=dict(config=config,validation=scores)
    assert winner is not None
    save(OUT/'parameter-grid.json',grid)
    # This lock is written BEFORE test metrics are constructed or consulted.
    save(OUT/'selection-lock.json',dict(**winner,baseline_validation=baseline,baseline_config=baseline_config,
        criterion=protocol['tuning']['objective'],grid_count=len(grid),accepted_count=sum(r['accepted']for r in grid),
        protocol_sha256=sha(OUT/'protocol.json'),amendment_sha256=sha(OUT/'protocol-amendment-before-training.json'),
        dataset_sha256=sha(OUT/'dataset.json'),test_used_for_selection=False,
        models={b:{w:read(OUT/'predictions'/f'{b}-{w}.json')for w in ['baseline','candidate']}for b in ['rgb','skeleton']}))
    predictions={};results={}
    configurations=dict(baseline=baseline_config,trained_same_defaults=dict(baseline_config,rgb='candidate',skeleton='candidate'),
        selected=winner['config'],rgb_without_skeleton_with_object=dict(winner['config'],alpha=1.))
    for name,config in configurations.items():
        p=predict(config);predictions[name]=p
        results[name]=dict(config=config,validation=score_ids(p,val),test=score_ids(p,test))
    np.savez_compressed(OUT/'final-predictions.npz',**predictions)
    save(OUT/'results.json',dict(comparisons=results,selection_lock_sha256=sha(OUT/'selection-lock.json'),
        test_group_caveat='One new acquisition session held out this round; historical data already inspected/calibrated. Descriptive evidence, no new blind generalization claim.',
        object_evidence_windows=len(covered),pseudo_labels_excluded_from_metrics=True))
    print(json.dumps(dict(winner=winner,results=results),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
