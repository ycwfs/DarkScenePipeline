"""Calibrate two global thresholds, lock, then report old train/test separately."""
from collections import Counter
import csv
import json
from pathlib import Path
import sys

import numpy as np
from evidence import OUT,ROOT,OLD,ABL,OP,save,digest,BEHAVIORS,filter_object_evidence
from darkpipe.stages.recognize import reject_probs


def csv_save(path: Path, rows: list) -> None:
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


class Evaluation:
    def __init__(self) -> None:
        self.protocol=json.loads((OUT/'protocol.json').read_text())
        self.rows=json.loads((OUT/'windows.json').read_text())
        assert digest(OUT/'windows.json')==self.protocol['files'][str((OUT/'windows.json').relative_to(ROOT))]
        self.evidence={};self.sources={s['id']:s for s in self.protocol['sources']}
        for s in self.protocol['sources']:
            d=json.loads((OUT/'evidence'/f'{s["id"]}.json').read_text())
            assert d['protocol_sha256']==digest(OUT/'protocol.json')
            for r in d['records']:assert r['id'] not in self.evidence;self.evidence[r['id']]=r
        self.o=BEHAVIORS.index('other');self.drink=BEHAVIORS.index('drink');self.pickup=BEHAVIORS.index('pick_up')
        self.top=np.array([BEHAVIORS.index(r['top_label']) for r in self.rows])
        self.score=np.array([r['top_score'] for r in self.rows])
        self.support=np.full(len(self.rows),np.inf)
        for i,r in enumerate(self.rows):
            if r['top_label'] in ('drink','pick_up') and r['top_score']>=.7:
                assert r['id'] in self.evidence,r['id'];e=self.evidence[r['id']]
                if e['object_observed']:self.support[i]=e['max_support']
        self.truth=np.array([self.o if r['truth']=='other' else BEHAVIORS.index(r['truth']) if r['truth'] is not None else -1 for r in self.rows])
        self.scope=np.array([r['scope'] for r in self.rows]);self.split=np.array([r['split'] for r in self.rows])
        self.new=self.scope=='new_multipeople'
        self.new_true=np.array([r['is_drink'] is True for r in self.rows])
        self.new_false=np.array([r['is_drink'] is False for r in self.rows])
        self.new_unknown=self.new & ~self.new_true & ~self.new_false
        self.oldval=(self.scope=='onsite') & (self.split=='val') & (self.truth>=0)
        self.onlineval=(self.scope=='online_supplement') & (self.split=='val') & (self.truth>=0)
        self.oldval_other=self.oldval & (self.truth==self.o)
        self.oldval_positive=self.oldval & (self.truth!=self.o)
        self.oldval_drink=self.oldval & (self.truth==self.drink)
        self.oldval_pickup=self.oldval & (self.truth==self.pickup)
        self.true_score=float(self.score[self.new_true][0]);assert self.new_true.sum()==1 and self.new_false.sum()==526

    def predict(self,action: float,objects: float | None) -> np.ndarray:
        p=self.top.copy();p[(self.score<action) & (p!=self.o)]=self.o
        if objects is not None:p[np.isin(p,[self.drink,self.pickup]) & (self.support<objects)]=self.o
        return p

    def metrics(self,p: np.ndarray,mask: np.ndarray) -> dict:
        known=mask & (self.truth>=0);y=self.truth[known];z=p[known]
        matrix=np.bincount(y*10+z,minlength=100).reshape(10,10)
        support=matrix.sum(1);predicted=matrix.sum(0);tp=np.diag(matrix)
        f1=np.divide(2*tp,support+predicted,out=np.zeros(10,float),where=(support+predicted)>0)
        return dict(windows=int(mask.sum()),known=int(known.sum()),unknown=int((mask & (self.truth<0)).sum()),
            other_count=int(support[self.o]),other_fp=int(support[self.o]-tp[self.o]),
            positive_count=int(support.sum()-support[self.o]),positive_correct=int(tp.sum()-tp[self.o]),
            macro_f1=float(f1[support>0].mean()) if np.any(support>0) else None,
            classes={c:dict(support=int(support[i]),tp=int(tp[i]),fp=int(predicted[i]-tp[i]),fn=int(support[i]-tp[i])) for i,c in enumerate(BEHAVIORS)},
            confusion=matrix.tolist())

    def calibration(self,action: float,objects: float) -> dict:
        p=self.predict(action,objects);m=self.metrics(p,self.oldval)
        return dict(action_threshold=float(action),object_min_conf=float(objects),
            old_val_drink_tp=int((p[self.oldval_drink]==self.drink).sum()),
            old_val_pickup_tp=int((p[self.oldval_pickup]==self.pickup).sum()),
            old_val_positive_correct=int((p[self.oldval_positive]==self.truth[self.oldval_positive]).sum()),
            old_val_other_fp=int((p[self.oldval_other]!=self.o).sum()),old_val_macro_f1=m['macro_f1'],
            online_val_other_fp=int((p[self.onlineval]!=self.o).sum()),
            new_known_drink_fp=int((p[self.new_false]==self.drink).sum()),
            new_true_drink_tp=int((p[self.new_true]==self.drink).sum()),
            new_unreviewed_drink=int((p[self.new_unknown]==self.drink).sum()),
            new_pickup_triggers=int((p[self.new]==self.pickup).sum()))

    def feasible(self,r: dict,b: dict) -> bool:
        return bool(all(r[k]>=b[k] for k in ['old_val_drink_tp','old_val_pickup_tp','old_val_positive_correct']) and
            all(r[k]<=b[k] for k in ['old_val_other_fp','online_val_other_fp','new_known_drink_fp','new_unreviewed_drink']) and
            r['action_threshold']<=self.true_score)

    def key(self,r: dict) -> tuple:
        cost=.5*(r['old_val_other_fp']/self.oldval_other.sum()+r['new_known_drink_fp']/526)
        distance=abs(r['action_threshold']-.85)+abs(r['object_min_conf']-.35)
        return cost,-r['old_val_macro_f1'],-r['old_val_positive_correct'],-r['new_true_drink_tp'],distance,r['action_threshold'],r['object_min_conf']

    def full(self,setting: dict) -> dict:
        p=self.predict(setting['action_threshold'],setting['object_min_conf']);out=dict(setting=setting)
        for scope in ['onsite','online_supplement']:
            out[scope]={split:self.metrics(p,(self.scope==scope) & ((self.split==split) if split!='all' else True)) for split in ['all','train','val','test']}
        out['new']=dict(windows=int(self.new.sum()),known_drink_positive=1,known_drink_negative=526,
            known_drink_tp=int((p[self.new_true]==self.drink).sum()),known_drink_fp=int((p[self.new_false]==self.drink).sum()),
            unknown_drink=int((p[self.new_unknown]==self.drink).sum()),pickup_triggers=int((p[self.new]==self.pickup).sum()),
            actions=dict(Counter(BEHAVIORS[i] for i in p[self.new])),per_source=[])
        for sid,s in self.sources.items():
            if s['scope']!='new_multipeople':continue
            mask=np.array([r['source_id']==sid for r in self.rows])
            out['new']['per_source'].append(dict(id=sid,known_drink_tp=int((p[mask & self.new_true]==self.drink).sum()),
                known_drink_fp=int((p[mask & self.new_false]==self.drink).sum()),unknown_drink=int((p[mask & self.new_unknown]==self.drink).sum()),
                pickup_triggers=int((p[mask]==self.pickup).sum()),actions=dict(Counter(BEHAVIORS[i] for i in p[mask]))))
        return out

    def verify(self) -> dict:
        p=self.predict(.85,.35);by_id={r['id']:int(v) for r,v in zip(self.rows,p)}
        old_checks=0
        for v in self.protocol['sources']:
            if v['scope']=='new_multipeople':continue
            for r in json.loads((OLD/'per-video'/f'{v["id"]}.json').read_text())['records']:
                assert BEHAVIORS[by_id[r['id']]]==r['after'],r['id'];old_checks+=1
        prior=json.loads((ABL/'action-results.json').read_text());new_checks=0
        for r in prior:
            if r['model']=='m_finetuned' and r['variant']=='small_enhanced' and r['split']=='feedback':
                assert BEHAVIORS[by_id[r['id']]]==r['decision']['label'],r['id'];new_checks+=1
        # Verify vectorized threshold math against the unmodified deployment policy.
        for i,r in enumerate(self.rows):
            for tau in [.7,.85,.94,.99]:
                label=int(reject_probs(np.array(r['fused_probs'],np.float32),BEHAVIORS,tau).argmax())
                assert label==self.predict_one(i,tau,None)
        checked=0
        by={r['id']:i for i,r in enumerate(self.rows)}
        for r in self.evidence.values():
            for obj in [.01,.1,.35,.9]:
                a=filter_object_evidence(r['label'],r['samples'],r['raw_luma'],obj)['label']
                assert BEHAVIORS[self.predict_one(by[r['id']],.7,obj)]==a,(r['id'],obj)
                checked+=1
        return dict(old_baseline_reproduced=old_checks,new_baseline_reproduced=new_checks,
            deployment_reject_checks=len(self.rows)*4,object_policy_checks=checked,passed=True)

    def predict_one(self,i: int,tau: float,obj: float | None) -> int:
        label=self.top[i]
        if label!=self.o and self.score[i]<tau:label=self.o
        if obj is not None and label in (self.drink,self.pickup) and self.support[i]<obj:label=self.o
        return int(label)


def main() -> None:
    ev=Evaluation();verification=ev.verify();save(OUT/'verification.json',verification)
    base=ev.calibration(.85,.35);grid=[]
    for tau in ev.protocol['grid']['action']:
        for obj in ev.protocol['grid']['object']:
            r=ev.calibration(tau,obj);r['feasible']=ev.feasible(r,base);r['objective']=ev.key(r)[0];grid.append(r)
    feasible=[r for r in grid if r['feasible']];assert feasible
    coarse=min(feasible,key=ev.key);tested={(round(r['action_threshold'],4),round(r['object_min_conf'],4)) for r in grid}
    fine=[]
    for tau in np.arange(round(coarse['action_threshold']*1000)-10,round(coarse['action_threshold']*1000)+11)/1000:
        for obj in np.arange(round(coarse['object_min_conf']*1000)-10,round(coarse['object_min_conf']*1000)+11)/1000:
            if not .7<=tau<=.99 or not .01<=obj<=.90 or (round(tau,4),round(obj,4)) in tested:continue
            r=ev.calibration(tau,obj);r['feasible']=ev.feasible(r,base);r['objective']=ev.key(r)[0];fine.append(r)
    candidates=feasible+[r for r in fine if r['feasible']];chosen=min(candidates,key=ev.key)
    recall=min(candidates,key=lambda r:(-r['old_val_positive_correct'],-r['new_true_drink_tp'],ev.key(r)))
    lock=dict(protocol_sha256=digest(OUT/'protocol.json'),coarse_combinations=len(grid),fine_additional=len(fine),
        baseline=base,coarse_best=coarse,recommended=chosen,recall_option=recall,
        feasible_combinations=len(candidates),selection_used='Old onsite val and reviewed new drinking feedback only; not old train/test.',
        confirmed_true_score=ev.true_score,confirmed_true_object_score=float(ev.support[ev.new_true][0]))
    save(OUT/'selection-lock.json',lock)  # Lock before any old-test metric is computed.
    csv_save(OUT/'coarse-grid.csv',grid);csv_save(OUT/'fine-grid.csv',fine)
    save(OUT/'recommended-settings.json',dict(reco_min_conf=chosen['action_threshold'],object_min_conf=chosen['object_min_conf'],
        skel_fusion_alpha=.95,lum_gate=30.,skel_no_person='soft',model='onsite-finetuned seven-class YOLO26m',
        input='720 Retinexformer enhanced person crops',state='Offline candidate preset; production per-person object prior not integrated'))
    settings=dict(baseline=dict(action_threshold=.85,object_min_conf=.35),
        no_prior_baseline=dict(action_threshold=.85,object_min_conf=None),
        recommended={k:chosen[k] for k in ['action_threshold','object_min_conf']},
        coarse_recommended={k:coarse[k] for k in ['action_threshold','object_min_conf']},
        recall_option={k:recall[k] for k in ['action_threshold','object_min_conf']})
    summary={name:ev.full(s) for name,s in settings.items()};save(OUT/'summary.json',summary)
    current=ev.predict(.85,.35);after=ev.predict(chosen['action_threshold'],chosen['object_min_conf'])
    window_rows=[]
    for i,r in enumerate(ev.rows):
        window_rows.append({k:r[k] for k in ['id','source_id','scope','split','group','track_id','history_generation','timestamp','truth','is_drink','top_label','top_score','raw_luma']} |
            dict(object_max_support=None if np.isinf(ev.support[i]) else float(ev.support[i]),prior_bypassed=bool(np.isinf(ev.support[i])),
                baseline=BEHAVIORS[current[i]],recommended=BEHAVIORS[after[i]],changed=bool(current[i]!=after[i])))
    csv_save(OUT/'window-results.csv',window_rows)
    save(OUT/'selected-predictions.json',dict(settings=settings,labels=BEHAVIORS,ids=[r['id'] for r in ev.rows],
        predictions={name:ev.predict(s['action_threshold'],s['object_min_conf']).tolist() for name,s in settings.items()}))
    print(json.dumps(lock,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
