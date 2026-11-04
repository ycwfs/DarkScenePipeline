"""Explicit post-audit amendment: protect every supported old-val action class.

The first aggregate-retention search hid class substitution. Preserve it verbatim;
do not rebrand repeatedly inspected historical test as an independent holdout.
"""
import json
import numpy as np
from tune import Evaluation,csv_save
from evidence import OUT,save,digest,BEHAVIORS


def main() -> None:
    ev=Evaluation();base=ev.calibration(.85,.35);bp=ev.predict(.85,.35)
    classes=[c for c in range(10) if c!=ev.o and np.any(ev.oldval & (ev.truth==c))]
    masks={c:ev.oldval & (ev.truth==c) for c in classes}
    minimum={c:int((bp[masks[c]]==c).sum()) for c in classes}
    amend=dict(reason='First locked aggregate guard allowed old-val running TP8→5 while drinking improved8→10; old test subsequently showed regression. Add per-class old-val retention.',
        selection_inputs='Old validation and reviewed new feedback only; no old train/test scores in selection function.',
        caveat='Protocol amended after inspecting first search and historical test; subsequent test is diagnostic, not pristine holdout.',
        baseline_support={BEHAVIORS[c]:dict(tp=minimum[c],support=int(masks[c].sum())) for c in classes},
        baseline_other_count=int(ev.oldval_other.sum()),baseline_positive_count=int(ev.oldval_positive.sum()),
        constraint='All original feasibility rules AND every old-val supported named-class TP >= baseline class TP.',
        objective='Same fixed equal-weight false-positive objective; no altered model or inference.',
        original_lock_sha256=digest(OUT/'selection-lock.json'),fine='Same +/-0.01 step0.001 near amended coarse choice')
    save(OUT/'protocol-amendment.json',amend)
    def score(tau,obj):
        row=ev.calibration(tau,obj);pred=ev.predict(tau,obj)
        for c in classes:row['val_tp_'+BEHAVIORS[c]]=int((pred[masks[c]]==c).sum())
        row['feasible']=ev.feasible(row,base) and all(row['val_tp_'+BEHAVIORS[c]]>=minimum[c] for c in classes)
        row['objective']=ev.key(row)[0]
        return row
    grid=[score(tau,obj) for tau in ev.protocol['grid']['action'] for obj in ev.protocol['grid']['object']]
    valid=[r for r in grid if r['feasible']];assert valid
    coarse=min(valid,key=ev.key);fine=[]
    oldkeys={(round(r['action_threshold'],4),round(r['object_min_conf'],4)) for r in grid}
    for tau in np.arange(round(coarse['action_threshold']*1000)-10,round(coarse['action_threshold']*1000)+11)/1000:
        for obj in np.arange(round(coarse['object_min_conf']*1000)-10,round(coarse['object_min_conf']*1000)+11)/1000:
            if not .7<=tau<=.99 or not .01<=obj<=.9 or (round(tau,4),round(obj,4)) in oldkeys:continue
            fine.append(score(tau,obj))
    valid += [r for r in fine if r['feasible']]
    chosen=min(valid,key=ev.key)
    true_kept=[r for r in valid if r['new_true_drink_tp']==1]
    retain=min(true_kept,key=ev.key) if true_kept else None
    object_only=[r for r in grid if r['action_threshold']==.85 and r['feasible']]
    object_choice=min(object_only,key=ev.key)
    action_only=[r for r in grid if r['object_min_conf']==.35 and r['feasible']]
    action_choice=min(action_only,key=ev.key)
    lock=dict(amendment_sha256=digest(OUT/'protocol-amendment.json'),baseline=base,
        coarse_best=coarse,recommended=chosen,true_drink_retention_option=retain,
        object_only=object_choice,action_only=action_choice,
        coarse_combinations=len(grid),fine_additional=len(fine),feasible=len(valid))
    save(OUT/'class-guard-selection-lock.json',lock)
    csv_save(OUT/'class-guard-grid.csv',grid);csv_save(OUT/'class-guard-fine-grid.csv',fine)
    settings={name:{k:r[k] for k in ['action_threshold','object_min_conf']} for name,r in dict(
        recommended=chosen,coarse_recommended=coarse,true_drink_retention=retain,object_only=object_choice,action_only=action_choice).items() if r is not None}
    results={name:ev.full(setting) for name,setting in settings.items()};save(OUT/'class-guard-summary.json',results)
    save(OUT/'recommended-settings.json',dict(reco_min_conf=chosen['action_threshold'],object_min_conf=chosen['object_min_conf'],
        skel_fusion_alpha=.95,lum_gate=30.,skel_no_person='soft',
        algorithm='unchanged onsite-finetuned seven-class YOLO26m +720enhanced personROI single-frame prior',
        selection_lock='class-guard-selection-lock.json',state='Offline candidate preset; platform object prior not integrated'))
    print(json.dumps(lock,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
