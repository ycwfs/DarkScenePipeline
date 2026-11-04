from collections import defaultdict,Counter
import numpy as np
from common import *
from review_state import CandidateSequence
from darkpipe.person.object_prior import filter_object_evidence

CLASSES=['wave','throw','fall','fight','talk','drink','pick_up','shake_hands']
METHODS=['model2','model3','prior2','prior3']

def object_support(r,objects):
    e=objects.get(r['id'])
    if not e or not e.get('exact_features_verified'):return False,'object_evidence_missing'
    label=r['raw_label'];samples=e.get('actions',{}).get(label)
    if samples is None:return False,'object_samples_missing'
    result=filter_object_evidence(label,samples,r.get('raw_luma'),.581,35.)
    return result['applied'] and result['label']==label,result['reason']

def eligible(r,kind,objects,details):
    if r['raw_label'] not in CLASSES or r['baseline']!='other' or not(.65<r['score']<.925):
        details[r['id']]={'eligible':False,'reason':'not_subthreshold_candidate'};return False
    if kind=='prior' and r['raw_label']=='wave' and not r['support']:
        details[r['id']]={'eligible':False,'reason':r.get('prior_reason','wave_pose_rejected')};return False
    if kind=='prior' and r['raw_label'] in ('drink','pick_up'):
        support,reason=object_support(r,objects);details[r['id']]={'eligible':support,'reason':reason};return support
    details[r['id']]={'eligible':True,'reason':'no_action_specific_prior'}
    return True

def replay(rows,objects):
    output={m:{}for m in METHODS};evidence={};
    states={}
    for r in sorted(rows,key=lambda r:(r['source_id'],r['track_id'],r['history_generation'],r['timestamp'],r.get('event_index',0))):
        key=(r['source_id'],r['track_id'],r['history_generation'])
        if key not in states:states[key]={m:CandidateSequence(2,.5)for m in ('model2','prior2')}|{m:CandidateSequence(3,1.)for m in ('model3','prior3')}
        for m in METHODS:
            is_prior=m.startswith('prior');ok=eligible(r,'prior' if is_prior else 'model',objects,evidence)
            decision=states[key][m].push(r,ok)
            output[m][r['id']]=decision
    return output,evidence

def count(rows,decisions,method):
    recovered=[r for r in rows if decisions[method][r['id']]['recovered']]
    known=[r for r in recovered if r['truth']is not None]
    tp=[r for r in known if r['truth']==r['raw_label']]
    fp=[r for r in known if r['truth']=='other' or r['truth'] not in (None,r['raw_label'])]
    unknown=[r for r in recovered if r['truth']is None]
    confirms=[r for r in recovered if decisions[method][r['id']].get('first_confirmation')]
    delays=[decisions[method][r['id']]['confirmation_delay_s']for r in confirms]
    return dict(recovered_windows=len(recovered),true_recovered=len(tp),false_alarms=len(fp),
        unknown=len(unknown),wrong_action=sum(r['truth'] not in (None,'other',r['raw_label'])for r in fp),
        by_action=dict(Counter(r['raw_label']for r in tp)),false_by_action=dict(Counter(r['raw_label']for r in fp)),
        first_confirmations=len(confirms),delay_s=dict(mean=sum(delays)/len(delays),median=float(np.median(delays)),max=max(delays))if delays else None)

def groups(rows,decisions,method):
    grouped=defaultdict(list);last={};serial=Counter()
    for r in sorted(rows,key=lambda r:(r['source_id'],r['track_id'],r['history_generation'],r['timestamp'])):
        if r['truth'] is None:continue
        if r['scope']=='new_multipeople' and r.get('review_ids'):
            for rid in r['review_ids']:grouped['review:'+rid].append(r)
            continue
        key=(r['scope'],r['source_id'],r['track_id'],r['history_generation']);prev=last.get(key)
        if prev is None or prev['truth']!=r['truth']or r['timestamp']-prev['timestamp']>.650001:serial[key]+=1
        last[key]=r
        grouped['seq:'+':'.join(map(str,key))+':'+str(serial[key])].append(r)
    detail=[]
    for gid,rr in grouped.items():
        truth=rr[0]['truth'];base=any(r['baseline']==truth for r in rr)if truth!='other'else any(r['baseline']!='other'for r in rr)
        accepted=[r for r in rr if decisions[method][r['id']]['recovered']]
        correct=any(r['raw_label']==truth for r in accepted)
        alarm=truth=='other'and bool(accepted)
        detail.append(dict(group=gid,scope=rr[0]['scope'],truth=truth,baseline_detected_or_alarm=base,
            candidate_correct=correct,candidate_alarm=alarm,newly_recovered_event=truth!='other'and not base and correct,
            newly_false_alarm_event=truth=='other'and not base and alarm))
    return detail

def main():
    data=rows();objects={p.stem:read(p)for p in (AUDIT/'object-evidence').glob('*.json')}
    objects.update({p.stem:read(p)for p in (OUT/'object-evidence').glob('*.json')})
    # Main evaluation excludes the training split; retain it separately as a diagnostic.
    partitions={
        'old_val':[r for r in data if r['scope']!='new_multipeople'and r['split']=='val'],
        'old_test':[r for r in data if r['scope']!='new_multipeople'and r['split']=='test'],
        'crowded_val':[r for r in data if r['scope']=='new_multipeople'and r['split']=='val'],
        'crowded_test':[r for r in data if r['scope']=='new_multipeople'and r['split']=='test'],
        'train_diagnostic':[r for r in data if r['split']=='train']}
    all_rows=[r for rr in partitions.values()for r in rr]
    decisions,prior_audit=replay(all_rows,objects)
    result={}
    for name,rr in partitions.items():
        classes={}
        for label in CLASSES:
            subset=[r for r in rr if r['raw_label']==label and .65<r['score']<.925]
            bins={}
            for i in range(len(BINS)):
                binned=[r for r in subset if r['bin']==i]
                bins[str(i)]=dict(candidate_windows=len(binned),
                    known=sum(r['truth']is not None for r in binned),
                    known_true=sum(r['truth']==label for r in binned),
                    known_other=sum(r['truth']=='other'for r in binned),
                    decisions={m:count(binned,decisions,m)for m in METHODS})
            classes[label]=dict(candidate_windows=len(subset),known=sum(r['truth']is not None for r in subset),
                known_true=sum(r['truth']==label for r in subset),known_other=sum(r['truth']=='other'for r in subset),
                decisions={m:count(subset,decisions,m)for m in METHODS},bins=bins)
        result[name]=dict(total=len(rr),known=sum(r['truth']is not None for r in rr),
            baseline_positive_tp=sum(r['baseline']==r['truth']and r['truth']not in (None,'other')for r in rr),
            baseline_other_fp=sum(r['baseline']!='other'and r['truth']=='other'for r in rr),
            classes=classes,totals={m:count(rr,decisions,m)for m in METHODS})
    event_results={};event_rows=[]
    for name,rr in partitions.items():
        event_results[name]={}
        for m in METHODS:
            g=groups(rr,decisions,m);event_rows.extend([dict(partition=name,method=m,**x)for x in g])
            event_results[name][m]=dict(positive_events=sum(x['truth']!='other'for x in g),
                newly_recovered_events=sum(x['newly_recovered_event']for x in g),
                other_events=sum(x['truth']=='other'for x in g),new_false_alarm_events=sum(x['newly_false_alarm_event']for x in g))
    save(OUT/'results.json',dict(protocol=read(OUT/'protocol.json'),partitions=result,events=event_results,
        prior_evidence_coverage=dict(object_windows=len(objects),missing_object_window_ids=sorted(
            r['id']for r in all_rows if r['raw_label']in ('drink','pick_up')and r['baseline']=='other'and .65<r['score']<.925 and r['id']not in objects))))
    save(OUT/'decisions.json',decisions);save(OUT/'prior-audit.json',prior_audit);save(OUT/'event-results.json',event_rows)
    detail=[]
    for r in all_rows:
        if r['raw_label'] not in CLASSES or r['bin'] is None:continue
        x={k:r.get(k)for k in ('id','source_id','scope','split','track_id','history_generation','timestamp','score','raw_label','truth','truth_source','baseline','bin','support','prior_reason')}
        obj=objects.get(r['id']);x['object_evidence_available']=obj is not None
        x['object_reason']=None
        for method in METHODS:x[method]=decisions[method][r['id']]['recovered'];x[method+'_delay_s']=decisions[method][r['id']].get('confirmation_delay_s')
        detail.append(x)
    import csv
    with(OUT/'window-results.csv').open('w')as f:
        w=csv.DictWriter(f,fieldnames=list(detail[0]));w.writeheader();w.writerows(detail)
    save(OUT/'frozen-input-hashes.json',read(AUDIT/'input-hashes.json'))
    print(json.dumps({k:{m:v['totals'][m]for m in ('prior2','prior3')}for k,v in result.items()},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
