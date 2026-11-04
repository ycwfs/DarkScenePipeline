"""Paired, causal replay with exact priors and explicit unlabelled accounting."""
from collections import defaultdict,Counter
import csv,time
from common import *
from policy import Continuation,RunningReview
from darkpipe.person.running_prior import speed_evidence
from darkpipe.person.wave_prior import wave_evidence
from darkpipe.person.object_prior import filter_object_evidence

METHODS=['continuation','running2','combined','running1','running3']

def evidence(rows):
    pot=set(read(OUT/'continuation-potential.json'));old=read(ROOT/'reports/running-speed-prior-20261027/old-paired-results.json')
    original={r['id']:r['candidate']for r in old};ev={};parity=[];timing=[]
    with np.load(ROOT/'reports/four-parameter-tuning-20261026/inputs.npz')as a:
        oldobj=a['object_support'][list(a['gates']).index(35)]
    for i,r in enumerate(rows):
        label=r['raw_label'];high=label!='other'and r['score']>=.925
        e=dict(support=False,reason='below_candidate_or_other');base=label if high else 'other'
        if label!='other'and r['score']>.65:
            path=OUT/'features'/(r['id']+'.npz');assert path.exists(),r['id'];ss=samples(path)
            assert [s.frame_index for s in ss]==r['source_frames']
            start=time.perf_counter()
            if label=='wave':
                p=wave_evidence(ss,r['raw_luma'],3,35.)
                e=dict(support=p['applied']and p['supported'],reason=p['reason'],wave_counts=p['counts'])
                if high and p['applied']and not p['supported']:base='other'
            elif label=='chase':
                p=speed_evidence(ss,r['raw_luma'],.35,35.,True,'bypass')
                e=dict(support=p['reason']=='speed_supported',reason=p['reason'],speed=p['speed'],pairs=len(p['pairs']))
                if high and p['applied']and not p['supported']:base='other'
            elif label in ('drink','pick_up'):
                obj=OUT/'object-evidence'/(r['id']+'.json')
                bright=r['raw_luma']is not None and r['raw_luma']>=35
                need=bright and(r['id']in pot or(r['scope']=='new_multipeople'and high))
                if need:
                    assert obj.exists(),('missing object evidence',r['id']);data=read(obj);assert data['exact_features_verified']
                    chosen=list(dict.fromkeys(r['source_frames'][j]for j in (0,8,16,23,31)))
                    assert[x['frame']for x in data['actions'][label]]==chosen
                    p=filter_object_evidence(label,data['actions'][label],r['raw_luma'],.581,35.)
                    e=dict(support=p['applied']and p['label']==label,reason=p['reason'],object_support=p['evidence'])
                    if high and p['applied']and p['label']=='other':base='other'
                else:e=dict(support=False,reason='below_luma_gate_or_not_causal_continuation')
                if r['scope']!='new_multipeople'and high:
                    base=label if oldobj[i,0 if label=='drink'else 1]>=.581 else 'other'
            else:e=dict(support=True,reason='no_existing_prior_for_class')
            timing.append(dict(id=r['id'],ms=(time.perf_counter()-start)*1000,action=label))
        if r['id']in original:
            parity.append(dict(id=r['id'],reconstructed=base,original=original[r['id']]))
            assert base==original[r['id']],('baseline parity',r['id'],base,original[r['id']])
        r.update(baseline=base,support=e['support'],prior_reason=e['reason']);ev[r['id']]=e
    save(OUT/'prior-evidence.json',ev);save(OUT/'cpu-prior-timing.json',timing)
    save(OUT/'baseline-parity.json',dict(old_checked=len(parity),old_mismatches=sum(r['reconstructed']!=r['original']for r in parity)))
    return ev

def replay(rows):
    state={};results={};cost=[]
    order=sorted(rows,key=lambda r:(r['source_id'],r['track_id'],r['history_generation'],r['timestamp'],r['event_index']))
    for r in order:
        key=(r['source_id'],r['track_id'],r['history_generation'])
        if key not in state:state[key]=(Continuation(),RunningReview(2,.5),RunningReview(1,0.),RunningReview(3,1.))
        c,r2,r1,r3=state[key];start=time.perf_counter_ns()
        a,b,d,e=c.push(r),r2.push(r),r1.push(r),r3.push(r)
        combined=a if a['recovered']else b
        cost.append((time.perf_counter_ns()-start)/1e6)
        results[r['id']]=dict(continuation=a,running2=b,combined=combined,running1=d,running3=e)
    return results,cost

def summary(rows,output,method):
    accepted=[r for r in rows if output[r['id']][method]['recovered']]
    known=[r for r in accepted if r['truth']is not None]
    true=[r for r in known if r['truth']==r['raw_label']]
    other=[r for r in known if r['truth']=='other']
    wrong=[r for r in known if r['truth']not in ('other',r['raw_label'])]
    waits=[output[r['id']][method]['wait_s']for r in accepted]
    confirms=[dict(id=r['id'],delay=output[r['id']][method]['confirmation_delay_s'],truth=r['truth'])for r in accepted if output[r['id']][method].get('first_confirmation')]
    delays=[x['delay']for x in confirms]
    return dict(recovered_windows=len(accepted),true_recovered=len(true),other_false_positives=len(other),wrong_action=len(wrong),
        added_false_positives=len(other)+len(wrong),unlabelled=len(accepted)-len(known),
        true_actions=dict(Counter(r['raw_label']for r in true)),false_actions=dict(Counter(r['raw_label']for r in other+wrong)),
        added_window_wait_s=dict(mean=float(np.mean(waits)),p95=float(np.percentile(waits,95)),max=max(waits))if waits else None,
        first_confirmations=len(confirms),confirmation_delay_s=dict(mean=float(np.mean(delays)),median=float(np.median(delays)),min=min(delays),max=max(delays))if delays else None,
        confirmed_episodes=confirms)

def event_metrics(rows,output):
    groups=defaultdict(list)
    # New: exact user review IDs. Old: same truth / source / actor / generation,
    # consecutive labelled windows. These are temporal groups, not independent people.
    old_last={};serial=Counter()
    for r in sorted(rows,key=lambda r:(r['source_id'],r['track_id'],r['history_generation'],r['timestamp'])):
        if r['scope']=='new_multipeople':
            if r['truth']is not None:
                for rid in r['review_ids']:groups['review:'+rid].append(r)
            continue
        key=(r['source_id'],r['track_id'],r['history_generation']);last=old_last.get(key)
        if last is None or last['truth']!=r['truth']or r['timestamp']-last['timestamp']>.650001:serial[key]+=1
        old_last[key]=r
        if r['truth']is not None:groups['old:'+':'.join(map(str,key))+':'+str(serial[key])].append(r)
    result=[]
    for gid,rr in groups.items():
        truth=rr[0]['truth'];assert all(r['truth']==truth for r in rr)
        baseline=any(r['baseline']==truth for r in rr)if truth!='other'else any(r['baseline']!='other'for r in rr)
        for m in METHODS:
            adds=[r for r in rr if output[r['id']][m]['recovered']]
            correct=[r for r in adds if r['raw_label']==truth]
            new_correct=truth!='other'and not baseline and bool(correct)
            new_alarm=truth=='other'and not baseline and bool(adds)
            first=correct[0]if new_correct else adds[0]if new_alarm else None
            result.append(dict(group=gid,source_id=rr[0]['source_id'],scope=rr[0]['scope'],truth=truth,method=m,
                baseline_detected_or_alarm=baseline,additional_correct_windows=len(correct),additional_false_windows=sum(r['raw_label']!=truth for r in adds),
                recovered_previously_missed_group=new_correct,new_false_other_group=new_alarm,
                first_new_bin=None if first is None else first['bin'],first_new_id=None if first is None else first['id']))
    save(OUT/'event-groups.json',result)
    return {m:{scope:dict(positive_groups=sum(x['truth']!='other'for x in result if x['method']==m and x['scope']==scope),
        recovered_previously_missed_groups=sum(x['recovered_previously_missed_group']for x in result if x['method']==m and x['scope']==scope),
        new_false_other_groups=sum(x['new_false_other_group']for x in result if x['method']==m and x['scope']==scope))
        for scope in sorted({r['scope']for r in rows})}for m in METHODS}

def main():
    rows,probs=load();ev=evidence(rows);out,cost=replay(rows)
    roles=dict(old_train=[r for r in rows if r['scope']!='new_multipeople'and r['split']=='train'],
        old_val=[r for r in rows if r['scope']!='new_multipeople'and r['split']=='val'],
        old_test=[r for r in rows if r['scope']!='new_multipeople'and r['split']=='test'],
        new_feedback=[r for r in rows if r['scope']=='new_multipeople'],all=rows)
    report={}
    for role,rr in roles.items():
        report[role]=dict(total=len(rr),known=sum(r['truth']is not None for r in rr),true_actions=sum(r['truth']not in (None,'other')for r in rr),
            baseline_correct=sum(r['baseline']==r['truth']and r['truth']not in (None,'other')for r in rr),
            baseline_other_fp=sum(r['baseline']!='other'and r['truth']=='other'for r in rr),
            baseline_wrong_action=sum(r['truth']not in (None,'other')and r['baseline']not in ('other',r['truth'])for r in rr),
            bins=[dict(range=[lo,hi],candidates=sum(r['bin']==j for r in rr),known_candidates=sum(r['bin']==j and r['truth']is not None for r in rr),
                methods={m:summary([r for r in rr if r['bin']==j],out,m)for m in METHODS})for j,(lo,hi)in enumerate(BINS)],
            totals={m:summary(rr,out,m)for m in METHODS})
    save(OUT/'results.json',dict(protocol=read(OUT/'protocol.json'),roles=report,events=event_metrics(rows,out),decision_cost_ms=dict(mean=float(np.mean(cost)),p95=float(np.percentile(cost,95)),max=max(cost))))
    detailed=[]
    for r in rows:
        if r['bin']is None:continue
        v={k:r[k]for k in ('id','source_id','scope','split','track_id','history_generation','timestamp','score','raw_label','truth','baseline','bin','support','prior_reason')}
        v['review_ids']=';'.join(r['review_ids']);v['speed']=ev[r['id']].get('speed')
        for m in METHODS:v[m]=out[r['id']][m]['recovered'];v[m+'_wait_s']=out[r['id']][m]['wait_s']
        detailed.append(v)
    with(OUT/'window-results.csv').open('w')as f:
        wr=csv.DictWriter(f,fieldnames=list(detailed[0]));wr.writeheader();wr.writerows(detailed)
    save(OUT/'decisions.json',out)
    save(OUT/'replay-rows.json',[{k:v for k,v in r.items()if k not in ('fused_probs','source_frames','top_score','top_label')}for r in rows])
    print(json.dumps({role:{m:{k:v for k,v in res.items()if k in ('true_recovered','added_false_positives','unlabelled','confirmation_delay_s')}for m,res in obj['totals'].items()}for role,obj in report.items()},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
