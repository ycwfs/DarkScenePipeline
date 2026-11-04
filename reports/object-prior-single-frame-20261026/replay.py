"""Compare frozen consecutive evidence against one sampled-frame support.

This is an exploratory replay; it does not detect on additional video frames,
retrain, tune the detector, or change the released operator.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
BASE = OUT.parent/'object-prior-finetune-20261025'
sys.path.insert(0, str(BASE/'scripts'))
from prior_rules import PriorConfig, decide, observations

POLICIES = ('baseline', 'conservative', 'strict', 'single_interaction', 'single_roi')


def save(name: str, value: object) -> None:
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def csv_rows(name: str, rows: list) -> None:
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def single_support(record: dict) -> dict:
    label=record['baseline']
    if label not in ('drink','pick_up'):
        return dict(single_interaction=label,single_roi=label,roi_evidence=[],interaction_frames=[])
    cfg=PriorConfig()
    compatible=(0,1) if label=='drink' else (0,1,2,3,4,5)
    evidence=[];interaction=[]
    for s in record['samples']:
        if not s['binding_valid']:
            continue
        # All valid cached samples retain the actual native pixel crop rectangle.
        rect=np.asarray(s['person_roi'])
        for d in s['detections']:
            center=(np.asarray(d[:2])+d[2:4])/2
            if int(d[5]) in compatible and d[4]>=cfg.support_conf and (
                (center>=rect[:2]).all() and (center<=rect[2:]).all()):
                evidence.append(dict(frame=s['frame'],timestamp=s['timestamp'],class_id=int(d[5]),
                    confidence=d[4],bbox=d[:4],person_roi=s['person_roi']))
        field='drink' if label=='drink' else 'pickup'
        if any(o[field] for o in observations(s,cfg)):
            interaction.append(s['frame'])
    # Relax positive evidence only. Never reject a person based on one phone frame.
    veto=record['conservative']!=label
    return dict(single_interaction=label if interaction and not veto else 'other',
        single_roi=label if evidence and not veto else 'other',
        roi_evidence=evidence,interaction_frames=interaction)


def bootstrap(rows: list, policy: str, reference: str) -> dict:
    groups=defaultdict(lambda:np.zeros(3))
    for r in rows:
        truth=r['truth']
        groups[r['group']]+=[r[reference]==truth,r[policy]==truth,1]
    a=np.asarray(list(groups.values()))
    rng=np.random.default_rng(25)
    draws=a[rng.integers(len(a),size=(5000,len(a)))].sum(1)
    delta=(draws[:,1]-draws[:,0])/draws[:,2]
    total=a.sum(0)
    return dict(groups=len(a),windows=int(total[2]),reference_correct=int(total[0]),policy_correct=int(total[1]),
        difference=float((total[1]-total[0])/total[2]),difference_ci95=np.quantile(delta,[.025,.975]).tolist())


def main() -> None:
    rows=[];sources=[]
    for path in sorted((BASE/'action-evaluation').glob('*.json')):
        data=json.loads(path.read_text());job=data['job']
        sources.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        for r in data['records']:
            old=decide(r['baseline'],r['samples'])
            assert all(old[k]==r[k] for k in ('conservative','strict','reason','evidence'))
            result=single_support(r)
            # Original strict support must survive both weaker positive gates.
            if r['strict']!='other':
                assert result['single_interaction']==r['strict'] and result['single_roi']==r['strict']
            if result['single_interaction']!='other':assert result['single_roi']==result['single_interaction']
            rows.append(dict(source_id=job['id'],split=job['split'],group=job['group'],
                event_index=r['event_index'],track_id=r['track_id'],history_generation=r['history_generation'],
                timestamp=r['timestamp'],truth=r['truth'],baseline=r['baseline'],
                conservative=r['conservative'],strict=r['strict'],**result))
    assert len(rows)==619
    summary=[]
    for split in ('val','test'):
        for truth in ('drink','pick_up','other'):
            selected=[r for r in rows if r['split']==split and r['truth']==truth]
            summary.append(dict(split=split,truth=truth,windows=len(selected),
                groups=len({r['group']for r in selected}),**{
                    p:sum(r[p]==truth for r in selected) for p in POLICIES}))
    feedback=[]
    for source_id in [f'new-{i:02d}' for i in range(1,10)]+['all']:
        for action in ('drink','pick_up'):
            selected=[r for r in rows if r['split']=='feedback' and r['baseline']==action and
                      (source_id=='all' or r['source_id']==source_id)]
            feedback.append(dict(source_id=source_id,action=action,windows=len(selected),**{
                p:sum(r[p]==action for r in selected) for p in POLICIES}))
    stats={}
    for split in ('val','test'):
        for truth in ('drink','pick_up','other'):
            selected=[r for r in rows if r['split']==split and r['truth']==truth]
            for p in ('single_interaction','single_roi'):
                for ref in ('baseline','strict'):
                    stats[f'{split}:{truth}:{p}-minus-{ref}']=bootstrap(selected,p,ref)
    save('results.json',dict(rows=rows,summary=summary,feedback=feedback,source_hashes=sources,
        rules_sha256=hashlib.sha256((BASE/'scripts/prior_rules.py').read_bytes()).hexdigest(),
        protocol_sha256=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()))
    save('statistics.json',dict(contrasts=stats,runs=1,seed=25,bootstrap_repetitions=5000,
        method='paired source-group percentile bootstrap; same-scene auxiliary GT; descriptive only',
        significance='No p-values: too few independent scenes and post-result exploratory comparison',
        feedback_truth='unknown; no accuracy or corrected-error metric'))
    csv_rows('old-correct-counts.csv',summary);csv_rows('feedback-retained-counts.csv',feedback)
    csv_rows('window-decisions.csv',[{k:v for k,v in r.items()if k not in ('roi_evidence','interaction_frames')}for r in rows])
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    palette=['#87919a','#cf995b','#4f8ca8','#087f80']
    names=['baseline','strict','single_interaction','single_roi']
    labels=['No prior','Consecutive','One interaction frame','One ROI frame']
    fig,ax=plt.subplots(figsize=(10,4.5))
    test=[r for r in summary if r['split']=='test']
    for j,(p,leg) in enumerate(zip(names,labels)):
        vals=[r[p]for r in test];xs=np.arange(3)+(j-1.5)*.19
        ax.bar(xs,vals,width=.18,label=leg,color=palette[j])
        for x,v,r in zip(xs,vals,test):ax.text(x,v+.1,f'{v}/{r["windows"]}',ha='center',fontsize=9)
    ax.set_xticks(range(3),['True drink','True pick-up','Known other'])
    ax.set_ylabel('Correct selected person windows');ax.set_ylim(0,14)
    ax.set_title('Single-frame support reduces missed actions in the old test subset')
    ax.legend(ncol=2);fig.tight_layout()
    for suffix in ('png','pdf'):fig.savefig(folder/f'01-true-action-retention.{suffix}',dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.5))
    selected=[r for r in feedback if r['source_id']=='all']
    for j,(p,leg) in enumerate(zip(names,labels)):
        vals=[r[p]for r in selected];xs=np.arange(2)+(j-1.5)*.19
        ax.bar(xs,vals,width=.18,label=leg,color=palette[j])
        for x,v,r in zip(xs,vals,selected):ax.text(x,v+5,f'{v}/{r["windows"]}',ha='center',fontsize=9)
    ax.set_xticks(range(2),['Predicted drink','Predicted pick-up'])
    ax.set_ylabel('Retained feedback windows (truth unknown)');ax.set_ylim(0,660)
    ax.set_title('New footage: more retained predictions do not mean higher accuracy')
    ax.legend(ncol=2);fig.tight_layout()
    for suffix in ('png','pdf'):fig.savefig(folder/f'02-feedback-retained.{suffix}',dpi=180)
    plt.close(fig)
    print(json.dumps(dict(old=summary,feedback=selected),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
