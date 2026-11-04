"""Calibration evidence only, with explicit missing-pose validation amendment."""
import json
from types import SimpleNamespace as S
from common import OUT, save, digest
from running_prior import speed_evidence


def pass_rule(e,tau,policy):
    if tau==0 or e['reason']in ('below_luma_gate','unknown_raw_luma','pose_branch_disabled'):return True
    if e['speed'] is None:return policy=='bypass'
    return e['speed']>=tau


def main():
    assert not (OUT/'selection.json').exists(), 'Selection is locked; do not retune against test'
    assert (OUT/'frontend-verification.json').exists()
    old=json.loads((OUT/'old-calibration-evidence.json').read_text())
    labels={r['frame']:r for r in json.loads((OUT/'review/labels.json').read_text())['labels']}
    new=[]
    for item in json.loads((OUT/'long-running-evidence.json').read_text()):
        event=item['event'];gt=labels[event['frame_index']]
        new.append(dict(**gt,baseline='chase',evidence=speed_evidence([S(**s)for s in item['samples']],event['raw_luma'],0)))
    save(OUT/'long-calibration-evidence.json',new)
    val=[r for r in old if r['split']=='val'];table=[]
    for policy in ['reject','bypass']:
        for tau in json.loads((OUT/'protocol.json').read_text())['grid']:
            m=dict(threshold=tau,missing_policy=policy)
            for name,rows in [('val',val),('long',new),('train',[r for r in old if r['split']=='train'])]:
                true=[r for r in rows if r['truth']=='chase'and r['baseline']=='chase']
                false=[r for r in rows if r['truth']is not None and r['truth']!='chase'and r['baseline']=='chase']
                m[name]=dict(true_baseline=len(true),true_kept=sum(pass_rule(r['evidence'],tau,policy)for r in true),
                    false_baseline=len(false),false_kept=sum(pass_rule(r['evidence'],tau,policy)for r in false),
                    unknown_baseline=sum(r['truth']is None and r['baseline']=='chase'for r in rows))
            table.append(m)
    eligible=[r for r in table if r['threshold']>0 and r['missing_policy']=='bypass'and
        r['val']['true_kept']>=.95*r['val']['true_baseline']and r['long']['true_kept']==r['long']['true_baseline']]
    choice=min(eligible,key=lambda r:(r['val']['false_kept']+r['long']['false_kept'],r['threshold']))
    save(OUT/'protocol-amendment.json',dict(before_test=True,
        reason='Strict missing-pose reject cannot preserve all reviewed child triggers at any positive threshold: frame613 has zero reliable pairs. Candidate uses fail-open for missing pose; strict variant retained in sweep. Geometry formula unchanged.',
        user_question='Asked optional missing-pose preference; pending at amendment. Proceed with explicitly stated conservative bypass assumption.',
        original_protocol_sha256=digest(OUT/'protocol.json'),current_rule_sha256=digest(OUT/'implementation/running_prior.py')))
    save(OUT/'threshold-sweep.json',table)
    save(OUT/'selection.json',dict(threshold=choice['threshold'],missing_policy='bypass',calibration=choice,
        selection='Historical val and user-labeled long triggers; preserve >=95% baseline val TP and all child TP, maximize FP removed, ties smallest threshold.',test_opened=False))
    print(json.dumps(choice,indent=2))

if __name__=='__main__':main()
