"""Cross-check cumulative sweep and chosen decisions against unchanged policies."""
from tune4 import Joint
from common import *
from darkpipe.stages.recognize import reject_probs


def main():
    j=Joint();rng=np.random.default_rng(26);checks=0;object_checks=0
    ts=[.70,.75,.85,.905,.94,.99];os=[.01,.1,.35,.358,.9]
    for g in j.gates:
        for a in [.5,.7,.95,.99]:
            r=j.surfaces(a,g,ts,os)
            for _ in range(3):
                ti=int(rng.integers(len(ts)));oi=int(rng.integers(len(os)))
                brute=j.calibration(dict(alpha=a,gate=g,action=ts[ti],object=os[oi]))
                for k,v in r.items():
                    if k!='feasible':assert np.isclose(v[ti,oi],brute[k],atol=1e-12,rtol=0),(g,a,k,v[ti,oi],brute[k])
                checks+=1
    settings=json.loads((OUT/'predictions.json').read_text())['settings']
    policy_checks=0
    for name,s in settings.items():
        gi=j.gates.index(s['gate']);top,score,support=j.values(s['alpha'],s['gate']);pred=j.predict(s)
        for i in range(len(j.rows)):
            raw=j.rows[i]['raw_luma']
            sk=j.sk[gi,i] if not np.array_equal(j.sk[gi,i],np.eye(10,dtype=np.float32)[9]) else None
            d=fuse_person(j.rgb[i],sk,[raw],alpha=s['alpha'],threshold=s['action'],lum_gate=s['gate'])
            label=BEHAVIORS.index(d['label'])
            if support[i]<s['object']:label=j.o
            assert label==pred[i],(name,j.rows[i]['id'])
            policy_checks+=1
    by={r['id']:i for i,r in enumerate(j.rows)}
    for path in (OUT/'evidence').glob('*.json'):
        for e in json.loads(path.read_text())['records']:
            i=by[e['window_id']];ci=['drink','pick_up'].index(e['label'])
            for gi,g in enumerate(j.gates):
                for conf in [.01,.35,.358,.9]:
                    actual=filter_object_evidence(e['label'],e['samples'],e['raw_luma'],conf,g)['label']
                    expected=e['label'] if j.support[gi,i,ci]>=conf else 'other'
                    assert actual==expected,(e['id'],g,conf);object_checks+=1
    p=json.loads((OUT/'protocol.json').read_text());models={}
    for k,m in p['models'].items():
        models[k]=digest(ROOT/m['path']);assert models[k]==m['sha256']
    result=dict(passed=True,baseline_windows_reproduced=len(j.rows),brute_force_grid_checks=checks,
        deployment_fusion_policy_checks=policy_checks,object_policy_checks=object_checks,model_hashes=models,
        gate_input_coverage='Per-frame new luma audited; old per-gate trackers and skeleton sequences replayed.',
        historical_rgb_batch_note='When old15 and old30 shared crop SHA but had tiny batch-dependent probability differences, old30 probabilities were retained consistently.')
    save(OUT/'verification.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
