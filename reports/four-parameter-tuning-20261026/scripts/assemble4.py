"""Align exact person windows and gate-aware probabilities/support for joint replay."""
from collections import Counter,defaultdict
from common import *


def main():
    protocol=json.loads((OUT/'protocol.json').read_text());fingerprint=digest(OUT/'protocol.json')
    audit=json.loads((OUT/'new-luma-audit.json').read_text())
    minimum=min(s['minimum'] for s in audit['sources'])
    gates=[g for g in GATES if g<=minimum]
    assert 30 in gates
    amendment=dict(reason='Actual raw-frame audit found frames below75 despite all cached window means above75. Exclude gates that change new-scene pose inputs until a separate full new-scene replay; old75 retained as diagnostic. Amendment uses input brightness only, before any joint metric.',
                   minimum_new_frame_luma=minimum,active_gates=gates,
                   excluded_gates=[g for g in GATES if g not in gates],
                   all_active_new_frame_masks_identical=True,protocol_sha256=fingerprint)
    save(OUT/'gate-coverage-amendment.json',amendment)
    rows=json.loads((PREV/'windows.json').read_text());idx={r['id']:i for i,r in enumerate(rows)}
    source_indices=defaultdict(list)
    for i,r in enumerate(rows):source_indices[r['source_id']].append(i)
    rgb=np.zeros((len(rows),10),np.float32)
    sk=np.zeros((len(gates),len(rows),10),np.float32);sk[:,:,9]=1
    valid=np.zeros((len(gates),len(rows)),bool)
    support=np.full((len(gates),len(rows),2),np.inf)
    covered=np.zeros_like(support,bool);stats=Counter()
    original={r['video_id']+'|'+r['crop_sha256']+'|'+str(r['source_frames']):r for r in json.loads((CACHE/'evaluated-windows.json').read_text())}
    for v in protocol['sources']:
        d=json.loads((ROOT/v['cache']).read_text());new=v['scope']=='new_multipeople'
        events=d if new else d['events']['30']
        inds=source_indices[v['id']];assert len(inds)==len(events)
        for i,e in zip(inds,events):rgb[i]=e['rgb_probs']
        rp=OUT/'replay'/f'{v["id"]}.json';replay=json.loads(rp.read_text()) if rp.exists() else None
        if replay:
            assert replay['protocol_sha256']==fingerprint
            for k,n in replay['stats'].items():stats[k]+=n
        for gi,g in enumerate(gates):
            selected=replay['events'][str(g)] if replay else events
            assert len(selected)==len(inds),(v['id'],g,'different window count')
            for k,(i,e) in enumerate(zip(inds,selected)):
                if replay:
                    assert e['baseline_id']==rows[i]['id'],(v['id'],g,k,'unmatched crop')
                    assert e['truth']==rows[i]['truth']
                    assert e['source_frames']==rows[i]['source_frames']
                    assert np.array_equal(rgb[i],e['rgb_probs'])
                if e['skeleton_probs'] is not None:sk[gi,i]=e['skeleton_probs'];valid[gi,i]=True
        evidence=json.loads((OUT/'evidence'/f'{v["id"]}.json').read_text());assert evidence['protocol_sha256']==fingerprint
        for k,n in evidence['stats'].items():stats['object_'+k]+=n
        for e in evidence['records']:
            i=idx[e['window_id']];ci=['drink','pick_up'].index(e['label'])
            for gi,g in enumerate(gates):
                decision=filter_object_evidence(e['label'],e['samples'],e['raw_luma'],.001,g)
                if decision['applied']:support[gi,i,ci]=max((b['confidence'] for b in decision['evidence']),default=0.)
                covered[gi,i,ci]=True
    luma=np.array([np.nan if r['raw_luma'] is None else r['raw_luma'] for r in rows])
    for gi,g in enumerate(gates):
        for alpha in [.5,.99]:
            p=alpha*rgb+(1-alpha)*sk[gi];p[luma<g]=rgb[luma<g]
            for ci,c in enumerate(['drink','pick_up']):
                reachable=p[:,BEHAVIORS.index(c)]>=.7
                assert np.all(covered[gi,:,ci][reachable]),(g,alpha,c,'missing candidate evidence')
                stats['candidate_endpoint_checks']+=int(reachable.sum())
    np.savez_compressed(OUT/'inputs.npz',rgb=rgb,skeleton=sk,pose_valid=valid,object_support=support,luma=luma,gates=np.array(gates))
    save(OUT/'coverage.json',dict(stats=dict(stats),windows=len(rows),sources=len(protocol['sources']),gates=gates,
         baseline_source_sha256=digest(PREV/'windows.json'),inputs_sha256=digest(OUT/'inputs.npz'),
         exact_windows_shared=True,all_active_new_masks_checked=True))
    print(json.dumps(dict(stats),indent=2),flush=True)


if __name__=='__main__':main()
