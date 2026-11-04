"""Plan actual object checks only where an anchor could causally extend an action."""
from collections import defaultdict,Counter
from common import *

def potential(rows):
    anchors={};prev={};result=[]
    for r in sorted(rows,key=lambda r:(r['source_id'],r['track_id'],r['history_generation'],r['timestamp'],r['event_index'])):
        key=(r['source_id'],r['track_id'],r['history_generation']);t=r['timestamp'];label=r['raw_label']
        if key in prev and t-prev[key]>.650001:anchors.pop(key,None)
        prev[key]=t
        if label=='other' or r['score']<=.65:anchors.pop(key,None);continue
        if r['score']>=.925:anchors[key]=(t,label);continue
        a=anchors.get(key)
        if a is None:continue
        if label!=a[1] or t-a[0]>1.200001:anchors.pop(key,None);continue
        result.append(r['id'])
    return set(result)

def main():
    rows,_=load();pot=potential(rows)
    wanted=[r for r in rows if r['raw_label']in ('drink','pick_up')and r['raw_luma']is not None and r['raw_luma']>=35
        and(r['id']in pot or(r['scope']=='new_multipeople'and r['score']>=.925))]
    save(OUT/'continuation-potential.json',sorted(pot))
    dataset=read(ROOT/'reports/manypeople-finetune-20261029/dataset.json')['records'];features={r['id']:r['feature_path']for r in dataset}
    wanted_ids={r['id']for r in wanted};cached={};provenance={}
    for folder in ['manypeople-finetune-20261029','manypeople-all85-finetune-20261030']:
        for path in (ROOT/'reports'/folder/'object-evidence').glob('*.json'):
            for e in read(path)['records']:
                rid=e['id']
                if rid not in wanted_ids:continue
                local=OUT/'features'/(rid+'.npz')
                if not local.exists():continue
                with np.load(local)as a,np.load(features[rid])as b:
                    assert all(np.array_equal(a[k],b[k],equal_nan=True)for k in ('pose','boxes','binding','source_frames','lumas','shape')),(rid,'feature cache mismatch')
                    chosen=list(dict.fromkeys(int(a['source_frames'][j])for j in (0,8,16,23,31)))
                    assert all([x['frame']for x in e['actions'][lab]]==chosen for lab in ('drink','pick_up'))
                cached[rid]=e;provenance[rid]=str(path.relative_to(ROOT))
    for rid,e in cached.items():save(OUT/'object-evidence'/(rid+'.json'),dict(**e,provenance=provenance[rid],exact_features_verified=True))
    pending=[r for r in wanted if r['id']not in cached]
    save(OUT/'object-needed.json',pending)
    save(OUT/'object-plan.json',dict(potential_continuation=len(pot),needed=len(wanted),reused=len(cached),new_inference=len(pending),sources=dict(Counter(r['source_id']for r in pending))))
    print(read(OUT/'object-plan.json'))

if __name__=='__main__':main()
