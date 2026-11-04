from collections import defaultdict,Counter
from common import *

def candidate_rows(data,scopes):
    return [r for r in data if r['scope'] in scopes and r['split'] in ('val','test')
        and r['raw_label'] in ('drink','pick_up') and r['baseline']=='other'
        and r['bin'] is not None]

def main():
    data=rows();scopes={'onsite','online_supplement','new_multipeople'}
    candidates=candidate_rows(data,scopes);groups=defaultdict(list)
    for r in candidates:groups[(r['source_id'],r['track_id'],r['history_generation'],r['raw_label'])].append(r)
    runs=[];needed=set()
    for key,rr in groups.items():
        rr.sort(key=lambda r:r['timestamp']);run=[]
        for r in rr:
            if run and r['timestamp']-run[-1]['timestamp']>.650001:
                if len(run)>=2 and run[-1]['timestamp']-run[0]['timestamp']>=.5:
                    runs.append(run);needed.update(x['id']for x in run)
                run=[]
            run.append(r)
        if len(run)>=2 and run[-1]['timestamp']-run[0]['timestamp']>=.5:
            runs.append(run);needed.update(x['id']for x in run)
    old=read(AUDIT/'object-evidence') if False else None
    prior_dir=AUDIT/'object-evidence';cached={}
    for p in prior_dir.glob('*.json'):
        r=read(p)
        if r.get('exact_features_verified'):
            cached[p.stem]=r
    exact_existing=[];missing=[]
    for rid in sorted(needed):
        if rid in cached:
            save(OUT/'object-evidence'/(rid+'.json'),cached[rid]);exact_existing.append(rid)
        else:missing.append(next(r for r in data if r['id']==rid))
    save(OUT/'object-needed.json',missing)
    save(OUT/'candidate-plan.json',dict(heldout_candidate_windows=len(candidates),
        repeated_candidate_runs=len(runs),object_sequence_windows=len(needed),
        exact_object_evidence_reused=len(exact_existing),object_inference_needed=len(missing),
        by_action=dict(Counter(r['raw_label']for r in missing)),by_source=dict(Counter(r['source_id']for r in missing)),
        runs=[dict(source_id=s[0]['source_id'],track_id=s[0]['track_id'],generation=s[0]['history_generation'],
            action=s[0]['raw_label'],ids=[r['id']for r in s],times=[r['timestamp']for r in s])for s in runs]))
    save(OUT/'source-meta.json',read(AUDIT/'source-meta.json'))
    hash_files=[AUDIT/'input-hashes.json',AUDIT/'decisions.json',AUDIT/'prior-evidence.json',OUT/'protocol.json',
        OP/'darkpipe/person/object_runtime.py',OP/'darkpipe/person/object_prior.py',OP/'ckpts/yolo26m_objects_20261026.onnx']
    save(OUT/'input-hashes.json',{str(p.relative_to(ROOT)):sha(p)for p in hash_files})
    print(json.dumps(read(OUT/'candidate-plan.json'),ensure_ascii=False))

if __name__=='__main__':main()
