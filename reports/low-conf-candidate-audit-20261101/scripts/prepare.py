"""Exact geometry: historical sequential cache or deduplicated same-identity features."""
from collections import defaultdict,Counter
from types import SimpleNamespace as S
from common import *

def iou(a,b):
    a=np.asarray(a);b=np.asarray(b);x=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    return x/max(1e-6,np.maximum(0,a[2:]-a[:2]).prod()+np.maximum(0,b[2:]-b[:2]).prod()-x)

def shape(v):
    h,w=v['height'],v['width']
    if v['processed']:
        candidates=[x for x in range(h-180,h)if x%2==0 and x+max(48,round(.08*x))+(max(48,round(.08*x))&1)==h]
        assert len(candidates)==1;h=candidates[0]
    if max(h,w)>720:
        ratio=720/max(h,w);h,w=round(h*ratio)//2*2,round(w*ratio)//2*2
    return h,w

def geometry(v,rows):
    # Exact reconstruction from running-speed-prior-20261027/scripts/old_evidence.py,
    # with production binding_valid added for object ROI evidence.
    cache=read(ROOT/v['cache']);rp=ROOT/'reports/four-parameter-tuning-20261026/replay'/f'{v["id"]}.json'
    replay=read(rp)if rp.exists()else None
    frames=read(ROOT/'reports/luma-gate15-20261024/data'/v['id']/'result.json')['frames']if replay else cache['frames']
    tracks=replay['tracks']['35']if replay else [f['tracks']['30']for f in frames]
    events=replay['events']['35']if replay else cache['events']['30'];h,w=shape(v)
    for r in rows:
        e=events[r['event_index']];assert e['source_frames']==r['source_frames'] and e['track_id']==r['track_id']
        ss=[]
        for fi in e['source_frames']:
            m=frames[fi];actors=[t for t in tracks[fi]if t['track_id']==e['track_id']];assert len(actors)==1
            box=actors[0]['bbox'];found=[j for j,b in enumerate(m['boxes'])if np.allclose(b[:4],box,atol=1e-4,rtol=0)];assert len(found)==1
            pose=m['poses'][found[0]];p=np.zeros((17,3),np.float32)if pose is None else np.array(pose,np.float32);p[:,:2]/=[w,h]
            if m['luma']is not None and m['luma']<35:p[:]=0
            binding=not any(iou(box,t['bbox'])>.4 for t in tracks[fi]if t['track_id']!=e['track_id'])
            ss.append(S(frame_index=fi,timestamp=m['timestamp'],pose=p,luma=m['luma'],person_box=np.array(box),frame_shape=(h,w),binding_valid=binding))
        yield r,ss

def main():
    rows,_=load();wanted=[r for r in rows if r['raw_label']!='other' and r['score']>.65]
    sources={v['id']:v for v in read(ROOT/'reports/wave-pose-prior-20261027/protocol.json')['old_sources']if v['scope']!='new_multipeople'}
    grouped=defaultdict(list);provenance={};source_meta={}
    for r in wanted:
        if r['scope']!='new_multipeople':grouped[r['source_id']].append(r)
    for sid,rr in grouped.items():
        v=sources[sid];source_meta[sid]=dict(source=str(ROOT/v['source']),fps=v['fps'],processed=v['processed'])
        for r,ss in geometry(v,rr):write_feature(r,ss);provenance[r['id']]='exact_historical_sequential_cache'
    union=defaultdict(dict)
    for r in read(ROOT/'reports/manypeople-finetune-20261029/dataset.json')['records']:
        if r['scope']!='new':continue
        source_meta[r['source_id']]=dict(source=r['source'],fps=r['fps'],processed=False)
        key=(r['source_id'],r['track_id'],r['generation'])
        for s in samples(r['feature_path']):
            prev=union[key].get(s.frame_index)
            if prev is not None:
                assert np.array_equal(prev.pose,s.pose) and np.array_equal(prev.person_box,s.person_box)
                assert prev.binding_valid==s.binding_valid and prev.timestamp==s.timestamp
            else:union[key][s.frame_index]=s
    missing=[]
    for r in wanted:
        if r['scope']!='new_multipeople':continue
        frames=union[(r['source_id'],r['track_id'],r['history_generation'])]
        if all(fi in frames for fi in r['source_frames']):
            write_feature(r,[frames[fi]for fi in r['source_frames']]);provenance[r['id']]='exact_identity_frame_union'
        else:missing.append(r)
    save(OUT/'frontend-needed.json',missing);save(OUT/'source-meta.json',source_meta);save(OUT/'feature-provenance.json',provenance)
    save(OUT/'input-audit.json',dict(windows=len(rows),known=sum(r['truth']is not None for r in rows),new_human=sum(r['scope']=='new_multipeople'and r['truth']is not None for r in rows),candidates=len(wanted),features=len(provenance),frontend_missing=len(missing),by_source=dict(Counter(r['source_id']for r in missing))))
    print(read(OUT/'input-audit.json'),flush=True)
    paths=[ROOT/'reports/four-parameter-tuning-20261026/inputs.npz',ROOT/'reports/threshold-tuning-20261026/windows.json',ROOT/'reports/manypeople-finetune-20261029/dataset.json',OUT/'protocol.json']
    paths+=list((OP/'darkpipe/person').glob('*.py'))+list((OP/'ckpts').glob('*'))
    save(OUT/'input-hashes.json',{str(p.relative_to(ROOT)):sha(p)for p in paths if p.is_file()})

if __name__=='__main__':main()
