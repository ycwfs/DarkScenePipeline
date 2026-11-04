from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from types import SimpleNamespace

OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parents[1]
OP=ROOT/'platform/dark-demo-v2'
sys.path.insert(0,str(OP))
LABELS=['wave','throw','chase','fall','fight','talk','drink','pick_up','shake_hands','other']
BINS=[(.65,.75),(.75,.85),(.85,.925)]

def read(p):return json.loads(Path(p).read_text())
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');q.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def load():
    rows=read(ROOT/'reports/threshold-tuning-20261026/windows.json')
    with np.load(ROOT/'reports/four-parameter-tuning-20261026/inputs.npz')as z:
        g=list(z['gates']).index(35);p=.972*z['rgb']+.028*z['skeleton'][g]
        m=z['luma']<35;p[m]=z['rgb'][m]
    d={r['id']:r for r in read(ROOT/'reports/manypeople-finetune-20261029/dataset.json')['records']if r['scope']=='new'}
    for i,r in enumerate(rows):
        r['raw_label']=LABELS[int(p[i].argmax())];r['score']=float(p[i].max());r['review_ids']=[]
        if r['scope']=='new_multipeople':
            rec=d.get(r['id']);r['truth']=None
            if rec and rec['label_kind']=='human':
                assert r['source_frames']==rec['source_frames'] and r['track_id']==rec['track_id'] and r['history_generation']==rec['generation']
                r.update(truth=rec['label'],review_ids=rec['review_ids'],split=rec['split'])
        r['bin']=next((j for j,(lo,hi)in enumerate(BINS)if r['score']>.65 and lo<=r['score']<hi and r['raw_label']!='other'),None)
    return rows,p

def samples(path):
    with np.load(path)as a:
        return [SimpleNamespace(frame_index=int(n),timestamp=float(t),pose=p,luma=None if np.isnan(l)else float(l),
            person_box=b,binding_valid=bool(v),frame_shape=tuple(a['shape']))
            for n,t,p,l,b,v in zip(a['source_frames'],a['timestamps'],a['pose'],a['lumas'],a['boxes'],a['binding'])]

def write_feature(row,ss):
    assert [s.frame_index for s in ss]==row['source_frames']
    assert np.allclose(ss[-1].person_box,row['bbox'],atol=.02,rtol=0),row['id']
    path=OUT/'features'/(row['id']+'.npz');path.parent.mkdir(exist_ok=True)
    np.savez_compressed(path,pose=np.stack([s.pose for s in ss]),boxes=np.stack([s.person_box for s in ss]),
        lumas=np.array([np.nan if s.luma is None else s.luma for s in ss]),binding=np.array([s.binding_valid for s in ss]),
        shape=np.array(ss[-1].frame_shape),source_frames=np.array([s.frame_index for s in ss]),timestamps=np.array([s.timestamp for s in ss]))
