"""Audit tensors and construct gate-specific features using production math."""
from collections import Counter
from types import SimpleNamespace
import sys
from common import *
sys.path.insert(0, str(OP))
import numpy as np
from darkpipe.stages.skeleton_aux import normalize_window, completeness
from darkpipe.person.wave_prior import wave_evidence
from darkpipe.person.running_prior import speed_evidence
from extract import write_npz


def legacy_features(row, legacy):
    r=legacy[row['legacy_record_id']]
    with np.load(row['path']) as a:pose=a['pose']
    manifest=read(OLD/'data'/r['video_id']/'manifest.json')
    boxes=[];binding=[]
    for fi in r['source_frames']:
        tracks=manifest['frames'][fi]['tracks']
        box=next(t['bbox'] for t in tracks if t['track_id']==r['track_id'])
        boxes.append(box)
        from darkpipe.person.core import iou
        binding.append(not any(iou(np.array(box),np.array(t['bbox']))>.4 for t in tracks if t['track_id']!=r['track_id']))
    return dict(pose=pose,boxes=np.array(boxes),binding=np.array(binding),shape=np.array([404,720]),
        lumas=np.array([np.nan if x is None else x for x in r['lumas']]),
        source_frames=np.array(r['source_frames']),timestamps=np.array(r['timestamps']))


def main():
    for scope in ('old','new'):
        for shard in range(3):assert (OUT/f'extract-{scope}-{shard}-complete.json').exists()
    records=read(OUT/'dataset-plan.json')['records']
    legacy={r['id']:r for r in read(OLD/'training-data.json')['records']}
    n=len(records);sk=np.zeros((len(GATES),n,3,32,17),np.float32)
    valid=np.zeros((len(GATES),n),bool);wave=np.zeros((len(GATES),n),np.int16);speed=np.full((len(GATES),n),np.nan,np.float32)
    rgb_hashes=[];covered=Counter()
    for i,r in enumerate(records):
        feature=OUT/'features'/(r['id']+'.npz')
        if not feature.exists():
            assert r.get('legacy_record_id')
            write_npz(feature,legacy_features(r,legacy))
        with np.load(feature) as a:data={k:a[k] for k in a.files}
        with np.load(r['path']) as a:rgb=a['rgb']
        assert rgb.shape==(32,224,400,3) and rgb.dtype==np.uint8
        digest=hashlib.sha256(rgb.tobytes()).hexdigest()
        if 'crop_sha256' in r:assert digest==r['crop_sha256']
        assert data['source_frames'].tolist()==r['source_frames']
        assert data['pose'].shape==(32,17,3) and np.isfinite(data['pose']).all()
        assert np.allclose(data['timestamps'],np.array(r['source_frames'])/r['fps'],atol=1e-6)
        r.update(feature_path=str(feature),crop_sha256=digest)
        rgb_hashes.append(digest)
        lumas=data['lumas'];raw=None if np.isnan(lumas).all() else float(np.nanmean(lumas))
        r['raw_luma']=raw
        for gi,g in enumerate(GATES):
            pose=data['pose'].copy();pose[np.isfinite(lumas)&(lumas<g)]=0
            normalized,presence=normalize_window(pose[:,None])
            visible=(pose[:,:,2]>.3).sum(1)>=10
            eligible=bool(presence[0]>=.5 and completeness(pose[:,None])[0]>=.5 and visible.mean()>=.5
                and pose[-1,:,2].mean()>=.5 and visible[-1] and (raw is None or raw>=g))
            sk[gi,i]=normalized[0];valid[gi,i]=eligible
            samples=[SimpleNamespace(pose=p,frame_index=int(fi),timestamp=float(t),
                luma=None if np.isnan(l) else float(l),person_box=b,frame_shape=tuple(data['shape']))
                for p,fi,t,l,b in zip(pose,data['source_frames'],data['timestamps'],lumas,data['boxes'])]
            w=wave_evidence(samples,raw,3,g)
            wave[gi,i]=max(w['counts']) if w['applied'] else 32767
            s=speed_evidence(samples,raw,.35,g)
            if s['speed'] is not None:speed[gi,i]=s['speed']
        covered[(r['scope'],r['split'],r['label_kind'])]+=1
        if i%250==0:print('ASSEMBLE',i,n,flush=True)
    np.savez_compressed(OUT/'skeleton-inputs.npz',features=sk,valid=valid,wave_counts=wave,speed=speed,gates=np.array(GATES))
    save(OUT/'dataset.json',dict(records=records,ready=True,protocol_sha256=sha(OUT/'protocol.json'),
        audit=dict(rgb_shape_dtype_hashes=True,pose_shapes_finite=True,same_source_frames=True,
            xclip_used=False,weak_labels_never_test_truth=True,all_old_source_splits_preserved=True),
        coverage=[dict(scope=k[0],split=k[1],kind=k[2],n=v) for k,v in covered.items()]))
    print('READY',n,dict(Counter(r['scope'] for r in records)),flush=True)


if __name__=='__main__':main()
