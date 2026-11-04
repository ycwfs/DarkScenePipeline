"""Compare vector calibration to actual deployed fusion/three-prior functions."""
import sys
from types import SimpleNamespace
from common import *
sys.path.insert(0,str(OP))
import numpy as np
from darkpipe.person.core import fuse_person
from darkpipe.person.object_prior import filter_object_evidence
from darkpipe.person.wave_prior import apply_wave_prior
from darkpipe.person.running_prior import apply_running_prior


def main():
    rows=read(OUT/'dataset.json')['records'];results=read(OUT/'results.json')['comparisons']
    final=np.load(OUT/'final-predictions.npz');features=np.load(OUT/'skeleton-inputs.npz')
    inputs={(b,w):np.load(OUT/'predictions'/f'{b}-{w}.npz')['probs']for b in ('rgb','skeleton')for w in ('baseline','candidate','selected')}
    evidence={e['id']:e for p in (OUT/'object-evidence').glob('*.json')for e in read(p)['records']}
    checked=0;counts={}
    for name in ['baseline','trained_same_defaults','selected']:
        c=results[name]['config'];gi=GATES.index(c['gate']);n=0
        for i,r in enumerate(rows):
            if r['id']not in evidence:continue
            with np.load(r['feature_path'])as a:f={k:a[k]for k in a.files}
            pose=f['pose'].copy();pose[np.isfinite(f['lumas'])&(f['lumas']<c['gate'])]=0
            samples=tuple(SimpleNamespace(pose=p,frame_index=int(fi),timestamp=float(t),luma=None if np.isnan(l)else float(l),
                person_box=b,frame_shape=tuple(f['shape']))for p,fi,t,l,b in zip(pose,f['source_frames'],f['timestamps'],f['lumas'],f['boxes']))
            sk=inputs['skeleton',c['skeleton']][gi,i]if features['valid'][gi,i]else None
            row=fuse_person(inputs['rgb',c['rgb']][i],sk,[s.luma for s in samples],alpha=c['alpha'],threshold=c['action'],lum_gate=c['gate'])
            label=row['label']
            obj=filter_object_evidence(label,evidence[r['id']]['actions'].get(label,[]),row['raw_luma'],c['object'],c['gate'])
            if obj['label']!=label:row['label']='other';row['confidence']=1.;row['decision_scores']=[float(x=='other')for x in LABELS]
            task=SimpleNamespace(window=samples)
            apply_wave_prior(task,row,3,c['gate'],pose_enabled=c['alpha']<.999)
            apply_running_prior(task,row,.35,c['gate'],pose_enabled=c['alpha']<.999)
            assert row['label']==LABELS[final[name][i]],(name,r['id'],row['label'],LABELS[final[name][i]])
            n+=1;checked+=1
        counts[name]=n
    save(OUT/'replay-parity.json',dict(checked=checked,configurations=counts,mismatched_labels=0,
        implementation='Production fuse_person, filter_object_evidence, apply_wave_prior and apply_running_prior vs vector calibration on same sampled windows. Full tracking not covered.'))
    print('PARITY',checked,flush=True)


if __name__=='__main__':main()
