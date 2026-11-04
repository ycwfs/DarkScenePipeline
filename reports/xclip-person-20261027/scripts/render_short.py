"""Short fixed-ROI classifier comparisons; enhancement rerun, probabilities frozen."""
import argparse
from collections import defaultdict
import json
from types import SimpleNamespace as S
import numpy as np
import cv2
import torch
from experiment import ROOT, OUT, init_stages, prepare_stages, core, cached_source, save, BEHAVIORS
from darkpipe.media import VideoWriter
from darkpipe.stages.recognize import reject_probs


def main():
    p=argparse.ArgumentParser();p.add_argument('--split',default='val');a=p.parse_args()
    torch.set_num_threads(2);cv2.setNumThreads(2)
    all_records=[json.loads(p.read_text())for p in sorted((OUT/'old'/a.split).glob('*.json'))]
    # Deterministic selection by existing ground truth and source name, not XCLIP outcomes.
    selected=[];seen=set()
    for label in ['wave','chase','drink','pick_up','other']:
        choices=sorted([r for r in all_records if any(e['truth']==label for e in r['events'])],key=lambda r:r['source']['source'])
        for r in choices[:2 if label=='chase' else 1]:
            if r['source']['id']not in seen:seen.add(r['source']['id']);selected.append(r)
    save(OUT/f'{a.split}-demo-selection.json',[r['source']for r in selected])
    cfg,stages=init_stages();tau=json.loads((OUT/'selection.json').read_text())['threshold']
    for data in selected:
        v=data['source'];dest=OUT/'short-videos'/a.split/v['id'];dest.mkdir(parents=True,exist_ok=True)
        tracks,_=cached_source(v);events=defaultdict(list)
        for r in data['events']:events[r['frame_index']].append(r)
        latest={};cfg.input_processed=v['processed'];cap=cv2.VideoCapture(str(ROOT/v['source']));assert cap.isOpened()
        fps=cap.get(cv2.CAP_PROP_FPS);writer=VideoWriter(str(dest/'comparison.mp4'),fps=fps);index=0
        try:
            while True:
                ok,raw=cap.read()
                if not ok:break
                frame,luma=prepare_stages(stages,cfg,raw)
                for row in events[index]:latest[row['track_id']]=row
                panels=[]
                for variant in ['baseline','xclip']:
                    actors=[]
                    for t in tracks[index]:
                        e=latest.get(t['track_id']);pred=None
                        if e is not None and index/fps-e['timestamp']<=1.:
                            if variant=='xclip':
                                prob=reject_probs(np.array(e['xclip_probs']),BEHAVIORS,tau);label=BEHAVIORS[int(prob.argmax())];conf=float(prob.max())
                            else:label=e['baseline'];conf=None
                            # Baseline display is categorical: probabilities came from a different final decision policy.
                            pred=dict(label=label,confidence=conf if conf is not None else 1.,threshold=tau if variant=='xclip'else 0.,mode='rgb_only')
                        actors.append(S(tid=t['track_id'],box=np.array(t['bbox']),pose=None,prediction=pred,last_inference=e['timestamp']if e else -1e9))
                    shown=core.render_people(frame,actors,frame.shape,index/fps,False)
                    if variant=='baseline':
                        # Hide artificial percent: renderer has no categorical-only option.
                        for actor in actors:
                            if actor.prediction and actor.prediction['label']!='other':actor.prediction['confidence']=float('nan')
                        shown=frame.copy()
                        for actor in actors:
                            b=actor.box.astype(int);cv2.rectangle(shown,tuple(b[:2]),tuple(b[2:]),(0,220,255),2)
                            label='warming up'if actor.prediction is None else actor.prediction['label']
                            cv2.putText(shown,f'T{actor.tid} {label}',(int(b[0]),min(shown.shape[0]-3,int(b[3])+14)),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,220,255),1)
                    title='VideoMamba current final labels'if variant=='baseline'else f'XCLIP zero-shot RGB tau={tau:.2f}'
                    cv2.putText(shown,title,(8,18),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1)
                    panels.append(shown)
                shown=np.hstack(panels);writer.write(shown)
                if index==int(fps):cv2.imwrite(str(dest/'preview.jpg'),shown)
                index+=1
        finally:
            cap.release();writer.close()
        assert index==v['frames'];save(dest/'summary.json',dict(source=v,frames=index,threshold=tau,method='Fixed historical actor/time boxes; both enhanced RGB; baseline categorical result, XCLIP probability; no skeleton drawn/inferred'))
        print('DEMO',a.split,v['source'],index,flush=True)
    for s in stages:s.close()

if __name__=='__main__':main()
