"""Public zero-shot XCLIP, enhancement/person ROI only; no pose/object network."""
import argparse
from collections import Counter, deque
import copy
import contextlib
import json
from pathlib import Path
import sys
import time
import hashlib

OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parents[1];OP=ROOT/'platform/dark-demo-v2'
sys.path.insert(0,str(OP))
import cv2
import numpy as np
import torch
from transformers import XCLIPModel, AutoTokenizer
from darkpipe.config import PipelineConfig
from darkpipe.constants import BEHAVIORS
from darkpipe.person import core
from darkpipe.person.runtime import prepare_stages, content_height
from darkpipe.stages import build_stages
from darkpipe.stages.object_prior import YoloOnnxDetector
from darkpipe.person.action_engine import MODEL_FILES
from darkpipe.stages.recognize import reject_probs


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


class XClip:
    def __init__(self,device='cuda:0'):
        self.device=device;model=ROOT/'ckpts/xclip-base-patch16-zero-shot'
        self.net=XCLIPModel.from_pretrained(str(model),local_files_only=True).to(device).eval()
        tokenizer=AutoTokenizer.from_pretrained(str(model),local_files_only=True)
        prompts=json.loads((OUT/'prompts.json').read_text());texts=[];groups=[]
        for i,label in enumerate(BEHAVIORS):
            texts.extend(prompts[label]);groups.extend([i]*len(prompts[label]))
        tokens=tokenizer(texts,padding=True,return_tensors='pt').to(device)
        with torch.no_grad():self.text=self.net.get_text_features(**tokens).float()
        self.tokens=tokens
        self.group=torch.tensor(groups,device=device)
        self.count=torch.bincount(self.group,minlength=len(BEHAVIORS)).float()
        self.mean=torch.tensor([.485,.456,.406],device=device)[None,None,:,None,None]
        self.std=torch.tensor([.229,.224,.225],device=device)[None,None,:,None,None]
        self.parity_checked=False

    @torch.inference_mode()
    def predict(self,arrays):
        x=torch.from_numpy(np.stack(arrays)).to(self.device).permute(0,1,4,2,3).float()/255
        x=(x-self.mean)/self.std
        b,t=x.shape[:2];m=self.net
        assert t==m.config.vision_config.num_frames==32
        with torch.autocast('cuda',dtype=torch.float16):
            out=m.vision_model(pixel_values=x.flatten(0,1))
            video=m.mit(m.visual_projection(out[1]).reshape(b,t,-1))[1]
            img=m.prompts_visual_layernorm(out[0][:,1:,:])@m.prompts_visual_projection
            img=img.reshape(b,t,-1,video.shape[-1]).mean(1)
            text=self.text.unsqueeze(0).expand(b,-1,-1)
            text=text+m.prompts_generator(text,img)
            video=video/video.norm(dim=-1,keepdim=True);text=text/text.norm(dim=-1,keepdim=True)
            cosine=torch.einsum('bd,bkd->bk',video,text).float()
        if not self.parity_checked:
            with torch.autocast('cuda',dtype=torch.float16):
                stock=m(**self.tokens,pixel_values=x[:1]).logits_per_video.float()
            err=float((stock-cosine[:1]*m.logit_scale.exp().float()).abs().max())
            save(OUT/f'forward-parity-{__import__("os").getpid()}.json',dict(max_logit_error=err,passed=err<.03))
            assert err<.03,err
            self.parity_checked=True
        scores=torch.zeros(b,len(BEHAVIORS),device=self.device)
        scores.index_add_(1,self.group,cosine)
        return (scores/self.count*m.logit_scale.exp().float()).softmax(1).cpu().numpy()


def init_stages(processed=False):
    cfg=PipelineConfig(recognize='off',device='cuda:0',ckpt_dir=str(OP/'ckpts'),proc_max_side=720,enhance_chunk=1,input_processed=processed)
    stages,_=build_stages(cfg);stages=[s for s in stages if not s.post_recognition]
    for s in stages:s.load('cuda:0')
    return cfg,stages


def cached_source(v):
    prior=ROOT/'reports/four-parameter-tuning-20261026/replay'/f'{v["id"]}.json'
    c=json.loads((ROOT/v['cache']).read_text())
    if prior.exists():
        r=json.loads(prior.read_text());return r['tracks']['35'],r['events']['35']
    return [f['tracks']['30']for f in c['frames']],c['events']['30']


def old(a,model):
    if a.split=='test':assert (OUT/'selection.json').exists(),'Test after val threshold lock'
    wave=ROOT/'reports/wave-pose-prior-20261027'
    sources=[v for v in json.loads((wave/'protocol.json').read_text())['old_sources']if v['scope']!='new_multipeople'and v['split']==a.split]
    all_rows=json.loads((ROOT/'reports/threshold-tuning-20261026/windows.json').read_text())
    predictions=json.loads((ROOT/'reports/four-parameter-tuning-20261026/predictions.json').read_text())
    baselines={r['id']:predictions['labels'][i]for r,i in zip(all_rows,predictions['predictions']['recommended'])}
    cfg,stages=init_stages()
    for n,v in enumerate(sources):
        dest=OUT/'old'/a.split/f'{v["id"]}.json'
        if dest.exists():continue
        rr=[r for r in all_rows if r['source_id']==v['id']]
        if not rr:save(dest,dict(source=v,events=[],frames=0,no_windows=True));continue
        tracks,events=cached_source(v)
        cfg.input_processed=v['processed'];cap=cv2.VideoCapture(str(ROOT/v['source']));assert cap.isOpened()
        needed={f for r in rr for f in r['source_frames']};pixels={};index=0;results=[]
        while True:
            ok,raw=cap.read()
            if not ok:break
            if index in needed:
                frame,luma=prepare_stages(stages,cfg,raw)
                for t in tracks[index]:
                    crop,_=core.crop_person(frame,np.asarray(t['bbox']),shape=(224,224))
                    pixels[(index,t['track_id'])]=crop
            index+=1
        cap.release();assert index==v['frames']
        for start in range(0,len(rr),2):
            batch=rr[start:start+2];arrays=[]
            for r in batch:
                e=events[r['event_index']];assert e['source_frames']==r['source_frames']
                arrays.append(np.stack([pixels[(f,r['track_id'])]for f in r['source_frames']]))
            probs=model.predict(arrays)
            for r,arr,p in zip(batch,arrays,probs):
                results.append(dict(r,baseline=baselines[r['id']],xclip_probs=p.tolist(),
                    input_sha256=hashlib.sha256(arr.tobytes()).hexdigest(),
                    labels={str(t):BEHAVIORS[int(reject_probs(p,BEHAVIORS,t).argmax())]for t in [0,.4,.75,.85,.925]}))
        save(dest,dict(source=v,events=results,frames=index,frontend='frozen exact gate35 track boxes; enhanced pixels regenerated; no pose or object forward'))
        print('OLD',a.split,n+1,len(sources),v['source'],len(results),flush=True)
    for s in stages:s.close()


def long(a,model):
    import onnxruntime as ort
    from darkpipe.media import VideoWriter
    record=next(v for v in json.loads((ROOT/'reports/manypeople-inference-20261025/input-manifest.json').read_text())if v['number']==a.number)
    dest=OUT/'long-videos'/f'new-{a.number:02}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'summary.json').exists():raise FileExistsError(dest)
    cfg,stages=init_stages();detector=YoloOnnxDetector(str(OP/'ckpts'),'cuda:0')
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    detector.sess=ort.InferenceSession(str(OP/'ckpts'/MODEL_FILES['detector']),sess_options=options,providers=[('CUDAExecutionProvider',{'device_id':0})])
    assert detector.sess.get_providers()[0]=='CUDAExecutionProvider'
    tracker=core.PersonTracker();last={};generations={};seen={};index=0;count=Counter();begin=time.monotonic()
    # Entire frontend uses boxes+appearance only; never loads pose/ST-GCN/VideoMamba/object weights.
    original_crop=core.crop_person
    core.crop_person=lambda frame,box,shape=(224,224),margin=.15:original_crop(frame,box,shape=shape,margin=margin)
    cap=cv2.VideoCapture(record['absolute_path']);assert cap.isOpened();fps=cap.get(cv2.CAP_PROP_FPS)
    writer=VideoWriter(str(dest/'xclip.mp4'),fps=fps)
    f=(dest/'events.jsonl').open('w');calibrated=json.loads((OUT/'selection.json').read_text())['threshold']if (OUT/'selection.json').exists()else None
    try:
        while True:
            ok,raw=cap.read()
            if not ok:break
            frame,luma=prepare_stages(stages,cfg,raw);boxes=core.person_boxes(detector._frames_to_dets([frame])[0])
            tracks=tracker.update(boxes,[None]*len(boxes),frame,index/fps);tasks=[]
            for t in tracks:
                if index/fps-seen.get(t.tid,index/fps)>.2:generations[t.tid]=generations.get(t.tid,0)+1
                seen[t.tid]=index/fps;t.append(frame,index/fps,index,luma)
                key=(t.tid,generations.get(t.tid,0));win=t.window(fps)
                if win is not None and index/fps-last.get(key,-1e9)>=.5-1e-6:
                    tasks.append((t,key,win));last[key]=index/fps
            for start in range(0,len(tasks),2):
                batch=tasks[start:start+2];pp=model.predict([np.stack([s.crop for s in win])for _,_,win in batch])
                for (t,key,win),p in zip(batch,pp):
                    decision=reject_probs(p,BEHAVIORS,.4);label=BEHAVIORS[int(decision.argmax())]
                    row=dict(label=label,confidence=float(decision.max()),mode='xclip_rgb_only',threshold=.4,
                        track_id=t.tid,history_generation=key[1],session=1,timestamp=index/fps,frame_index=index,
                        bbox=t.box.tolist(),raw_luma=luma,source_frames=[s.frame_index for s in win],rgb_probs=p.tolist(),
                        skeleton_probs=None,fused_probs=p.tolist(),decision_scores=decision.tolist(),pose_valid=False,
                        labels={str(tau):BEHAVIORS[int(reject_probs(p,BEHAVIORS,tau).argmax())]for tau in [0,.4,.75,.85,.925]},
                        object_prior=dict(applied=False,reason='disabled'),wave_prior=dict(applied=False,reason='disabled'),running_prior=dict(applied=False,reason='disabled'))
                    if calibrated is not None:row['calibrated_label']=BEHAVIORS[int(reject_probs(p,BEHAVIORS,calibrated).argmax())]
                    t.prediction=row;t.last_inference=index/fps;count[label]+=1;f.write(json.dumps(row)+'\n')
            shown=core.render_people(frame,tracks,frame.shape,index/fps,False)
            cv2.putText(shown,'XCLIP zero-shot | RGB only | tau=0.40',(8,20),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,255),1)
            writer.write(shown);index+=1
            if index%250==0:
                save(dest/'progress.json',dict(frames=index,events=sum(count.values()),elapsed=time.monotonic()-begin))
                print('LONG',a.number,index,sum(count.values()),flush=True)
        assert index==record['frames']
    finally:
        cap.release();writer.close();f.close()
        for s in stages:s.close()
    save(dest/'summary.json',dict(source=record,frames=index,events=sum(count.values()),labels=dict(count),elapsed=time.monotonic()-begin,
        loaded_models=['NTIRE.pth',MODEL_FILES['detector'],'microsoft/xclip-base-patch16-zero-shot'],no_pose=True,no_object=True,
        primary_threshold=.4,visualization_size=[720,404],method='Full sequence, independent box+appearance tracker; no skeleton in frontend/decision.'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['old','long'],required=True);p.add_argument('--split',default='val');p.add_argument('--number',type=int,default=1);a=p.parse_args()
    torch.set_num_threads(2);cv2.setNumThreads(2);model=XClip()
    (old if a.mode=='old'else long)(a,model)

if __name__=='__main__':main()
