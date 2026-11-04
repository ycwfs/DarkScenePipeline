"""Four-way, full-source-frame comparisons; no test-based checkpoint tuning."""
from collections import defaultdict
import json
import subprocess
import time

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from functools import lru_cache
import torch

from inference import Frontend,CausalPoseFilter,OUT,ROOT
from prepare import EDGES,save
from darkpipe.stages.enhance_retinexformer import RetinexformerStage


@lru_cache(None)
def _cjk_font(size):
    # The operator's font is a small action-label subset, not a full report font.
    return ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',size)


def downscale(image):
    h,w=image.shape[:2];ratio=min(720/max(h,w),1)
    return cv2.resize(image,(round(w*ratio)//2*2,round(h*ratio)//2*2),interpolation=cv2.INTER_AREA)


def draw_panel(image,tracks,poses,title):
    canvas=image.copy()
    for t in tracks:
        color=(60,225,100);box=np.round(t.box).astype(int)
        cv2.rectangle(canvas,tuple(box[:2]),tuple(box[2:]),color,2,cv2.LINE_AA)
        cv2.putText(canvas,f'P{t.tid:02d}',(box[0],max(14,box[1]-4)),0,.5,color,1,cv2.LINE_AA)
        pose=poses.get(t.tid)
        if pose is None:continue
        for a,b in EDGES:
            if min(pose[a,2],pose[b,2])>=.3:
                cv2.line(canvas,tuple(np.round(pose[a,:2]).astype(int)),tuple(np.round(pose[b,:2]).astype(int)),(60,205,255),2,cv2.LINE_AA)
        for x,y,score in pose:
            if score>=.3:cv2.circle(canvas,(round(x),round(y)),3,(255,120,70),-1,cv2.LINE_AA)
    h,w=canvas.shape[:2];scale=min(720/w,406/h);new=cv2.resize(canvas,(round(w*scale),round(h*scale)))
    panel=np.zeros((446,720,3),np.uint8);panel[:]=(20,25,32)
    left=(720-new.shape[1])//2;top=40+(406-new.shape[0])//2;panel[top:top+new.shape[0],left:left+new.shape[1]]=new
    pil=Image.fromarray(cv2.cvtColor(panel,cv2.COLOR_BGR2RGB));d=ImageDraw.Draw(pil)
    d.text((12,8),title,font=_cjk_font(20),fill=(235,241,248))
    return cv2.cvtColor(np.asarray(pil),cv2.COLOR_RGB2BGR)


def frame_record(tracks,poses):
    return [dict(id=t.tid,box=t.box.tolist(),score=float(t.score),
                 raw_pose=None if t.pose is None else t.pose.tolist(),
                 filtered_pose=poses[t.tid].tolist()) for t in tracks]


def jitter(records,model):
    history=defaultdict(list);raw_values=[];filtered_values=[];displayed=0;raw_displayed=0;person_frames=0
    for f in records:
        for p in f[model]['tracks']:
            person_frames+=1
            if p['raw_pose'] is None:continue
            raw=np.array(p['raw_pose']);smooth=np.array(p['filtered_pose']);box=np.array(p['box'])
            raw_displayed+=int((raw[:,2]>=.3).sum()>=10);displayed+=int((smooth[:,2]>=.3).sum()>=10)
            center=(box[:2]+box[2:])/2;diag=max(np.linalg.norm(box[2:]-box[:2]),1)
            h=history[p['id']];h.append((f['frame'],(raw[:,:2]-center)/diag,(smooth[:,:2]-center)/diag,(raw[:,2]>=.3)&(smooth[:,2]>=.3)))
            if len(h)>=3 and h[-1][0]-h[-3][0]==2:
                a,b,c=h[-3:];mask=a[3]&b[3]&c[3]
                raw_values.extend(np.linalg.norm(a[1]-2*b[1]+c[1],axis=1)[mask].tolist())
                filtered_values.extend(np.linalg.norm(a[2]-2*b[2]+c[2],axis=1)[mask].tolist())
    return dict(person_frames=person_frames,raw_pose_person_frames=raw_displayed,
        filtered_pose_person_frames=displayed,common_joint_triplets=len(raw_values),
        mean_normalized_second_difference_raw=float(np.mean(raw_values)) if raw_values else None,
        mean_normalized_second_difference_filtered=float(np.mean(filtered_values)) if filtered_values else None,
        note='same visible joint triplets; second difference is a stability proxy, not pose accuracy')


def main():
    cv2.setNumThreads(1);torch.set_num_threads(2)
    manifest=json.loads((OUT/'manifest.json').read_text())
    selected=[v['video'] for v in manifest['videos'] if v['video']['effective_split']['split']=='test']
    inventory=json.loads((ROOT/'reports/dark-annotations-20261023/inventory.json').read_text())['raw_videos']
    extras=['real-actiondata-all/挥手18.mp4','real-actiondata-all/捡东西1.mp4',
            'real-actiondata-all/挥手28.mp4','real-actiondata-all/喝水26.mp4',
            'real-actiondata-all/捡东西23.mp4','real-actiondata-all/other/other100.mp4']
    for source in extras:selected.append(next(v for v in inventory if v['source']==source))
    op=ROOT/'platform/dark-demo-v2/ckpts'
    models={'original':Frontend(op/'yolo26m_detect.onnx',op/'rtmo-s_end2end.onnx'),
        'candidate':Frontend(OUT/'models/yolo26m_dark_candidate.onnx',OUT/'models/rtmo-s_dark_candidate.onnx')}
    enhancer=RetinexformerStage(str(op),chunk=1);enhancer.load('cuda:0')
    summaries=[]
    for video in selected:
        uid=video['id'];source=video['source'];dest=OUT/'videos'/f'{uid}.mp4';dest.parent.mkdir(exist_ok=True)
        cache=OUT/'data'/uid/'demo.json'
        if cache.exists() and dest.exists():
            summaries.append(json.loads(cache.read_text())['summary']);continue
        for model in models.values():model.tracker.__init__(max_gap=.12)
        smoothers={name:CausalPoseFilter() for name in models}
        cap=cv2.VideoCapture(str(ROOT/source));fps=cap.get(cv2.CAP_PROP_FPS)
        encoder=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo',
            '-pix_fmt','bgr24','-s','1440x940','-r',str(fps),'-i','-','-an','-c:v','libx264',
            '-preset','fast','-crf','20','-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',str(dest)],stdin=subprocess.PIPE)
        records=[];index=-1
        try:
            while True:
                ok,native=cap.read();index+=1
                if not ok:break
                raw=downscale(native);b,g,r=cv2.mean(raw)[:3];luma=.114*b+.587*g+.299*r
                enhanced=enhancer([raw])[0];stamp=index/fps
                row=dict(frame=index,timestamp=stamp,luma=luma);panels=[]
                for name,model in models.items():
                    start=time.perf_counter();tracks,result=model(enhanced,stamp);elapsed=time.perf_counter()-start
                    filtered=smoothers[name].update(tracks,stamp)
                    row[name]=dict(tracks=frame_record(tracks,filtered),frontend_ms=elapsed*1000,**result)
                    prefix='原模型' if name=='original' else '微调候选'
                    panels.append(draw_panel(enhanced,tracks,{t.tid:t.pose for t in tracks},prefix+' · 未平滑'))
                    panels.append(draw_panel(enhanced,tracks,filtered,prefix+' · 连续确认＋因果平滑'))
                mosaic=np.vstack([np.hstack(panels[:2]),np.hstack(panels[2:]),np.full((48,1440,3),(20,25,32),np.uint8)])
                pil=Image.fromarray(cv2.cvtColor(mosaic,cv2.COLOR_BGR2RGB));d=ImageDraw.Draw(pil)
                split=video.get('effective_split',video['frozen_split'])['split']
                d.text((12,900),f'{source.split("/")[-1]}  |  {split}  |  F{index:04d}  L={luma:.1f}  |  姿态诊断展示：临时显示暗场骨架，不改门75',font=_cjk_font(19),fill=(235,241,248))
                mosaic=cv2.cvtColor(np.asarray(pil),cv2.COLOR_RGB2BGR)
                encoder.stdin.write(mosaic.tobytes());records.append(row)
                if index==int(video['frames']/2):
                    poster=OUT/'posters'/f'{uid}.jpg';poster.parent.mkdir(exist_ok=True);cv2.imwrite(str(poster),mosaic)
        finally:
            cap.release();encoder.stdin.close();code=encoder.wait(timeout=60)
        if code:raise RuntimeError(f'encoder failed {dest}')
        decoded=cv2.VideoCapture(str(dest));count=0
        while decoded.read()[0]:count+=1
        decoded.release();assert count==len(records)==video['frames'],(source,count,len(records),video['frames'])
        check=subprocess.run(['ffmpeg','-v','error','-threads','2','-i',str(dest),'-f','null','-'],capture_output=True,check=True)
        assert not check.stderr,check.stderr
        summary=dict(source=source,id=uid,split=split,frames=count,fps=fps,video=str(dest.relative_to(OUT)),
            raw_dark_frames=sum(r['luma']<75 for r in records),
            statistics={name:jitter(records,name) for name in models},full_decode=True)
        save(cache,dict(summary=summary,frames=records));summaries.append(summary)
        save(OUT/'demo-summary.json',dict(videos=summaries))
        print('[demo]',source,count,summary['statistics'],flush=True)
    enhancer.close()


if __name__=='__main__':main()
