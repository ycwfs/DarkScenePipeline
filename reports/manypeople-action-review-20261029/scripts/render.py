"""Render raw-source review excerpts with a target box and person zoom; CPU only."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from collect import OUT, ROOT, read, save


def render_one(g: dict, source: dict) -> dict:
    target=OUT/g['video']; poster=OUT/g['poster']
    evidence=OUT/'render-metadata'/f'{g["id"]}.json'
    version=3 if source['number']==3 else 2
    fps=float(Fraction(str(source['fps'])))
    if target.exists() and poster.exists() and evidence.exists() and read(evidence).get('render_version') == version:
        return read(evidence)
    start=max(0,math.floor(g['preview_start']*fps+1e-5))
    end=min(source['frames'],math.ceil(g['preview_end']*fps-1e-5))
    cap=cv2.VideoCapture(source['absolute_path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])
    if not cap.isOpened():raise RuntimeError(source['absolute_path'])
    cap.set(cv2.CAP_PROP_POS_FRAMES,start)
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',21)
    header=Image.new('RGB',(1280,56),(19,28,43));d=ImageDraw.Draw(header)
    d.text((14,5),f"{g['id']}  |  原片 {g['number']:02d}  |  {g['representative_track']}  |  预测：{g['label_zh']}（待确认）",font=font,fill=(239,245,255))
    d.text((974,5),'目标人物放大',font=font,fill=(255,211,92))
    header=cv2.cvtColor(np.array(header),cv2.COLOR_RGB2BGR)
    tmp=target.with_suffix('.partial.mp4')
    cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24',
         '-s','1280x624','-r',str(fps/2),'-i','pipe:0','-an','-c:v','libx264','-preset','veryfast',
         '-threads','1','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(tmp)]
    p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    written=0; visible=0; best=1e9; poster_frame=None
    try:
        for frame_idx in range(start,end):
            ok,raw=cap.read()
            if not ok:raise RuntimeError(f'{g["id"]}: unexpected EOF at {frame_idx}')
            if (frame_idx-start)%2:continue
            t=frame_idx/fps
            canvas=np.full((624,1280,3),(19,28,43),np.uint8)
            canvas[:56]=header
            canvas[56:596,:960]=cv2.resize(raw,(960,540),interpolation=cv2.INTER_AREA)
            trace=min(g['trace'],key=lambda x:abs(x['t']-t)) if g['trace'] else None
            if trace and abs(trace['t']-t)<=.60:
                b=np.asarray(trace['box'],dtype=np.float64)
                x1,y1,x2,y2=b*np.array([raw.shape[1]/720,raw.shape[0]/404]*2)
                bx1,by1,bx2,by2=b*np.array([960/720,540/404]*2)
                cv2.rectangle(canvas,(round(bx1),round(by1)+56),(round(bx2),round(by2)+56),(65,220,255),2)
                cv2.putText(canvas,g['representative_track'],(max(0,round(bx1)),max(70,round(by1)+52)),cv2.FONT_HERSHEY_SIMPLEX,.48,(65,220,255),1,cv2.LINE_AA)
                mx=(x2-x1)*.15;my=(y2-y1)*.08
                x1,y1,x2,y2=max(0,int(x1-mx)),max(0,int(y1-my)),min(raw.shape[1],int(x2+mx)),min(raw.shape[0],int(y2+my))
                crop=raw[y1:y2,x1:x2]
                if crop.size:
                    scale=min(310/crop.shape[1],530/crop.shape[0]);w=max(1,round(crop.shape[1]*scale));h=max(1,round(crop.shape[0]*scale))
                    crop=cv2.resize(crop,(w,h));left=960+(320-w)//2;top=56+(540-h)//2
                    canvas[top:top+h,left:left+w]=crop;visible+=1
            else:
                cv2.putText(canvas,'No nearby target box',(968,310),cv2.FONT_HERSHEY_SIMPLEX,.5,(175,185,199),1,cv2.LINE_AA)
            active=any(abs(o['t']-t)<=.30 for o in g['observations'])
            text=f'SOURCE {t:07.2f}s   |   '+('PREDICTION TRIGGER' if active else 'context')+'   |   Raw pixels; box from nearby prediction (+/-0.60s)'
            cv2.putText(canvas,text,(12,616),cv2.FONT_HERSHEY_SIMPLEX,.48,(65,220,255) if active else (192,202,218),1,cv2.LINE_AA)
            distance=abs(t-(g['trigger_start']+g['trigger_end'])/2)
            if trace and abs(trace['t']-t)<=.6 and distance<best:
                best=distance;poster_frame=canvas.copy()
            p.stdin.write(canvas.tobytes());written+=1
        p.stdin.close();error=p.stderr.read().decode();code=p.wait()
        if code:raise RuntimeError(f'{g["id"]}: ffmpeg {error}')
        if not written:raise RuntimeError('empty preview')
        if poster_frame is None:poster_frame=canvas
        cv2.imwrite(str(poster),poster_frame,[cv2.IMWRITE_JPEG_QUALITY,86]);tmp.replace(target)
    finally:
        cap.release()
        if p.poll() is None:p.kill();p.wait()
    result=dict(render_version=version,id=g['id'],source_number=g['number'],source_sha256=source['sha256'],start_frame=start,
                source_fps=fps,source_start_time=start/fps,end_frame_exclusive=end,frames=written,fps=fps/2,seconds=written/(fps/2),
                target_box_visible_frames=visible,video=g['video'],bytes=target.stat().st_size,
                review_only=True,training_input=False,box_policy='nearest same-run same-track action event within0.60s; no synthetic pose, no interpolation',
                source_pixels='original recording; context scaled to960x540, crop from original resolution')
    evidence.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=6);ap.add_argument('--limit',type=int);ap.add_argument('--scope',choices=['all','videomamba','xclip'],default='all');args=ap.parse_args()
    cv2.setNumThreads(1)
    for d in ('clips','posters','render-metadata'):(OUT/d).mkdir(exist_ok=True)
    groups=read(OUT/'review-items.json');sources={s['number']:s for s in read(OUT/'sources.json')}
    if args.scope!='all':groups=[g for g in groups if g['category']==args.scope]
    if args.limit:groups=groups[:args.limit]
    began=time.monotonic();results=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(render_one,g,sources[g['number']]):g['id'] for g in groups}
        for future in as_completed(futures):
            results.append(future.result())
            if len(results)%20==0 or len(results)==len(groups):
                print(f'{len(results)}/{len(groups)} clips; elapsed {time.monotonic()-began:.1f}s',flush=True)
                save('render-progress.json',dict(done=len(results),total=len(groups),elapsed=time.monotonic()-began))
    save('render-results.json',sorted(results,key=lambda x:x['id']))


if __name__=='__main__':main()
