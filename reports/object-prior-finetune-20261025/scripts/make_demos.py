"""Render original motion beside explicitly frozen sampled object evidence.

The labels describe the selected inference window, not a new continuous run.
Object boxes are displayed only on the actual sampled stills that produced them.
"""
import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from build_dataset import OUT, save
from evaluate_actions import bright, crop_rect


FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
LABELS = dict(drink='喝水', pick_up='捡物', other='其他', wave='挥手')
CLASSES = ['瓶', '杯', '手机', '相机', '纸张', '可移动小物', '机器人/展品']
PLAN = [
    ('old-drink-supported','3b3cd82b736befd7',2,'旧测试：真实喝水，严格规则保留'),
    ('old-drink-missed','5da1038bd2222f4b',1,'旧测试：真实喝水，严格规则误杀'),
    ('old-pickup-missed','7ce9e264c4e2c972',0,'旧测试：真实捡物，严格规则误杀'),
    ('new05-phone','new-05',1223,'新05：手机候选的连续证据检查'),
    ('new03-exhibit','new-03',3331,'新03：展品候选与人物绑定检查'),
    ('new09-supported','new-09',2113,'新09：容器候选，严格规则保留'),
    ('new03-near','new-03',None,'新03：近处喝水触发复核'),
    ('new07-pickup','new-07',None,'新07：捡物触发复核'),
]


def fit(image: Image.Image, size: tuple) -> Image.Image:
    im = image.copy(); im.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', size, '#141e2c')
    canvas.paste(im, ((size[0]-im.width)//2, (size[1]-im.height)//2))
    return canvas


def source_frame(cap, index: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    assert ok and abs(cap.get(cv2.CAP_PROP_POS_FRAMES)-index-1)<.51
    return frame


def sampled_panel(raw: np.ndarray, sample: dict, size: tuple) -> Image.Image:
    rect = np.asarray(sample.get('person_roi', crop_rect(np.asarray(sample['person_box']),raw.shape)),int)
    x1,y1,x2,y2=rect
    crop=bright(raw[y1:y2,x1:x2])
    im=Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB))
    scale=min(size[0]/im.width,(size[1]-70)/im.height)
    w,h=round(im.width*scale),round(im.height*scale)
    left,top=(size[0]-w)//2,55+(size[1]-70-h)//2
    canvas=Image.new('RGB',size,'#141e2c');canvas.paste(im.resize((w,h)),(left,top))
    draw=ImageDraw.Draw(canvas);font=ImageFont.truetype(FONT,17)
    draw.text((8,4),f"采样帧 {sample['timestamp']:.2f}s（冻结）",font=font,fill='white')
    draw.text((8,27),'有效绑定' if sample['binding_valid'] else '绑定不明确：规则弃权',font=font,fill='#ffc578')
    def coords(box):return ((np.asarray(box)-np.tile(rect[:2],2))*scale+np.array([left,top]*2)).tolist()
    draw.rectangle(coords(sample['person_box']),outline='#60ceee',width=2)
    for b in sample['detections']:
        if b[4]<.35:continue
        bounds=coords(b[:4]);draw.rectangle(bounds,outline='#ffda70',width=3)
        tx=max(0,min(size[0]-140,bounds[0]));ty=max(51,bounds[1]-22)
        text=f'{CLASSES[int(b[5])]} {b[4]:.2f}'
        draw.rectangle((tx,ty,tx+160,ty+23),fill='#141e2c')
        draw.text((tx,ty),text,font=font,fill='#ffda70')
    if not any(b[4]>=.35 for b in sample['detections']):
        draw.text((8,size[1]-24),'无 ≥0.35 的物体框',font=font,fill='#ffda70')
    return canvas


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--sheets-only',action='store_true');args=parser.parse_args()
    cv2.setNumThreads(1)
    folder=OUT/'demos';folder.mkdir(exist_ok=True)
    manifest=[]
    font=ImageFont.truetype(FONT,24);small=ImageFont.truetype(FONT,20)
    for slug,source_id,index,title in PLAN:
        data=json.loads((OUT/'action-evaluation'/f'{source_id}.json').read_text());job=data['job']
        if index is None:
            target=9849 if source_id=='new-03' else 596
            event=next(r for r in data['records']if abs(r['frame_index']-target)<2)
        else:event=next(r for r in data['records']if r['event_index']==index)
        cap=cv2.VideoCapture(job['source']);assert cap.isOpened();fps=job['fps']
        nframes=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        panels=[sampled_panel(source_frame(cap,s['frame']),s,(360,490))for s in event['samples']]
        sheet=Image.new('RGB',(1800,1140),'#101926');draw=ImageDraw.Draw(sheet)
        draw.text((20,12),title+f" | T{event['track_id']} | 窗口结束 {event['timestamp']:.2f}s",font=font,fill='white')
        draw.text((20,50),f"原结果：{LABELS.get(event['baseline'],event['baseline'])}  保守：{LABELS.get(event['conservative'])}  严格：{LABELS.get(event['strict'])}；非连续检测展示",font=small,fill='#b8cde1')
        for i,panel in enumerate(panels):sheet.paste(panel,(i*360,95))
        raw=source_frame(cap,event['frame_index'])
        scene=Image.fromarray(cv2.cvtColor(raw,cv2.COLOR_BGR2RGB));scene=fit(scene,(960,540))
        sheet.paste(scene,(0,600));draw.text((980,625),f"原始画面 / {source_id}",font=font,fill='white')
        draw.text((980,675),'蓝框：原识别人框；黄框：物体模型输出',font=small,fill='#b8cde1')
        draw.text((980,712),'右侧/上方ROI只作显示放大和gamma提亮',font=small,fill='#b8cde1')
        draw.text((980,749),'物体类别是预测，不是人工确认标签',font=small,fill='#b8cde1')
        draw.text((980,786),'新视频动作真值待复核，不计准确率',font=small,fill='#b8cde1')
        sheet.save(folder/f'{slug}.jpg',quality=94)
        row=dict(id=slug,title=title,source_id=source_id,source=job['source'],event_index=event['event_index'],
            person_id=event['track_id'],history_generation=event['history_generation'],timestamp=event['timestamp'],
            truth=event['truth'],baseline=event['baseline'],conservative=event['conservative'],strict=event['strict'],
            sheet=f'demos/{slug}.jpg',video=f'demos/{slug}.mp4',
            display='Original motion plus frozen sampled ROI evidence. Labels refer to selected window, not continuous predictions.')
        if not args.sheets_only:
            start=max(0,int(event['timestamp']*fps)-int(3*fps))
            stop=min(nframes,int(event['timestamp']*fps)+int(3*fps)+1)
            target=folder/f'{slug}.mp4'
            proc=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1440x850',
                '-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','20',
                '-pix_fmt','yuv420p','-movflags','+faststart',str(target)],stdin=subprocess.PIPE)
            cap.set(cv2.CAP_PROP_POS_FRAMES,start)
            saved=0
            try:
                for fi in range(start,stop):
                    ok,frame=cap.read();assert ok
                    canvas=Image.new('RGB',(1440,850),'#101926');paint=ImageDraw.Draw(canvas)
                    paint.text((20,8),title,font=font,fill='white')
                    paint.text((20,49),f"所选窗口 T{event['track_id']} @ {event['timestamp']:.2f}s | 原：{LABELS.get(event['baseline'])}  保守：{LABELS.get(event['conservative'])}  严格：{LABELS.get(event['strict'])}",font=small,fill='#a5dce1')
                    canvas.paste(fit(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)),(960,540)),(0,100))
                    available=[i for i,s in enumerate(event['samples'])if s['frame']<=fi]
                    if available:
                        i=available[-1];panel=fit(panels[i],(460,580))
                        canvas.paste(panel,(970,100))
                    else:paint.text((1000,180),'等待第一张证据采样帧',font=small,fill='white')
                    paint.text((20,658),f'原视频连续播放  {fi/fps:.2f}s；物体框仅显示在右侧冻结证据帧',font=small,fill='white')
                    paint.text((20,700),'蓝色：该人物原检测框  黄色：候选物体预测（≥0.35）',font=small,fill='#b8cde1')
                    paint.text((20,739),'新视频采用最近事件框近似绑定；此为离线规则对照，非生产实时效果。',font=small,fill='#b8cde1')
                    paint.text((20,778),'真实动作可被严格规则误杀；缺物体证据时保守规则保留原动作。',font=small,fill='#b8cde1')
                    proc.stdin.write(np.asarray(canvas).tobytes());saved+=1
            finally:
                proc.stdin.close()
            assert proc.wait()==0
            row.update(frames=saved,fps=fps,start_time=start/fps,end_time=(stop-1)/fps)
        cap.release();manifest.append(row)
        print(slug,flush=True)
    save(OUT/('demo-plan.json' if args.sheets_only else 'demo-manifest.json'),manifest)


if __name__=='__main__':
    main()
