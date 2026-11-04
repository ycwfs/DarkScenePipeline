"""Real-source brightness/person examples, rendered with the actual packaged engine."""
from collections import Counter
import copy
import html
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'platform/dark-demo-v2'))
from darkpipe.config import PipelineConfig
from darkpipe.person.core import render_people
from darkpipe.person.runtime import PersonEngine
from darkpipe.media import VideoWriter


def choose():
    candidates=[]
    for v in json.loads((OUT/'protocol.json').read_text())['videos']:
        if v['split']!='train':continue
        d=json.loads((OUT/'data'/v['id']/'result.json').read_text())
        counts=Counter(len(f['boxes']) for f in d['frames'])
        candidates.append(dict(**v,people_mode=counts.most_common(1)[0][0],people_max=max(counts)))
    selected=[];missing=[]
    for lo,hi in [(0,15),(15,30),(30,45),(45,60),(60,75),(75,256)]:
        for count in (1,2):
            group=[v for v in candidates if v['raw_mean_luma'] is not None and lo<=v['raw_mean_luma']<hi
                   and v['people_mode']==count and 1.8<=v['seconds']<=9]
            if not group:
                missing.append(dict(brightness=[lo,hi],people_mode=count,reason='No source in the fixed train subset with this sustained detector count and duration2–9s'))
                continue
            center=95 if lo==75 else (lo+hi)/2
            v=min(group,key=lambda r:abs(r['raw_mean_luma']-center))
            selected.append(dict(**v,band=f'{lo}-{hi}' if hi<256 else '75+',selection='Raw source brightness and modal detector count; no outcome filtering'))
    for count in (3,):
        group=[v for v in candidates if v['raw_mean_luma'] is not None and v['people_mode']==count and v['seconds']<=10]
        if group:selected.append(dict(**min(group,key=lambda r:r['seconds']),band='75+',selection='Additional raw multi-person example'))
    for count in (4,8,13):
        group=[v for v in candidates if v['processed'] and v['seconds']>=1.8]
        v=min(group,key=lambda r:(abs(r['people_mode']-count),abs(r['seconds']-3)))
        if not any(r['id']==v['id'] for r in selected):
            selected.append(dict(**v,band='原图亮度未知',selection='Online negative; detector count is not manual person GT'))
    # Include the exact crowded negative source previously requested by the user.
    v=next(v for v in candidates if '095918_drinking_water_0060' in v['source'])
    if not any(r['id']==v['id'] for r in selected):selected.append(dict(**v,band='原图亮度未知',selection='User-specified crowded negative'))
    (OUT/'demo-selection.json').write_text(json.dumps(dict(records=selected,missing=missing),ensure_ascii=False,indent=2)+'\n')
    return selected


def main():
    torch.set_num_threads(2);cv2.setNumThreads(2)
    selected=choose()
    cfgdata=json.loads((ROOT/'platform/dark-demo-v2/models.json').read_text())['defaults']
    cfg=PipelineConfig(person_mode=True,recognize='behavior',ckpt_dir=str(ROOT/'platform/dark-demo-v2/ckpts'),
        proc_max_side=720,enhance_chunk=1,sr='bicubic',sr_scale=2,denoise='fast',**cfgdata)
    folder=OUT/'demos';folder.mkdir(exist_ok=True)
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',22)
    results=[];engine=None;current_events=[]
    try:
        engine=PersonEngine(cfg,time.monotonic(),lambda row:current_events.append(row),False)
        for epoch,v in enumerate(selected,1):
            current_events.clear();cfg.input_processed=v['processed']
            cap=cv2.VideoCapture(str(ROOT/v['source']));assert cap.isOpened()
            destination=folder/f'{v["band"].replace("原图亮度未知","unknown")}_{v["id"]}.mp4'
            writer=VideoWriter(str(destination),fps=v['fps'])
            index=0;started=time.monotonic();frame_rows=[];poster=False
            try:
                while True:
                    ok,raw=cap.read()
                    if not ok:break
                    stamp=index/v['fps']
                    frame,tracks,record=engine.process(raw,index,stamp,epoch,v['fps'])
                    shown=render_people(engine.picture(frame),tracks,frame.shape,stamp,cfg.pose_overlay)
                    luma='未知（已增强回传）' if record['raw_luma'] is None else f'{record["raw_luma"]:.1f}'
                    canvas=np.full((shown.shape[0]+48,shown.shape[1],3),22,np.uint8);canvas[48:]=shown
                    pil=Image.fromarray(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB));d=ImageDraw.Draw(pil)
                    d.text((14,8),f'{Path(v["source"]).name} | 原图L={luma} | 候选人体框 {len(tracks)} | α={cfg.skel_fusion_alpha:g} / 阈值{cfg.reco_min_conf:g}',font=font,fill='white')
                    canvas=cv2.cvtColor(np.asarray(pil),cv2.COLOR_RGB2BGR)
                    writer.write(canvas)
                    frame_rows.append(dict(frame=index,raw_luma=record['raw_luma'],people=len(tracks),
                        actors=[dict(track_id=t.tid,bbox=t.box.tolist(),label=t.prediction['label'] if t.prediction else None,
                            pose=t.pose is not None,mode=t.prediction['mode'] if t.prediction else None) for t in tracks]))
                    if not poster and stamp>=min(1.6,v['seconds']*.7):
                        cv2.imwrite(str(destination.with_suffix('.jpg')),canvas);poster=True
                    index+=1
            finally:
                cap.release();writer.close()
            assert index==v['frames']==writer.count
            evidence=dict(source=v,frames=index,events=list(current_events),frame_records=frame_rows,
                elapsed=time.monotonic()-started,configuration=cfgdata,engine='packaged darkpipe.person.runtime.PersonEngine')
            destination.with_suffix('.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
            results.append(dict(id=v['id'],source=v['source'],video=str(destination.relative_to(OUT)),
                poster=str(destination.with_suffix('.jpg').relative_to(OUT)),band=v['band'],people_mode=v['people_mode'],
                people_max=v['people_max'],raw_mean_luma=v['raw_mean_luma'],frames=index,
                labels=dict(Counter(e['label'] for e in current_events))))
            print('[demo]',v['source'],index,results[-1]['labels'],flush=True)
    finally:
        if engine:engine.close()
    (OUT/'demo-results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    cards=[]
    for r in results:
        title=f"{r['band']} · 候选框通常{r['people_mode']}个 · {Path(r['source']).name}"
        cards.append(f'<article><h2>{html.escape(title)}</h2><video controls preload="metadata" poster="{r["poster"]}" src="{r["video"]}"></video><p>{html.escape(r["source"])}</p></article>')
    (OUT/'demos.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>逐人算法真实样本</title>'
        '<style>body{font-family:system-ui,sans-serif;background:#f4f6f8;color:#243548;margin:32px auto;max-width:1500px}h1{font-size:28px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(540px,1fr));gap:22px}article{background:white;padding:18px;border-radius:10px}h2{font-size:18px}video{width:100%}p{overflow-wrap:anywhere;font-size:14px}</style>'
        f'<h1>现场逐人识别 · 亮度门30 / RGB权重{cfg.skel_fusion_alpha:g} / 阈值{cfg.reco_min_conf:g}</h1>'
        '<p>全部是真实来源，没有人工压暗。低于30仅RGB；骨架未满窗或不可用时不显示。人数为检测候选数，可能有漏框、误框或短暂换ID；不等同人工人数真值。已增强回传的原图亮度未知。演示来自训练来源，只展示功能，准确率另见冻结测试报告。</p><main>'+''.join(cards)+'</main></html>')


if __name__=='__main__':
    main()
