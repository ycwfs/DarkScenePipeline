"""Render aligned detector evidence; never turn detector labels into ground truth."""
import html
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT=Path(__file__).resolve().parents[1]
MAIN=['full_enh720','person_enh720','person_native']
TITLES=['整帧检测结果（同一区域放大）','增强图人物裁剪后检测','原始像素人物裁剪后检测']
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def render(row: dict, classes: dict) -> Image.Image:
    dest=OUT/'data'/row['video_id'];fi=row['frame']
    enhanced=cv2.imread(str(dest/f'{fi:06d}_enhanced.png'))
    native=cv2.imread(str(dest/f'{fi:06d}_native.jpg'))
    canvas=Image.new('RGB',(1440,620),'#171e29');draw=ImageDraw.Draw(canvas)
    title=ImageFont.truetype(FONT,22);font=ImageFont.truetype(FONT,16)
    draw.text((12,8),f'{Path(row["source"]).stem}  T{row["track_id"]:02d}  {row["timestamp"]:.2f}s  人物动作标注: {row["truth"]}',font=title,fill='white')
    for i,(v,heading)in enumerate(zip(MAIN,TITLES)):
        native_view=v=='person_native'
        image=native.copy()if native_view else enhanced.copy()
        scale=np.array([image.shape[1]/enhanced.shape[1],image.shape[0]/enhanced.shape[0]])
        roi=np.array(row['native_roi']if native_view else row['roi'])
        e=row['variants'][v]
        for j,det in enumerate(e['detections']):
            if det[4]<=.25 or int(det[5])==0:continue
            coords=np.rint(np.array(det[:4])*np.tile(scale,2)).astype(int)
            center=(coords[:2]+coords[2:])/2
            if not ((center>=roi[:2])&(center<=roi[2:])).all():continue
            color=(80,230,100)if j in e['drink_rows']else ((70,190,255)if j in e['pickup_rows']else (255,180,70))
            cv2.rectangle(image,tuple(coords[:2]),tuple(coords[2:]),color,2)
        crop=image[roi[1]:roi[3],roi[0]:roi[2]]
        patch=Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB))
        display_scale=min(450/patch.width,420/patch.height)
        patch=patch.resize((round(patch.width*display_scale),round(patch.height*display_scale)),Image.Resampling.BILINEAR)
        canvas.paste(patch,(i*480+(480-patch.width)//2,90+(420-patch.height)//2))
        draw.text((i*480+12,48),heading,font=font,fill='#dce6ed')
        draw.text((i*480+12,72),f'容器分数 {e["container_score"]:.3f} / 手边 {e["drink_score"]:.3f} / 脚边 {e["pickup_score"]:.3f}',font=font,fill='#dce6ed')
        listed=[]
        for j,d in enumerate(e['detections']):
            center=(np.array(d[:2])+d[2:4])/2
            if d[4]>.25 and int(d[5])!=0 and ((center>=np.array(row['roi'])[:2])&(center<=np.array(row['roi'])[2:])).all():
                suffix=' 手边'if j in e['drink_rows']else(' 脚边'if j in e['pickup_rows']else'')
                listed.append(f'{classes[str(int(d[5]))]} {d[4]:.3f}{suffix}')
        for line,text in enumerate(listed[:4]or['未检出非人框（>0.25）']):
            draw.text((i*480+12,520+line*22),text,font=font,fill='#e5e8eb')
    return canvas


def main() -> None:
    protocol=json.loads((OUT/'protocol.json').read_text())
    data=[json.loads((OUT/'data'/v['id']/'results.json').read_text())for v in protocol['cases']]
    rows=[r for d in data for r in d['records']]
    classes=json.loads((OUT/'model-provenance.json').read_text())['class_names']
    positive=[r for r in rows if (r['truth']=='drink'and any(r['variants'][v]['container_score']>.25 for v in MAIN))
              or(r['truth']=='pick_up'and any(r['variants'][v]['pickup_score']>.25 for v in MAIN))]
    other_drink=[r for r in rows if r['truth']=='other'and any(r['variants'][v]['drink_score']>.25 for v in MAIN)]
    # Negative ankle examples: deterministic first candidate per video, at most twelve videos.
    negative=[];seen=set()
    for r in rows:
        if r['truth']=='other'and r['video_id']not in seen and any(r['variants'][v]['pickup_score']>.25 for v in MAIN):
            negative.append(r);seen.add(r['video_id'])
        if len(negative)>=12:break
    failures=[]
    for stem in ['喝水16','喝水34','捡东西2']:
        candidates=[r for r in rows if r['source']==f'real-actiondata-all/{stem}.mp4'and r['truth']in{'drink','pick_up'}]
        if candidates:failures.append(candidates[len(candidates)//2])
    selected={f'{r["video_id"]}_{r["frame"]:06d}_T{r["track_id"]:03d}':r for r in positive+other_drink+negative+failures}
    visual=OUT/'visual';visual.mkdir(exist_ok=True)
    manifest=[]
    for i,(uid,row)in enumerate(selected.items(),1):
        output=visual/f'{i:03d}_{uid}.jpg';render(row,classes).save(output,quality=94)
        manifest.append(dict(number=i,id=uid,image=output.relative_to(OUT).as_posix(),
            source=row['source'],frame=row['frame'],track_id=row['track_id'],truth=row['truth'],
            scores={v:{k:row['variants'][v][k]for k in ['container_score','drink_score','pickup_score']}for v in MAIN}))
    (OUT/'visual-index.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    body=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>人物ROI物体先验对照</title>',
        '<style>body{font:16px system-ui;max-width:1480px;margin:30px auto;padding:12px;line-height:1.6}img{width:100%}section{border-top:1px solid #bbb;padding-top:12px}</style>',
        '<h1>整帧与人物裁剪：物体先验对照</h1><p>固定检测模型和阈值0.25；左侧是整帧检测后把同一区域放大，'+
        '中间是在增强图上裁人物后重新检测，右侧是映射到原视频像素后裁剪检测。现场原视频未增强，线上切片已经增强。</p>',
        '<p>框表示模型预测，不等于正确物体。绿色满足手边容器规则，橙色满足脚边物体规则；其他非人框为蓝色。'+
        '展示全部已标注正动作中至少一种方法检出容器/脚边物体的候选，以及全部other手边容器候选、12段other脚边候选和3个既有失败例。'+
        '这是用于核对的诊断集合，不用于计算整体准确率。</p><p><a href="README.md">结论与报告</a></p>']
    for r in manifest:
        body.append(f'<section><h2>{r["number"]:03d} · {html.escape(r["source"])} · T{r["track_id"]:02d} · frame {r["frame"]}</h2>'+
                    f'<img loading="lazy" src="{r["image"]}" alt="三种检测输入的同人物对照"></section>')
    body.append('</html>');(OUT/'index.html').write_text('\n'.join(body))
    print('Visual comparisons:',len(manifest),'positive candidates:',len(positive),'other wrist candidates:',len(other_drink))


if __name__=='__main__':main()
