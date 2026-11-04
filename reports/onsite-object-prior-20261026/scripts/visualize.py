"""Full-source paired review videos; cached action/track replay is labeled explicitly."""
from collections import defaultdict
import html
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from evaluate import OUT, ROOT, OP, CACHE, save, actor, DownscaleStage, RetinexformerStage
from darkpipe.person.core import render_people
from darkpipe.media import VideoWriter
from darkpipe.constants import BEHAVIOR_ZH

FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
REASONS=dict(single_frame_support='检出兼容物体，保留动作',no_compatible_object='未检出兼容物体，改为其他',
    below_luma_gate='低于照度门，跳过先验',no_valid_bright_observation='无有效亮帧观测，先验弃权',
    unaffected_action='此动作不使用物体先验',unknown_raw_luma='原图亮度未知，先验弃权')


def text(canvas: np.ndarray, lines: list[tuple]) -> np.ndarray:
    image=Image.fromarray(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB));draw=ImageDraw.Draw(image)
    for xy,message,size,color in lines:draw.text(xy,message,font=ImageFont.truetype(FONT,size),fill=color)
    return cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)


def select(results: list[dict]) -> list[dict]:
    candidates=[]
    for d in results:
        if d['source']['scope']!='onsite':continue
        for r in d['records']:
            candidates.append((d,r))
    chosen=[];used=set()
    predicates=[]
    for c in ['drink','pick_up']:
        predicates.extend([
            (f'{BEHAVIOR_ZH[c]}：真实动作获保留',lambda r,c=c:r['truth']==c and r['baseline']['label']==c and r['after']==c and r['prior']['applied']),
            (f'{BEHAVIOR_ZH[c]}：真实动作新增漏报',lambda r,c=c:r['truth']==c and r['baseline']['label']==c and r['after']=='other'),
            (f'{BEHAVIOR_ZH[c]}：other误报被过滤',lambda r,c=c:r['truth']=='other' and r['baseline']['label']==c and r['after']=='other')])
    predicates.extend([
        ('暗场：照度门跳过先验',lambda r:r['prior']['reason']=='below_luma_gate' and r['truth']==r['baseline']['label']),
        ('多人：其他动作保持原结果',lambda r:r['people']>=2 and r['truth']==r['after'] and r['after'] in ('wave','chase'))])
    for title,predicate in predicates:
        possible=[(d,r) for d,r in candidates if d['source']['id'] not in used and predicate(r)]
        if not possible:continue
        possible.sort(key=lambda item:(dict(test=0,val=1,train=2)[item[0]['source']['split']],
            item[0]['source']['seconds']>12, abs(item[0]['source']['seconds']-4),item[1]['timestamp']))
        d,r=possible[0];used.add(d['source']['id'])
        chosen.append(dict(title=title,video_id=d['source']['id'],event_id=r['id'],event_index=r['event_index'],
            track_id=r['track_id'],timestamp=r['timestamp'],truth=r['truth'],before=r['baseline']['label'],
            after=r['after'],reason=r['prior']['reason'],source=d['source']['source'],split=d['source']['split']))
    save(OUT/'demo-selection.json',chosen)
    return chosen


def contact_sheet(d: dict, focus: dict, enhancer) -> np.ndarray:
    record=next(r for r in d['records'] if r['id']==focus['event_id'])
    samples=record['samples']
    if not samples:return None
    wanted={s['frame']:s for s in samples};cap=cv2.VideoCapture(str(ROOT/d['source']['source']))
    assert cap.isOpened();tiles=[];down=DownscaleStage(720);fi=0
    while fi<=max(wanted):
        ok,raw=cap.read();assert ok
        if fi in wanted:
            sample=wanted[fi];small=down([raw])[0];enh=enhancer([small])[0]
            rect=sample['person_roi']
            if rect is not None:
                ratio=np.array([enh.shape[1]/raw.shape[1],enh.shape[0]/raw.shape[0]])
                lo=np.floor(np.array(rect[:2])*ratio).astype(int);hi=np.ceil(np.array(rect[2:])*ratio).astype(int)
                crop=enh[lo[1]:hi[1],lo[0]:hi[0]].copy()
                for box in sample['detections']:
                    if box[4]<.35:continue
                    xy=(np.array(box[:4])*np.tile(ratio,2)-np.tile(lo,2)).round().astype(int)
                    cv2.rectangle(crop,tuple(xy[:2]),tuple(xy[2:]),(40,200,255),2)
                scale=min(300/crop.shape[1],310/crop.shape[0]);im=cv2.resize(crop,None,fx=scale,fy=scale)
                tile=np.full((390,320,3),28,np.uint8);tile[10:10+im.shape[0],10:10+im.shape[1]]=im
                valid=sample['binding_valid'] and sample['available']
                objects=[f'{int(b[5])}:{b[4]:.2f}' for b in sample['detections'] if b[4]>=.35]
                tile=text(tile,[((10,325),f'帧{fi}  L={sample["raw_luma"]:.1f}  有效={valid}',15,'white'),
                    ((10,351),'检出 '+(', '.join(objects) if objects else '无 ≥0.35'),14,'#ffcf66')])
                tiles.append(tile)
        fi+=1
    cap.release()
    if not tiles:return None
    canvas=np.concatenate(tiles,axis=1)
    head=np.full((100,canvas.shape[1],3),28,np.uint8)
    head=text(head,[((12,6),focus['title']+f' | {Path(focus["source"]).name} | T{focus["track_id"]} @ {focus["timestamp"]:.2f}s',21,'white'),
        ((12,40),'五个固定采样位置；0瓶 1杯 2手机 3相机 4纸 5便携物 6展品。框是检测预测，并非物体真值。',17,'#cccccc'),
        ((12,69),REASONS[focus['reason']],18,'#ffcf66')])
    return np.concatenate([head,canvas])


def render(d: dict, focus: dict, number: int, enhancer) -> dict:
    v=d['source'];data=json.loads((ROOT/v['cache']).read_text());events=defaultdict(list)
    for row in d['records']:events[row['frame_index']].append(row)
    folder=OUT/'demos';folder.mkdir(exist_ok=True)
    target=folder/f'{number:02d}_{v["id"]}.mp4';writer=VideoWriter(str(target),v['fps'])
    cap=cv2.VideoCapture(str(ROOT/v['source']));assert cap.isOpened();latest={};index=0
    down=DownscaleStage(720);poster=None
    try:
        while True:
            ok,raw=cap.read()
            if not ok:break
            small=down([raw])[0];enhanced=enhancer([small])[0];meta=data['frames'][index];ts=index/v['fps']
            for row in events[index]:latest[row['track_id']]=row
            halves=[]
            for side in ['before','after']:
                tracks=[]
                for t in meta['tracks']['30']:
                    a=actor(meta,t['track_id'],np.ones(2));row=latest.get(t['track_id']);pred=None
                    if row is not None:
                        pred=dict(row['baseline']);pred['label']=row['baseline']['label'] if side=='before' else row['after']
                    tracks.append(SimpleNamespace(tid=t['track_id'],box=np.array(t['bbox']),pose=None if a is None else a['pose'],
                        prediction=pred,last_inference=-1e10 if row is None else row['timestamp']))
                halves.append(render_people(enhanced,tracks,enhanced.shape,ts))
            body=np.concatenate(halves,axis=1);header=np.full((112,body.shape[1],3),24,np.uint8)
            subtitle=f'{Path(v["source"]).name} | {v["split"]} | t={ts:.2f}s | 原图L={meta["luma"]:.1f} | 固定窗口离线回放'
            header=text(header,[((12,4),focus['title'],22,'white'),((12,34),subtitle,17,'#cccccc'),
                ((14,75),'加入物体先验前 · α0.95 / 动作阈值0.85',20,'white'),
                ((enhanced.shape[1]+14,75),'加入物体先验后 · 现场m / 720增强 / 物体0.35',20,'#ffcf66')])
            foot=np.full((58,body.shape[1],3),24,np.uint8)
            current=latest.get(focus['track_id']);status='等待完整动作窗'
            if current is not None and ts-current['timestamp']<=1:
                status=REASONS[current['prior']['reason']]
            foot=text(foot,[((12,4),f'关注T{focus["track_id"]}：{status}',19,'#ffcf66'),
                ((12,31),f'审核标签仅指选中窗口 @ {focus["timestamp"]:.2f}s：{BEHAVIOR_ZH[focus["truth"]]}。框/姿态/动作为缓存回放，非新实时性能测试。',16,'#cccccc')])
            canvas=np.concatenate([header,body,foot]);writer.write(canvas)
            if index==next(r['frame_index'] for r in d['records'] if r['id']==focus['event_id']):
                poster=target.with_suffix('.jpg');assert cv2.imwrite(str(poster),canvas)
            index+=1
    finally:cap.release();writer.close()
    assert index==v['frames']==writer.count
    # Strict entire-file decode, not metadata-only duration checks.
    subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(target),'-f','null','-'],check=True,capture_output=True)
    sheet=contact_sheet(d,focus,enhancer);sheet_path=None
    if sheet is not None:
        sheet_path=target.with_name(target.stem+'_evidence.jpg');assert cv2.imwrite(str(sheet_path),sheet)
    result=dict(focus,video=str(target.relative_to(OUT)),poster=str(poster.relative_to(OUT)),
        evidence=None if sheet_path is None else str(sheet_path.relative_to(OUT)),frames=index,seconds=index/v['fps'],decoded_ok=True)
    save(target.with_suffix('.json'),result)
    print('[demo]',number,focus['title'],v['source'],index,flush=True)
    return result


def index_page(demos: list[dict], results: list[dict]) -> None:
    cards=[]
    for d in demos:
        evidence='' if d['evidence'] is None else f'<p><a href="{d["evidence"]}">查看五帧物体检测证据</a></p>'
        cards.append(f'<article><h3>{html.escape(d["title"])}</h3><video controls preload="metadata" poster="{d["poster"]}" src="{d["video"]}"></video>'
            f'<p>{html.escape(d["source"])} · {d["split"]} · T{d["track_id"]} @ {d["timestamp"]:.2f}s</p>{evidence}</article>')
    table=[]
    for data in results:
        v=data['source']
        if v['scope']!='onsite':continue
        before=sum(r['baseline']['label']!='other' for r in data['records']);after=sum(r['after']!='other' for r in data['records'])
        changed=[r for r in data['records'] if r['after']!=r['baseline']['label']]
        fp=sum(r['truth']=='other' for r in changed);fn=sum(r['truth']==r['baseline']['label'] for r in changed)
        table.append(f'<tr><td>{html.escape(v["source"])}</td><td>{v["split"]}</td><td>{len(data["records"])}</td>'
            f'<td>{before} → {after}</td><td>{fp} / {fn}</td><td><a href="per-video/{v["id"]}.json">JSON</a></td></tr>')
    summary=json.loads((OUT/'summary.json').read_text())['onsite']['all']
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>旧现场物体先验评估</title><style>body{font-family:system-ui,sans-serif;margin:32px auto;max-width:1500px;padding:0 20px;color:#233344;background:#f5f7fa}h1{font-size:28px}p{line-height:1.7}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:20px}article{padding:16px;background:white;border-radius:10px}video,img{width:100%}td,th{border-bottom:1px solid #d2d9e0;padding:9px;text-align:left}table{width:100%;background:white;font-size:14px}a{color:#246c9e}input{font:inherit;padding:10px;width:90%;margin-bottom:15px}.notice{background:#fff2dc;padding:18px;border-radius:8px}small{color:#536575}</style>
<h1>407段旧现场 · 逐人动作＋物体先验</h1>
<p>现场微调YOLO26m，720增强图裁剪；动作0.85 / RGB权重0.95 / 照度门30 / 物体0.35。左：无先验；右：加入先验。新物体推理＋固定动作窗口回放，非重新运行全量实时流水线。</p>'''
    page+=f'<p class="notice">共过滤{summary["filtered"]}个触发窗口：已知other误报纠正{summary["corrected_other_false_positives"]}个；新增真动作漏报{summary["newly_missed_true_actions"]}个；未知标签{summary["filtered_unknown"]}个。未知标签不能算误报收益。训练／验证／测试详见报告。</p>'
    page+='<p><a href="README.md">详细报告</a> · <a href="video-results.csv">全部逐视频CSV</a> · <a href="window-results.csv">逐人窗口CSV</a> · <a href="class-results.csv">分类统计</a></p>'
    page+='<img src="figures/01-true-and-false.png" alt="真实动作保留和误报对比"><h2>完整来源并排回放</h2><p>按结果类型选例，仅作错误分析，不按这些视频估计准确率。每段保持原速并完整解码通过。</p><main>'+''.join(cards)+'</main>'
    page+='<h2>全部407段现场清单</h2><input id="search" placeholder="按视频名称、类别或划分筛选"><table><thead><tr><th>来源</th><th>划分</th><th>窗口数</th><th>非other触发 前→后</th><th>纠正误报 / 新增漏报</th><th>证据</th></tr></thead><tbody>'+''.join(table)+'</tbody></table>'
    page+='<script>document.getElementById("search").addEventListener("input",e=>{const q=e.target.value.toLowerCase();document.querySelectorAll("tbody tr").forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q));});</script></html>'
    (OUT/'index.html').write_text(page)


def main() -> None:
    torch.set_num_threads(2);cv2.setNumThreads(1)
    protocol=json.loads((OUT/'protocol.json').read_text())
    results=[json.loads((OUT/'per-video'/f'{v["id"]}.json').read_text()) for v in protocol['sources']]
    chosen=select(results);by_id={d['source']['id']:d for d in results}
    enhancer=RetinexformerStage(str(OP/'ckpts'),chunk=1);enhancer.load('cuda:0')
    demos=[]
    try:
        for i,focus in enumerate(chosen,1):demos.append(render(by_id[focus['video_id']],focus,i,enhancer))
    finally:enhancer.close()
    save(OUT/'demo-results.json',demos);index_page(demos,results)


if __name__=='__main__':main()
