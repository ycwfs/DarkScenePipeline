"""Decode complete outputs, audit object records, and build an honest review inventory."""
import collections,hashlib,html,json,subprocess,zipfile
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parents[1]
items=json.loads((OUT/'inputs.json').read_text()); summaries=[];videos=[];cards=[]
for item in items:
 folder=OUT/'demos'/item['id'];result=json.loads((folder/'summary.json').read_text())
 events=json.loads((folder/'events.json').read_text());frames=[json.loads(s)for s in (folder/'frames.jsonl').read_text().splitlines()]
 n=result['stats']['frames'];assert len(frames)==n
 objects=[(f,b)for f in frames for b in f['object_overlays']]
 assert all(0<=b['age']<=.600001 and b['confidence']>=.581 and f['raw_luma']>=35 and all(0<=v<=1 for v in b['bbox']) for f,b in objects)
 assert all(e['threshold']==.925 for e in events)
 stats=result['stats']['clips'];assert all(stats.get(k,0)==0 for k in ['write_errors','clips_unfinished','clips_failed','clips_forced','frames_dropped','clips_abandoned','dispatcher_frames_dropped'])
 for path in [folder/'output.mp4',*sorted((folder/'clips').rglob('*.mp4'))]:
  run=subprocess.run(['ffmpeg','-nostdin','-v','error','-xerror','-i',str(path),'-f','null','-'],capture_output=True)
  assert run.returncode==0,(str(path),run.stderr.decode())
  cap=cv2.VideoCapture(str(path));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
  if path.name=='output.mp4':assert count==n,(count,n)
  videos.append(dict(path=str(path.relative_to(OUT)),frames=count,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
 exact=sum(not b['followed']for f,b in objects)
 summary=dict(id=item['id'],title=item['title'],split=item['split'],source=item['source'],start=item['start'],seconds=item['seconds'],frames=n,
  people_max=result['people_max'],object_frames=result['stats']['object_overlay']['frames_with_objects'],exact_boxes=exact,
  followed_boxes=len(objects)-exact,clips=stats['clips_saved'],labels=result['labels'],prior_reasons=result['prior_reasons'],clip_object_frames=stats['object_overlay'].get('frames_with_objects',0))
 summaries.append(summary)
 links=' '.join(f'<a href="{html.escape(str(p.relative_to(OUT)))}">片段{i+1}</a>'for i,p in enumerate(sorted((folder/'clips').rglob('*.mp4')))) or '未触发动作片段'
 d='demos/'+item['id']
 first_object=next((f['timestamp']for f in frames if f['object_overlays']),None)
 jump=(f"<button onclick=\"const v=document.getElementById('v-{item['id']}');v.pause();v.currentTime={first_object};\">定位第一个物体框（{first_object:.2f}s）</button>" if first_object is not None else '本段未检出可显示物体')
 cards.append(f'''<article><h2>{html.escape(item['id']+' · '+item['title'])}</h2><p>{html.escape(item['split'])} · {n}帧 · 最多{result['people_max']}个人体候选 · {summary['object_frames']}帧显示物体 · {stats['clips_saved']}个动作片段</p><video id="v-{item['id']}" controls preload="none" poster="{d}/preview.jpg" src="{d}/output.mp4"></video><p>{jump}</p><p>{links} · <a href="{d}/events.json">完整事件</a> · <a href="{d}/summary.json">统计/源视频</a></p></article>''')
(OUT/'index.html').write_text('''<!doctype html><html lang="zh"><meta charset="utf-8"><title>现场物体框可视化 · 2026-10-27</title><style>body{max-width:1400px;margin:32px auto;padding:0 20px;background:#f5f7fa;color:#182a38;font:16px system-ui}h1{font-size:28px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:20px}article{background:white;padding:18px;border-radius:12px}video{width:100%}h2{font-size:21px}a{color:#036b91}p{line-height:1.6}.note{padding:20px;background:#e7f3f7;border-radius:10px}</style><h1>逐人动作＋物体框：现场可视化</h1><div class="note"><p>默认保持 RGB=0.972、照度门35、动作阈值0.925、物体阈值0.581。青色框：物体类别＋检测置信度＋所属人物ID。bottle瓶、cup杯、phone手机、camera相机、paper纸、item便携物、exhibit机器人/展品。末尾“~”表示最多0.6秒的局部光流位置跟随，非重新检测。</p><p>物体只在满足照度门的喝水/捡物候选中按需检测。离线与片段缓冲将晚到检测画回原采集帧，动作标签仍使用原事件时间；实时流不增加该缓冲，框可能更晚出现或更少。未显示物体框不等于不存在物品。框表示模型预测，不是人工确认。</p><p>9段旧现场完整短片（8 train、1 val）＋3段多人原录像选段，共85秒多人场景，未重跑全部9段/53.54分钟。保留无检出、误报与真实喝水漏检，不能作为独立准确率成绩。人数是人体候选峰值，可能包含展品误检和重复框。</p><p><a href="README.md">说明与验证</a> · <a href="inputs.json">来源与时间段</a> · <a href="object-overlay-demos-20261027.zip">下载演示合集</a> · <a href="full-output-stream.mp4">真实推流接收录像</a></p></div><main>'''+''.join(cards)+'</main></html>')
report=dict(passed=True,summary=summaries,decoded_videos=videos,total_frames=sum(s['frames']for s in summaries),
 total_clips=sum(s['clips']for s in summaries),object_frames=sum(s['object_frames']for s in summaries))
(OUT/'video-verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items()if k not in ['decoded_videos','summary']},ensure_ascii=False))
