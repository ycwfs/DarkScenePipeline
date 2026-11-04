"""Validate complete demo videos and publish a readable per-source index."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import html,json,subprocess
import numpy as np
OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parents[1]
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(s)for s in p.read_text().splitlines()if s.strip()]
def decode(p):
 run=subprocess.run(['ffmpeg','-nostdin','-v','error','-xerror','-threads','2','-i',str(p),'-f','null','-'],capture_output=True,text=True)
 probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames,width,height,avg_frame_rate:format=duration','-of','json',str(p)]))
 return dict(path=str(p.relative_to(OUT)),passed=run.returncode==0,stderr=run.stderr,probe=probe)
def main():
 inputs=read(OUT/'inputs.json');results=[];videos=[]
 for item in inputs:
  dest=OUT/'demos'/item['id'];s=read(dest/'summary.json');f=lines(dest/'frames.jsonl');events=read(dest/'events.json');assert len(f)==s['stats']['frames']
  assert all(e['label']==e['running_prior']['label']for e in events)
  for key in ('object_prior','wave_prior','running_prior'):
   for e in events:
    prior=e[key]
    if prior.get('applied') and prior.get('label')=='other':assert e['label']=='other'
  cs=s['stats']['clips']
  for key in ('clips_failed','clips_forced','clips_abandoned','frames_dropped','write_errors','clips_unfinished'):
   assert cs.get(key,0)==0,(item['id'],key,cs.get(key))
  clips=[(p,read(p))for p in(dest/'clips').rglob('*.json')if p.with_suffix('.mp4').exists()]
  assert all(c['action_key']!='other'for _,c in clips)
  luma=[r['raw_luma']for r in f if r['raw_luma']is not None];people=Counter(r['people']for r in f)
  s.update(raw_luma_mean=float(np.mean(luma))if luma else None,raw_luma_min=float(min(luma))if luma else None,raw_luma_max=float(max(luma))if luma else None,
   people_mode=people.most_common(1)[0][0],people_histogram=dict(people),below_gate_frames=sum(v<35 for v in luma),total_frames=len(f),
   events=len(events),clips=[dict(path=str(p.with_suffix('.mp4').relative_to(OUT)),metadata=str(p.relative_to(OUT)),context_complete=c['context_complete'],person_id=c['person_id'],label=c['action_key'])for p,c in clips])
  results.append(s);videos+=[dest/'output.mp4']+[p.with_suffix('.mp4')for p,_ in clips]
 with ThreadPoolExecutor(max_workers=2)as pool:checks=list(pool.map(decode,videos))
 save(OUT/'demo-video-verification.json',checks);assert all(c['passed']for c in checks)
 bypath={c['path']:c for c in checks}
 for r in results:assert int(bypath['demos/'+r['input']['id']+'/output.mp4']['probe']['streams'][0]['nb_read_frames'])==r['total_frames']
 save(OUT/'demo-results.json',results)
 body=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>完整先验方案演示</title><style>body{font-family:system-ui,sans-serif;max-width:1200px;margin:36px auto;color:#213547;padding:0 20px}h1{font-size:28px}article{border-top:1px solid #ccd5df;margin-top:28px;padding-top:18px}video{width:100%;max-height:650px;background:#111}table{border-collapse:collapse}td,th{padding:9px;border:1px solid #ddd}a{color:#2463b6}p{line-height:1.7}summary{cursor:pointer}</style><h1>完整先验方案 · 2026-10-27</h1><p>最终归档在原0.3.0镜像内实际推理。RGB α=0.972，原图门35，动作0.925，物体0.581；挥手3帧，跑动0.35框高/秒。人物框、骨架与物体框共用正式绘制。未加测试条码。</p><p>这是历史现场的功能演示，不是独立准确率测试。原图亮度未知的已增强回传会跳过三类先验；短片开头不足1秒没有动作窗口，演示中有13/19个片段的前后2秒上下文未完全满足，并在片段元数据标记。人数是检测统计，不是人工真值。</p><p><a href="README.md">报告</a> · <a href="../algorithm-flow-20261027/README.md">可编辑流程图</a> · <a href="live/full-output-5min.mp4">5分钟真实推流完整接收录像</a> · <a href="live/clips-index.html">实时保存片段索引</a></p><table><tr><th>样本</th><th>原图平均L</th><th>人数众数/峰值</th><th>事件/片段</th></tr>']
 for r in results:
  i=r['input'];l='未知'if r['raw_luma_mean']is None else f"{r['raw_luma_mean']:.1f}"
  body.append(f'<tr><td><a href="#{i["id"]}">{html.escape(i["title"])}</a></td><td>{l}</td><td>{r["people_mode"]}/{r["people_max"]}</td><td>{r["events"]}/{len(r["clips"])}</td></tr>')
 body.append('</table>')
 for r in results:
  i=r['input'];base='demos/'+i['id'];source_href=('../'+html.escape(i['input'][len('reports/'):]) if i['input'].startswith('reports/') else '../../'+html.escape(i['input']));body.append(f'<article id="{i["id"]}"><h2>{html.escape(i["title"])}</h2><p>{html.escape(i["source"])}；照度门下 {r["below_gate_frames"]}/{r["total_frames"]} 帧；最终标签 {html.escape(str(r["labels"]))}</p><video controls preload="none" poster="{base}/preview.jpg" src="{base}/output.mp4"></video><p><a href="{base}/output.mp4">完整处理视频</a> · <a href="../../{html.escape(i["input"])}">输入</a> · <a href="{base}/events.json">事件/先验证据</a></p><details><summary>自动保存片段与先验统计</summary>')
  for c in r['clips']:body.append(f'<p><a href="{c["path"]}">T{c["person_id"]} {c["label"]}</a>；前后上下文完整：{c["context_complete"]} · <a href="{c["metadata"]}">元数据</a></p>')
  body.append('<pre>'+html.escape(json.dumps(r['prior_reasons'],ensure_ascii=False,indent=2))+'</pre></details></article>')
 body.append('<p>覆盖空缺：固定历史短片中没有持续双人的15–30、45–60、60–75亮度组合；未人工压暗多人亮场视频来填补。多人新现场另外提供四段30秒原速视频。</p></html>');(OUT/'index.html').write_text('\n'.join(body))
 print(json.dumps(dict(inputs=len(results),frames=sum(r['total_frames']for r in results),events=sum(r['events']for r in results),clips=sum(len(r['clips'])for r in results),decoded_videos=len(checks)),indent=2))
if __name__=='__main__':main()
