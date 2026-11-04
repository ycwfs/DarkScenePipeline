"""Summaries and a local video index; explicitly keep negative findings visible."""
from collections import defaultdict
import csv
import hashlib
import html
import json
from pathlib import Path
import zipfile

import numpy as np

from prepare import OUT,ROOT,save,iou


def aggregate(videos):
    result={}
    for name in ['original','candidate']:
        items=[v['statistics'][name] for v in videos]
        total=lambda k:sum(r[k] for r in items)
        n=total('common_joint_triplets')
        raw=sum((r['mean_normalized_second_difference_raw'] or 0)*r['common_joint_triplets'] for r in items)/max(n,1)
        smoothed=sum((r['mean_normalized_second_difference_filtered'] or 0)*r['common_joint_triplets'] for r in items)/max(n,1)
        result[name]=dict(person_frames=total('person_frames'),raw_pose_person_frames=total('raw_pose_person_frames'),
            filtered_pose_person_frames=total('filtered_pose_person_frames'),common_joint_triplets=n,
            jitter_raw=raw,jitter_filtered=smoothed,jitter_reduction=1-smoothed/raw if raw else None)
    return result


def main():
    videos=json.loads((OUT/'demo-summary.json').read_text())['videos']
    dark_test=[v for v in videos if v['split']=='test' and v['raw_dark_frames']==v['frames']]
    bright=[v for v in videos if v['raw_dark_frames']==0]
    aggregate_result=dict(dark_test=aggregate(dark_test),bright_test=aggregate(bright),all=aggregate(videos),
        caveat='Detection counts include false positives; jitter compares identical surviving joint triplets only. No human tracking GT.')
    save(OUT/'stability-summary.json',aggregate_result)
    # The stationary human-shaped exhibit is visually identifiable on these two
    # fixed-camera videos. This is an assistant visual audit, not user-labeled GT.
    audit=[]
    for uid in ['56f605e0ad40c73a','da3c570dcc033d1b']:
        data=json.loads((OUT/'data'/uid/'demo.json').read_text())
        region=[45,195,140,405]  # enhanced720 image pixels, same camera framing
        row=dict(id=uid,source=data['summary']['source'],frames=len(data['frames']),
            object='stationary human-shaped exhibit in left foreground',bbox_720=region,
            annotation='assistant visual inspection; not added to training')
        for model in ['original','candidate']:
            hits=[];pose_hits=0;smooth_hits=0
            for f in data['frames']:
                people=[p for p in f[model]['tracks'] if iou(p['box'],region)>.5]
                if people:hits.append(f['frame'])
                pose_hits+=int(any(p['raw_pose'] is not None and (np.array(p['raw_pose'])[:,2]>=.3).sum()>=10 for p in people))
                smooth_hits+=int(any((np.array(p['filtered_pose'])[:,2]>=.3).sum()>=10 for p in people))
            row[model]=dict(exhibit_box_frames=len(hits),exhibit_pose_frames=pose_hits,
                            smoothed_exhibit_pose_frames=smooth_hits,frame_indices=hits)
        audit.append(row)
    save(OUT/'visual-regression-audit.json',dict(records=audit,production_replacement_recommended=False))
    test=json.loads((OUT/'evaluation/paired-test.json').read_text())['summary']
    notes={
        '02cfb195aba51c8e':'暗场漏框减少，但骨架仍会缺点；平滑会隐藏短暂结果。',
        '168083707a4147ee':'困难暗场，新增框也不够连续，平滑后没有完整骨架。',
        '35ec19ec97d6d0f0':'极暗弯腰失败案例，未进入有可靠标签的44帧指标。',
        '58f5536a4ea9f55b':'困难片段仍漏检；原/新均不稳定，未进入44帧指标。',
        '5b9fe350fe2eb33b':'极暗人物，只在部分时间检出。',
        '7e9bc4c0e9a12906':'暗场框和骨架覆盖改善。',
        'd7c5ba82952df18d':'可观察新模型与平滑的分别作用。',
        'f50f562c1c39d257':'新增骨架仍短暂、断续，持续时间筛选后明显减少。',
        '4078760fdadd05f6':'此前确认示例；训练内回放，不能代表泛化。',
        '1d4d072b22d10fe1':'此前确认示例；框改善，弯腰骨架仍不完整。',
        '56f605e0ad40c73a':'亮场回归：左侧人形展品新增误检，平滑无法消除。',
        'da3c570dcc033d1b':'亮场回归：左侧人形展品被持续画出骨架。',
        '7ce9e264c4e2c972':'亮场动作姿态回归；固定动作模型未参与本视频。',
        'fb6b017c01b4e300':'亮场other回归，供检查额外假人。'}
    cards=[]
    for v in videos:
        name=html.escape(Path(v['source']).name);uid=v['id']
        category='暗场留出' if v in dark_test else '亮场回归' if v in bright else '训练内示例'
        cards.append(f'''<article data-category="{category}"><div class="meta">{category} · {v['frames']} 帧 · {v['frames']/v['fps']:.2f} 秒</div>
<h2>{name}</h2><p>{html.escape(notes.get(uid,''))}</p>
<video controls preload="none" poster="posters/{uid}.jpg" src="videos/{uid}.mp4"></video>
<a href="videos/{uid}.mp4" download>下载对照视频 ↗</a></article>''')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>暗场人体与姿态微调 · 2026-10-24</title><style>
:root{color-scheme:dark;--bg:#10191d;--panel:#1a272c;--ink:#ecf2ef;--muted:#afc3c2;--accent:#a8e6b7}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.7 system-ui,sans-serif}
main{max-width:1480px;margin:auto;padding:48px 28px}header{max-width:1000px;margin-bottom:32px}.eyebrow{letter-spacing:.16em;color:var(--accent);font-size:13px}h1{font-size:clamp(30px,4vw,50px);line-height:1.2;margin:14px 0 24px}p{color:var(--muted)}.notice{padding:18px 22px;border-left:3px solid #e5bc79;background:#302d24;color:#f1dfc0}
.stats{display:flex;gap:24px;flex-wrap:wrap;margin:28px 0}.stat{padding:16px 24px;background:var(--panel);min-width:210px}.stat b{display:block;font-size:28px;color:var(--accent)}
nav{display:flex;gap:10px;flex-wrap:wrap;margin:30px 0}button{font:inherit;background:transparent;border:1px solid #617672;color:var(--ink);padding:7px 18px;cursor:pointer}button[aria-pressed=true]{background:var(--accent);color:#142219}button:focus-visible,a:focus-visible{outline:3px solid #e5bc79;outline-offset:3px}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:28px}article{background:var(--panel);padding:22px;min-width:0}article h2{font-size:24px;margin:4px 0}article p{min-height:54px;margin-top:8px}.meta{font-size:13px;color:var(--accent)}video{width:100%;aspect-ratio:1440/940;background:#080d10}a{color:var(--accent);text-underline-offset:4px}article a{display:inline-block;margin-top:12px}.legend{font-size:15px}footer{margin-top:40px;color:var(--muted)}[hidden]{display:none!important}@media(max-width:800px){main{padding:28px 16px}.grid{grid-template-columns:1fr}.stats{gap:10px}.stat{flex:1;padding:12px}.stat b{font-size:24px}}
</style><main><header><div class="eyebrow">DARKSCENE / EXPERIMENT 2026.10.24</div>
<h1>暗场检出改善，<br>还不能直接上线。</h1>
<p>使用总体审核认可的辅助框与骨架，经连续性、最短持续时间筛选后，微调 YOLO26m 与 RTMO-s。14 段完整视频展示训练和平滑各自的效果与代价。</p>
<div class="notice">亮场中出现人形展品的持续误检；平滑不能纠正稳定的假人。极暗弯腰仍漏检。本轮保留候选与对照，生产权重和算子包未替换。</div>
<div class="stats"><div class="stat">暗场测试辅助框命中<b>18/44 → 43/44</b></div><div class="stat">辅助关节点匹配<b>162/201 → 176/201</b></div><div class="stat">完整对照视频<b>14 段</b></div></div>
<p>以上是对筛选后辅助标签的一致性，不是独立人工精标准确率。44 帧来自5段视频；全部8段暗场测试视频均在下面，含没有可靠标签的失败案例。</p>
<p class="legend">四格读法：左上原模型，右上原模型＋平滑；左下微调模型，右下微调模型＋平滑。人体框立即显示；平滑骨架需人物确认0.4秒、关节点连续0.16秒。为诊断效果，视频主动显示暗场姿态；运行照度门仍为75。</p>
<a href="README.md">实验报告</a> · <a href="dark-finetune-video-review.zip">下载全部视频</a></header>
<nav aria-label="筛选视频"><button data-filter="全部" aria-pressed="true">全部</button><button data-filter="暗场留出" aria-pressed="false">暗场留出</button><button data-filter="亮场回归" aria-pressed="false">亮场回归</button><button data-filter="训练内示例" aria-pressed="false">训练内示例</button></nav>
<div class="grid">'''+''.join(cards)+'''</div><footer>候选权重在 models/。人物ID仅为短期跟踪号。动作分类和生产部署未在本轮替换。</footer></main>
<script>document.querySelectorAll('button[data-filter]').forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('button[data-filter]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));document.querySelectorAll('article').forEach(x=>{x.hidden=b.dataset.filter!=='全部'&&x.dataset.category!==b.dataset.filter;if(x.hidden)x.querySelector('video').pause()})}));document.querySelectorAll('video').forEach(v=>v.addEventListener('play',()=>document.querySelectorAll('video').forEach(x=>{if(x!==v)x.pause()})));</script></html>'''
    (OUT/'index.html').write_text(page)
    with (OUT/'video-index.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['source','id','split','frames','fps','raw_dark_frames','video','note'])
        writer.writeheader()
        for v in videos:writer.writerow({**{k:v[k] for k in writer.fieldnames if k!='note'},'note':notes.get(v['id'],'')})
    print(json.dumps(dict(stability=aggregate_result,audit=audit),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
