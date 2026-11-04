"""Descriptive paired object counts, gated action replay, and actual detection crops."""
from collections import Counter, defaultdict
import csv
import html
import json
from pathlib import Path
import sys

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0,str(Path(__file__).parent))
from compare import OUT, BASE, MODELS, VARIANTS, MAPPING, save, digest

CLASSES=['bottle','cup','phone','camera','paper','portable','exhibit']


def csv_save(name, rows):
    with (OUT/name).open('w',newline='',encoding='utf-8-sig')as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def bootstrap(objects, left, right, cls):
    by={}
    for r in objects:
        if r['split']!='test':continue
        if (r['model'],r['variant'])not in (left,right):continue
        by.setdefault(r['id'],{})[(r['model'],r['variant'])]=r
    groups=defaultdict(lambda:np.zeros(3))
    for d in by.values():
        a,b=d[left],d[right];aa,bb=a['counts'][str(cls)],b['counts'][str(cls)]
        assert aa['tp']+aa['fn']==bb['tp']+bb['fn']
        if aa['tp']+aa['fn']:
            groups[a['group']]+=[aa['tp'],bb['tp'],aa['tp']+aa['fn']]
    arr=np.asarray(list(groups.values())); rng=np.random.default_rng(25)
    draws=arr[rng.integers(len(arr),size=(5000,len(arr)))].sum(1)
    delta=(draws[:,0]-draws[:,1])/draws[:,2]; total=arr.sum(0)
    return dict(left=left,right=right,cls=CLASSES[cls],groups=len(arr),GT=int(total[2]),
        recall_delta=float((total[0]-total[1])/total[2]),CI95=np.quantile(delta,[.025,.975]).tolist())


def render_examples(inputs, actions):
    predictions={m:json.loads((OUT/'predictions'/f'{m}.json').read_text())['predictions'] for m in MODELS}
    windows={r['id']:r for r in inputs['windows']}
    plans=json.loads((BASE/'demo-plan.json').read_text())
    cards=[];rendered=[]
    for plan in plans:
        window=windows[f"{plan['source_id']}_{plan['event_index']}"]
        samples=[s for s in window['samples'] if s['views']]
        if not samples:continue
        # Predefined middle sampled frame and whole-person ROI; not selected by scores.
        sample=samples[len(samples)//2]; key=sample['views'][0]; meta=inputs['images'][key]
        panel=np.full((4*340,4*340,3),245,np.uint8)
        for i,variant in enumerate(VARIANTS):
            for j,name in enumerate(MODELS):
                image=cv2.imread(str(OUT/meta['variants'][variant]['path']));h,w=image.shape[:2]
                ratio=min(326/w,278/h);nw,nh=round(w*ratio),round(h*ratio)
                tile=np.full((340,340,3),245,np.uint8);x,y=(340-nw)//2,44
                tile[y:y+nh,x:x+nw]=cv2.resize(image,(nw,nh))
                for box in predictions[name][key][variant]:
                    cls=int(box[5]);conf=box[4]
                    if conf<.35:continue
                    if name!='m_finetuned':
                        if cls not in MAPPING:continue
                        cls=MAPPING[cls]
                    xy=np.asarray(box[:4])*ratio+np.array([x,y,x,y]);xy=xy.round().astype(int)
                    color=(40,100,230) if cls==0 else (180,100,10)
                    cv2.rectangle(tile,tuple(xy[:2]),tuple(xy[2:]),color,2)
                    cv2.putText(tile,f'{CLASSES[cls]} {conf:.2f}',(max(1,xy[0]),max(y+12,xy[1]-3)),
                        cv2.FONT_HERSHEY_SIMPLEX,.40,color,1,cv2.LINE_AA)
                cv2.putText(tile,f'{name} / {variant}',(6,18),cv2.FONT_HERSHEY_SIMPLEX,.43,(30,30,30),1,cv2.LINE_AA)
                cv2.putText(tile,f'input crop {w}x{h} px',(6,35),cv2.FONT_HERSHEY_SIMPLEX,.42,(50,50,50),1,cv2.LINE_AA)
                panel[i*340:(i+1)*340,j*340:(j+1)*340]=tile
        path=OUT/'examples'/f'{plan["id"]}.jpg';path.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(path),panel,[cv2.IMWRITE_JPEG_QUALITY,92])
        title=plan['title']
        if plan['id']=='old-drink-supported':title='旧测试：真实喝水（喝水38）'
        if plan['id']=='old-drink-missed':title='旧测试：真实喝水（喝水36），检查漏检'
        if plan['id']=='old-pickup-missed':title='旧测试：真实捡物（捡东西23），检查类别覆盖'
        if plan['id']=='new09-supported':title='新09：手持相机，核对是否错分为瓶'
        decisions=[r for r in actions if r['id']==window['id']]
        table='<table><tr><th>模型</th>'+''.join(f'<th>{v}</th>' for v in VARIANTS)+'</tr>'
        for name in MODELS:
            table+='<tr><td>'+name+'</td>'
            for variant in VARIANTS:
                result=next(r['decision'] for r in decisions if r['model']==name and r['variant']==variant)
                result_label={'drink':'喝水','pick_up':'捡物','other':'其他'}[result['label']]
                reason={'single_frame_support':'有物体支持','no_compatible_object':'无相容物体',
                        'no_valid_bright_observation':'观测不可用，跳过','below_luma_gate':'低亮，跳过',
                        'unaffected_action':'原动作不触发先验'}.get(result['reason'],result['reason'])
                table+=f'<td>{result_label}<br><small>{reason}</small></td>'
            table+='</tr>'
        table+='</table>'
        cards.append(f'<article><h2>{html.escape(title)}</h2><p>固定事件 {plan["event_index"]}；原片帧 {sample["frame"]}；同一人物裁剪。图中文字是物体模型预测。</p><a href="examples/{path.name}"><img loading="lazy" src="examples/{path.name}"></a><p>下表按该窗口全部有效采样帧及局部ROI决定，上图只展示预先选定的中间帧全人ROI。</p>{table}</article>')
        rendered.append(dict(case=plan['id'],source_id=plan['source_id'],event_index=plan['event_index'],
            frame=sample['frame'],view_id=key,image=str(path.relative_to(OUT))))
    save(OUT/'examples.json',rendered)
    return cards


def main():
    objects=json.loads((OUT/'object-results.json').read_text())
    actions=json.loads((OUT/'action-results.json').read_text())
    inputs=json.loads((OUT/'inputs.json').read_text())
    summary=[];old=[];feedback=[]
    for split in ('val','test'):
        for name in MODELS:
            for variant in VARIANTS:
                for cls in range(7):
                    rows=[r for r in objects if (r['split'],r['model'],r['variant'])==(split,name,variant)]
                    tp=sum(r['counts'][str(cls)]['tp']for r in rows);fp=sum(r['counts'][str(cls)]['fp']for r in rows)
                    fn=sum(r['counts'][str(cls)]['fn']for r in rows)
                    summary.append(dict(split=split,model=name,variant=variant,cls=CLASSES[cls],
                        supported=name=='m_finetuned'or cls<3,images=len(rows),tp=tp,fp=fp,fn=fn,
                        recall=tp/(tp+fn)if tp+fn else None,precision=tp/(tp+fp)if tp+fp else None))
                for truth in ('drink','pick_up','other'):
                    rows=[r for r in actions if(r['split'],r['model'],r['variant'],r['truth'])==(split,name,variant,truth)]
                    old.append(dict(split=split,model=name,variant=variant,truth=truth,windows=len(rows),
                        baseline_correct=sum(r['baseline']==truth for r in rows),
                        correct=sum(r['decision']['label']==truth for r in rows),
                        common_correct=sum(r['common_decision']['label']==truth for r in rows),
                        gate_bypass=sum(r['decision']['reason']=='below_luma_gate'for r in rows)))
    for name in MODELS:
        for variant in VARIANTS:
            for action in ('drink','pick_up'):
                rows=[r for r in actions if(r['split'],r['model'],r['variant'],r['baseline'])==('feedback',name,variant,action)]
                feedback.append(dict(model=name,variant=variant,action=action,windows=len(rows),
                    retained=sum(r['decision']['label']==action for r in rows),
                    supported=sum(r['decision']['reason']=='single_frame_support'for r in rows),
                    bypassed=sum(not r['decision']['applied']for r in rows),
                    common_retained=sum(r['common_decision']['label']==action for r in rows)))
    csv_save('object-counts.csv',summary);csv_save('old-actions.csv',old);csv_save('feedback-counts.csv',feedback)
    native_ids={key for key,r in inputs['images'].items()
                if r['kind']=='holdout' and not r['processed'] and r['raw_luma']>=30}
    native_summary=[]
    for row in summary:
        cls=CLASSES.index(row['cls'])
        selected=[r for r in objects if r['id'] in native_ids and
                  (r['split'],r['model'],r['variant'])==(row['split'],row['model'],row['variant'])]
        tp=sum(r['counts'][str(cls)]['tp'] for r in selected)
        fp=sum(r['counts'][str(cls)]['fp'] for r in selected)
        fn=sum(r['counts'][str(cls)]['fn'] for r in selected)
        native_summary.append(dict(row,images=len(selected),tp=tp,fp=fp,fn=fn,
            recall=tp/(tp+fn) if tp+fn else None,precision=tp/(tp+fp) if tp+fp else None))
    csv_save('object-counts-raw-source.csv',native_summary)
    contrasts=[]
    for left,right in [(('n','native_raw'),('m','native_raw')),(('s','native_raw'),('m','native_raw')),
                       (('s','native_raw'),('s','small_raw')),(('s','small_enhanced'),('s','small_raw')),
                       (('m_finetuned','native_raw'),('m_finetuned','small_enhanced'))]:
        for cls in (0,2):contrasts.append(bootstrap(objects,left,right,cls))
    save(OUT/'statistics.json',dict(runs=1,training_runs=0,bootstrap_seed=25,bootstrap_repeats=5000,
        contrasts=contrasts,method='paired source-group percentile bootstrap; descriptive same-scene auxiliary labels',
        significance='No p values: single fixed model run, sparse classes, correlated scenes, exploratory feedback'))
    figures=OUT/'figures';figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(12,4.2));palette=['#1f77b4','#70b1c5','#e1ad62','#ab635b']
    for ax,cls in zip(axes,('bottle','phone')):
        for j,variant in enumerate(VARIANTS):
            rr=[next(r for r in summary if(r['split'],r['model'],r['variant'],r['cls'])==('test',m,variant,cls))for m in MODELS]
            xs=np.arange(4)+(j-1.5)*.19;vals=[r['tp']for r in rr]
            ax.bar(xs,vals,width=.18,label=variant,color=palette[j])
            for x,r in zip(xs,rr):ax.text(x,r['tp']+.08,f"{r['tp']}/{r['tp']+r['fn']}",ha='center',fontsize=7)
        ax.set_xticks(range(4),list(MODELS));ax.set_title(cls+' / old test reviewed ROIs');ax.set_ylim(0,14 if cls=='bottle'else 4)
        ax.set_ylabel('Matched GT boxes (IoU >= 0.5)')
    axes[0].legend(fontsize=7);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(figures/f'01-object-recall.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.2))
    for ax,action in zip(axes,('drink','pick_up')):
        for j,variant in enumerate(VARIANTS):
            rr=[next(r for r in feedback if(r['model'],r['variant'],r['action'])==(m,variant,action))for m in MODELS]
            xs=np.arange(4)+(j-1.5)*.19
            ax.bar(xs,[r['supported']for r in rr],width=.18,label=variant,color=palette[j])
        ax.set_xticks(range(4),list(MODELS));ax.set_title(f'New feedback: {action}');ax.set_ylabel('Windows with object support (not accuracy)')
    axes[0].legend(fontsize=7);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(figures/f'02-feedback-support.{ext}',dpi=180)
    plt.close(fig)
    cards=render_examples(inputs,actions)
    table=['|模型|输入|瓶检对/真值；误框|手机检对/真值；误框|新喝水支持/527|新捡物支持/37|','|---|---|---|---|---:|---:|']
    for name in MODELS:
        for variant in VARIANTS:
            obj=[next(r for r in summary if(r['split'],r['model'],r['variant'],r['cls'])==('test',name,variant,c))for c in ('bottle','phone')]
            act=[next(r for r in feedback if(r['model'],r['variant'],r['action'])==(name,variant,c))for c in ('drink','pick_up')]
            table.append(f"|{name}|{variant}|{obj[0]['tp']}/{obj[0]['tp']+obj[0]['fn']}；{obj[0]['fp']}|{obj[1]['tp']}/{obj[1]['tp']+obj[1]['fn']}；{obj[1]['fp']}|{act[0]['supported']}|{act[1]['supported']}|")
    (OUT/'numeric-summary.md').write_text('\n'.join(table)+'\n')
    gate_rows=[r for r in actions if r['model']=='n'and r['variant']=='native_raw'and r['decision']['reason']=='below_luma_gate']
    assert len(gate_rows)==8 and all(r['decision']['label']==r['baseline'] for r in gate_rows)
    dark_ids={r['id'] for r in gate_rows}
    assert all(not r['samples'] for r in inputs['windows'] if r['id'] in dark_ids)
    assert all(r['decision']['label']==r['baseline'] and not r['decision']['applied']
               for r in actions if r['id'] in dark_ids)
    save(OUT/'summary.json',dict(object_summary=summary,native_object_summary=native_summary,old_actions=old,feedback=feedback,
        gate_skipped_windows=[{k:r[k]for k in ('id','raw_luma','truth','baseline')}for r in gate_rows],
        images=len(inputs['images']),window_count=len(inputs['windows'])))
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>YOLO26n/s 与高清裁剪对照</title><style>body{font:16px/1.65 system-ui;max-width:1400px;margin:auto;padding:24px;color:#203040;background:#f4f6f8}article{padding:20px;background:white;margin:20px 0}img{width:100%;height:auto}a{color:#135ca4}h2{font-size:20px}table{border-collapse:collapse;width:100%}td,th{padding:8px;border:1px solid #ccd5dd;text-align:left}</style>
<h1>YOLO26n／s／m 与物体检测输入对照</h1><p>物体阈值0.35、动作阈值0.85、原图照度门30。窗口或采样原图太暗时跳过物体先验，不因缺失物体证据否决动作。</p>
<article><h2>用户确认后的误报收益与漏报代价</h2><p>已确认全9段的喝水触发只有原片01/T317/149.92秒是真的。当前527窗为1真、526误报：高清原图n/s/现场微调m分别过滤517/522/436个误报，同时均过滤这一个真喝水。<a href="false-positive-tradeoff.md">查看全部16组合、审核样本和连续候选段统计</a>。</p><img src="figures/03-false-positive-tradeoff.png"></article>
<article><h2>捡物：候选过滤与旧标签漏报权衡</h2><p>新数据37个捡物触发，您认为基本都是误报；逐窗标签尚未完成，因此这里只称候选抑制量。微调m高清过滤33/37、旧test保留真捡物4/4并过滤1/3误报。<a href="pickup-tradeoff.md">查看捡物统计与逐窗结果</a>。微调m高清仍放行的4个窗口见<a href="examples/pickup-native-retained.jpg">采样帧复核图</a>。</p><img src="figures/04-pickup-tradeoff.png"></article>
<p>native_raw：增强前高清裁剪；native_gamma：高清裁剪加gamma；small_raw：长边720原图裁剪；small_enhanced：长边720＋Retinexformer后裁剪。历史已处理回传无法还原增强前像素，单独记录processed；新9段均有原始高清视频。</p>
<p>n／s／m为公开COCO权重，只比较共同的瓶／杯／手机；m_finetuned为现场七类物体候选。公开模型不支持相机、纸、展品标签。下图对公开模型只画共同三类预测，微调模型画七类；没有把预测当真值。</p>
<p><a href="analysis-report.md">结论报告</a> · <a href="false-positive-tradeoff.md">喝水误报／漏报</a> · <a href="pickup-tradeoff.md">捡物误报／漏报</a> · <a href="numeric-summary.md">全部数字</a> · <a href="object-counts.csv">逐类检测表</a> · <a href="old-actions.csv">旧动作真值对照</a> · <a href="feedback-counts.csv">新录像候选保留表</a> · <a href="pickup-counts.csv">捡物统计</a> · <a href="pickup-window-results.csv">捡物逐窗口</a></p>
<img src="figures/01-object-recall.png"><p>同一旧test审核区域，比较IoU≥0.5的检对框；真值为辅助视觉标注，小样本同场景，误框见表。</p>
<img src="figures/02-feedback-support.png"><p>仅显示有物体支持的窗口；绑定不可靠而跳过过滤的窗口不混入支持数。喝水已获用户触发范围确认，误报与漏报请看上方补充；捡物图只反映支持数量，新触发逐窗真值未确认。</p>
''' + ''.join(cards)+'</html>'
    (OUT/'index.html').write_text(page)
    print('\n'.join(table),flush=True)


if __name__=='__main__':main()
