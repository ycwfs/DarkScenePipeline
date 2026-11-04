"""Analysis from completed predictions; no prompt/model/threshold tuning here."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import html
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from calibrate import OUT,LABELS,metrics

ROOT=OUT.parents[1]


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def predict(prob,tau):
    p=np.asarray(prob);i=p.argmax(1);i[(i!=9)&(p.max(1)<tau)]=9
    return i


def measured(rows,pred):
    return metrics(rows,np.eye(10)[pred],0)


def decode(p):
    c=subprocess.run(['ffmpeg','-nostdin','-v','error','-xerror','-threads','2','-i',str(p),'-f','null','-'],capture_output=True,text=True)
    assert c.returncode==0 and not c.stderr,(p,c.stderr)
    q=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames,width,height,codec_name','-of','json',str(p)],capture_output=True,text=True,check=True)
    return dict(path=str(p.relative_to(OUT)),decode_passed=True,**json.loads(q.stdout)['streams'][0])


def main():
    tau=json.loads((OUT/'selection.json').read_text())['threshold'];assert tau==.41
    metadata=json.loads((ROOT/'reports/threshold-tuning-20261026/windows.json').read_text());lookup={r['id']:i for i,r in enumerate(metadata)}
    inputs=np.load(ROOT/'reports/four-parameter-tuning-20261026/inputs.npz')
    wave={r['id']:r['variants']['3_total']for r in json.loads((ROOT/'reports/wave-pose-prior-20261027/old-window-results.json').read_text())}
    results={};all_rows={};all_preds={}
    for split,n in [('val',60),('test',58)]:
        files=list((OUT/'old'/split).glob('*.json'));assert len(files)==n
        rows=[r for p in sorted(files)for r in json.loads(p.read_text())['events']];all_rows[split]=rows
        rgb=inputs['rgb'][[lookup[r['id']]for r in rows]];xp=np.array([r['xclip_probs']for r in rows])
        preds={f'xclip_{t}':predict(xp,t)for t in [0,.4,.41,.75,.85,.925]}
        preds.update(videomamba_rgb_0925=predict(rgb,.925),videomamba_rgb_075=predict(rgb,.75),
            current_full=np.array([LABELS.index(wave[r['id']])for r in rows]))
        all_preds[split]=preds
        results[split]=dict(sources=n,sources_with_windows=sum(bool(json.loads(p.read_text())['events'])for p in files),
            person_windows=len(rows),known=sum(r['truth']is not None for r in rows),
            methods={k:measured(rows,p)for k,p in preds.items()})
    # Recording-group bootstrap on the paired historical test windows; single model run, not seed variance.
    rows=all_rows['test'];known=[i for i,r in enumerate(rows)if r['truth']is not None]
    group_names=sorted({rows[i]['group']for i in known});groups={g:[i for i in known if rows[i]['group']==g]for g in group_names}
    y=np.array([LABELS.index(r['truth'])if r['truth']is not None else -1 for r in rows]);classes=sorted(set(y[known]));rng=np.random.default_rng(27)
    scores=[]
    def mf(pred,idx):
        yy=y[idx];pp=pred[idx];vals=[]
        for c in classes:
            tp=((pp==c)&(yy==c)).sum();den=(pp==c).sum()+(yy==c).sum();vals.append(2*tp/den if den else np.nan)
        return float(np.nanmean(vals))
    for _ in range(2000):
        idx=np.array([i for g in rng.choice(group_names,len(group_names),replace=True)for i in groups[g]])
        scores.append(mf(all_preds['test']['xclip_0.41'],idx)-mf(all_preds['test']['current_full'],idx))
    boot=dict(groups=len(groups),replicates=2000,seed=27,macro_f1_difference_xclip_minus_full=float(measured(rows,all_preds['test']['xclip_0.41'])['macro_f1']-measured(rows,all_preds['test']['current_full'])['macro_f1']),
        percentile95=np.percentile(scores,[2.5,97.5]).tolist(),method='Resample original source groups. Descriptive uncertainty, no significance or external-generalization claim.')
    longs=[]
    for n in [1,7,9]:
        dest=OUT/'long-videos'/f'new-{n:02}';summary=json.loads((dest/'summary.json').read_text());events=[json.loads(s)for s in (dest/'events.jsonl').read_text().splitlines()]
        baseline=[json.loads(s)for s in (ROOT/'reports/wave-pose-prior-20261027/long-videos'/f'new-{n:02}'/'candidate-events.jsonl').read_text().splitlines()]
        summary['threshold_counts']={str(t):dict(Counter(LABELS[i]for i in predict([r['rgb_probs']for r in events],t)))for t in [.4,.41,.75,.925]}
        summary['videomamba_full_counts']=dict(Counter(r['label']for r in baseline));summary['baseline_windows']=len(baseline)
        summary['comparison_limit']='Independent no-pose tracking changes windows; counts are triggers, not false-positive counts. Only reviewed anchors below have labels.'
        longs.append(summary)
    # Match the ten already reviewed running candidates by time+box, never by coincidental track number.
    base_dir=ROOT/'reports/wave-pose-prior-20261027/long-videos/new-01'
    baseline=[json.loads(s)for s in (base_dir/'candidate-events.jsonl').read_text().splitlines()]
    xrows=[json.loads(s)for s in (OUT/'long-videos/new-01/events.jsonl').read_text().splitlines()]
    gt={r['frame']:r for r in json.loads((ROOT/'reports/running-speed-prior-20261027/review/labels.json').read_text())['labels']}
    def overlap(a,b):
        a,b=np.array(a),np.array(b);inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
        return float(inter/max(1e-8,(a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod()-inter))
    anchors=[]
    for b in baseline:
        if b['frame_index']not in gt or b['label']!='chase':continue
        match=sorted([(overlap(b['bbox'],x['bbox']),abs(b['timestamp']-x['timestamp']),x)for x in xrows if abs(b['timestamp']-x['timestamp'])<=.28],key=lambda t:(-t[0],t[1]))
        r=dict(**gt[b['frame_index']],timestamp=b['timestamp'],matched=bool(match and match[0][0]>=.5))
        if r['matched']:
            ov,dt,x=match[0];r.update(iou=ov,time_delta=dt,xclip_track=x['track_id'],xclip_timestamp=x['timestamp'],
                labels={str(t):LABELS[predict([x['rgb_probs']],t)[0]]for t in [.4,.41,.75,.925]})
        anchors.append(r)
    save(OUT/'analysis-results.json',dict(old=results,long=longs,bootstrap=boot,reviewed_running_anchors=anchors))
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    methods=['current_full','videomamba_rgb_0925','xclip_0.41','xclip_0.75','xclip_0.925'];names=['Current full','VideoMamba RGB .925','XCLIP .41 (val)','XCLIP .75 (fixed)','XCLIP .925 (fixed)']
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    mm=results['test']['methods'];axs[0].barh(names,[mm[k]['macro_f1']for k in methods]);axs[0].set_xlim(0,1);axs[0].set_xlabel('Historical test macro F1')
    axs[1].barh(names,[mm[k]['other_false']for k in methods],color='#ce6548');axs[1].set_xlabel('False action on 163 known other windows')
    fig.tight_layout();fig.savefig(figdir/'test-comparison.png',dpi=180);fig.savefig(figdir/'test-comparison.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.5));classes=['wave','chase','drink','pick_up'];xx=np.arange(len(classes));width=.22
    for j,key in enumerate(['current_full','xclip_0.41','xclip_0.75']):
        ax.bar(xx+(j-1)*width,[mm[key]['per_class'][c]['recall']for c in classes],width,label=key)
    ax.set_xticks(xx,['Wave (11)','Running (25)','Drink (11)','Pick up (4)']);ax.set_ylim(0,1.1);ax.set_ylabel('Historical test recall');ax.legend();fig.tight_layout();fig.savefig(figdir/'class-recall.png',dpi=180);fig.savefig(figdir/'class-recall.pdf');plt.close(fig)
    videos=list((OUT/'long-videos').rglob('*.mp4'))+list((OUT/'short-videos').rglob('*.mp4'))
    with ThreadPoolExecutor(max_workers=3)as pool:verified=list(pool.map(decode,videos))
    for v in verified:
        summary=json.loads((OUT/v['path']).parent.joinpath('summary.json').read_text());assert int(v['nb_frames'])==summary['frames']
    save(OUT/'verification.json',dict(passed=True,video_count=len(verified),videos=verified,old_source_files=118,
        known_windows=results['val']['known']+results['test']['known'],full_frames=sum(r['frames']for r in longs),
        no_pose_network=True,no_skeleton_classifier=True,no_object_model=True,no_finetuning=True,
        optimized_forward='Every inference process asserted max logit error<.03 vs stock HF fp16 forward before scoring. PID1 output filename collided across containers; retained one parity JSON only.'))
    table='\n'.join(f'|{name}|{mm[k]["macro_f1"]:.3f}|{mm[k]["other_false"]}/163|'+ '/'.join(str(mm[k]['per_class'][c]['tp'])for c in classes)+'|' for k,name in zip(methods,names))
    report=f'''# 逐人 XCLIP 零样本实验（2026-10-27）

使用公开 microsoft/xclip-base-patch16-zero-shot，未微调。增强＋现场人体检测＋逐人RGB裁剪；完整长视频中不加载姿态模型、ST-GCN、物体模型，也不使用任何骨架或物体先验。结论：当前零样本设置不能直接替换现有方案来优先压低误报。

验证集选定拒识阈值 **{tau}**。默认公开对照 .4 也完整保留；.75/.85/.925 是评分前固定的诊断档，没有根据测试结果重选。提示词仅在推理前修正“chase”为单人跑动、删除other中的独自跑步，其余不调。

|方法|测试宏F1|other误报|挥手/跑动/喝水/捡物正确窗（分母11/25/11/4）|
|---|---:|---:|---|
{table}

60个旧val源、58个旧test源均列入清单；分别{results['val']['sources_with_windows']}、{results['test']['sources_with_windows']}源有窗口。val226窗中198有真值；test252窗中214有真值，其余未知不计准确率。没有稳定人体窗口不能算正确other。固定人物/时间窗对照沿用旧gate35跟踪框；这些框来自原姿态辅助跟踪，因此该部分是**分类器对照**，与完整无姿态跟踪的长视频分开。

XCLIP 32帧/1秒，逐人15%外扩、保持比例填充224×224；VideoMamba现有输入224×400。均来自同帧720增强像素，XCLIP使用其预训练mean/std。XCLIP概率与现场微调VideoMamba不具有同样的置信度标尺。Current full包含ST-GCN、7类物体和挥手先验；单列纯VideoMamba RGB以暴露这项区别。

完整01/07/09共{sum(r['frames']for r in longs)}帧、{sum(r['source']['duration']for r in longs):.2f}秒，XCLIP得到{sum(r['events']for r in longs)}人物窗，原方案16913窗。完整视频绘制事前主阈值.4，短视频绘制val选定.41。不同跟踪流程的人物编号不可直接比较；10个已审核跑动触发按时间差≤.28秒和IoU≥.5匹配，明细见analysis-results.json，未匹配保留。长视频总触发数不能直接叫误报，未穷尽标注所有动作与时段。

可查看[index.html](index.html)中的三段完整视频与12段旧现场短视频对照。全部{len(verified)}视频完整解码及帧数核验通过。不是实时吞吐、NFS或无限长流验收。

这是单次公开权重推理；历史测试集已被前期研究查看，VideoMamba也曾用现场train微调，不能称全新场景的公平训练预算对照。{boot['groups']}个来源组配对bootstrap的宏F1差（XCLIP.41−当前完整方案）为{boot['macro_f1_difference_xclip_minus_full']:.3f}，95%描述区间{boot['percentile95']}；不作显著性或泛化保证。

模型参考：Ni et al., Expanding Language-Image Pretrained Models for General Video Recognition, ECCV 2022（X-CLIP）；公开权重 microsoft/xclip-base-patch16-zero-shot。模型SHA、提示词SHA、协议见protocol.json；本轮未替换正式算子包和镜像。
'''
    (OUT/'analysis-report.md').write_text(report)
    (OUT/'stats-appendix.md').write_text('# 统计口径\n\n单次推理，不存在多seed训练方差。单位是重叠人物时间窗，按旧原视频来源group抽样2000次给出配对宏F1差的百分位描述区间。未做p值检验或多参数显著性筛选。阈值只以val选择；历史test仅报告，已历史查看。未知GT和无跟踪窗口排除且单列。\n\n'+json.dumps(boot,ensure_ascii=False,indent=2)+'\n')
    (OUT/'figure-catalog.md').write_text('# 图表\n\n- test-comparison.png/.pdf：214个已知测试窗的宏F1及163个other误报。无误差条；确切计数，不能把高阈值造成的少误报等同完整动作召回。\n- class-recall.png/.pdf：各类召回及分母，强调捡物只有4窗、零样本阈值升高时会漏动作。无推断统计。\n')
    cards=[]
    for v in verified:
        path=v['path'];meta=json.loads((OUT/path).parent.joinpath('summary.json').read_text())
        source=meta['source'];name=Path(source.get('source',source.get('absolute_path',path))).name
        title=f'{name} / '+('完整无骨架流程，阈值0.4'if path.startswith('long')else f'旧{source["split"]}，固定人物窗对照，XCLIP阈值0.41')
        cards.append(f'<article><h3>{html.escape(title)}</h3><video controls preload="metadata" width="900" src="{html.escape(path)}"></video></article>')
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>XCLIP逐人实验</title><style>body{font-family:sans-serif;max-width:1100px;margin:30px auto;background:#f5f6f8}article{padding:16px;margin:18px 0;background:white}video,img{max-width:100%}</style><h1>XCLIP逐人零样本对照</h1><p>完整长视频阈值0.4；旧短视频对照阈值0.41（验证集选择）。增强＋逐人RGB，无骨架、物体先验。短片左边为当前VideoMamba完整流程的最终标签，右边为XCLIP。</p><p><a href="analysis-report.md">报告</a> · <a href="analysis-results.json">全部数值</a> · <a href="../running-speed-prior-20261027/index.html">跑动速度先验</a></p><img src="figures/test-comparison.png"><img src="figures/class-recall.png">'+''.join(cards))
    print(json.dumps(dict(verification=True,videos=len(verified),test=mm['xclip_0.41'],baseline=mm['current_full']),ensure_ascii=False))

if __name__=='__main__':main()
