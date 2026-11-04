"""Summarize paired ROI detection evidence, including unwanted priors on Other."""
from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path

import numpy as np
os.environ.setdefault('MPLCONFIGDIR', '/tmp/darkpipe-roi-prior-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import binomtest

OUT = Path(__file__).resolve().parents[1]
VARIANTS = ['full_enh720', 'person_enh720', 'person_native', 'person_raw720', 'full_native']
MAIN = VARIANTS[:3]
NAMES = ['Full enhanced frame', 'Enhanced person ROI', 'Native person ROI']


def counts(rows: list, variant: str, metric: str, threshold: float) -> dict:
    hits = [r for r in rows if r['variants'][variant][metric] > threshold]
    return dict(hits=len(hits), n=len(rows), rate=len(hits)/len(rows) if rows else None,
                videos_hit=len({r['video_id'] for r in hits}), videos=len({r['video_id'] for r in rows}),
                groups=len({r['group'] for r in rows}))


def cluster_compare(rows: list, variant: str, metric: str) -> dict:
    grouped = defaultdict(list)
    for r in rows:
        grouped[r['group']].append([r['variants'][v][metric] > .25 for v in ['full_enh720', variant]])
    pairs = np.array([np.mean(grouped[k], axis=0) for k in sorted(grouped)])
    if not len(pairs):
        return dict(groups=0)
    delta = pairs[:, 1]-pairs[:, 0]
    rng = np.random.default_rng(2302)
    boot = delta[rng.integers(0, len(delta), (5000, len(delta)))].mean(1)
    wins, losses = int((delta > 1e-12).sum()), int((delta < -1e-12).sum())
    return dict(groups=len(delta), before_mean=float(pairs[:, 0].mean()), after_mean=float(pairs[:, 1].mean()),
        before_sd=float(pairs[:, 0].std(ddof=1)) if len(delta)>1 else None,
        after_sd=float(pairs[:, 1].std(ddof=1)) if len(delta)>1 else None,
        delta=float(delta.mean()), delta_ci95=np.quantile(boot, [.025,.975]).tolist(),
        increased=wins, decreased=losses, tied=len(delta)-wins-losses,
        p=float(binomtest(wins, wins+losses, .5).pvalue) if wins+losses else 1.)


def main() -> None:
    protocol = json.loads((OUT/'protocol.json').read_text())
    assert json.loads((OUT/'completion.json').read_text())['completed']
    data = [json.loads((OUT/'data'/v['id']/'results.json').read_text()) for v in protocol['cases']]
    rows = [r for d in data for r in d['records']]
    active = [r for r in rows if r['gate_active'] and r['variants']['full_enh720']['pose_usable']]
    cohorts = {
        'drink_all': [r for r in rows if r['truth']=='drink'],
        'drink_active_pose': [r for r in active if r['truth']=='drink'],
        'pickup_all': [r for r in rows if r['truth']=='pick_up'],
        'pickup_active_pose': [r for r in active if r['truth']=='pick_up'],
        'other_all': [r for r in rows if r['truth']=='other'],
        'other_active_pose': [r for r in active if r['truth']=='other'],
        'all': rows}
    results = {}
    for name, subset in cohorts.items():
        results[name] = {v: {metric: {str(t): counts(subset,v,metric,t) for t in [.25,.1,.05]}
                                   for metric in ['container_score','drink_score','pickup_score']} for v in VARIANTS}
    coverage = {}
    for label in ['drink','pick_up','wave','other']:
        subset = [d for d in data if d['item']['video_label']==label]
        coverage[label] = dict(videos=len(subset), frames=sum(len(d['frames'])for d in subset),
            frames_with_person=sum(f['tracks']>0 for d in subset for f in d['frames']),
            no_person_videos=sum(not d['records'] for d in subset),
            videos_with_reviewed_action=sum(any(r['truth']==label for r in d['records'])for d in subset))
    stats = {}
    for cohort, metric in [('drink_active_pose','drink_score'),('pickup_active_pose','pickup_score'),
                            ('other_active_pose','drink_score'),('other_active_pose','pickup_score')]:
        for variant in MAIN[1:]:
            stats[f'{cohort}/{metric}/{variant}'] = cluster_compare(cohorts[cohort],variant,metric)
    ordered = sorted(stats, key=lambda k:stats[k].get('p',1.)); previous=0.
    for i,k in enumerate(ordered):
        previous=max(previous,min(1.,(len(ordered)-i)*stats[k].get('p',1.)))
        stats[k]['holm_p']=previous
    costs={v:dict(calls=sum(d['detector_calls'].get(v,0)for d in data),
                  seconds=sum(d['detector_seconds'].get(v,0)for d in data))for v in VARIANTS}
    result=dict(coverage=coverage, cohorts=results, group_statistics=stats, detector_costs=costs,
                truth_counts=dict(Counter(r['truth']for r in rows)),
                total_person_frames=len(rows), active_person_frames=len(active),
                production_parity_videos=sum(bool(d['production_parity_frames'])for d in data))
    (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    with (OUT/'paired-person-frames.csv').open('w')as f:
        fields=['source','video_id','group','frame','timestamp','track_id','truth','luma','gate_active','pose_usable']
        fields += [f'{v}_{m}'for v in VARIANTS for m in ['container_score','drink_score','pickup_score']]
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in rows:
            entry={k:r[k]for k in fields[:9]};entry['pose_usable']=r['variants']['full_enh720']['pose_usable']
            entry.update({f'{v}_{m}':r['variants'][v][m]for v in VARIANTS for m in ['container_score','drink_score','pickup_score']})
            writer.writerow(entry)
    table=['| 样本/观测（阈值0.25） | 整帧增强 | 增强人物ROI | 原始像素人物ROI |', '|---|---:|---:|---:|']
    specs=[('drink_all','container_score','已标注喝水人物：ROI内出现容器候选'),
           ('drink_active_pose','drink_score','喝水人物：容器靠近本人手腕'),
           ('pickup_active_pose','pickup_score','捡物人物：任意非人框靠近脚踝'),
           ('other_active_pose','drink_score','other人物：也满足容器靠近手腕'),
           ('other_active_pose','pickup_score','other人物：也满足非人框靠近脚踝')]
    for cohort,metric,label in specs:
        cells=[results[cohort][v][metric]['0.25']for v in MAIN]
        table.append('| '+label+' | '+' | '.join(f'{c["hits"]}/{c["n"]}'for c in cells)+' |')
    table='\n'.join(table)
    text=f'''# 人物裁剪区域物体检测：配对诊断

固定YOLO26m COCO-80 ONNX、640输入和物体阈值0.25；动作阈值0.75与本实验无关。
222段视频（57喝水、33捡物、128线上other、4挥手），每段6帧，共1332帧。
形成{len(rows)}个人物帧；以下正动作真值来自既有逐人时间标注，负样本沿用用户确认的other标签。
人物帧是检测/跟踪候选，仍包含把展品识别成人的情况；动作按人物时间段划分，不直接使用视频文件名。
抽样横跨整段视频，不是实际部署的每1秒窗口6帧，不能把帧级先验直接当作线上事件成绩。

{table}

前三列始终使用同一人物框与姿态。容器候选= bottle / wine glass / cup / bowl；
“靠近手腕/脚踝”沿用现有归一化躯干距离规则，但绑定到该人，且仅统计照度门允许、单帧姿态合格的行。
这里是检测输出和规则支持计数，尚不是经逐框真值验证的物体准确率、召回率或动作准确率。
other可能真的拿着杯子；那属于有效物体检测、无效喝水证据，不应一概算作检测误报。

## 输入对照

1. `full_enh720`：当前720增强整帧送640检测器；整帧结果绑定到同一人物。
2. `person_enh720`：在同一增强帧按人体框外扩15%裁剪后送640；这是主要裁剪效应对照。
3. `person_native`：将同一ROI映射回原视频像素再检测。现场视频为原图，平台切片已增强；后者不能恢复原图。
4. `person_raw720`：原图先缩720再裁剪，用于区分增强与原生像素差异；平台样本与增强ROI相同。
5. `full_native`：原生整帧检测作为附加对照；仍会整体缩到640，不能保留所有小物体细节。

没有使用动作模型的400×224缓存图作为检测输入，也没有根据物体结果调整裁剪位置。
人体框复用本次已冻结的逐人跟踪结果；没有检出人时无法生成自动ROI，不能用人工框掩盖这一失败。
对暗帧额外执行姿态只用于诊断，主要几何表仍按照度门筛选。生产先验和alpha融合均未改动。

## 覆盖与统计边界

各类采样帧的人体覆盖见`summary.json/coverage`；不确定人物/未标注动作时间不纳入正负对比。
数据来自同一摄像机，部分旧片是同次录制的裁片；沿用已有录制家族/平台会话分组。
按组的配对区间和符号检验只描述“证据出现率”的变化，不是检测精度显著性；全部8项比较做Holm校正。
无完整物体框真值，不能报告mAP。未改变或再次训练任何模型，未使用本实验调动作阈值。

逐人每帧数据见[CSV](paired-person-frames.csv)，低分候选与映射坐标见`data/*/results.json`；
模型哈希、输入形状及逐视频生产后处理对齐验证见`model-provenance.json`和`summary.json`。
'''
    (OUT/'analysis-report.md').write_text(text)
    appendix=['# 分组统计附录','',
      '单位是录制家族/平台会话内的证据出现比例；同一人物帧的两种检测输入成对比较。',
      '二元数据稀疏且包含大量平局，使用双侧精确符号检验；无需假设正态分布。5000次来源组配对bootstrap，seed2302。',
      '组间仍可能有相邻录制相关性；没有随机独立采样、物体框真值或多种子训练，不解释为总体精度推断。',
      '差值是候选/几何证据出现率的变化；other上的上升不是收益。8项比较采用Holm校正。','',
      '| 比较 | 组数 | 旧均值±SD | 新均值±SD | 差值[95%CI] | 增/减/平 | p / Holm p |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for name,s in stats.items():
        if s['groups']>1:
            appendix.append(f'| {name} | {s["groups"]} | {s["before_mean"]:.4f}±{s["before_sd"]:.4f} | '
              f'{s["after_mean"]:.4f}±{s["after_sd"]:.4f} | {s["delta"]:+.4f} [{s["delta_ci95"][0]:+.4f},{s["delta_ci95"][1]:+.4f}] | '
              f'{s["increased"]}/{s["decreased"]}/{s["tied"]} | {s["p"]:.4g} / {s["holm_p"]:.4g} |')
    (OUT/'stats-appendix.md').write_text('\n'.join(appendix)+'\n')
    figs=OUT/'figures';figs.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#6A788E','#168A81','#C98B26']
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,selection,title in [(axes[0],specs[:3],'Labeled action person-frames'),(axes[1],specs[3:],'Other: unwanted action evidence')]:
        x=np.arange(len(selection))
        for j,(v,name,color)in enumerate(zip(MAIN,NAMES,colors)):
            cells=[results[c][v][m]['0.25']for c,m,_ in selection]
            bars=ax.bar(x+(j-1)*.25,[100*(c['rate']or 0)for c in cells],width=.25,label=name,color=color)
            ax.bar_label(bars,labels=[f'{c["hits"]}/{c["n"]}'for c in cells],fontsize=7,rotation=65,padding=3)
        labels=['Container in ROI','Container near wrist','Object near ankle']if len(selection)==3 else ['Container near wrist','Object near ankle']
        ax.set(xticks=x,xticklabels=labels,ylim=(0,22),ylabel='Evidence present (%)',title=title)
        ax.tick_params(axis='x',labelsize=9)
    axes[0].legend(fontsize=8,loc='upper left');fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(figs/f'01-prior-evidence.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,c,title in [(axes[0],'drink_active_pose','Drink: container near wrist'),(axes[1],'other_active_pose','Other: container near wrist')]:
        for v,name,color in zip(MAIN,NAMES,colors):
            ax.plot([.05,.1,.25],[100*(results[c][v]['drink_score'][str(t)]['rate']or 0)for t in [.05,.1,.25]],marker='o',color=color,label=name)
        ax.set(xlabel='Object confidence threshold (strict >)',ylabel='Evidence present (%)',title=title,
               ylim=(0,20)if c=='drink_active_pose'else(0,1.5))
        ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(figs/f'02-threshold-diagnostic.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)
    (OUT/'figure-catalog.md').write_text('''# 图表目录

所有图均由本轮实际检测数据生成，提供PNG、PDF、SVG。

- `01-prior-evidence`：固定0.25时整帧/增强ROI/原生ROI的候选与几何证据出现率；柱上是实际分子/分母。
  应同时看正动作和other，不能只看候选数增加。没有误差条；分组区间单列在统计附录。
- `02-threshold-diagnostic`：物体阈值0.05/0.10/0.25下的手腕附近容器证据。应观察降门是否同时增加other支持，
  不能把其解释为动作ROC或选择新的生产阈值。

均无物体框真值，检测出现率不是准确率；原生ROI对原视频同时改变细节与增强状态，纯裁剪效应看增强ROI对增强整帧。
''')
    print(table)
    print('Coverage:',json.dumps(coverage,ensure_ascii=False))


if __name__=='__main__':
    main()
