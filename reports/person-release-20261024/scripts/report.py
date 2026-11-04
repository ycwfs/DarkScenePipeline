"""Reproducible source-cluster statistics and figures for the locked configuration."""
from collections import defaultdict
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

OUT = Path(__file__).resolve().parents[1]


def band(value):
    if value is None:
        return 'raw unknown'
    for hi, name in [(15,'0–15'),(30,'15–30'),(45,'30–45'),(60,'45–60'),(75,'60–75')]:
        if value < hi:
            return name
    return '75+'


def paired(rows):
    groups=defaultdict(list)
    for r in rows:
        if r['truth'] is not None:
            groups[r['group']].append((r['before']==r['truth'],r['after']==r['truth']))
    if not groups:
        return dict(groups=0)
    means=np.asarray([np.mean(v,axis=0) for v in groups.values()])
    delta=means[:,1]-means[:,0]
    rng=np.random.default_rng(24)
    boot=delta[rng.integers(len(delta),size=(5000,len(delta)))].mean(1)
    wins,losses=int((delta>1e-12).sum()),int((delta< -1e-12).sum())
    return dict(groups=len(groups),before_mean=float(means[:,0].mean()),after_mean=float(means[:,1].mean()),
        before_sd=float(means[:,0].std(ddof=1)),after_sd=float(means[:,1].std(ddof=1)),
        difference=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),wins=wins,losses=losses,
        ties=int((abs(delta)<1e-12).sum()),sign_p=float(binomtest(wins,wins+losses).pvalue) if wins+losses else 1.)


def main():
    rows=json.loads((OUT/'evaluated-windows.json').read_text())
    result=json.loads((OUT/'results.json').read_text())
    lock=json.loads((OUT/'selection-lock.json').read_text());c=lock['config']
    grid=json.loads((OUT/'parameter-grid.json').read_text())
    stats={split:paired([r for r in rows if r['split']==split]) for split in ('val','test','train')}
    (OUT/'stats.json').write_text(json.dumps(stats,indent=2)+'\n')
    figures=OUT/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,metric,title in zip(axes,['macro_f1','other_false_positives'],['Supported-class macro F1','Other false positives (windows)']):
        xs=np.arange(3)
        for offset,method,name,color in [(-.18,'baseline','Gate30 / previous settings','#7e8ca0'),(.18,'selected','Validation-selected settings','#218c79')]:
            values=[result['scores'][s][method][metric] for s in ('val','test','train')]
            bars=ax.bar(xs+offset,values,.34,label=name,color=color)
            for rect,value in zip(bars,values):
                ax.text(rect.get_x()+.17,rect.get_height(),f'{value:.3f}' if metric=='macro_f1' else str(value),ha='center',va='bottom',fontsize=9)
        ax.set_xticks(xs,['Validation\n(selection)','Test\n(after lock)','Train\n(diagnostic)']);ax.set_title(title)
        ax.legend(fontsize=8,loc='best')
    fig.savefig(figures/'01-main-comparison.png');fig.savefig(figures/'01-main-comparison.pdf');plt.close(fig)
    selected=[r for r in grid if r['model']==c['model'] and r['no_pose']==c['no_pose']]
    matrix=np.full((46,41),np.nan)
    for r in selected:
        if r['accepted']:
            matrix[round(r['threshold']*100)-50,round(r['alpha']*40)]=r['macro_f1']
    fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True)
    im=ax.imshow(matrix,origin='lower',aspect='auto',extent=[-.0125,1.0125,.495,.955],vmin=0,vmax=1,cmap='viridis')
    ax.scatter(c['alpha'],c['threshold'],marker='*',s=160,c='white',edgecolors='black')
    ax.set(xlabel='RGB weight alpha',ylabel='Action threshold',title='Validation search; white cells exceed baseline other false positives')
    fig.colorbar(im,ax=ax,label='Supported-class macro F1')
    fig.savefig(figures/'02-validation-grid.png');fig.savefig(figures/'02-validation-grid.pdf');plt.close(fig)
    summaries=[]
    for split in ('val','test','train'):
        for b in ('0–15','15–30','30–45','45–60','60–75','75+','raw unknown'):
            rr=[r for r in rows if r['split']==split and band(r['raw_luma'])==b and r['truth'] is not None]
            positive=[r for r in rr if r['truth']!='other'];negative=[r for r in rr if r['truth']=='other']
            summaries.append(dict(split=split,brightness=b,n=len(rr),positives=len(positive),other=len(negative),
                positive_before=sum(r['before']==r['truth'] for r in positive),positive_after=sum(r['after']==r['truth'] for r in positive),
                other_fp_before=sum(r['before']!='other' for r in negative),other_fp_after=sum(r['after']!='other' for r in negative)))
    with (OUT/'brightness-results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    table='| 划分 | 有标签窗 | 宏F1 原→选定 | other误报 原→选定 | 正动作正确 原→选定 |\n|---|---:|---:|---:|---:|\n'
    for split in ('val','test','train'):
        b,s=result['scores'][split]['baseline'],result['scores'][split]['selected']
        table+=f"| {split} | {s['n']} | {b['macro_f1']:.4f} → {s['macro_f1']:.4f} | {b['other_false_positives']} → {s['other_false_positives']} / {s['other_count']} | {b['positive_correct']} → {s['positive_correct']} / {s['positive_count']} |\n"
    report=f'''# 逐人方案：现场模型适配与参数选择

默认原图亮度门为 **30**；验证集选定 RGB 权重 **{c['alpha']:g}**、动作阈值 **{c['threshold']:g}**、无合格骨架策略 **{c['no_pose']}**，骨架模型选择 **{c['model']}**。

比较问题是：在最新清理展品人体模型、暗场姿态和逐人 RGB 固定后，骨架适配及参数选择能否改善动作识别，同时不增加验证集 other 误报。基线为同样门30、旧现场ST-GCN、α0.7、阈值0.75、soft策略。

{table}

535段去重来源共49160帧，按原录制家族／线上会话固定417训练、60验证、58测试；逐人窗口随检测结果变化，不能与旧固定窗成绩直接混比。人物标签按原逐人时间标注做空间迁移，迁移不可靠者保留未知，不当其他；3条先前无预测预览审核标签仅按完全一致的裁剪哈希和帧索引复用。覆盖细节见 results.json，未检出的人不计为正确。

训练只更新ST-GCN；仅一个随机种子24。候选训练20轮，验证选轮后搜索41种RGB权重×46种阈值×2种缺骨架策略×2套骨架模型。约束为验证集other误报不高于基线，再最大化已覆盖类别宏F1。完整网格见 parameter-grid.json；selection-lock.json写入后才统计测试集。大量网格选择可能对小验证集过拟合，所选参数是本批协议下的选择，不是普适最优。

![主要比较](figures/01-main-comparison.png)

图1应按验证、测试和训练三个口径分别阅读。柱形是单次固定模型的人物窗统计，没有种子误差条；训练回放不能支持泛化结论。

![验证网格](figures/02-validation-grid.png)

图2白色空格表示other误报超过预设上限，星号是锁定选择。它展示召回与误报的取舍，不是测试集调参图。各亮度段的真实正／负样本数在 brightness-results.csv；没有正样本的格子无法证明动作召回。

增强仍为公开NTIRE预训练Retinexformer；此前现场增强微调损害识别，未采用。人体YOLO26m、姿态RTMO-s、逐人VideoMamba及骨架动作模型均具有现场训练来源。暗场姿态标签是辅助标注，经质量筛选，不是独立人工精标。

历史初始化已见过部分旧视频，测试集此前也被查看。当前指标只覆盖有真值且能形成时序窗的人，不能代表全摄像头端到端召回，不能宣称9类都充分验证。线上增强片段原图L未知，不可归入亮场档位。实时丢帧与模型更新会改变窗口，需另看正式包真实流验收。
'''
    (OUT/'analysis-report.md').write_text(report)
    test=stats['test']
    (OUT/'stats-appendix.md').write_text(f'''# 统计口径

单种子训练；统计单位为原始录制家族／会话组。窗口高度相关，不将窗口当独立重复，不用正态t检验。每组先取预测正确率，再对配对组差值做5000次有放回组bootstrap（seed24），并用组差值正负的精确符号检验。唯一预先确认的测试集主比较为选定方案对门30基线；亮度分层、训练和验证仅描述，不作额外显著性主张。

测试集 {test['groups']} 组：基线组正确率均值±标准差 {test['before_mean']:.4f}±{test['before_sd']:.4f}；选定 {test['after_mean']:.4f}±{test['after_sd']:.4f}。配对绝对增量 {test['difference']:.4f}，95%CI {test['ci95']}，组符号检验p={test['sign_p']:.6g}，获益／退化／持平={test['wins']}/{test['losses']}/{test['ties']}。这是来源组准确率增量，不是宏F1置信区间，也不包含重新抽样调参的不确定性。完整每划分统计见stats.json。只一个主检验，因此未做多重校正；网格不逐点报告p值。

测试有历史暴露，bootstrap不能消除这一局限。没有多种子结果，不能估计训练随机性的方差；没有独立完整逐帧真人GT，不能报告检测mAP或整体漏识率。
''')
    (OUT/'figure-catalog.md').write_text('''# 图表说明

01-main-comparison.png/.pdf：来自results.json，分别比较验证、测试、训练人物窗的已支持类别宏F1与other误报。观察正动作取舍和误报是否同时改善；不要将训练增益外推或将柱形视为多种子均值。

02-validation-grid.png/.pdf：来自parameter-grid.json，仅显示满足误报约束的验证网格。观察可接受参数区域及选中点，说明阈值／RGB权重存在共同取舍；白格不是缺数据，星号不是测试集最优。无误差条，因为每格为一次固定样本评估。
''')
    print(table)


if __name__=='__main__':
    main()
