import json,statistics
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parents[1]
(OUT/'figures').mkdir(parents=True, exist_ok=True)
rows=json.loads((OUT/'records.json').read_text())
sweep=json.loads((OUT/'threshold-sweep.json').read_text())
conds=['raw_native','raw720','ntire720','ft720']; labels=['Raw native','Raw 720','NTIRE 720','Onsite FT 720']
colors=['#64748b','#2563eb','#16a34a','#dc2626']
fig,ax=plt.subplots(figsize=(9,5.2),dpi=180)
for det,label,ls in [('formal','Formal YOLO26m','-'),('original','Original YOLO26m','--')]:
 vals=[]
 for c in conds:
  gt=sum(len(r['gt_boxes_native']) for r in rows)
  vals.append(100*sum(r['conditions'][c][det]['matched'] for r in rows)/gt)
 ax.plot(labels,vals,marker='o',linewidth=2,label=label,linestyle=ls)
 ax.set_ylim(0,105); ax.set_ylabel('Matched auxiliary person boxes (%)'); ax.set_title('Dark-frame person recall by input and detector')
 ax.grid(axis='y',alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(OUT/'figures/figure-01-recall.png'); fig.savefig(OUT/'figures/figure-01-recall.pdf'); plt.close(fig)
# source heatmap for formal detector
sources=sorted(set(r['source'].split('/')[-1] for r in rows)); mat=np.zeros((len(sources),len(conds)))
for i,s in enumerate(sources):
 rr=[r for r in rows if r['source'].split('/')[-1]==s]; gt=sum(len(r['gt_boxes_native']) for r in rr)
 for j,c in enumerate(conds): mat[i,j]=100*sum(r['conditions'][c]['formal']['matched'] for r in rr)/gt
fig,ax=plt.subplots(figsize=(9,4),dpi=180); im=ax.imshow(mat,vmin=0,vmax=100,cmap='YlGn'); ax.set_xticks(range(4),labels); ax.set_yticks(range(len(sources)),sources)
for i in range(len(sources)):
 for j in range(4): ax.text(j,i,f'{mat[i,j]:.0f}',ha='center',va='center',color='black' if mat[i,j]>55 else 'white',fontsize=9)
ax.set_title('Formal detector recall by source and input'); fig.colorbar(im,ax=ax,label='Recall (%)'); fig.tight_layout(); fig.savefig(OUT/'figures/figure-02-source-heatmap.png'); fig.savefig(OUT/'figures/figure-02-source-heatmap.pdf'); plt.close(fig)
# GT height distributions and threshold sweep
fig,axs=plt.subplots(1,2,figsize=(11,4.4),dpi=180)
heights=[]
for c in conds: heights.append([h for r in rows for h in r['conditions'][c]['gt_heights']])
axs[0].boxplot(heights,labels=labels,showfliers=False); axs[0].set_ylabel('GT person box height (pixels)'); axs[0].set_title('Person scale at each input stage'); axs[0].tick_params(axis='x',rotation=25); axs[0].grid(axis='y',alpha=.25)
for c,col in zip(['raw720','ntire720','ft720'],colors[1:]): axs[1].plot([float(x) for x in sweep[c]], [100*sweep[c][x]['recall'] for x in sweep[c]], marker='o',label=c,color=col)
axs[1].set_xlabel('Person confidence threshold'); axs[1].set_ylabel('Matched recall (%)'); axs[1].set_ylim(0,105); axs[1].set_title('Formal detector threshold sweep'); axs[1].grid(alpha=.25); axs[1].legend()
fig.tight_layout(); fig.savefig(OUT/'figures/figure-03-scale-threshold.png'); fig.savefig(OUT/'figures/figure-03-scale-threshold.pdf'); plt.close(fig)
# stats appendix

def pct(x,n): return f'{x}/{n} ({100*x/n:.2f}%)'
lines=['# 统计附录：暗场增强输入与人体检测','', '单位是采样帧；44帧来自5个来源。辅助框经过筛选，非穷尽人工GT，因此 precision、误检和置信区间只描述“未匹配辅助框候选”，不能当真实误检率。', '', '## 正式检测器', '', '| 输入 | 命中 | 召回 | 预测 | 未匹配预测 | 平均IoU |', '|---|---:|---:|---:|---:|---:|']
for c,l in zip(conds,labels):
 gt=44; ma=sum(r['conditions'][c]['formal']['matched'] for r in rows); pr=sum(r['conditions'][c]['formal']['predictions'] for r in rows); un=sum(r['conditions'][c]['formal']['unmatched_predictions'] for r in rows); i=[v for r in rows for v in r['conditions'][c]['formal']['ious']]
 lines.append(f'| {l} | {ma}/44 | {ma/44:.4f} | {pr} | {un} | {np.mean(i):.4f} |')
lines += ['', '## 原始检测器对照', '', '| 输入 | 命中 | 召回 | 预测 | 未匹配预测 | 平均IoU |', '|---|---:|---:|---:|---:|---:|']
for c,l in zip(conds,labels):
 ma=sum(r['conditions'][c]['original']['matched'] for r in rows); pr=sum(r['conditions'][c]['original']['predictions'] for r in rows); un=sum(r['conditions'][c]['original']['unmatched_predictions'] for r in rows); i=[v for r in rows for v in r['conditions'][c]['original']['ious']]
 lines.append(f'| {l} | {ma}/44 | {ma/44:.4f} | {pr} | {un} | {np.mean(i):.4f} |')
lines += ['', '## 配对差异（正式检测器）', '', '| 对照 | 逐帧命中增加 | 逐帧命中减少 | 净变化 |', '|---|---:|---:|---:|']
base=[r['conditions']['ntire720']['formal']['matched'] for r in rows]
for c,l in [('raw_native','Raw native vs NTIRE'),('raw720','Raw 720 vs NTIRE'),('ft720','Onsite FT 720 vs NTIRE')]:
 vals=[r['conditions'][c]['formal']['matched'] for r in rows]; inc=sum(a>b for a,b in zip(vals,base)); dec=sum(a<b for a,b in zip(vals,base)); lines.append(f'| {l} | {inc} | {dec} | {sum(vals)-sum(base)} |')
lines += ['', '## 置信度扫描', '', '阈值扫描只在 raw720、NTIRE720、现场微调720上进行，固定正式检测器和同一NMS。']
for c in ['raw720','ntire720','ft720']:
 lines.append(f'\n### {c}\n\n| 阈值 | 召回 | 预测 | 未匹配预测 |\n|---:|---:|---:|---:|')
 for t,v in sweep[c].items(): lines.append(f'| {t} | {v["matched"]}/44 ({v["recall"]:.4f}) | {v["predictions"]} | {v["unmatched_predictions"]} |')
lines += ['', '## 统计口径与限制', '', '- 主要比较是同帧配对描述统计；没有多个随机种子，也没有完整人工标注，未进行模型级显著性声明。', '- 未匹配预测不是确认误检，因为标签没有穷尽所有人物；真正的误检需要人工逐框审核。', '- 原始分辨率路径只用于诊断；生产仍按720缩放。', '- 现场微调增强在本轮CPU推理，未用于速度结论。']
(OUT/'stats-appendix.md').write_text('\n'.join(lines)+'\n')
(OUT/'figure-catalog.md').write_text('''# 图目录\n\n## figure-01-recall\n\n- 目的：比较增强输入与检测器对辅助人体框的命中率。\n- 数据：44个冻结暗场采样帧；线条为正式检测器与原始检测器。\n- 应注意：正式检测器在NTIRE720和现场微调720都达到44/44；现场增强没有额外召回收益。\n- 决策影响：优先检查检测器和失败帧，而不是直接替换增强模型。\n\n## figure-02-source-heatmap\n\n- 目的：显示收益是否集中在某一个视频来源。\n- 数据：5个来源，正式检测器，百分比为源内辅助框召回。\n- 应注意：个别来源对增强敏感，不能只看总平均。\n- 决策影响：后续应补充不同机位、遮挡和更极暗场景。\n\n## figure-03-scale-threshold\n\n- 目的：同时展示720缩放后人物尺寸和置信度阈值影响。\n- 数据：44个辅助框高度及正式检测器0.10–0.50阈值扫描。\n- 应注意：人物在720图中平均约131像素高；NTIRE/现场微调在0.35阈值都保持全命中。\n- 决策影响：暂不降低生产person阈值；若要覆盖原始720单个漏检，应先做人工误检审计。\n''')
