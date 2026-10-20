"""Describe paired results without treating overlapping windows as independent trials."""
from collections import defaultdict
import csv
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'reports/feedback-finetune-20261019'


def read(group: str, name: str) -> list:
    return [json.loads(s) for s in (OUT/'evaluation'/group/(name+'.jsonl')).read_text().splitlines()]


def paired(old: list, new: list) -> list:
    a={(r['suite'],r['name']):r for r in old}
    b={(r['suite'],r['name']):r for r in new}
    assert a.keys()==b.keys()
    out=[]
    for key,r in a.items():
        s=b[key]
        assert r['source']==s['source'] and r['expect']==s['expect']
        out.append(dict(suite=r['suite'],name=r['name'],source=r['source'],session=r.get('session'),
                        expected=r['expect'],old=r['label'],new=s['label'],
                        old_named=max(r['decision'][:9]),new_named=max(s['decision'][:9]),
                        old_rgb_named=max(r['rgb'][:9]),new_rgb_named=max(s['rgb'][:9])))
    return out


def main() -> None:
    selection=json.loads((OUT/'selection.json').read_text())
    selected=selection['selected']
    assert selected
    pairs=paired(read('regression','baseline'),read('regression',selected))
    pairs+=paired(read('holdout','baseline'),read('holdout',selected))
    target=OUT/'analysis'
    figures=target/'figures'
    figures.mkdir(parents=True,exist_ok=True)
    with (target/'paired-windows.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(pairs[0]))
        writer.writeheader();writer.writerows(pairs)
    summaries={}
    table=['| Suite | Windows | Correct before → after | False actions before → after |',
           '|---|---:|---:|---:|']
    for suite in ('A','B','C','P','Qtrain','Qholdout'):
        sub=[r for r in pairs if r['suite']==suite]
        a=sum(r['old']==r['expected'] for r in sub)
        b=sum(r['new']==r['expected'] for r in sub)
        negatives=all(r['expected']=='other' for r in sub)
        fa=sum(r['old']!='other' for r in sub) if negatives else None
        fb=sum(r['new']!='other' for r in sub) if negatives else None
        summaries[suite]=dict(n=len(sub),old_correct=a,new_correct=b,old_false=fa,new_false=fb)
        table.append(f'| {suite} | {len(sub)} | {a} → {b} | {fa} → {fb} |' if negatives
                     else f'| {suite} | {len(sub)} | {a} → {b} | — |')
    (target/'metrics.json').write_text(json.dumps(summaries,indent=2)+'\n')
    # Exact finite-benchmark error counts, no invented error bars over correlated windows.
    labels=list(summaries)
    x=np.arange(len(labels))
    old=[summaries[s]['n']-summaries[s]['old_correct'] for s in labels]
    new=[summaries[s]['n']-summaries[s]['new_correct'] for s in labels]
    fig,ax=plt.subplots(figsize=(9,4.5))
    ax.bar(x-.18,old,.36,label='Before',color='#7b8794')
    ax.bar(x+.18,new,.36,label='Fine-tuned',color='#198a73')
    for xx,aa,bb in zip(x,old,new):
        ax.text(xx-.18,aa+.25,str(aa),ha='center',fontsize=10)
        ax.text(xx+.18,bb+.25,str(bb),ha='center',fontsize=10)
    ax.set_xticks(x,labels);ax.set_ylabel('Incorrect windows (lower is better)')
    ax.set_title('Paired replay at rejection threshold 0.75')
    ax.legend();ax.spines[['top','right']].set_visible(False)
    ax.set_ylim(0,max(old+new+[1])*1.2+1)
    fig.tight_layout()
    for suffix in ('png','pdf'):fig.savefig(figures/f'01-paired-errors.{suffix}',dpi=180)
    plt.close(fig)
    history=json.loads((OUT/'training-history.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    axes[0].plot([r['epoch'] for r in history],[r['loss'] for r in history],'o-',color='#198a73')
    axes[0].set(xlabel='Epoch',ylabel='Training loss',title='One seed, mixed replay training')
    axes[1].plot([r['epoch'] for r in history],[r['train_top1'] for r in history],'o-',color='#3875b0')
    axes[1].set(xlabel='Epoch',ylabel='Sampled training accuracy',title='Not a held-out metric')
    fig.tight_layout()
    for suffix in ('png','pdf'):fig.savefig(figures/f'02-training.{suffix}',dpi=180)
    plt.close(fig)
    sessions=defaultdict(list)
    for row in pairs:
        if row['suite']=='Qholdout':sessions[row['session']].append(row)
    session_stats=[]
    for name,rows in sessions.items():
        fa=sum(r['old']!='other' for r in rows)
        fb=sum(r['new']!='other' for r in rows)
        session_stats.append(dict(session=name,windows=len(rows),before=fa,after=fb,
                                  rate_delta=(fb-fa)/len(rows)))
    delta=np.array([r['rate_delta'] for r in session_stats])
    permutations=[float(np.mean(delta*np.array(signs))) for signs in itertools.product((-1,1),repeat=len(delta))]
    # Diagnostic only: a two-session sign-flip has four possible assignments and very little resolution.
    p=float(np.mean(np.abs(permutations)>=abs(float(delta.mean()))-1e-12))
    stats=dict(seed_count=1,seed=19,holdout_sessions=len(sessions),holdout_videos=9,
               session_stats=session_stats,mean_session_rate_delta=float(delta.mean()),
               exploratory_two_sided_session_sign_flip_p=p,assignments=len(permutations),
               population_ci=None,ci_reason='Only two same-camera sessions; insufficient independent clusters for a reliable population interval.',
               warning='No window-level significance test: windows overlap. No across-seed variance estimate from one run.')
    (target/'stats.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n')
    report=f'''# 新负样本微调：训练前后配对分析

比较问题：在默认阈值 0.75、alpha=0.7 和既有骨架模型不变时，仅微调 VideoMamba 能否减少新负样本误报，同时守住原有动作回归。

选中检查点：`{selected}`。所有 4 轮均保留并评估；选择只使用 ABCP 回归和 Qtrain，之后才运行 Qholdout。

{chr(10).join(table)}

A/B/C/P 为已有数据回归，不是独立泛化集。Qtrain 是本轮 23 段训练负样本的回放；Qholdout 为 2 个未用于本轮训练或选模型的会话，共 9 段、68 窗。原模型是否曾见过同一场景，不等于这些新视频参加过本轮训练；现场背景仍相同。

现有现场正样本回归只覆盖喝水、挥手、追逐、捡物；没有投掷、跌倒、打架、交谈、握手的正样本，因此不能宣称已验证全部 9 类动作的保持情况。

P 和 Q 均先裁掉已烧录标签条，不再次增强；缺少原始输入，无法恢复照度门输入或增强前画面。这轮 P 的预处理纠正了早期脚本的重复增强，因此仅比较这次重新生成的配对基线。不能将这些窗口与 11,593 条旧线上事件一一对应，也不能声称旧日志中的 23 次误报已被全部重算。

主图 `01-paired-errors` 展示固定数据集的错误窗口数；没有误差条。需要同时看正样本套件与新负样本，避免将统一压成 Other 误称为改进。`02-training` 展示单个种子的训练变化，只说明优化过程，不代表泛化。

所有逐窗结果保留完整精度。回放缓存是部署 push 实际使用的 400×224 输入，重建归一化后逐位核对；骨架权重未改，候选复用同一骨架概率和 presence，融合仍调用部署实现。后续独立解包测试用于核对实际交付。
'''
    (target/'analysis-report.md').write_text(report)
    (target/'stats-appendix.md').write_text(f'''# 统计口径

- 训练种子 1 个（19）；不报告不存在的多种子均值±标准差。
- 主表是固定测试数据上的精确计数，窗口存在重叠，不能当作独立 Bernoulli 样本。
- 留出会话数 {len(sessions)}，视频数 9。按会话计算的平均误报率变化为 {delta.mean():.6f}（after−before）。
- 探索性会话配对符号翻转检验：双侧 p={p:.6f}，只有 {len(permutations)} 种排列，假设配对差值在零假设下可交换符号；样本量不足以作强显著性结论。
- 未为重叠窗口构造虚假的窄置信区间。仅两个同摄像机会话，可靠总体 95% 区间与跨场景推断均缺乏足够独立样本。
- 只有一个探索性主检验，不对次要套件逐个做显著性检验；不声称已消除所有场景误报。

会话明细见 stats.json；全部配对窗见 paired-windows.csv。完整差值及套件计数见 metrics.json。
''')
    (target/'figure-catalog.md').write_text('''# 图表说明

## 01-paired-errors.png / .pdf

数据：metrics.json；横轴套件，纵轴错误窗口数，越低越好。Before 与 Fine-tuned 使用同一窗口；阈值 0.75。固定集计数无误差条。
观察：检查新负样本改善是否伴随 A/C 正样本退化。决策：仅凭 Qtrain 下降不能证明泛化，必须同时报告 Qholdout。
限制：ABCP 含训练内数据；P/Q 为处理后视频回放，不能重建原始线上时序；各套件窗口量不同，柱高不能横向当作难度排序。

## 02-training.png / .pdf

数据：training-history.json；横轴 epoch，纵轴分别为混合训练 loss 和采样训练准确率。单种子，无误差条。
观察：优化是否推进、是否出现数值异常。决策：训练曲线只能排查训练行为，选模型依据仍为配对回归与预先设定门限。
限制：训练采样 20% 来自新负样本，其余为按类别平衡的旧数据；训练准确率不能与自然发生率下的平台准确率直接比较。
''')
    print(chr(10).join(table))


if __name__=='__main__': main()
