"""Exact diagnostics with training-fit and source-held-out results kept separate."""
from collections import Counter, defaultdict
import csv
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/darkpipe-all85-matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *


NAMES=dict(baseline='原正式方案',previous_same_defaults='上轮首次续训＋原参数',
    rebalanced_same_defaults='39正例减少other轮＋原参数',selected='本轮旧验证选定组合',
    best_trained='本轮两分支均微调组合',last_epoch='本轮固定末轮组合',individual_branch_selections='各分支独立选轮组合')
ROLES=dict(old_val='旧验证（选轮）',old_test='旧测试回归',new_positive_training_fit='多人85正例（已训练）',
    new_other_training_fit='多人85 other（已训练）',new_other_untrained_same_sources='多人566 other（未训练、同采集组）')


def main() -> None:
    data=read(OUT/'dataset.json');rows=data['records'];results=read(OUT/'results.json')['comparisons']
    lock=read(OUT/'selection-lock.json');pred=dict(np.load(OUT/'final-predictions.npz'));ids=scored_indices(rows)
    methods=list(NAMES);important=['baseline','rebalanced_same_defaults','selected','best_trained','last_epoch']
    fields=['id','role','source','track_id','generation','start','end','human_label','train_selected',*methods]
    records=[]
    for i in ids:
        r=rows[i]
        records.append(dict(id=r['id'],role=score_role(r),source=r['source_id'],track_id=r['track_id'],
            generation=r['generation'],start=r['start'],end=r['end'],human_label=r['label'],train_selected=r['train_selected'],
            **{m:LABELS[int(pred[m][i])] for m in methods}))
    for filename,subset in [('window-results.csv',records),('all85-positive-results.csv',
        [r for r in records if r['role']=='new_positive_training_fit'])]:
        with (OUT/filename).open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(subset)
    table=['|集合|方案|other误报／other数|正确动作／动作数|宏F1|','|---|---|---:|---:|---:|']
    for role in ROLES:
        for method in important:
            m=results[method]['groups'][role]
            table.append(f"|{ROLES[role]}|{NAMES[method]}|{m['other_fp']}/{m['other_n']}|{m['positive_tp']}/{m['positive_n']}|{m['macro_f1']:.4f}|")
    def effect(method: str, role: str) -> dict:
        a=results[method]['groups'][role];b=results['baseline']['groups'][role]
        return dict(other_fp=a['other_fp']-b['other_fp'],positive_tp=a['positive_tp']-b['positive_tp'])
    effects={m:{r:effect(m,r) for r in ROLES} for m in important[1:]}
    save(OUT/'effects.json',effects)
    annotations={a['review_id']:a for a in read(ROOT/'reports/manypeople-finetune-20261029/annotations.json')['annotations']}
    groups=defaultdict(list)
    for i in ids:
        if rows[i]['scope']!='new':continue
        for rid in rows[i]['review_ids']:
            if annotations[rid]['human_label']==rows[i]['label']:groups[rid].append(i)
    group_metrics={}
    for method in important:
        out=[]
        for rid,members in groups.items():
            truth=annotations[rid]['human_label']
            out.append(dict(review_id=rid,truth=truth,any_correct_or_false_alarm=any(
                pred[method][i]!=9 if truth=='other' else LABELS[int(pred[method][i])]==truth for i in members),
                any_training_window=any(rows[i]['train_selected'] for i in members)))
        group_metrics[method]=dict(positive_events=sum(r['truth']!='other' for r in out),
            correct_positive_events=sum(r['truth']!='other' and r['any_correct_or_false_alarm'] for r in out),
            other_events=sum(r['truth']=='other' for r in out),
            false_other_events=sum(r['truth']=='other' and r['any_correct_or_false_alarm'] for r in out),records=out)
    save(OUT/'review-group-results.json',group_metrics)
    # Paired source bootstrap describes this fixed model on old regression only.
    source_groups=defaultdict(list)
    for i in ids:
        if score_role(rows[i])=='old_test':source_groups[rows[i]['group']].append(i)
    intervals={}
    for method in ['selected','best_trained','last_epoch']:
        counts=[]
        for members in source_groups.values():
            neg=[i for i in members if rows[i]['label']=='other'];pos=[i for i in members if rows[i]['label']!='other']
            counts.append([len(neg),sum(pred['baseline'][i]!=9 for i in neg),sum(pred[method][i]!=9 for i in neg),
                len(pos),sum(LABELS[pred['baseline'][i]]==rows[i]['label'] for i in pos),
                sum(LABELS[pred[method][i]]==rows[i]['label'] for i in pos)])
        a=np.array(counts);rng=np.random.default_rng(29);boot=[]
        for _ in range(2000):
            s=a[rng.integers(0,len(a),len(a))].sum(0)
            if s[0] and s[3]:boot.append([(s[2]-s[1])/s[0],(s[5]-s[4])/s[3]])
        q=np.percentile(boot,[2.5,97.5],axis=0).T.tolist()
        intervals[method]=dict(other_fpr_difference=q[0],positive_recall_difference=q[1])
    stats=dict(seed=29,training_seeds=1,old_test_source_groups=len(source_groups),bootstrap_replicates=2000,
        old_test_paired_source_95ci=intervals,crowded_independent_positive_holdout=0,pvalues=None,
        limitations='Previously inspected old sources, one training seed; no untouched-test or significant-generalization claim. Crowded positives all trained, other shares source groups.')
    save(OUT/'statistics.json',stats)
    f=OUT/'figures';f.mkdir(exist_ok=True)
    display=['Baseline','39-pos run','Old-val selected','Both trained','Last epoch'];x=np.arange(len(important))
    fig,axes=plt.subplots(1,3,figsize=(15,4.6))
    for bars in [axes[0].bar(x-.18,[results[m]['groups']['old_test']['other_fp'] for m in important],.36,color='#D55E00',label='False other action'),
                 axes[0].bar(x+.18,[results[m]['groups']['old_test']['positive_tp'] for m in important],.36,color='#0072B2',label='Correct action')]:axes[0].bar_label(bars,padding=2)
    axes[0].set_title('Old source-held-out regression\n163 other / 51 positive');axes[0].legend(fontsize=8)
    bars=axes[1].bar(x,[results[m]['groups']['new_positive_training_fit']['positive_tp'] for m in important],color='#0072B2');axes[1].bar_label(bars,padding=2)
    axes[1].set_ylim(0,94);axes[1].set_title('Crowded positive TRAINING FIT\nAll 85 used by current training')
    bars=axes[2].bar(x,[results[m]['groups']['new_other_untrained_same_sources']['other_fp'] for m in important],color='#D55E00');axes[2].bar_label(bars,padding=2)
    axes[2].set_title('Crowded untrained other / 566\nSame capture groups as training')
    for ax in axes:ax.set_xticks(x,display,rotation=22,ha='right');ax.set_ylabel('Windows');ax.spines[['top','right']].set_visible(False);ax.margins(y=.2)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(f/f'01-full-scheme-comparison.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for ri,branch in enumerate(['rgb','skeleton']):
        for folder,style,title in [('manypeople-positive-rebalance-20261030','--','39 positives'),(OUT.name,'-','85 positives')]:
            history=read(ROOT/'reports'/folder/branch/'history.json')
            for ci,(metric,label) in enumerate([('other_fpr','Other FPR'),('positive_macro_recall','Positive macro recall')]):
                axes[ri,ci].plot([r['epoch'] for r in history],[r['metrics']['old'][metric] for r in history],style,label=title)
                axes[ri,ci].set_title(f'{branch}: old validation, branch threshold .75');axes[ri,ci].set_ylabel(label)
        for ax in axes[ri]:ax.set_ylim(0,1.03);ax.set_xlabel('Epoch');ax.legend();ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(f/f'02-old-validation-dynamics.{ext}',dpi=180)
    plt.close(fig)
    br=read(OUT/'branch-results.json')['comparisons'];branch_lines=['|分支／阈值.75|权重|旧误报／163或有效骨架数|旧正确／正例数|多人已训练真动作正确／有效窗|',
        '|---|---|---:|---:|---:|']
    for branch in ['rgb','skeleton']:
        for which in ['baseline','trained','last']:
            old=br[branch][which]['0.75']['old_test'];new=br[branch][which]['0.75']['new_positive_training_fit']
            branch_lines.append(f"|{branch}|{which}|{old['other_fp']}/{old['other_n']}|{old['positive_tp']}/{old['positive_n']}|{new['positive_tp']}/{new['positive_n']}|")
    selected_epochs=lock['epochs'];trained_epochs={b:lock['best_trained_models'][b]['epoch'] for b in ['rgb','skeleton']}
    selection_text=f"旧验证选中RGB {selected_epochs['rgb']}／骨架 {selected_epochs['skeleton']}；两分支均微调组合选中RGB {trained_epochs['rgb']}／骨架 {trained_epochs['skeleton']}（满足旧验证约束：{lock['best_trained']['accepted']}）。epoch0代表原权重。"
    effect_text='；'.join(f"{NAMES[m]}：旧test误报{effects[m]['old_test']['other_fp']:+d}、正确动作{effects[m]['old_test']['positive_tp']:+d}，多人训练正例正确{results[m]['groups']['new_positive_training_fit']['positive_tp']}/85，未训练同源other误报{results[m]['groups']['new_other_untrained_same_sources']['other_fp']}/566" for m in ['selected','best_trained','last_epoch'])
    decision=dict(status='experiment_complete_not_promoted',production_modified=False,selected_epochs=selected_epochs,
        trained_epochs=trained_epochs,reason='All crowded positives were trained; improvements there demonstrate fitting only. Old regression and actual end-to-end deployment must be assessed separately.',
        effects=effects,parameters_changed=False)
    save(OUT/'adoption-decision.json',decision)
    parity=read(OUT/'replay-parity.json')
    exported={which:read(OUT/f'export-{which}-verification.json') for which in ['trained','selected','last']}
    save(OUT/'export-verification.json',exported)
    (OUT/'analysis-report.md').write_text(f'''# 全部85个多人正例微调对照

{selection_text}

{effect_text}。本轮已完成两模型训练和固定参数比较，未替换正式模型/包/镜像。

用户明确指定旧训练5009窗、85个人工多人真动作全部加入，再选择85个人工other。RGB训练5179窗；骨架3379窗，其中新正例70窗合格。85窗来自44个人工审核事件，类别为跑动34、交谈33、挥手12、喝水6。预测other、XCLIP、标注冲突及非真人不训练。

这次原多人val/test的46个正例已转为训练，因此**85个正例的结果只能称训练拟合**；并无新多人真动作留出集。85个已训练other与566个未训练other分报，后者也来自已见采集组，不能当独立泛化。旧数据原来源分组保持；旧验证选轮，旧test仅作历史回归，原数据已被多轮查看。

从原正式模型训练RGB6轮/ST-GCN20轮，seed29，旧ROI重放与蒸馏。新other采样1.28%、全部other10.26%，不按文件比例解释训练影响。四参数固定alpha.972/门35/动作.925/物体.581，挥手3帧、跑动.35；不根据新test/训练拟合调参数。147组合仅按旧198窗验证选择，并在查看评估结果前锁定“全部权重候选”“两分支均微调”和末轮。以最后一轮测试得分再挑轮会有偏差，本报告不这样做。

{chr(10).join(table)}

单分支另用.75作诊断，不加融合/物体/挥手/跑动先验，和完整方案口径不同。骨架只在有效姿态窗上统计；RGB包含全部85个正例。两分支的分母不能直接当相同难度比较。

{chr(10).join(branch_lines)}

[逐窗完整对照](window-results.csv)及[85个正例](all85-positive-results.csv)可定位人物、原视频和起止时间。`review-group-results.json`给出事件组any-trigger，窗口重叠不能当独立事件。`branch-results.json`同时包含.75/.925原始分支结果，便于区分分类器变化与先验过滤。

生产融合和三先验函数回放{parity['checked']}次、标签差异{parity['mismatched_labels']}。三种预声明组合的权重均导出并实际GPU加载，逐张量一致及缓存窗口概率复现见`export-verification.json`；没有重建镜像。旧数据、原正式模型与新数据像素保持。

本轮单seed，不宣称显著性或无条件泛化；旧来源组配对bootstrap仅描述固定模型来源变化，不包含训练随机性/历史调参偏差。85个正例均见过，不能据此判断全新多人场景；这85个正例及旧测试的51个正例均为L≥75，不能据此确认暗场真动作能力维持。也不是实时推流、NFS、Box或无限流稳定性验收。生产发布需额外新数据和完整原片验证，当前仅保存独立实验权重。
''')
    (OUT/'stats-appendix.md').write_text('# 统计边界\n\n'+json.dumps(stats,ensure_ascii=False,indent=2)+
        '\n\n比较单位为人物窗口，原来源组为旧回归重采样单位。单seed不报告训练均值±标准差，重叠窗口不构造伪独立p值；没有多重比较显著性声明。新85个正例为训练拟合，566个other为同源未训练窗口，均不报告独立泛化置信区间。\n')
    (OUT/'figure-catalog.md').write_text('''# 图表说明

- `01-full-scheme-comparison.png/pdf`：固定模型输入与阈值，分别展示旧来源回归、多人正例训练拟合、同源未训练other误报。来源为`results.json`。蓝色为正确动作，橙色为误报；无误差条（单seed）。应看旧能力是否退化、训练正例能否拟合、负例误报是否增加；多人面板不能作留出泛化证据。
- `02-old-validation-dynamics.png/pdf`：39正例上一轮与85正例本轮的RGB/骨架旧验证曲线，来源为两轮`history.json`。单分支阈值.75、无先验，用于查看学习过程和旧能力变化；不能等同最终完整方案指标。只比较旧验证，因为多人正例已参加训练。

数字对照见`analysis-report.md`、`window-results.csv`，统计边界见`stats-appendix.md`。
''')
    print(json.dumps(dict(selected_epochs=selected_epochs,trained_epochs=trained_epochs,effects=effects),ensure_ascii=False))


if __name__=='__main__':main()
