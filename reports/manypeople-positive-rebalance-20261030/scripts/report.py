"""Descriptive fixed-threshold comparison; repeated held-out sources are disclosed."""
from collections import Counter, defaultdict
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *
from evaluation_metrics import metrics

PREVIOUS = ROOT / 'reports/manypeople-finetune-20261029'


def main() -> None:
    rows = read(OUT / 'dataset.json')['records']
    lock = read(OUT / 'selection-lock.json')
    final = dict(np.load(OUT / 'final-predictions.npz'))
    old_rows = read(PREVIOUS / 'dataset.json')['records']
    assert [r['id'] for r in rows] == [r['id'] for r in old_rows]
    final['previous_same_defaults'] = np.load(PREVIOUS / 'final-predictions.npz')['trained_same_defaults']
    methods = ['baseline', 'previous_same_defaults', 'selected', 'last_epoch']
    titles = dict(baseline='原正式方案', previous_same_defaults='上一轮选定权重＋原参数',
                  selected='本轮验证选定权重＋原参数', last_epoch='本轮末轮权重＋原参数（诊断）')
    def indices(scope: str, split: str = 'test') -> list[int]:
        return [i for i, r in enumerate(rows) if r['scope'] == scope and r['split'] == split
                and r['label_kind'] != 'pseudo' and (scope == 'new' or r.get('current_index') is not None)]
    comparison = {}
    table = ['|集合|方案|other误报／other数|正确动作／动作数|宏F1|', '|---|---|---:|---:|---:|']
    for method in methods:
        comparison[method] = {}
        for split in ['val', 'test']:
            comparison[method][split] = {}
            for domain in ['old', 'new']:
                ids = indices(domain, split)
                m = metrics([rows[i] for i in ids], final[method][ids])
                comparison[method][split][domain] = m
                table.append(f"|{split}/{domain}|{titles[method]}|{m['other_fp']}/{m['other_n']}|{m['positive_tp']}/{m['positive_n']}|{m['macro_f1']:.4f}|")
    save(OUT / 'fixed-threshold-comparison.json', comparison)
    effects = {d: dict(other_fp=comparison['selected']['test'][d]['other_fp']-comparison['baseline']['test'][d]['other_fp'],
                       positive_tp=comparison['selected']['test'][d]['positive_tp']-comparison['baseline']['test'][d]['positive_tp'])
               for d in ['old', 'new']}
    with (OUT / 'window-results.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['id','scope','split','source_id','group','truth',*methods])
        writer.writeheader()
        for split in ['val', 'test']:
            for domain in ['old', 'new']:
                for i in indices(domain, split):
                    r = rows[i]
                    writer.writerow(dict(id=r['id'],scope=domain,split=split,source_id=r['source_id'],group=r['group'],
                                         truth=r['label'],**{m:LABELS[int(final[m][i])] for m in methods}))
    branches = {}
    valid = np.load(OUT / 'skeleton-inputs.npz')['valid'][GATES.index(35.)]
    for branch in ['rgb', 'skeleton']:
        branches[branch] = {}
        for which in ['baseline', 'candidate', 'selected', 'last']:
            probs = np.load(OUT / 'predictions' / f'{branch}-{which}.npz')['probs']
            if branch == 'skeleton':probs = probs[GATES.index(35.)]
            labels = probs.argmax(1); labels[(labels != 9) & (probs.max(1) < .75)] = 9
            branches[branch][which] = {}
            for domain in ['old', 'new']:
                ids = [i for i in indices(domain) if branch == 'rgb' or valid[i]]
                branches[branch][which][domain] = metrics([rows[i] for i in ids], labels[ids])
    save(OUT / 'branch-test-results.json', dict(action_threshold=.75,gate=35.,priors_applied=False,
        skeleton_requires_valid_pose=True,comparisons=branches))
    diagnostic=[]
    for h in read(OUT/'rgb/history.json'):
        cached=np.load(OUT/'rgb'/f"val-{h['epoch']:02d}.npz")
        for label in ['wave','chase']:
            mask=np.array([rows[i]['scope']=='new' and rows[i]['label']==label for i in cached['indices']])
            probs=cached['probs'][mask];truth=LABELS.index(label);top=probs.argmax(1)
            diagnostic.append(dict(epoch=h['epoch'],label=label,n=len(probs),
                raw_top_counts=dict(Counter(LABELS[i] for i in top)),
                raw_correct=int(sum(top==truth)),correct_above_075=int(sum((top==truth)&(probs.max(1)>=.75))),
                true_class_mean_probability=float(probs[:,truth].mean())))
    save(OUT/'validation-positive-diagnostics.json',diagnostic)
    annotations = {r['review_id']:r for r in read(PREVIOUS / 'annotations.json')['annotations']}
    grouped = defaultdict(list)
    for i in indices('new'):
        for rid in rows[i]['review_ids']:
            if annotations[rid]['human_label'] == rows[i]['label']:grouped[rid].append(i)
    groups = {}
    for method in methods:
        records = []
        for rid, ids in grouped.items():
            truth = annotations[rid]['human_label']
            records.append(dict(review_id=rid,truth=truth,
                triggered=any(final[method][i] != 9 for i in ids) if truth == 'other'
                else any(LABELS[final[method][i]] == truth for i in ids)))
        groups[method] = dict(other_groups=sum(r['truth']=='other' for r in records),
            false_alarm_groups=sum(r['truth']=='other' and r['triggered'] for r in records),
            positive_groups=sum(r['truth']!='other' for r in records),
            positive_detected_groups=sum(r['truth']!='other' and r['triggered'] for r in records),records=records)
    save(OUT / 'review-group-results.json', groups)
    previous_differences = {}
    for split in ['val', 'test']:
        previous_differences[split] = {}
        for domain in ['old', 'new']:
            ids = indices(domain, split)
            previous_differences[split][domain] = dict(
                n=len(ids), changed_labels=int(np.count_nonzero(
                    final['selected'][ids] != final['previous_same_defaults'][ids])))
    save(OUT / 'previous-comparison.json', previous_differences)
    # Source-group bootstrap is descriptive, not an independent-window test.
    by_source = defaultdict(list)
    for i in indices('old'):by_source[rows[i]['group']].append(i)
    counts = []
    for ids in by_source.values():
        neg = [i for i in ids if rows[i]['label']=='other'];pos=[i for i in ids if rows[i]['label']!='other']
        counts.append([len(neg),sum(final['baseline'][i]!=9 for i in neg),sum(final['selected'][i]!=9 for i in neg),
            len(pos),sum(LABELS[final['baseline'][i]]==rows[i]['label'] for i in pos),
            sum(LABELS[final['selected'][i]]==rows[i]['label'] for i in pos)])
    counts=np.array(counts);rng=np.random.default_rng(29);boot=[]
    for _ in range(2000):
        a=counts[rng.integers(0,len(counts),len(counts))].sum(0)
        if a[0] and a[3]:boot.append([(a[2]-a[1])/a[0],(a[5]-a[4])/a[3]])
    ci=np.percentile(boot,[2.5,97.5],axis=0).T.tolist()
    stats=dict(seed=29,seeds=1,old_source_groups=len(counts),new_acquisition_groups=1,
        old_selected_minus_baseline_95ci=dict(other_fpr=ci[0],positive_recall=ci[1]),
        bootstrap_replicates=2000,pvalues=None,
        limitation='Earlier test outcomes informed the revision. Overlapping windows are correlated; no untouched-test or significance claim.')
    save(OUT / 'statistics.json',stats)
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    figure,axes=plt.subplots(1,2,figsize=(11,4.5))
    names=['Baseline','Previous','Rebalanced','Last epoch'];x=np.arange(4);width=.36
    for ax,domain in zip(axes,['old','new']):
        fp=[comparison[m]['test'][domain]['other_fp'] for m in methods]
        tp=[comparison[m]['test'][domain]['positive_tp'] for m in methods]
        for bars in [ax.bar(x-width/2,fp,width,label='False action on other',color='#d55e00'),
                     ax.bar(x+width/2,tp,width,label='Correct true action',color='#0072b2')]:
            ax.bar_label(bars,padding=2,fontsize=9)
        ax.set_xticks(x,names,rotation=15);ax.set_title('Old ROI regression' if domain=='old' else 'Crowded regression')
        ax.set_ylabel('Windows');ax.set_ylim(0,max(fp+tp+[1])*1.25);ax.spines[['top','right']].set_visible(False)
        ax.legend(fontsize=8)
    figure.tight_layout()
    for ext in ['png','pdf']:figure.savefig(figdir/f'01-fixed-threshold-comparison.{ext}',dpi=180)
    plt.close(figure)
    figure,axes=plt.subplots(2,2,figsize=(11,7))
    for ri,branch in enumerate(['rgb','skeleton']):
        for run,style,title in [(PREVIOUS,'--','Previous'),(OUT,'-','Rebalanced')]:
            hist=read(run/branch/'history.json')
            for d,color in [('old','#0072b2'),('new','#e69f00')]:
                for ci,metric in enumerate(['other_fpr','positive_macro_recall']):
                    axes[ri,ci].plot([h['epoch'] for h in hist],[h['metrics'][d][metric] for h in hist],
                        style,color=color,label=f'{title}/{d}',linewidth=1.5)
        for ci in range(2):
            ax=axes[ri,ci];ax.axvline(lock['epochs'][branch],color='grey',linestyle=':',label='Full-scheme chosen epoch')
            ax.set_ylim(0,1.03);ax.set_xlabel('Epoch');ax.set_title(branch+' / validation / branch threshold .75')
            ax.set_ylabel('Other FPR' if ci==0 else 'Positive macro recall');ax.legend(fontsize=7);ax.spines[['top','right']].set_visible(False)
    figure.tight_layout()
    for ext in ['png','pdf']:figure.savefig(figdir/f'02-training-diagnostics.{ext}',dpi=180)
    plt.close(figure)
    acceptable = all(e['other_fp']<=0 and e['positive_tp']>=0 for e in effects.values())
    trained = any(lock['epochs'][b]>0 for b in ['rgb','skeleton'])
    regression = ('固定窗口回归通过，可作为进一步端到端验证候选；尚未替换正式包。' if acceptable and trained
               else '没有形成满足误报／召回要求的新权重组合，保留正式方案。')
    same_as_previous = all(previous_differences['test'][d]['changed_labels'] == 0 for d in ['old', 'new'])
    verdict = ('减少other后，固定参数测试预测与上一轮候选完全相同，未证明新增收益；保留正式模型和交付包。'
               if same_as_previous else regression)
    save(OUT/'adoption-decision.json',dict(status='retain_production' if same_as_previous or not (acceptable and trained)
            else 'candidate_requires_end_to_end_validation',
        epochs=lock['epochs'],effects=effects,production_modified=False,decision=verdict,
        window_regression_passed=acceptable,release_decision='retain_production',
        same_test_predictions_as_previous=same_as_previous,
        export_verification='passed' if (OUT/'export-verification.json').exists() else 'not_completed'))
    parity = read(OUT/'replay-parity.json')
    export_status = ('候选导出文件的安全加载和真实缓存窗口推理已验证，见export-verification.json。'
                     if (OUT/'export-verification.json').exists() else
                     '附加的候选文件导出、安全加载及概率复现检查尚未完成；自动审批服务503使验证任务未能启动。原检查点已完成上述镜像内预测。')
    (OUT/'analysis-report.md').write_text(f'''# 减少多人 other 的续训结果

{verdict}

用户确认保留原来源分组。训练保留旧5009窗、新人工真动作39窗，人工other由281减到39，预测other由278减到0。新other采样概率3.75%→1.28%，全部other12.5%→10.26%。上一轮已经类别均衡，不能解释为“other文件很多，所以每批都被other淹没”。此次同时改变了负例覆盖、伪标签使用与采样权重，属于用户提出策略的组合对照，不是单一因素因果消融。

两模型分别从正式权重训练：VideoMamba6轮、ST-GCN20轮，seed29。旧数据蒸馏、学习率、全局批次及更新步数保持；RGB从双卡改为四卡，全局抽样顺序已核对相同，每rank/worker的随机翻转分配和数值归约存在差异，不能称单一因素严格消融。新训练正例是交谈33、喝水4、挥手2，没有新跑动；同动作窗相关、有效独立事件更少。验证集没有真喝水／捡物，测试只有一个新采集时段；正例覆盖不足仍未解决。

所有完整方案均固定原四参数0.972／35／0.925／0.581及挥手3帧、跑动0.35。本轮仅在验证集比较147个权重组合，选中RGB第{lock['epochs']['rgb']}轮、骨架第{lock['epochs']['skeleton']}轮。第0轮表示保留原权重，不能称为微调新模型。未根据本轮测试结果再选轮或调整阈值。

{chr(10).join(table)}

相对原正式方案，旧测试other误报变化{effects['old']['other_fp']:+d}窗、正确动作{effects['old']['positive_tp']:+d}窗；新多人other误报变化{effects['new']['other_fp']:+d}窗、正确动作{effects['new']['positive_tp']:+d}窗。应同时看误报和漏检，不能只用总体准确率或F1作决定。

**与上一轮同参数候选相比，本轮旧test和新test最终预测标签均完全相同**，不是仅汇总计数碰巧一致。验证集多恢复一个挥手窗口，未在测试集表现出新增收益。多人审核事件组的any-trigger统计也保持：143个other组中5组误报，16个正动作组中8组正确触发。因此相对正式模型增加的1个正确跑动窗口，不能称为多识别了1次独立动作。旧测试有1窗真喝水从误判挥手变为other，仍未识别出其真实类别。

末轮组合虽然测试窗口计数更好，但验证集新正例仅正确1/18（正式3/18、本轮选定5/18），没有通过本轮预定选择条件；不能看完test后改选末轮。其多人正动作审核组检出为7/16，低于选定组合8/16，窗口数改善也不代表事件检出改善。

新正训练39窗来自17个人工审核事件：交谈12个事件、喝水4个、挥手1个，均已覆盖。训练期RGB新验证跑动的阈值前正确类别从初始10/10下降，说明部分退化发生在类别预测本身，不能只靠降阈值解释。减少other仍未使RGB通过联合选模；这并不能单独证明退化原因，但现有数据缺少多人跑动训练正例且多人正例覆盖很窄，应优先补充不同人物/场景的真动作，继续保留旧ROI重放与适量人工困难负例。

`last_epoch`是预先声明的末轮诊断，未用测试选取；`previous_same_defaults`沿用上一轮选定RGB0／骨架2且保持正式阈值，避免把上一轮物体阈值降到0.25造成的新增误报混作训练影响。单分支阈值0.75的诊断保存在branch-test-results.json，与完整方案口径不同。审核组any-trigger结果见review-group-results.json，不等于实际保存片段数。

验证／测试的人工other保持完整，没有通过删负例美化评分。原数据抽取与标签不变，7020记录保持同顺序，仅改变训练选中标记。所有预测other在本轮不参与训练或评分，XCLIP不用。人体／姿态检测、增强、物体检测权重均保持原值。

该测试已在上一轮被查看，本轮为历史来源回归，不是新盲测。窗口高度相关，单seed不报告训练均值±标准差或显著性结论。旧来源配对bootstrap区间仅描述此次固定模型的来源变化，不涵盖训练随机性和历史调参偏差；新测试只有1个采集组，不构造伪独立置信区间。

生产函数回放{parity['checked']}次判定、标签差异{parity['mismatched_labels']}，见replay-parity.json。{export_status}本轮不是实时推流、NFS或Box性能验收；缓存ROI上的识别变化不能直接等同原片端到端每小时误报率。正式包保持，未覆盖正式权重。
''')
    (OUT/'stats-appendix.md').write_text('# 统计边界\n\n'+json.dumps(stats,ensure_ascii=False,indent=2)+
        '\n\n基于原来源组配对bootstrap，2000次；不把重叠窗口当独立重复。不提供窗口级p值，无多重比较显著性声明；147组合仅用验证集选轮，已看过的旧测试仅作回归描述。\n')
    (OUT/'figure-catalog.md').write_text('''# 图表说明

- 01-fixed-threshold-comparison：同一旧／新测试集合、相同完整方案阈值，展示误报数和真动作正确数。无误差条，单训练种子。应同时比较橙色与蓝色柱，不能只看误报。
- 02-training-diagnostics：上一轮与本轮各分支验证集误报率／正类宏召回随轮数变化。各分支单独阈值0.75，虚线标本轮完整方案选轮；用于判断减少other是否缓解正例召回损失，不能将分支曲线当完整方案结果。

数据来源为results/fixed-threshold-comparison及两轮history.json；所有图提供PNG和PDF，未绘制虚构误差条。
''')
    print(json.dumps(dict(epochs=lock['epochs'],effects=effects,decision=verdict),ensure_ascii=False))


if __name__=='__main__':main()
