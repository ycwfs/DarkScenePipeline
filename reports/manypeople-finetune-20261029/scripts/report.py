"""Descriptive paired comparisons, source-cluster intervals and real figures."""
from collections import defaultdict,Counter
import csv
import sys
from common import *
sys.path.insert(0,str(OP))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluation_metrics import metrics


def main():
    rows=read(OUT/'dataset.json')['records'];results=read(OUT/'results.json');lock=read(OUT/'selection-lock.json')
    pred=np.load(OUT/'final-predictions.npz');fig=OUT/'figures';fig.mkdir(exist_ok=True)
    methods=['baseline','trained_same_defaults','selected']
    titles=['Production baseline','Selected weights, old defaults','Validation selection']
    colors=['#777777','#56B4E9','#0072B2']
    def indices(scope,split='test'):
        return [i for i,r in enumerate(rows)if r['scope']==scope and r['split']==split and r['label_kind']!='pseudo'
            and (scope=='new' or r.get('current_index')is not None)]
    table=['|集合|配置|窗口数|other误报/other数|动作正确/动作数|宏F1|', '|---|---|---:|---:|---:|---:|']
    for split in ['validation','test']:
        for domain in ('old','new'):
            for name,title in zip(methods,['原正式方案','选定权重＋原参数','验证集选定组合']):
                m=results['comparisons'][name][split][domain]
                table.append(f'|{split}/{domain}|{title}|{m["n"]}|{m["other_fp"]}/{m["other_n"]}|{m["positive_tp"]}/{m["positive_n"]}|{m["macro_f1"]:.4f}|')
    figure,axes=plt.subplots(1,2,figsize=(10,4.2))
    for ax,domain in zip(axes,['old','new']):
        values=[results['comparisons'][m]['test'][domain]['other_fp']for m in methods]
        bars=ax.bar(range(3),values,color=colors)
        for bar,v in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.1,str(v),ha='center')
        ax.set_xticks(range(3),['Baseline','New weights','Selected']);ax.set_ylabel('False action windows on labelled other')
        ax.set_title('Old ROI holdout'if domain=='old'else 'Crowded session holdout')
        ax.set_ylim(0,max(values+[1])*1.25);ax.spines[['top','right']].set_visible(False)
    figure.tight_layout();figure.savefig(fig/'01-false-positives.pdf');figure.savefig(fig/'01-false-positives.png',dpi=200);plt.close(figure)
    figure,axes=plt.subplots(2,2,figsize=(10,7))
    for ri,branch in enumerate(['rgb','skeleton']):
        hist=read(OUT/branch/'history.json')
        selected=(lock['epochs'][branch] if 'epochs' in lock else
            read(OUT/branch/'selection.json')['epoch'] if (OUT/branch/'selection.json').exists() else 0)
        for domain,color in [('old','#0072B2'),('new','#E69F00')]:
            axes[ri,0].plot([h['epoch']for h in hist],[h['metrics'][domain]['other_fpr']for h in hist],'-o',ms=3,label=domain,color=color)
            axes[ri,1].plot([h['epoch']for h in hist],[h['metrics'][domain]['positive_macro_recall']for h in hist],'-o',ms=3,label=domain,color=color)
        for ci,ax in enumerate(axes[ri]):
            ax.axvline(selected,color='#555555',linestyle=':',label='selected epoch');ax.set_xlabel('Epoch')
            ax.set_title(branch+' / validation');ax.set_ylabel('Other FPR'if ci==0 else 'Positive class macro recall')
            ax.set_ylim(0,1.03);ax.legend();ax.spines[['top','right']].set_visible(False)
    figure.tight_layout();figure.savefig(fig/'02-validation-training.pdf');figure.savefig(fig/'02-validation-training.png',dpi=200);plt.close(figure)
    evaluated=set(indices('old','val')+indices('new','val')+indices('old')+indices('new'))
    with (OUT/'window-results.csv').open('w')as stream:
        fields=['id','scope','split','group','source_id','track_id','label_kind','truth','baseline','trained_same_defaults','selected']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for i,r in enumerate(rows):
            # Full-prior arrays outside audited val/test have incomplete object evidence.
            if i not in evaluated:continue
            writer.writerow({**{k:r[k]for k in fields[:7]},'truth':r['label'],**{name:LABELS[int(pred[name][i])]for name in methods}})
    # Cluster on original source group. New holdout has ONE acquisition group: no fake CI.
    old_ids=indices('old');groups=defaultdict(list)
    for i in old_ids:groups[rows[i]['group']].append(i)
    pergroup=[]
    for group,ids in groups.items():
        negatives=[i for i in ids if rows[i]['label']=='other'];positives=[i for i in ids if rows[i]['label']!='other']
        pergroup.append([len(negatives),sum(pred['baseline'][i]!=9 for i in negatives),sum(pred['selected'][i]!=9 for i in negatives),
            len(positives),sum(LABELS[pred['baseline'][i]]==rows[i]['label']for i in positives),sum(LABELS[pred['selected'][i]]==rows[i]['label']for i in positives)])
    a=np.array(pergroup);rng=np.random.default_rng(29);samples=[]
    for _ in range(2000):
        s=a[rng.integers(0,len(a),len(a))].sum(0)
        if s[0] and s[3]:samples.append([(s[2]-s[1])/s[0],(s[5]-s[4])/s[3]])
    intervals=np.percentile(samples,[2.5,97.5],axis=0).T.tolist()
    stats=dict(seed_count=1,training_seed=29,old_test_source_groups=len(groups),
        new_test_acquisition_groups=len({rows[i]['group']for i in indices('new')}),
        paired_source_bootstrap_replicates=2000,old_test_selected_minus_baseline_95ci=dict(other_fpr=intervals[0],positive_recall=intervals[1]),
        inferential_pvalues=None,reason='Single training seed, historical sources, only one new held-out acquisition session. No independent-window significance test or multiple-search p-value.',
        weak_labels_excluded=True)
    save(OUT/'statistics.json',stats)
    # Report both trained branches separately; the full-system result alone can hide
    # a worse branch behind a very high RGB fusion weight.
    branch_results={};pose_valid=np.load(OUT/'skeleton-inputs.npz')['valid'][GATES.index(35.)]
    branch_table=['|单分支（门35、阈值.75）|集合|窗口数|other误报|动作正确/动作数|宏F1|',
        '|---|---|---:|---:|---:|---:|']
    for branch in ['rgb','skeleton']:
        for which in ['baseline','selected']:
            probs=np.load(OUT/'predictions'/f'{branch}-{which}.npz')['probs']
            if branch=='skeleton':probs=probs[GATES.index(35.)]
            labels=probs.argmax(1);labels[(labels!=9)&(probs.max(1)<.75)]=9
            key=branch+'-'+which;branch_results[key]={}
            for domain in ['old','new']:
                inds=[i for i in indices(domain)if branch=='rgb'or pose_valid[i]]
                m=metrics([rows[i]for i in inds],labels[inds]);branch_results[key][domain]=m
                branch_table.append(f'|{key}|{domain}/test|{m["n"]}|{m["other_fp"]}|{m["positive_tp"]}/{m["positive_n"]}|{m["macro_f1"]:.4f}|')
    save(OUT/'branch-test-results.json',dict(threshold=.75,gate=35.,skeleton_requires_valid_pose=True,
        no_priors_applied=True,comparisons=branch_results,test_not_used_for_selection=True))
    annotations={r['review_id']:r for r in read(OUT/'annotations.json')['annotations']}
    groups_review=defaultdict(list)
    for i in indices('new'):
        for uid in rows[i]['review_ids']:
            if annotations[uid]['human_label']==rows[i]['label']:groups_review[uid].append(i)
    reviewed=[]
    for uid,inds in groups_review.items():
        truth=annotations[uid]['human_label']
        entry=dict(review_id=uid,truth=truth,windows=len(inds))
        for method in methods:
            entry[method]=dict(any_false_action=any(pred[method][i]!=9 for i in inds)if truth=='other'else None,
                any_correct_action=any(LABELS[pred[method][i]]==truth for i in inds)if truth!='other'else None)
        reviewed.append(entry)
    group_summary={m:dict(other_groups=sum(r['truth']=='other'for r in reviewed),
        false_alarm_groups=sum(r[m]['any_false_action']is True for r in reviewed),
        positive_groups=sum(r['truth']!='other'for r in reviewed),
        correctly_detected_positive_groups=sum(r[m]['any_correct_action']is True for r in reviewed))for m in methods}
    save(OUT/'review-group-results.json',dict(scope='New test human review groups with at least one eligible window; any-trigger aggregation, not actual saved clip counts.',
        summaries=group_summary,records=reviewed))
    selections={b:(read(OUT/b/'selection.json')if (OUT/b/'selection.json').exists()else None)for b in ('rgb','skeleton')}
    config=lock['config'];selected=results['comparisons']['selected']['test'];baseline=results['comparisons']['baseline']['test']
    changes={d:dict(false_positive_delta=selected[d]['other_fp']-baseline[d]['other_fp'],
        positive_correct_delta=selected[d]['positive_tp']-baseline[d]['positive_tp'])for d in ('old','new')}
    save(OUT/'effect-sizes.json',changes)
    demo_note='实际管线视频尚未校验完成。'
    if (OUT/'demo-verification.json').exists():
        verification=read(OUT/'demo-verification.json')
        demo_note=(f"已完成{len(verification['runs'])}次实际共享生产离线入口运行，"
            f"{len(verification['videos'])}个完整输出/动作片段均通过严格完整解码；"
            "视频与事件入口见DEMO_VIDEOS.md。写错、强制释放、unfinished和片段写队列丢帧均为0。"
            "这些是短文件功能验证，源文件边界不足时前后2秒上下文仍可能不全，不能据此声称7×24推流稳定。"
            "旧test没有L<75正例，暗场取val、微暗取train，仅作功能演示，不添加到测试分母。"
            "七个输入两配置的461个同人同源帧窗口可配对，最终动作类别0变化；"
            "new-drink与new-chase两段依然全部输出other，保留这些漏检展示。"
            "短片重新启动跟踪与采样，不能把固定ROI测试指标直接对应成这些视频中的改善量。")
    report=f'''# 多人现场人工标注续训与联合参数验证

**当前交付建议：保持原生产模型和参数，不晋升本轮联合候选。** 联合搜索只在验证集选优；锁定后的多人测试显示other误报5/271→11/271，新增6窗均为误喝水，而真实动作只多正确1窗。按“优先压低误报”的目标，这组候选未通过采用判断。没有根据测试结果继续搜索或改写selection-lock.json。

VideoMamba已实际训练6轮，但续训检查点未在冻结约束下胜过原模型，因此最终RGB保留epoch0；不能将导出的RGB文件称为新的微调成果。ST-GCN第2轮在单分支测试上减少了误报，旧场景other误报28→4、新多人38→26，真实动作正确分别23→32和6→6；这是分支收益，不能等同完整方案同比改善。预先列出的“新骨架＋原参数”对照完整方案，旧测试保持34/51动作正确与2/163 other误报，新测试正确10→11/28、other误报仍5/271，值得进一步独立现场验证，但本轮没有据此再次选参或替换交付包。

降低物体阈值主要恢复旧场景喝水：旧测试真喝水3→5/11，同时多人新增6个误喝水。新验证集只有挥手8窗和跑动10窗，没有真喝水或捡物正例，且只有一个采集时段；不能把该验证集上的最优阈值解释为跨场景通用最优。

上传473条审核记录，385确认、47排除、41不确定；两处冲突按用户确认排除。仅使用主VideoMamba＋骨架审核，XCLIP未用于训练、排除规则、选模或评分。8766审核服务已停止。

本轮分别微调VideoMamba与ST-GCN，旧数据重放、原模型蒸馏约束。RGB完成6轮、骨架完成20轮；单次seed29。单分支诊断选择RGB epoch{selections['rgb']['epoch']}／骨架epoch{selections['skeleton']['epoch']}；最终完整方案联合选择 **RGB epoch{lock['epochs']['rgb']}／骨架epoch{lock['epochs']['skeleton']}**。epoch0代表验证后保留原权重，不能称为新的微调权重。

验证集搜索{lock['grid_count']}组、{lock['accepted_count']}组满足约束，选中：RGB={config['rgb']}，骨架={config['skeleton']}，RGB融合权重 **{config['alpha']}**，原图照度门 **{config['gate']}**，动作阈值 **{config['action']}**，物体阈值 **{config['object']}**。保持挥手3个不同源帧、跑动0.35人体框高度/秒，七类物体模型及增强/人体/姿态模型不变。alpha=1只作消融，不参与完整方案晋升。

{chr(10).join(table)}

{chr(10).join(branch_table)}

单分支表用于观察两个动作模型本身的变化；骨架只在有合格姿态的同人窗口上计分，分母不能与RGB直接互比。该表未加物体/挥手/跑动先验，不代表正式最终输出。

按人工审核条目合并相邻窗口，新测试中有可用窗口的other条目共{group_summary['baseline']['other_groups']}条，出现至少一次误动作的条目数原方案{group_summary['baseline']['false_alarm_groups']}→选定组合{group_summary['selected']['false_alarm_groups']}；有可用窗口的正动作条目共{group_summary['baseline']['positive_groups']}条，至少一次正确识别{group_summary['baseline']['correctly_detected_positive_groups']}→{group_summary['selected']['correctly_detected_positive_groups']}。这是审核组内any-trigger口径，不能等同实际保存clip数量或每小时报警数；明细review-group-results.json。

旧测试动作正确变化{changes['old']['positive_correct_delta']:+d}窗、other误报变化{changes['old']['false_positive_delta']:+d}窗；新多人测试动作正确变化{changes['new']['positive_correct_delta']:+d}窗、other误报变化{changes['new']['false_positive_delta']:+d}窗。以上变化是固定同人输入、当前已审核窗口上的配对结果。

{demo_note}

## 数据及划分

旧535源保持原train/val/test分组，覆盖所有旧ROI中有可用标注的源；同人同源帧重叠缓存优先取当前前端，旧版独有窗口作为额外重放。计划旧5504窗，新1516窗，其中736人工窗口、780弱标签other。训练从原片/原增强缓存提取720增强图上的同人ROI，32帧、1秒、15%外扩、224×400；不是带字幕审核预览。

实际RGB训练5607窗＝旧5009＋新人工320＋新弱标签278；其余502弱标签归属val/test采集组，仅留作记录，从未作为训练或评分真值。骨架训练额外要求姿态质量合格，共3788窗。新正动作训练只有talk33、drink4、wave2窗，没有新跑动训练正例；因此不能认为新多人所有动作已得到均衡适配。

新train=05/07/09、val=04/06/08、test=01/02/03，同采集时段不同视角不跨集合。新talk只有训练来源，没有独立talk召回验证。弱标签other仅限已确认真人的同轨迹、原RGB other≥.90且检测≥.5、远离审核时段；损失权重.25，不算验证/测试真值。12条other与同轨迹排除实体冲突，整轨隔离；明细见protocol-amendment-before-training.json。

此次适配没有重新纳入HMDB/Kinetics等公共视频；旧少人能力保持的证据限于此次有验证/测试支持的现场动作类别。fight、shake_hands无新增正例，throw/fall等类别缺少可靠独立测试支持，不能宣称九类能力全部保持。

主指标只计当前前端旧ROI和新人工审核ROI；历史前端独有窗口仅参与训练重放，避免将相似重复视图重复计分。原始未检出人物、未知动作时间段不计为正确；本次未给整段所有帧补全真值，因此不能据此声称整片召回或每小时误报率。

原RGB重新推理与历史缓存另作输入复现检查，见baseline-reproduction-audit.json；训练/验证域有少量AMP批次浮点差，不宣称概率逐位相同。本文所有前后比较均使用本轮同输入重推的原模型概率。

## 参数选择和证据范围

先在验证集保住旧各有支持动作类别召回（下降不超过5个百分点）及新动作宏召回（下降不超过10个百分点），再最小化新旧域平均other误报率，之后比较宏F1/正确动作数/与原参数距离。最终将RGB全部7个检查点（含原模型）与骨架全部21个检查点及四参数一起比较，避免单分支阈值.75的选轮提前排除有利于完整融合的候选。联合选择协议在RGB训练期间、尚未查看任何test指标时补充，原因和时间见joint-selection-protocol.json；单分支选择保留作诊断。全部选择锁定后才计算测试指标。所谓最优仅指列出的检查点与网格及这次验证集。

所有评分窗口均补齐两种物体动作的原图对应增强ROI检测证据，使用正式同一帧支持策略；骨架按每个照度门逐帧遮罩并重新计算归一化、挥手、速度证据。这里是固定ROI对照；改变照度门可能改变实际跟踪，端到端视频检查另见demo-verification.json（文件存在时）。不能将固定ROI指标当作完整实时系统的准确率或实时性能。

所有九段新片已参与过历史算法校准，旧模型也见过部分旧来源；本轮做采集组隔离，但不是全新盲测。新测试仅1个采集时段，不能用数百个相关窗口冒充数百个独立样本。单训练种子不报告均值±标准差或显著优越性；旧源组配对bootstrap区间见stats-appendix.md。

模型、配置与哈希见candidate-config.json（生成后）；原生产模型、平台/Box归档和镜像未自动替换。审核与训练证据、失败/过渡窗口、参数全网格均保留在本目录。
'''
    (OUT/'analysis-report.md').write_text(report)
    (OUT/'stats-appendix.md').write_text('# 统计附录\n\n'+json.dumps(stats,ensure_ascii=False,indent=2)+'\n\n逐窗配对但窗高度相关；old按原source group重采样2000次，报告selected−baseline的百分比差95%百分位区间。该区间只描述固定这一次训练模型在历史源组上的抽样差异，不涵盖训练随机性、参数搜索和旧数据被看过的偏差。new只有1采集组，拒绝构造虚假的组间CI。没有执行以窗口为独立样本的McNemar/t检验，无p值或多重比较显著性声明。具体支持数、混淆矩阵见results.json；各方法绝对差见effect-sizes.json。\n')
    (OUT/'figure-catalog.md').write_text('''# 图表说明

- `figures/01-false-positives.pdf/png`：相同人工/历史真值other窗口上，原正式方案、微调同参数、最终选择的误报数；数据来自results.json/test。无误差条，因为仅一次训练与固定集合，不以窗口构造独立重复。阅读时同时核对表中动作正确数，不能仅看误报下降就认定改进。
- `figures/02-validation-training.pdf/png`：两分支每轮在旧/新验证集上的other误报率和正动作类别宏召回；epoch0为原模型，虚线为联合验证选择轮（RGB0、骨架2）。曲线是各分支阈值.75的诊断，未含test或pseudo真值；不能据此把分支指标当成完整方案。用于检查减少误报时丢失真实动作的程度，以及旧能力是否保持；验证选择不代表已经晋升生产。
''')
    print('REPORT',changes,config,flush=True)


if __name__=='__main__':main()
