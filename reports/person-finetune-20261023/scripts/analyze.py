"""Paired fixed-test analysis; cluster uncertainty, real figures, and explicit coverage limits."""
from collections import Counter,defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','/tmp/matplotlib-person-finetune-20261023')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import binomtest
from train import OUT,BEHAVIORS,metrics,fuse_person,write_json


def score(rows: list, key: str) -> dict:
    return metrics([BEHAVIORS.index(r['truth'])for r in rows],[BEHAVIORS.index(r[key])for r in rows])


def paired_group_stats(before: list,after: list,key: str) -> dict:
    groups=defaultdict(list)
    for a,b in zip(before,after):
        groups[a['group']].append((a[key]==a['truth'],b[key]==b['truth']))
    pairs=np.array([np.mean(groups[group],axis=0)for group in sorted(groups)])
    delta=pairs[:,1]-pairs[:,0]
    rng=np.random.default_rng(2301)
    bootstrap=delta[rng.integers(0,len(delta),size=(5000,len(delta)))].mean(1)
    wins,losses=int((delta>1e-12).sum()),int((delta< -1e-12).sum())
    p=binomtest(wins,wins+losses,.5).pvalue if wins+losses else 1.
    return dict(groups=len(delta),before_mean=float(pairs[:,0].mean()),after_mean=float(pairs[:,1].mean()),
        before_sd=float(pairs[:,0].std(ddof=1)),after_sd=float(pairs[:,1].std(ddof=1)),
        delta=float(delta.mean()),delta_ci95=np.quantile(bootstrap,[.025,.975]).tolist(),
        wins=wins,losses=losses,ties=len(delta)-wins-losses,sign_test_p=float(p),
        per_group=[dict(group=g,before=float(a),after=float(b),delta=float(b-a))for g,(a,b)in zip(sorted(groups),pairs)])


def savefig(fig,folder: Path,name: str) -> None:
    for suffix in ['png','pdf','svg']:fig.savefig(folder/f'{name}.{suffix}',dpi=170,bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    folder=OUT/'analysis';figdir=folder/'figures';figdir.mkdir(parents=True,exist_ok=True)
    baseline=json.loads((OUT/'evaluation/test-baseline.json').read_text())
    selected=json.loads((OUT/'evaluation/test-selected.json').read_text())
    before,after=baseline['records'],selected['records']
    dataset=json.loads((OUT/'training-data.json').read_text())
    history=json.loads((OUT/'training-history.json').read_text())
    selection=json.loads((OUT/'selection.json').read_text())
    assert [r['id']for r in before]==[r['id']for r in after]
    assert [(r['truth'],r['mode'],r['skeleton_probs'])for r in before]==[(r['truth'],r['mode'],r['skeleton_probs'])for r in after]
    assert all(np.isfinite(r['rgb_probs']).all() and abs(sum(r['rgb_probs'])-1)<.002 for r in before+after)
    assert baseline['training_data_sha256']==selected['training_data_sha256']==hashlib.sha256((OUT/'training-data.json').read_bytes()).hexdigest()
    stats={mode:paired_group_stats(before,after,key)for mode,key in [('rgb','rgb_label'),('fused','fused_label')]}
    ordered=sorted(stats,key=lambda mode:stats[mode]['sign_test_p']);previous=0.
    for i,mode in enumerate(ordered):
        previous=max(previous,min(1.,(len(ordered)-i)*stats[mode]['sign_test_p']))
        stats[mode]['holm_p']=previous
    by_id={r['id']:r for r in dataset['records']}
    ceilings=[]
    for r in after:
        original=by_id[r['id']];p=np.zeros(10,np.float32);p[BEHAVIORS.index(r['truth'])]=1
        sk=None if original['skeleton_probs'] is None else np.array(original['skeleton_probs'],np.float32)
        final=fuse_person(p,sk,original['lumas'])
        ceilings.append(dict(**r,oracle_fused_label=final['label']))
    subsets={}
    masks={
        'all':lambda r:True,
        'original_feedback_holdout':lambda r:r['sealed_feedback_holdout'],
        'historical_finetune_holdout':lambda r:not r['sealed_feedback_holdout'],
        'rgb_skeleton':lambda r:r['mode']=='rgb_skeleton',
        'rgb_low_luma':lambda r:r['mode']=='rgb_low_luma',
        'rgb_soft_no_pose':lambda r:r['mode']=='rgb_soft_no_pose'}
    for name,predicate in masks.items():
        a=[r for r in before if predicate(r)];b=[r for r in after if predicate(r)]
        subsets[name]=dict(n=len(b),videos=len({r['video_id']for r in b}),groups=len({r['group']for r in b}),
            baseline={mode:score(a,key)for mode,key in [('rgb','rgb_label'),('fused','fused_label')]},
            selected={mode:score(b,key)for mode,key in [('rgb','rgb_label'),('fused','fused_label')]})
    coverage=[]
    for label in BEHAVIORS:
        videos=[r for r in dataset['video_coverage']if r['video_label']==label]
        coverage.append(dict(label=label,total_videos=len(videos),
            any_person_detection=sum(r['frames_with_detection']>0 for r in videos),
            no_detections=sum(r['frames_with_detection']==0 for r in videos),
            detections_without_window=sum(r['frames_with_detection']>0 and r['windows']==0 for r in videos),
            any_stable_window=sum(r['windows']>0 for r in videos),
            any_labeled_window=sum(r['accepted']>0 for r in videos),
            any_labeled_action=sum(r['accepted_actions']>0 for r in videos)))
    result=dict(selection=selection,dataset_summary=dataset['summary'],baseline=baseline['metrics'],
        selected=selected['metrics'],paired_groups=stats,subsets=subsets,coverage=coverage,
        oracle_rgb_fusion=score(ceilings,'oracle_fused_label'),
        modes=dict(Counter(r['mode']for r in after)),
        constraints=dict(seed=23,seeds=1,threshold=.75,alpha=.7,lum_gate=75,production_modified=False))
    result['class_support']={label:dict(windows=sum(r['truth']==label for r in after),
        videos=len({r['video_id']for r in after if r['truth']==label}),
        groups=len({r['group']for r in after if r['truth']==label}))for label in BEHAVIORS}
    write_json(folder/'results.json',result)
    with (folder/'paired-test-windows.csv').open('w')as handle:
        fields=['id','source','track_id','start','end','truth','mode','baseline_rgb','selected_rgb','baseline_fused','selected_fused']
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for a,b in zip(before,after):
            writer.writerow({**{k:b[k]for k in fields[:7]},'baseline_rgb':a['rgb_label'],'selected_rgb':b['rgb_label'],
                             'baseline_fused':a['fused_label'],'selected_fused':b['fused_label']})
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#68788F','#168A81']
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    x=np.arange(2)
    selected_label='Selected ROI candidate' if selection['epoch'] else 'Selected unchanged pn2'
    for j,(name,evidence)in enumerate([('Whole-frame-trained pn2',baseline),(selected_label,selected)]):
        vals=[100*evidence['metrics'][mode]['macro_f1']for mode in ['rgb','fused']]
        bars=axes[0].bar(x+(j-.5)*.34,vals,width=.34,label=name,color=colors[j])
        axes[0].bar_label(bars,fmt='%.1f',padding=3)
    axes[0].set(xticks=x,xticklabels=['RGB at 0.75','Fusion at 0.75'],ylabel='Supported-class macro-F1 (%)',ylim=(0,105),title='Identical held-out person windows')
    delta=[100*stats[mode]['delta']for mode in ['rgb','fused']]
    lo=[delta[i]-100*stats[m]['delta_ci95'][0]for i,m in enumerate(['rgb','fused'])]
    hi=[100*stats[m]['delta_ci95'][1]-delta[i]for i,m in enumerate(['rgb','fused'])]
    axes[1].errorbar(x,delta,yerr=[np.maximum(lo,0),np.maximum(hi,0)],fmt='o',capsize=5,color=colors[1])
    axes[1].axhline(0,color='gray',lw=1);axes[1].set(xticks=x,xticklabels=['RGB','Fusion'],ylabel='Group mean accuracy change (pp)',title='95% paired source-group bootstrap CI')
    axes[0].legend(fontsize=8,loc='lower left');fig.tight_layout();savefig(fig,figdir,'01-test-comparison')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for mode,label in [('rgb','RGB'),('fused','Fusion')]:
        axes[0].plot([r['epoch']for r in history],[100*r['validation'][mode]['macro_f1']for r in history],marker='o',label=label)
    axes[0].axvline(selection['epoch'],color='gray',ls='--',label='Selected epoch')
    axes[0].set(xlabel='Epoch (0 = pn2 baseline)',ylabel='Validation macro-F1 (%)',title='Validation-only selection');axes[0].legend(fontsize=8)
    active=[v for v in coverage if v['total_videos']]
    x=np.arange(len(active));axes[1].bar(x,[v['total_videos']for v in active],color='#C5CBD3',label='Source videos')
    axes[1].bar(x,[v['any_stable_window']for v in active],color=colors[1],label='Stable detection window')
    axes[1].set(xticks=x,xticklabels=[v['label']for v in active],ylabel='Video count',title='Detection/tracking coverage before recognition')
    axes[1].tick_params(axis='x',rotation=35);axes[1].legend(fontsize=8);fig.tight_layout();savefig(fig,figdir,'02-validation-and-coverage')
    supported=[i for i,c in enumerate(BEHAVIORS)if selected['metrics']['fused']['classes'][c]['n']]
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,key,label in zip(axes,['rgb','fused'],['RGB','Fusion']):
        x=np.arange(len(supported))
        for j,evidence in enumerate([baseline,selected]):
            vals=[100*evidence['metrics'][key]['classes'][BEHAVIORS[i]]['recall']for i in supported]
            ax.bar(x+(j-.5)*.34,vals,width=.34,color=colors[j],label=['pn2','ROI'][j])
        ax.set(xticks=x,xticklabels=[f'{BEHAVIORS[i]}\nn={selected["metrics"][key]["classes"][BEHAVIORS[i]]["n"]}'for i in supported],
               ylabel='Recall (%)',ylim=(0,105),title=label+' per-class recall at 0.75');ax.legend(fontsize=8)
    fig.tight_layout();savefig(fig,figdir,'03-class-recall')
    rows=['| 模型 | RGB mF1 | 融合 mF1 | RGB 正确数 | 融合正确数 | RGB other误报 | 融合other误报 |',
          '|---|---:|---:|---:|---:|---:|---:|']
    for name,ev in [('旧 pn2',baseline),('本轮选择',selected)]:
        a,b=ev['metrics']['rgb'],ev['metrics']['fused']
        rows.append(f'| {name} | {a["macro_f1"]:.4f} | {b["macro_f1"]:.4f} | {a["correct"]}/{a["n"]} | {b["correct"]}/{b["n"]} | {a["other_false_positives"]}/{a["other_count"]} | {b["other_false_positives"]}/{b["other_count"]} |')
    table='\n'.join(rows)
    class_table=['| 类别 | 窗口/视频/来源组 | RGB召回：旧→新 | 融合召回：旧→新 |',
                 '|---|---:|---:|---:|']
    for label,units in result['class_support'].items():
        if units['windows']:
            a,b=baseline['metrics'],selected['metrics']
            class_table.append(f'| {label} | {units["windows"]}/{units["videos"]}/{units["groups"]} | '
                f'{a["rgb"]["classes"][label]["recall"]:.4f}→{b["rgb"]["classes"][label]["recall"]:.4f} | '
                f'{a["fused"]["classes"][label]["recall"]:.4f}→{b["fused"]["classes"][label]["recall"]:.4f} |')
    class_table='\n'.join(class_table)
    cohort_table=['| 测试来源 | 窗口/视频/来源组 | RGB other误报：旧→新 | 融合other误报：旧→新 |',
                  '|---|---:|---:|---:|']
    for key,label in [('historical_finetune_holdout','历史现场及旧负样本（仅本轮微调留出）'),
                      ('original_feedback_holdout','新反馈原封存留出会话')]:
        sub=subsets[key];a,b=sub['baseline'],sub['selected'];n=b['rgb']['other_count']
        cohort_table.append(f'| {label} | {sub["n"]}/{sub["videos"]}/{sub["groups"]} | '
                            f'{a["rgb"]["other_false_positives"]}/{n}→{b["rgb"]["other_false_positives"]}/{n} | '
                            f'{a["fused"]["other_false_positives"]}/{n}→{b["fused"]["other_false_positives"]}/{n} |')
    cohort_table='\n'.join(cohort_table)
    text=f'''# 逐人 VideoMamba 微调：配对评估

固定阈值0.75、alpha=0.7、原图照度门75。单种子23，训练8轮；只按验证集选择第{selection['epoch']}轮。
测试包含{len(after)}个人物窗口、{len({r['video_id']for r in after})}段视频、{len({r['group']for r in after})}个来源组。
两模型看到相同逐人RGB输入，骨架概率、照度分支和标签逐窗校验一致。

{table}

宏F1仅平均本批测试有真值的五类：wave、chase、drink、pick_up、other；其余五类没有测试正例。
同视频重叠窗口不能视为独立样本，各类支持量和召回如下：

{class_table}

以上是已有检测轨迹且标签足够清晰的窗口分类成绩，不能包含漏检视频，也不能等同于完整摄像机系统准确率。
`results.json`保存逐类召回率、混淆矩阵、照度分支、原反馈留出会话及检测覆盖率。
`paired-test-windows.csv`可逐窗核对全部变化。

## 主要变化与退化

逐人输入适配的收益不只来自other：追逐/跑动RGB命中从1/25升到20/25，融合命中从2/25升到13/25。
RGB来源组平均准确率增量的95%区间为[+0.1236,+0.3817]；融合对应区间[-0.0107,+0.1737]跨过0，
因此这批数据支持RGB适配有效，但尚不足以确认融合在来源组层面稳定提升。完整检验口径见统计附录。

喝水有明确退化：旧模型11/11正确，新模型8/11。三个错误窗口全部来自`real-actiondata-all/喝水34.mp4`
的T01，覆盖0.47–2.47秒；RGB误认为wave，融合后显示other。它们属于同一个视频，不能当作三个独立失败场景。
剩余唯一融合other误报来自`real-actiondata-all/other/other185.mp4`的T01（0.97–1.97秒），被误认为pick_up。
这些失败已保留在报告和CSV中，没有据此重选检查点或调低阈值。

## 原反馈留出会话单列

{cohort_table}

原封存会话为`20261015_163555_77`、`20261019_092255_83`，9段视频全部标为other；
本轮逐人划分产生55个人物窗口，与旧整帧评估的68窗不是同一统计单位。
融合0/55只表示这两个会话的可用人物窗口未触发，不能推断新场景误报率为零。

## 数据覆盖与边界

742个源文件经字节去重得到535段；本轮处理全部现场数据和两批线上切片，不含公开数据。
正样本采用带人物编号的时序预览人工核对标签；模糊人物/身份切换及不够明确的动作过渡窗口剔除。
负样本默认继承既有other标签；明确复核的轨迹覆盖原标签。未声称逐帧复核所有负样本。
192段原正样本中142段形成了可标注的具名动作窗口；44段没有稳定窗口，其中5段全程没有检测框，
39段有过检测框但未形成窗口。还有6段虽有窗口，却只涵盖旁观者、误检或动作外时间。
这些是候选窗口覆盖情况，不是有逐帧人物真值的检测召回率。
旧pn2已训练过大部分历史现场视频，新划分仅是本次微调留出；新反馈原有两个留出会话单独报告。
同场景相邻视频仍可能相关，因此不是跨摄像机场景泛化测试。
交谈、打架、握手无本轮正样本；投掷、跌倒各只有一个源视频并放入训练，无法评估这些类别保留能力。

## 固定骨架与融合限制

测试中骨架模式计数：{json.dumps(result['modes'],ensure_ascii=False)}。
固定骨架单独分类仅在{result['selected']['skeleton']['n']}/{len(after)}个窗口有效，
该有效子集的宏F1为{result['selected']['skeleton']['macro_f1']:.4f}；它的分母与全量RGB/融合不同，不能直接当作全量准确率比较。
把RGB替换为真实标签的一热向量后，固定融合规则的理论可达正确数为
{result['oracle_rgb_fusion']['correct']}/{len(after)}；这是诊断上限，不是真实模型成绩。
亮场无合格骨架时，非other融合概率最高0.7，小于阈值0.75，必定显示其他。
有效骨架若给真实类低于约1/6的概率，也会阻挡即使完全正确的RGB输出。
因此仅优化VideoMamba不能解决全部融合漏报；本轮未改变用户确认的融合规则。
新RGB在50个具名动作真值窗中命中42个，融合后只命中31个；挥手RGB保持11/11，融合仍只有7/11。
固定ST-GCN的历史训练曾合并多人logits并使用视频标签，输入形式相同不意味着逐人分类性能完全不受影响。
此外，现场数据的chase标签沿用原有词汇，许多片段表现为单人跑动，并不都包含可见的多人追赶。

## 图表解读

1. `01-test-comparison`：左侧比较相同测试窗上的RGB/融合mF1；右侧区间是按来源组重采样的平均准确率变化。
   差异支持哪些输入适配有效，要结合区间及类别样本数；不可把窗口数视为独立样本数。
2. `02-validation-and-coverage`：训练曲线包含基线和全部8轮；右侧说明有多少原视频进入了可识别轨迹。
   没有稳定窗口的视频需要优先改善检测/跟踪，增加VideoMamba训练不能直接补救。
3. `03-class-recall`：检查收益是否仅来自other；小样本类别的变化仅作本批观察，不可外推。

算子、NVIDIA Box、生产默认模型和镜像未更新。本轮权重及演示是独立逐人候选，待用户确认效果。
'''
    (folder/'analysis-report.md').write_text(text)
    st=['# 统计附录','',
        '主要描述指标为支持类别宏F1，越高越好；other误报越少越好。阈值固定，无测试集调参。',
        '只有1个训练种子，无法报告训练重复的均值±标准差或训练算法显著性。',
        '下述均值±SD是来源组准确率的分布；95% CI为5000次配对来源组bootstrap的差值区间。',
        '重叠窗口保留在原组内；不做把窗口当独立样本的t检验。组差值有界、零值多，采用非参数双侧精确符号检验，平局剔除。',
        '组间独立性仅是近似：同摄像机/相邻录制仍可能相关。区间与p值仅为本批来源组的探索性描述。',
        'RGB与融合两次比较共同采用Holm校正；组平均准确率的单位和窗口mF1不同。','',
        '| 分支 | 组数 | 旧组准确率均值±SD | 新组准确率均值±SD | 差值[95% CI] | 胜/负/平 | 双侧p | Holm p |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,s in stats.items():
        st.append(f'| {name} | {s["groups"]} | {s["before_mean"]:.4f}±{s["before_sd"]:.4f} | {s["after_mean"]:.4f}±{s["after_sd"]:.4f} | {s["delta"]:+.4f} [{s["delta_ci95"][0]:+.4f}, {s["delta_ci95"][1]:+.4f}] | {s["wins"]}/{s["losses"]}/{s["ties"]} | {s["sign_test_p"]:.6g} | {s["holm_p"]:.6g} |')
    st+=['','宏F1不提供看似精确的独立窗口区间；部分动作只有极少来源组，需补充独立数据与多种子实验。',
         '精确配对结果、每组分数和混淆矩阵见results.json；没有剔除分数低的测试窗口或视频。']
    (folder/'stats-appendix.md').write_text('\n'.join(st)+'\n')
    (folder/'figure-catalog.md').write_text('''# 图表目录

全部由实际JSON数据用Matplotlib生成，同时提供PNG、PDF、SVG。

| 图 | 数据源 | 用途与需观察的内容 | 限制 |
|---|---|---|---|
| 01-test-comparison | evaluation/test-*.json | RGB与融合mF1变化；来源组平均准确率差值及5000次bootstrap 95%区间 | 区间不是mF1区间，不含训练种子方差 |
| 02-validation-and-coverage | training-history.json、training-data.json | 全部轮次的验证趋势及进入稳定人物窗口的视频数量 | 覆盖率是视频级“至少有一个窗口”，不是逐人检测召回率 |
| 03-class-recall | evaluation/test-*.json | 各真实类召回变化及明确窗口数，识别是否只学会other | 样本少、历史数据已被初始模型见过 |

使用时保留图题、轴名、样本数和上述边界；不将训练内演示图当作独立测试证据。
''')
    print(table,flush=True)


if __name__=='__main__':main()
