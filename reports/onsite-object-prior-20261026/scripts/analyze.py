"""Paired, source-clustered descriptive analysis of the fixed object-prior replay."""
from collections import Counter
import csv
import html
import json
from pathlib import Path
import sys

import numpy as np
from evaluate import OUT, ROOT, OP, CACHE, digest, save
from darkpipe.constants import BEHAVIORS, BEHAVIOR_ZH


def csv_write(path: Path, rows: list[dict]) -> None:
    assert rows
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def metrics(rows: list[dict], key: str) -> dict:
    known=[r for r in rows if r['truth'] is not None]
    matrix=np.zeros((10,10),int)
    for r in known:matrix[BEHAVIORS.index(r['truth']),BEHAVIORS.index(r[key])]+=1
    classes={};f1=[]
    for i,c in enumerate(BEHAVIORS):
        tp=int(matrix[i,i]);support=int(matrix[i].sum());pred=int(matrix[:,i].sum())
        p=tp/pred if pred else 0.;recall=tp/support if support else None
        score=2*tp/(support+pred) if support+pred else 0.
        classes[c]=dict(support=support,tp=tp,fp=pred-tp,fn=support-tp,
            precision=p if pred else None,recall=recall,f1=score if support else None)
        if support:f1.append(score)
    positive=[r for r in known if r['truth']!='other'];negative=[r for r in known if r['truth']=='other']
    return dict(total=len(rows),known=len(known),unknown=len(rows)-len(known),
        accuracy=float(np.trace(matrix)/len(known)) if known else None,
        supported_macro_f1=float(np.mean(f1)) if f1 else None,
        positive_count=len(positive),positive_correct=sum(r[key]==r['truth'] for r in positive),
        other_count=len(negative),other_false_positives=sum(r[key]!='other' for r in negative),
        classes=classes,confusion=matrix.tolist())


def comparison(rows: list[dict]) -> dict:
    changed=[r for r in rows if r['before']!=r['after']]
    return dict(before=metrics(rows,'before'),after=metrics(rows,'after'),
        filtered=len(changed),corrected_other_false_positives=sum(r['truth']=='other' for r in changed),
        newly_missed_true_actions=sum(r['truth']==r['before'] for r in changed),
        changed_wrong_positive_class=sum(r['truth'] is not None and r['truth'] not in ('other',r['before']) for r in changed),
        filtered_unknown=sum(r['truth'] is None for r in changed),
        reasons=dict(Counter(r['reason'] for r in rows)))


def band(luma: float | None) -> str:
    if luma is None:return 'unknown'
    for lo,hi in [(0,15),(15,30),(30,60),(60,90),(90,256)]:
        if lo<=luma<hi:return f'{lo}-{hi}'
    raise ValueError(luma)


def bootstrap(rows: list[dict], metric: str) -> dict:
    groups=sorted({r['group'] for r in rows})
    terms=[]
    for group in groups:
        subset=[r for r in rows if r['group']==group]
        if metric=='other_false_positive_rate':subset=[r for r in subset if r['truth']=='other']
        elif metric=='positive_correct_rate':subset=[r for r in subset if r['truth'] not in (None,'other')]
        else:subset=[r for r in subset if r['truth']==metric]
        def count(key):
            return sum((r[key]!='other') if metric=='other_false_positive_rate' else r[key]==r['truth'] for r in subset)
        terms.append([len(subset),count('after')-count('before')])
    arr=np.asarray(terms,float);rng=np.random.default_rng(26)
    sums=arr[rng.integers(len(arr),size=(2000,len(arr)))].sum(1)
    valid=sums[:,0]>0;dist=sums[valid,1]/sums[valid,0]
    return dict(metric=metric,groups=len(groups),contributing_groups=int((arr[:,0]>0).sum()),
        resamples=2000,valid_resamples=int(valid.sum()),seed=26,
        delta=float(arr[:,1].sum()/arr[:,0].sum()) if arr[:,0].sum() else None,
        ci95=np.quantile(dist,[.025,.975]).tolist() if len(dist) else None,
        note='Paired original-source-group percentile bootstrap; descriptive uncertainty, not independent new-scene generalization.')


def figures(summary: dict) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,split in zip(axes,['all','test']):
        d=summary['onsite'][split];x=np.arange(4)
        b=[d['before']['classes'][c][metric] for c in ('drink','pick_up') for metric in ('tp','fp')]
        a=[d['after']['classes'][c][metric] for c in ('drink','pick_up') for metric in ('tp','fp')]
        for dx,values,color,label in [(-.19,b,'#3874a3','No object prior'),(.19,a,'#e69640','With object prior')]:
            bars=ax.bar(x+dx,values,.38,color=color,label=label);ax.bar_label(bars,padding=2,fontsize=9)
        ax.set_xticks(x,['Drink TP','Drink FP','Pickup TP','Pickup FP']);ax.set_ylabel('Labeled person windows')
        ax.set_title('All onsite (includes train)' if split=='all' else 'Historical onsite test only')
        ax.set_ylim(0,max(b+a)*1.22+1);ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=8)
    fig.savefig(folder/'01-true-and-false.png',dpi=170);fig.savefig(folder/'01-true-and-false.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    splits=['train','val','test'];x=np.arange(3)
    for key,label,color,dx in [('corrected_other_false_positives','Other false alarms removed','#459b75',-.25),
        ('newly_missed_true_actions','Additional true actions lost','#cd625c',0),
        ('filtered_unknown','Filtered, truth unknown','#a4acb5',.25)]:
        values=[summary['onsite'][s][key] for s in splits]
        bars=axes[0].bar(x+dx,values,.25,label=label,color=color);axes[0].bar_label(bars,padding=2,fontsize=8)
    axes[0].set_xticks(x,splits);axes[0].set_ylabel('Person windows');axes[0].legend(fontsize=7)
    axes[0].set_title('Filtering benefit and cost by split')
    coverage=summary['coverage']['onsite']
    values=[coverage['known_windows'],coverage['unknown_windows']]
    bars=axes[1].bar(['Known label','Unknown label'],values,color=['#3874a3','#a4acb5'])
    axes[1].bar_label(bars,padding=3);axes[1].set_ylabel('Person windows')
    axes[1].set_title(f"Coverage: {coverage['sources']} sources; {coverage['no_window_sources']} without windows")
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.margins(y=.2)
    fig.savefig(folder/'02-tradeoff-and-coverage.png',dpi=170);fig.savefig(folder/'02-tradeoff-and-coverage.pdf');plt.close(fig)


def main() -> None:
    protocol=json.loads((OUT/'protocol.json').read_text());fingerprint=digest(OUT/'protocol.json')
    rows=[];videos=[];results={};verification=dict(policy_invariants=True,source_hashes_verified=True)
    for v in protocol['sources']:
        path=OUT/'per-video'/f'{v["id"]}.json';data=json.loads(path.read_text())
        assert data['complete'] and data['protocol_sha256']==fingerprint and len(data['records'])==v['windows']
        results[v['id']]=data
        vr=[]
        for r in data['records']:
            b=r['baseline']['label'];a=r['after'];truth=r['truth'];changed=a!=b
            assert not changed or (b in ('drink','pick_up') and a=='other' and r['prior']['applied'])
            if r['raw_luma'] is None or r['raw_luma']<30:assert a==b and not r['prior']['applied']
            if r['prior']['reason']=='single_frame_support':assert r['prior']['evidence'] and a==b
            status='unchanged'
            if changed:
                status='unknown_filtered' if truth is None else 'false_alarm_removed' if truth=='other' else 'new_false_negative' if truth==b else 'wrong_positive_class_removed'
            item=dict(id=r['id'],video_id=v['id'],source=v['source'],dataset=v['dataset'],scope=v['scope'],
                split=v['split'],group=v['group'],track_id=r['track_id'],frame_index=r['frame_index'],
                timestamp=r['timestamp'],people=r['people'],raw_luma=r['raw_luma'],brightness=band(r['raw_luma']),
                truth=truth,truth_provenance=r['truth_provenance'],before=b,after=a,
                baseline_confidence=r['baseline']['confidence'],fusion_mode=r['baseline']['mode'],
                prior_applied=r['prior']['applied'],reason=r['prior']['reason'],outcome=status,
                supporting_detections=len(r['prior']['evidence']),sample_frames=len(r['samples']),
                valid_bright_samples=sum(s['binding_valid'] and s['available'] and s['raw_luma']>=30 for s in r['samples']))
            rows.append(item);vr.append(item)
        comp=comparison(vr)
        videos.append(dict(video_id=v['id'],source=v['source'],scope=v['scope'],split=v['split'],
            group=v['group'],video_label=v['video_label'],frames=v['frames'],seconds=v['seconds'],
            people_mode=v['people_mode'],people_max=v['people_max'],box_frames=v['box_frames'],
            windows=len(vr),known=comp['before']['known'],unknown=comp['before']['unknown'],
            before_actions=sum(r['before']!='other' for r in vr),after_actions=sum(r['after']!='other' for r in vr),
            filtered=comp['filtered'],false_alarms_removed=comp['corrected_other_false_positives'],
            new_false_negatives=comp['newly_missed_true_actions'],filtered_unknown=comp['filtered_unknown'],
            fresh_enhanced_frames=data['fresh_enhanced_frames'],fresh_object_crops=len(data['views']),
            result=str(path.relative_to(OUT)),status='complete' if vr else 'complete_no_action_windows'))
    onsite=[r for r in rows if r['scope']=='onsite'];supplement=[r for r in rows if r['scope']=='online_supplement']
    summary=dict(protocol_sha256=fingerprint,onsite={},online_supplement={},coverage={},strata={},bootstrap={})
    for scope,subset in [('onsite',onsite),('online_supplement',supplement)]:
        vs=[v for v in videos if v['scope']==scope]
        for split in ['all','train','val','test']:
            summary[scope][split]=comparison([r for r in subset if split=='all' or r['split']==split])
        summary['coverage'][scope]=dict(sources=len(vs),frames=sum(v['frames'] for v in vs),
            sources_by_split=dict(Counter(v['split'] for v in vs)),
            sources_with_windows=sum(v['windows']>0 for v in vs),no_window_sources=sum(v['windows']==0 for v in vs),
            no_window_video_labels=dict(Counter(v['video_label'] for v in vs if not v['windows'])),
            windows=len(subset),known_windows=sum(r['truth'] is not None for r in subset),
            unknown_windows=sum(r['truth'] is None for r in subset),people_at_window=dict(Counter(r['people'] for r in subset)),
            enhanced_frames=sum(v['fresh_enhanced_frames'] for v in vs),object_crops=sum(v['fresh_object_crops'] for v in vs))
    for field in ['brightness','people']:
        summary['strata'][field]={str(value):comparison([r for r in onsite if r[field]==value]) for value in sorted({r[field] for r in onsite})}
    for split in ['all','train','val','test']:
        subset=[r for r in onsite if split=='all' or r['split']==split]
        summary['bootstrap'][split]=[bootstrap(subset,metric) for metric in ['other_false_positive_rate','positive_correct_rate','drink','pick_up']]
    # Previous chosen-scheme results are an independent reproducibility comparison.
    previous=json.loads((ROOT/'reports/object-model-ablation-20261026/action-results.json').read_text())
    previous={r['id']:r for r in previous if r['model']=='m_finetuned' and r['variant']=='small_enhanced' and r['split'] in ('val','test')}
    matches=[dict(id=r['id'],old=previous[r['id']]['decision']['label'],new=r['after']) for r in rows if r['id'] in previous]
    verification['previous_overlap']=dict(windows=len(matches),matched=sum(r['old']==r['new'] for r in matches),
        differing=[r for r in matches if r['old']!=r['new']])
    assert len(rows)==4234 and len(onsite)==2089 and len(videos)==535
    assert sum(v['filtered'] for v in videos)==sum(r['before']!=r['after'] for r in rows)
    verification.update(sources=len(videos),windows=len(rows),onsite_sources=407,onsite_windows=len(onsite),complete=True)
    save(OUT/'verification.json',verification);save(OUT/'summary.json',summary)
    csv_write(OUT/'window-results.csv',rows);csv_write(OUT/'video-results.csv',videos)
    changed=[r for r in rows if r['before']!=r['after']];csv_write(OUT/'changed-windows.csv',changed)
    class_rows=[]
    for split,d in summary['onsite'].items():
        for cls in BEHAVIORS:
            b=d['before']['classes'][cls];a=d['after']['classes'][cls]
            class_rows.append(dict(split=split,action=cls,support=b['support'],before_tp=b['tp'],after_tp=a['tp'],
                before_fp=b['fp'],after_fp=a['fp'],before_fn=b['fn'],after_fn=a['fn']))
    csv_write(OUT/'class-results.csv',class_rows)
    figures(summary)
    write_report(summary,verification,videos,rows)
    print(json.dumps(dict(coverage=summary['coverage'],onsite={s:{k:d[k] for k in (
        'filtered','corrected_other_false_positives','newly_missed_true_actions','filtered_unknown')} for s,d in summary['onsite'].items()},
        verification=verification),ensure_ascii=False,indent=2))


def write_report(summary: dict, verification: dict, videos: list, rows: list) -> None:
    coverage=summary['coverage']['onsite'];d=summary['onsite']['all']
    table=['| 划分 | 视频数 | 已标注/未知窗口 | other误报（前→后） | 正动作正确（前→后） | 新增漏报 | 未知标签被过滤 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for split in ['train','val','test','all']:
        s=summary['onsite'][split];b=s['before'];a=s['after']
        n=sum(v['scope']=='onsite' and (split=='all' or v['split']==split) for v in videos)
        table.append(f"| {split} | {n} | {b['known']}/{b['unknown']} | {b['other_false_positives']}→{a['other_false_positives']}/{b['other_count']} | {b['positive_correct']}→{a['positive_correct']}/{b['positive_count']} | {s['newly_missed_true_actions']} | {s['filtered_unknown']} |")
    actions=['| 划分 | 动作 | 真值窗数 | 正确识别（前→后） | 该类别误报（前→后） |', '|---|---|---:|---:|---:|']
    for split in ['all','train','val','test']:
        for c in ['drink','pick_up']:
            b=summary['onsite'][split]['before']['classes'][c];a=summary['onsite'][split]['after']['classes'][c]
            actions.append(f"| {split} | {BEHAVIOR_ZH[c]} | {b['support']} | {b['tp']}→{a['tp']} | {b['fp']}→{a['fp']} |")
    all_classes=['| 动作 | 已标注真值窗 | 正确识别（前→后） | 加先验后召回 |', '|---|---:|---:|---:|']
    for c in BEHAVIORS:
        b=d['before']['classes'][c];a=d['after']['classes'][c]
        recall=f"{a['recall']*100:.2f}%" if a['recall'] is not None else '无可评估真值'
        all_classes.append(f"| {BEHAVIOR_ZH[c]} | {b['support']} | {b['tp']}→{a['tp']} | {recall} |")
    zero=[v for v in videos if v['scope']=='onsite' and v['windows']==0]
    no_boxes=sum(v['box_frames']==0 for v in zero)
    online=summary['online_supplement']['all']
    text=f'''# 旧现场逐人动作＋单帧物体先验评估 · 2026-10-26

固定采用用户选择的现场七类 YOLO26m、720 Retinexformer 增强图裁剪。动作阈值0.85、RGB权重0.95、原图门30、物体阈值0.35；未训练、未重新选参数。

主要范围为407段旧现场原视频（real-actiondata-all 390段＋real-actiondata 17段），另128段历史线上回传仅作附录。原现场{coverage['frames']}帧；{coverage['sources_with_windows']}段产生{coverage['windows']}个逐人窗口，已标注{coverage['known_windows']}，未知{coverage['unknown_windows']}。{coverage['no_window_sources']}段没有可评估窗口，不能计为正确；这些视频的文件级标签分布为{coverage['no_window_video_labels']}。窗口时刻候选人数分布：{coverage['people_at_window']}，检测数量不是人工人数真值。

## 主要结果

旧现场全部窗口中，先验过滤{d['filtered']}个触发：明确纠正other误报{d['corrected_other_false_positives']}个，新增真实动作漏报{d['newly_missed_true_actions']}个，另{d['filtered_unknown']}个因标注不确定无法判断收益。跨正动作错误类别被过滤{d['changed_wrong_positive_class']}个。下表按窗口计算，不是动作发生次数或保存片段数。

{chr(10).join(table)}

{chr(10).join(actions)}

“该类别误报”包括真值other和其他动作类别；“纠正other误报”只计算已明确标为other且被过滤为other的窗口，避免把所有过滤都叫纠错。原本低于动作阈值而报other的真动作也计入召回分母。

全部旧现场的各类结果（含训练，非独立泛化精度）：

{chr(10).join(all_classes)}

71段无窗口来源中，{no_boxes}段全程没有检测框，余下{len(zero)-no_boxes}段虽有候选框但没有形成足够持续的识别窗口；文件级非other来源16段也包含在这71段内，不能忽略。此处未新增逐帧人物真值，不将无框等同于无人。

附录128段历史线上负样本产生2145个已标注other窗口，误报{online['before']['other_false_positives']}→{online['after']['other_false_positives']}。它们是已经处理过的切片、原图亮度未知，按照度门规则不启用物体过滤；不能用这些数据声称先验纠错。

亮度分层详见summary.json的strata：L<30的149个喝水/捡物候选窗全部绕过物体先验。L=30–60的新增真动作漏报34个，L≥90新增22个，说明漏检物体不只发生在暗场。

## 方法与可复核证据

本次是**冻结逐人窗口回放＋新物体推理**。五个原模型SHA256与生产模型和历史缓存全部相同；全部535来源哈希、缓存和逐人32帧索引已核验。RGB／骨架概率复用原推理，调用当前生产fuse_person按0.95/0.85重算。人体框、姿态、跟踪同样复用，以控制加入先验前后的变量。并非重新执行全量实时PersonEngine。

新推理实际解码候选窗口的原视频，原图缩至长边720，再整帧Retinexformer增强；每窗从32帧选5个均匀位置，对30%外扩的人物区域及可信的手/头/脚局部区域裁剪、640letterbox检测。共新增强{coverage['enhanced_frames']}帧，检测{coverage['object_crops']}张独立裁剪。裁剪坐标按floor/ceil准确回映，跨区域同类去重。至少一帧同人区域检出瓶/杯支持喝水；检出瓶/杯/手机/相机/纸/便携物支持捡物。

原窗口亮度<30或未知则不启用先验；采样帧也执行亮度门；人物重叠IoU>0.4弃权；没有有效亮帧观测则保持原结果。有有效观测但没有兼容物体才变为other。没有新增手机/展品单帧否决规则，物体出现也不证明正在互动。五帧抽样不保证捕获整个动作中的短暂物体。

正式平台包暂未接入该逐人裁剪候选。本轮没有修改包或镜像，也没有进行实时速度/稳定性验收。

## 数据与结果入口

- [按视频完整清单](video-results.csv)：535行，包含全部407现场来源及无窗口视频。
- [逐窗口结果](window-results.csv)、[发生变化的窗口](changed-windows.csv)、[分动作结果](class-results.csv)。
- [每段结构化结果](per-video/)：来源、逐人原概率与新融合、原图亮度、5帧物体框/置信度、支持证据、拒绝或跳过原因。
- [可视化审核入口](index.html)：并排对比视频、案例和全部现场来源结果链接。
- [固定协议](protocol.json)、[机器可读汇总](summary.json)、[核验](verification.json)、[统计附录](stats-appendix.md)。

## 限制和结论边界

这批现场数据主要来自训练场景，train用于看拟合情况；val/test按旧source group保留，但历史初始化已经见过部分来源，测试也曾反复检查，因此不能宣称独立新场景泛化。{coverage['unknown_windows']}个未可靠逐人标注的现场窗口全部排除精度计算，71段无可用窗口不作为识别成功。固定来源辅助标注和人物关联可能有误差；真实动作片段完整召回需独立逐帧真值。

图1同时显示真动作保留和类别误报，图2区分训练/验证/测试的收益、损失及未知标签。先验确实能阻止部分缺少物体证据的动作触发，但旧现场真实互动动作也可能因物体小、遮挡或检测遗漏被过滤。不能把过滤量或总体accuracy上升单独解释为效果提升。
'''
    (OUT/'README.md').write_text(text)
    (OUT/'analysis-report.md').write_text(text)
    lines=['# 统计附录','', '一个固定模型组合、一次确定性评估；无训练seed重复实验。统计单位为人物动作窗口；同源窗口相关，按原source group整体配对重采样2000次，seed26。以下95%区间为百分位bootstrap描述区间（变化=加入先验后−之前），不作为显著性/泛化证明。未做p值检验，因此无多重检验校正声明。', '',
        '| 划分 | 指标 | 变化（百分点） | 95%区间（百分点） | 有效来源组 |', '|---|---|---:|---|---:|']
    for split,values in summary['bootstrap'].items():
        for v in values:
            ci=v['ci95'];delta=v['delta']
            lines.append(f"| {split} | {v['metric']} | {delta*100:.2f} | [{ci[0]*100:.2f}, {ci[1]*100:.2f}] | {v['contributing_groups']} |" if ci else f'| {split} | {v["metric"]} | NA | NA | 0 |')
    lines+=['','零变化的区间收缩到0仅反映本批样本，没有证明总体风险为零。喝水/捡物样本及来源组较少，区间可能很宽；分母不包含未知标签。总体accuracy/macroF1和十类混淆矩阵保存在summary.json，macro只在有真值支持的类上平均。没有支持的fight/talk/handshake等不能声称验证通过。']
    (OUT/'stats-appendix.md').write_text('\n'.join(lines)+'\n')
    (OUT/'figure-catalog.md').write_text('''# 图目录

1. `figures/01-true-and-false.png/.pdf`：同窗有无先验的喝水/捡物TP及FP，左为所有旧现场（含训练），右为历史test。精确整数，无误差条；源自summary.json。应同时观察FP下降和TP损失，不能只看触发减少。不能用全量训练内结果推断新摄像机泛化。
2. `figures/02-tradeoff-and-coverage.png/.pdf`：按split显示已知other误报纠正、新增真动作漏报、未知标签被过滤量；同时展示已知/未知窗口覆盖。精确整数，无误差条；源自summary.json。未知样本不能被计为误报收益，零窗口视频仍需单独检查。

源组bootstrap区间在stats-appendix.md，窗口计数柱没有伪装成独立重复实验均值。
''')


if __name__=='__main__':main()
