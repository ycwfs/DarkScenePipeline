"""Report threshold selection, observed benefit/cost, and incomplete feedback labels."""
from collections import Counter
import html
import json
from pathlib import Path

import numpy as np
from tune import Evaluation,csv_save
from evidence import OUT,ROOT,save,BEHAVIORS
from darkpipe.constants import BEHAVIOR_ZH


def bootstrap(ev: Evaluation,b: np.ndarray,a: np.ndarray,mask: np.ndarray,positive: bool) -> dict:
    groups=sorted({r['group'] for i,r in enumerate(ev.rows) if mask[i]})
    values=[]
    for group in groups:
        chosen=mask & np.array([r['group']==group for r in ev.rows])
        if positive:
            count=int(chosen.sum());delta=int((a[chosen]==ev.truth[chosen]).sum()-(b[chosen]==ev.truth[chosen]).sum())
        elif any(ev.new[chosen]):
            count=int(chosen.sum());delta=int((a[chosen]==ev.drink).sum()-(b[chosen]==ev.drink).sum())
        else:
            count=int(chosen.sum());delta=int((a[chosen]!=ev.o).sum()-(b[chosen]!=ev.o).sum())
        values.append([count,delta])
    arr=np.array(values,float);rng=np.random.default_rng(26)
    sums=arr[rng.integers(len(arr),size=(2000,len(arr)))].sum(1)
    rates=sums[sums[:,0]>0,1]/sums[sums[:,0]>0,0]
    return dict(groups=len(groups),windows=int(arr[:,0].sum()),delta=float(arr[:,1].sum()/arr[:,0].sum()),
        ci95=np.quantile(rates,[.025,.975]).tolist(),resamples=2000,seed=26)


def charts(ev: Evaluation,summary: dict,grid: list,chosen: dict) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=OUT/'figures';directory.mkdir(exist_ok=True)
    baseline=summary['baseline'];best=summary['recommended']
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    axes=axes.ravel()
    for ax,split in zip(axes[:2],['val','test']):
        d=baseline['onsite'][split];s=best['onsite'][split]
        bv=[d['classes'][c]['tp'] for c in ['drink','pick_up','wave','chase']]+[d['other_fp']]
        av=[s['classes'][c]['tp'] for c in ['drink','pick_up','wave','chase']]+[s['other_fp']]
        for dx,values,color,label in [(-.18,bv,'#44799e','Current .85/.35'),(.18,av,'#d99b42','Tuned thresholds')]:
            bars=ax.bar(np.arange(5)+dx,values,.36,color=color,label=label);ax.bar_label(bars,padding=3)
        ax.set_xticks(range(5),['Drink TP','Pickup TP','Wave TP','Run TP','Other FP']);ax.set_title(f'Old onsite {split}')
        ax.set_ylim(0,max(bv+av)*1.3+1);ax.legend(fontsize=7)
    d=baseline['new'];s=best['new'];bv=[d['known_drink_fp'],d['known_drink_tp']];av=[s['known_drink_fp'],s['known_drink_tp']]
    for dx,values,color in [(-.18,bv,'#44799e'),(.18,av,'#d99b42')]:
        bars=axes[2].bar(np.arange(2)+dx,values,.36,color=color);axes[2].bar_label(bars,padding=3)
    axes[2].set_xticks(range(2),['Known drink FP','Known drink TP'])
    axes[2].set_title('New9 feedback (used for tuning)');axes[2].set_ylim(0,max(bv+av)*1.3+1)
    bv=[d['actions'].get(c,0) for c in ['chase','wave','pick_up']];av=[s['actions'].get(c,0) for c in ['chase','wave','pick_up']]
    for dx,values,color in [(-.18,bv,'#44799e'),(.18,av,'#d99b42')]:
        bars=axes[3].bar(np.arange(3)+dx,values,.36,color=color);axes[3].bar_label(bars,padding=3)
    axes[3].set_xticks(range(3),['Run triggers','Wave triggers','Pickup triggers'])
    axes[3].set_title('New9 unreviewed actions: counts are NOT FP metrics');axes[3].set_ylim(0,max(bv+av)*1.3+1)
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.set_ylabel('Person windows')
    fig.savefig(directory/'01-baseline-vs-tuned.png',dpi=170);fig.savefig(directory/'01-baseline-vs-tuned.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    tau=ev.protocol['grid']['action'];obj=ev.protocol['grid']['object']
    for ax,field,title in zip(axes,['new_known_drink_fp','old_val_positive_correct'],
        ['Known new drink false positives','Old validation positive actions retained']):
        data=np.array([r[field] for r in grid]).reshape(len(tau),len(obj))
        im=ax.imshow(data,origin='lower',aspect='auto',extent=[.005,.905,.695,.995],cmap='viridis')
        ax.plot(.35,.85,'wx',ms=9,label='Current');ax.plot(chosen['object_min_conf'],chosen['action_threshold'],'r*',ms=12,label='Selected')
        ax.set_xlabel('Object confidence threshold');ax.set_ylabel('Action confidence threshold');ax.set_title(title)
        fig.colorbar(im,ax=ax,label='Windows');ax.legend(fontsize=8)
    fig.savefig(directory/'02-threshold-grid.png',dpi=170);fig.savefig(directory/'02-threshold-grid.pdf');plt.close(fig)


def main() -> None:
    import csv
    ev=Evaluation();original=json.loads((OUT/'summary.json').read_text());initial_lock=json.loads((OUT/'selection-lock.json').read_text())
    guarded=json.loads((OUT/'class-guard-summary.json').read_text());guard_lock=json.loads((OUT/'class-guard-selection-lock.json').read_text())
    summary=dict(original,initial_aggregate=original['recommended'],**{k:v for k,v in guarded.items() if k not in original})
    summary.update(recommended=guarded['recommended'],coarse_recommended=guarded['coarse_recommended'])
    lock=dict(initial_lock,**{k:guard_lock[k] for k in ['recommended','coarse_best','coarse_combinations','fine_additional']},
        feasible_combinations=guard_lock['feasible'])
    save(OUT/'final-summary.json',summary)
    chosen=lock['recommended'];baseline=summary['baseline'];best=summary['recommended']
    raw=list(csv.DictReader((OUT/'class-guard-grid.csv').open(encoding='utf-8-sig')))
    grid=[{k:(v=='True' if k=='feasible' else float(v)) for k,v in r.items()} for r in raw]
    charts(ev,summary,grid,chosen)
    b=ev.predict(.85,.35);a=ev.predict(chosen['action_threshold'],chosen['object_min_conf'])
    intervals={}
    for split in ['val','test','all']:
        mask=(ev.scope=='onsite') & ((ev.split==split) if split!='all' else True)
        intervals[split+'_other_fp_rate']=bootstrap(ev,b,a,mask & (ev.truth==ev.o),False)
        intervals[split+'_positive_correct_rate']=bootstrap(ev,b,a,mask & (ev.truth>=0) & (ev.truth!=ev.o),True)
    intervals['new_known_drink_fp_rate']=bootstrap(ev,b,a,ev.new_false,False);save(OUT/'uncertainty.json',intervals)
    tradeoff={};changed=[]
    for name,mask in [('old_all',ev.scope=='onsite'),('old_val',ev.oldval),
        ('old_test',(ev.scope=='onsite') & (ev.split=='test'))]:
        known=mask & (ev.truth>=0);positive=known & (ev.truth!=ev.o);other=known & (ev.truth==ev.o)
        tradeoff[name]=dict(other_errors_removed=int((other & (b!=ev.o) & (a==ev.o)).sum()),
            other_errors_added=int((other & (b==ev.o) & (a!=ev.o)).sum()),
            true_actions_recovered=int((positive & (b!=ev.truth) & (a==ev.truth)).sum()),
            new_true_action_losses=int((positive & (b==ev.truth) & (a!=ev.truth)).sum()),
            unknown_changes=int((mask & (ev.truth<0) & (a!=b)).sum()))
    for i,r in enumerate(ev.rows):
        if a[i]==b[i]:continue
        changed.append({k:r[k] for k in ['id','source_id','scope','split','timestamp','track_id','truth','is_drink','top_label','top_score']} |
            dict(before=BEHAVIORS[b[i]],after=BEHAVIORS[a[i]],object_score=None if np.isinf(ev.support[i]) else float(ev.support[i])))
    if changed:csv_save(OUT/'changed-windows.csv',changed)
    save(OUT/'tradeoff.json',tradeoff)
    # Exact per-source counts preserve no-window sources and separate unknown predictions.
    video_rows=[]
    for sid,s in ev.sources.items():
        mask=np.array([r['source_id']==sid for r in ev.rows]);known=mask & (ev.truth>=0)
        video_rows.append(dict(source_id=sid,source=s['source'],scope=s['scope'],split=s['split'],windows=int(mask.sum()),
            known_multiclass_windows=int(known.sum()),before_triggers=int((b[mask]!=ev.o).sum()),after_triggers=int((a[mask]!=ev.o).sum()),
            before_positive_correct=int(((b==ev.truth) & known & (ev.truth!=ev.o)).sum()),
            after_positive_correct=int(((a==ev.truth) & known & (ev.truth!=ev.o)).sum()),
            before_other_fp=int((known & (ev.truth==ev.o) & (b!=ev.o)).sum()),after_other_fp=int((known & (ev.truth==ev.o) & (a!=ev.o)).sum()),
            before_confirmed_drink_fp=int((mask & ev.new_false & (b==ev.drink)).sum()),after_confirmed_drink_fp=int((mask & ev.new_false & (a==ev.drink)).sum())))
    csv_save(OUT/'video-results.csv',video_rows)
    final_rows=[]
    for i,r in enumerate(ev.rows):
        final_rows.append({k:r[k] for k in ['id','source_id','scope','split','group','track_id','history_generation','timestamp','truth','is_drink','top_label','top_score','raw_luma']} |
            dict(object_max_support=None if np.isinf(ev.support[i]) else float(ev.support[i]),
                baseline=BEHAVIORS[b[i]],recommended=BEHAVIORS[a[i]],changed=bool(b[i]!=a[i])))
    csv_save(OUT/'final-window-results.csv',final_rows)
    old_prediction=OUT/'selected-predictions.json'
    initial_prediction=OUT/'initial-selected-predictions.json'
    if old_prediction.exists() and not initial_prediction.exists():old_prediction.rename(initial_prediction)
    selected={k:v['setting'] for k,v in summary.items()}
    save(OUT/'selected-predictions.json',dict(settings=selected,labels=BEHAVIORS,ids=[r['id'] for r in ev.rows],
        predictions={name:ev.predict(s['action_threshold'],s['object_min_conf']).tolist() for name,s in selected.items()}))
    # Pareto frontier over calibration loss and positive retention, no old-test peeking.
    feasible=[r for r in grid if r['feasible']];pareto=[]
    for retained in sorted({r['old_val_positive_correct'] for r in feasible},reverse=True):
        candidate=min((r for r in feasible if r['old_val_positive_correct']==retained),key=ev.key)
        if not pareto or candidate['objective']<min(r['objective'] for r in pareto):pareto.append(candidate)
    csv_save(OUT/'pareto-options.csv',pareto)
    render_report(ev,summary,lock,tradeoff,intervals,video_rows)
    print(json.dumps(dict(recommended=chosen,tradeoff=tradeoff,new_before=baseline['new'],new_after=best['new']),ensure_ascii=False,indent=2))


def render_report(ev: Evaluation,summary: dict,lock: dict,tradeoff: dict,intervals: dict,video_rows: list) -> None:
    base=summary['baseline'];best=summary['recommended'];setting=best['setting'];recall=summary['recall_option']
    table=['| 数据 | 已标注窗 | other误报 当前→调优 | 真动作正确 当前→调优 |', '|---|---:|---:|---:|']
    for split in ['train','val','test','all']:
        b=base['onsite'][split];a=best['onsite'][split]
        table.append(f"| 旧现场 {split} | {b['known']} | {b['other_fp']}→{a['other_fp']} / {b['other_count']} | {b['positive_correct']}→{a['positive_correct']} / {b['positive_count']} |")
    classes=['| 数据 | 动作 | 真值数 | 正确 当前→调优 | 类别误报 当前→调优 |', '|---|---|---:|---:|---:|']
    for split in ['all','val','test']:
        for c in ['drink','pick_up','wave','chase','fall','throw']:
            b=base['onsite'][split]['classes'][c];a=best['onsite'][split]['classes'][c]
            classes.append(f"| 旧 {split} | {BEHAVIOR_ZH[c]} | {b['support']} | {b['tp']}→{a['tp']} | {b['fp']}→{a['fp']} |")
    nb=base['new'];na=best['new'];total=tradeoff['old_all']
    true_support=lock['confirmed_true_object_score'];selection=lock['recommended']
    simple=summary['coarse_recommended'];action_only=summary['action_only'];object_only=summary['object_only']
    option_table=['| 方案 | 动作阈值 | 物体阈值 | 旧val各类保护 | 旧全量真动作正确/555 | 旧全量other误报/1009 | 新已知喝水误报/526 | 新真喝水保留/1 |',
        '|---|---:|---:|---|---:|---:|---:|---:|']
    for name,title in [('baseline','当前'),('recommended','误报优先推荐'),('coarse_recommended','两位小数对照'),
        ('action_only','只调动作阈值'),('object_only','只调物体阈值'),('initial_aggregate','初轮强压误报，放弃'),('recall_option','保留唯一真喝水，非主推荐')]:
        d=summary[name];s=d['setting']
        keep=all(d['onsite']['val']['classes'][c]['tp']>=base['onsite']['val']['classes'][c]['tp'] for c in BEHAVIORS if c!='other')
        option_table.append(f"| {title} | {s['action_threshold']:.3f} | {s['object_min_conf']:.3f} | {'满足' if keep else '不满足'} | {d['onsite']['all']['positive_correct']} | {d['onsite']['all']['other_fp']} | {d['new']['known_drink_fp']} | {d['new']['known_drink_tp']} |")
    new_actions=['| 新现场动作 | 当前触发窗 | 调优后触发窗 | 真值状态 |', '|---|---:|---:|---|']
    for c in BEHAVIORS:
        if c=='other' or not (nb['actions'].get(c,0) or na['actions'].get(c,0)):continue
        new_actions.append(f"| {BEHAVIOR_ZH[c]} | {nb['actions'].get(c,0)} | {na['actions'].get(c,0)} | {'原527喝水触发有二元审核' if c=='drink' else '未逐窗审核，不能把减少量称为误报收益'} |")
    text=f'''# 保持算法不变的阈值调优 · 2026-10-26

推荐离线候选参数：**动作阈值{setting['action_threshold']:.3f}、物体阈值{setting['object_min_conf']:.3f}**。RGB权重0.95、照度门30、soft、五帧抽样、同人ROI、现场七类YOLO26m、720 Retinexformer增强图裁剪全部固定。没有训练或换模型，没有改推理分支/先验判断规则。

原参数为动作0.85、物体0.35。粗扫动作0.70–0.99、物体0.01–0.90，步长0.01，共{lock['coarse_combinations']}组；再在候选附近±0.01按0.001细扫，新增{lock['fine_additional']}组。粗扫推荐{lock['coarse_best']['action_threshold']:.2f}/{lock['coarse_best']['object_min_conf']:.2f}；细扫只提高现有数据上的分辨率，不代表阈值精确到千分位就能泛化。初轮只约束正动作总量，发现类别间损失被抵消，随后加入逐类别保护；两个实验锁均保留，详见后文。

## 当前方案与调优后比较

{chr(10).join(table)}

旧现场全量相对当前先验方案：纠正other误报{total['other_errors_removed']}个、新增other误报{total['other_errors_added']}个；恢复真实动作{total['true_actions_recovered']}个、新增真实动作漏报{total['new_true_action_losses']}个。未知标签变化{total['unknown_changes']}个，不计收益。这里全量包含训练，不能用作独立泛化结论。

{chr(10).join(classes)}

## 新多人现场9段

- 已确认的526个非喝水触发：残留喝水误报 **{nb['known_drink_fp']}→{na['known_drink_fp']}**；相对最初不加先验的526误报，累计过滤{526-na['known_drink_fp']}个。
- 已确认的1个真喝水：保留 **{nb['known_drink_tp']}→{na['known_drink_tp']} / 1**。
- 新出现但尚未审核的喝水触发：**{nb['unknown_drink']}→{na['unknown_drink']}**。
- 捡物触发：**{nb['pickup_triggers']}→{na['pickup_triggers']}**。用户说“基本全为误报”，但没有逐窗真值；这里称触发量，不能称准确误报数。

{chr(10).join(new_actions)}

**统一动作阈值也影响非物品动作**：新现场跑动64→23、挥手115→42，这些窗口未逐一审核，减少量不能计为误报收益，可能包含真实动作。旧test已确认跑动19→17/25。若要保持这两类当前输出，使用下面“只调物体阈值”的0.85/0.37对照；它对非物品动作的判断逐窗保持不变，但新喝水误报仅49→42。这是全局阈值的取舍，不额外引入按类阈值或改变算法。

唯一确认真喝水的动作融合分数为{lock['confirmed_true_score']:.6f}，五帧物体最大兼容分数仅 **{true_support:.6f}**。保持单帧物体先验规则时，只有物体阈值不高于该分数才可能放行；因此只调动作阈值无法救回这个例子。较低物体阈值会同时放行低分错误物体，不能为了单个正例忽略整体误报。该分数来自相同裁剪重新执行检测，保留到0.001，未把原缓存未保存的低分当作零。

## 选参协议与替代选项

选参函数只使用旧现场val49段和新现场527个已审核喝水触发。旧train用于报告拟合；每轮选参锁写出后才计算旧test49段成绩。新9段根据本轮请求作为反馈调参数据，从此不是本轮独立测试；未用于模型训练。

**过程变更如实记录**：初轮0.934/0.075在旧val保住正动作总数，却把跑动TP从8降到5，用喝水提升抵消，旧test也随后观察到退化。该结果保存在selection-lock.json，不能作为主推荐。第二轮在protocol-amendment.json提前写明逐类保护后再选参：挥手TP≥6/9、跑动≥8/13、喝水≥8/10、捡物≥1/4。已看过首轮历史test，第二轮test只是诊断，不能声称未接触的独立验收；不隐藏这次协议修订。

其余可行条件：旧val other误报≤6/162、正动作正确≥23/36；新已知喝水FP≤49/526，不新增未经审核的喝水触发；旧线上val误报不增加；动作阈值不高于唯一真喝水分数。注意初轮protocol.json文字误写为35/163，程序一直使用实际36/162，修订协议已记录实际分母。最小化“旧val other误报率”和“新已知喝水残留比例”的等权平均；平局优先旧val宏F1、正动作数、真喝水保留，再优先接近现参数。不是通过压到全other来获得零误报。

逐类保护后可行组合共{lock['feasible_combinations']}个，扫描范围内没有同时满足所有保护且救回唯一新真喝水的参数。主推荐遵循用户“优先减少误报”的目标。

{chr(10).join(option_table)}

保真喝水诊断候选0.92/0.01能放行唯一正例，但旧val跑动8→6，违反逐类保护，因此不作为主推荐。即使主推荐通过旧val逐类保护，旧test仍有少量退化，说明阈值调优不能保证两域无代价同时提升。

## 数据、推理与完整性

全部544来源：407旧现场＋128历史线上负样本＋9新多人录像；共51834个固定人物窗口。所有来源哈希、五模型和物体模型哈希核验。旧现场2089窗中1564有标签、525未知，71来源没有动作窗，全部保留清单。新47600窗仅527有二元喝水标签，不能把not_drink自动当作other或其他动作真值。

阈值下限0.70覆盖1322个可能的喝水/捡物窗口。原始动作概率保持不动；物体检测重新保留到0.001以支持0.01起扫，已存在的新现场候选复用冻结裁剪像素，新增候选按相同最近事件≤0.55秒绑定与RTMO规则提取。新场景没有完整逐帧跟踪缓存，该绑定为离线近似；不可冒充线上实时先验验收。

0.85/0.35基线逐窗复现旧4234窗和新564个原候选；向量化阈值计算与生产reject_probs、filter_object_evidence逐项核对。低亮/未知亮度跳过、对象不兼容/无可用帧的规则均保持。

## 文件入口

- [推荐参数](recommended-settings.json)、[最终选参锁定](class-guard-selection-lock.json)、[完整协议](protocol.json)、[逐类保护修订](protocol-amendment.json)。
- [逐视频结果](video-results.csv)、[最终逐窗口结果](final-window-results.csv)、[变化窗口](changed-windows.csv)。
- [最终粗扫](class-guard-grid.csv)、[最终局部细扫](class-guard-fine-grid.csv)、[Pareto候选](pareto-options.csv)。
- [图表页面](index.html)、[最终完整分组统计](final-summary.json)、[置信区间](stats-appendix.md)、[核验](verification.json)。

初轮coarse-grid.csv、fine-grid.csv、summary.json、window-results.csv和initial-selected-predictions.json保留诊断过程，不是最终推荐。

当前保存的是可复核的离线调参结果和参数预设。正式逐人引擎仍未接入该物体候选，本轮不改算法、不打包、不改镜像；不能声称新参数已在平台自动生效。
'''
    (OUT/'README.md').write_text(text);(OUT/'analysis-report.md').write_text(text)
    stats=['# 统计附录','','单个固定模型组合的一次回放，无训练seed重复；不做显著性p值检验。对推荐参数相对当前参数的差值按原source group整体配对重采样2000次，seed26，报告百分位95%区间。旧val和新9均参与调参，区间是选择后的描述区间，未校正择优偏差，不是独立置信保证。逐类保护是在首轮test诊断后修订，后续test也不能当作未接触验收。', '',
        '| 指标 | 来源组 | 窗口数 | 后−前（百分点） | 95%区间（百分点） |','|---|---:|---:|---:|---:|']
    for name,d in intervals.items():stats.append(f"| {name} | {d['groups']} | {d['windows']} | {d['delta']*100:.2f} | [{d['ci95'][0]*100:.2f}, {d['ci95'][1]*100:.2f}] |")
    stats+=['','新真喝水只有一个已确认窗，不给出貌似稳健的总体召回估计。旧未知525窗不计精度，新未审窗口及捡物触发不计误报率。窗有时间相关性，按来源组重采样避免独立窗假设；同相机跨视频仍相关。没有模型/seed重复，不报告伪造的均值±标准差。']
    (OUT/'stats-appendix.md').write_text('\n'.join(stats)+'\n')
    (OUT/'figure-catalog.md').write_text('''# 图目录

1. `figures/01-baseline-vs-tuned.png/pdf`：旧val/test的喝水/捡物/挥手/跑动TP和other误报，以及新反馈的喝水FP/TP、未审跑动/挥手/捡物触发量。精确窗口数，无误差条。目的：同时展示减误报与漏报代价；非物品动作触发减少不能叫纠错；新反馈已经用于调优，不能当作独立验收。
2. `figures/02-threshold-grid.png/pdf`：动作阈值×物体阈值下的新已知喝水误报、旧val正动作保留。格点来自coarse-grid.csv，非插值预测；标注当前和推荐参数。目的：展示调参的约束与折中，不能用全other角落称最优，也不说明千分位阈值跨现场稳定。

每张图都基于同一固定模型/数据窗口；辅助描述区间见stats-appendix.md。
''')
    table_html=''.join(f'<tr><td>{html.escape(r["source"])}</td><td>{r["split"]}</td><td>{r["windows"]}</td><td>{r["before_triggers"]}→{r["after_triggers"]}</td><td>{r["before_other_fp"]}→{r["after_other_fp"]}</td><td>{r["before_confirmed_drink_fp"]}→{r["after_confirmed_drink_fp"]}</td></tr>' for r in video_rows)
    page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>阈值调优对照</title>
<style>body{{max-width:1500px;margin:32px auto;padding:0 20px;font-family:system-ui,sans-serif;color:#263846;background:#f6f8fa}}p{{line-height:1.7}}img{{max-width:100%}}table{{border-collapse:collapse;font-size:13px}}td,th{{border-bottom:1px solid #d5dde4;padding:9px;text-align:left}}a{{color:#246a9b}}input{{padding:10px;font:inherit;width:90%}}.note{{padding:18px;background:#fff1d8}}</style>
<h1>固定算法 · 旧现场＋新多人现场阈值调优</h1><p>推荐动作阈值 <b>{setting['action_threshold']:.3f}</b>，物体阈值 <b>{setting['object_min_conf']:.3f}</b>；融合权重0.95、照度门30及模型/流程固定。</p>
<p class="note">新9段已作为反馈调参，非独立测试。只有527个原喝水触发有二元真值；新增触发和捡物不自动记为误报。正式平台参数未改。</p>
<p><a href="README.md">详细报告</a> · <a href="recommended-settings.json">推荐参数</a> · <a href="video-results.csv">逐视频CSV</a> · <a href="final-window-results.csv">最终逐窗CSV</a> · <a href="class-guard-grid.csv">完整扫描</a></p>
<img src="figures/01-baseline-vs-tuned.png" alt="当前参数和调优参数对照"><img src="figures/02-threshold-grid.png" alt="阈值扫描热图">
<h2>544来源完整清单</h2><input id="filter" placeholder="输入视频名或划分筛选"><table><thead><tr><th>来源</th><th>划分</th><th>窗口数</th><th>所有触发 前→后</th><th>旧已知other误报</th><th>新已知喝水误报</th></tr></thead><tbody>{table_html}</tbody></table>
<script>document.getElementById('filter').addEventListener('input',e=>{{const q=e.target.value.toLowerCase();document.querySelectorAll('tbody tr').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q));}});</script></html>'''
    (OUT/'index.html').write_text(page)


if __name__=='__main__':main()
