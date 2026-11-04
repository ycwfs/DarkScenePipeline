"""Descriptive paired counts; no significance claim from overlapping windows."""
from collections import defaultdict,Counter
import csv,os
os.environ.setdefault('MPLCONFIGDIR','/tmp/darkpipe-lowconf-matplotlib')
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CN={'continuation':'已确认动作续接','running2':'跑动双窗口复核','combined':'两项联合','running1':'跑动单窗口（诊断）','running3':'跑动三窗口（诊断）'}

def main():
    result=read(OUT/'results.json');rows=read(OUT/'replay-rows.json');out=read(OUT/'decisions.json');groups=read(OUT/'event-groups.json')
    verification=read(OUT/'verification.json');role=result['roles']['all'];bins=role['bins'];tables=[];numeric=[]
    tables.append('|候选置信度|规则|补回真动作窗|新增误报窗（other／错类）|恢复但未标注窗|首次确认等待|')
    tables.append('|---|---|---:|---:|---:|---|')
    for b in bins:
        for m in ('continuation','running2','combined'):
            s=b['methods'][m];d=s['confirmation_delay_s'];delay='0秒（无需等待未来窗口）'if m=='continuation'else('—（无新确认）'if d is None else f"{d['median']:.2f}秒中位，{d['min']:.2f}–{d['max']:.2f}秒")
            tables.append(f"|{b['range'][0]}–{b['range'][1]}|{CN[m]}|{s['true_recovered']}|{s['added_false_positives']}（{s['other_false_positives']}／{s['wrong_action']}）|{s['unlabelled']}|{delay}|")
            numeric.append(dict(low=b['range'][0],high=b['range'][1],method=m,**{k:s[k]for k in ('true_recovered','other_false_positives','wrong_action','added_false_positives','unlabelled','first_confirmations')},delay_median_s=None if d is None else d['median']))
    with(OUT/'summary.csv').open('w')as f:
        w=csv.DictWriter(f,fieldnames=list(numeric[0]));w.writeheader();w.writerows(numeric)
    separated=['|数据范围|有标签窗（正类窗）|规则|补回真窗|新增误报窗|原本漏掉、现补回的动作组|新增other报警组|','|---|---:|---|---:|---:|---:|---:|']
    for key,name in [('old_train','旧训练回归'),('old_val','旧验证回归'),('old_test','旧测试回归'),('new_feedback','新多人反馈')]:
        rr=result['roles'][key]
        for m in ('continuation','running2','combined'):
            s=rr['totals'][m]
            subset=[g for g in groups if g['method']==m and((g['scope']=='new_multipeople')if key=='new_feedback'else(g['scope']!='new_multipeople'and next(r['split']for r in rows if r['source_id']==g['source_id'])==key.split('_')[1]))]
            recovered=sum(g['recovered_previously_missed_group']for g in subset);fp=sum(g['new_false_other_group']for g in subset)
            separated.append(f"|{name}|{rr['known']}（{rr['true_actions']}）|{CN[m]}|{s['true_recovered']}|{s['added_false_positives']}|{recovered}|{fp}|")
    # Source-cluster bootstrap for old test only; fixed model, descriptive coverage.
    bysource=defaultdict(list)
    for r in rows:
        if r['scope']!='new_multipeople'and r['split']=='test'and r['truth']is not None:bysource[r['group']].append(r)
    stats={}
    for m in ('continuation','running2','combined'):
        counts=[]
        for rr in bysource.values():
            pos=[r for r in rr if r['truth']!='other'];neg=[r for r in rr if r['truth']=='other']
            counts.append([len(pos),sum(out[r['id']][m]['recovered']and r['raw_label']==r['truth']for r in pos),len(neg),sum(out[r['id']][m]['recovered']for r in neg)])
        a=np.array(counts);rng=np.random.default_rng(1101);boot=[]
        for _ in range(5000):
            s=a[rng.integers(len(a),size=len(a))].sum(0)
            if s[0]and s[2]:boot.append([s[1]/s[0],s[3]/s[2]])
        stats[m]=dict(positive_recall_gain_95ci=np.percentile(boot,[2.5,97.5],axis=0)[:,0].tolist(),
            other_fpr_gain_95ci=np.percentile(boot,[2.5,97.5],axis=0)[:,1].tolist())
    save(OUT/'statistics.json',dict(old_test_source_groups=len(bysource),bootstrap_repetitions=5000,seed=1101,results=stats,pvalues=None,
        inference_limit='Previously inspected fixed predictions, overlapping windows, only three new capture groups; no untouched-test or significance claim. Zero bootstrap bounds do not establish zero risk.'))
    f=OUT/'figures';f.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(11.6,4.3))
    for ax,m,title in zip(axes,['continuation','running2'],['Confirmed-action continuation','Running review: two windows']):
        x=np.arange(3)
        for offset,key,color,label in [(-.25,'true_recovered','#0072B2','Correct recovery'),(0,'added_false_positives','#D55E00','Added false action'),(.25,'unlabelled','#999999','Unlabelled recovery')]:
            bars=ax.bar(x+offset,[b['methods'][m][key]for b in bins],.25,color=color,label=label);ax.bar_label(bars,padding=2,fontsize=8)
        ax.set_xticks(x,['0.65–0.75','0.75–0.85','0.85–0.925']);ax.set_title(title);ax.set_ylabel('Windows');ax.margins(y=.23);ax.spines[['top','right']].set_visible(False)
    axes[0].legend(fontsize=8);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(f/f'01-recovery-false-alarms.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11.6,4.3));x=np.arange(3);methods=['running1','running2','running3']
    for off,key,col,label in [(-.16,'true_recovered','#0072B2','Correct recovery'),(.16,'added_false_positives','#D55E00','Added false action')]:
        bars=axes[0].bar(x+off,[role['totals'][m][key]for m in methods],.32,color=col,label=label);axes[0].bar_label(bars,padding=2)
    axes[0].set_xticks(x,['1 window','2 windows','3 windows']);axes[0].set_ylabel('Labelled windows');axes[0].set_title('Running continuity trade-off');axes[0].legend(fontsize=8);axes[0].margins(y=.2)
    for j,m in enumerate(methods):
        ds=[e['delay']for e in role['totals'][m]['confirmed_episodes']]
        axes[1].scatter(np.full(len(ds),j),ds,s=45,color='#0072B2',alpha=.55)
        if ds:axes[1].plot([j-.15,j+.15],[np.median(ds)]*2,color='#222222',lw=2)
        else:axes[1].text(j,.03,'No confirmation',ha='center',fontsize=8)
    axes[1].set_xticks(x,['1 window','2 windows','3 windows']);axes[1].set_ylabel('Source-time wait (seconds)');axes[1].set_title('First candidate → first confirmation');axes[1].set_ylim(bottom=-.04)
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(f/f'02-running-wait-tradeoff.{ext}',dpi=180)
    plt.close(fig)
    # Summary per raw source for diagnosing concentrated errors.
    per_source=[]
    for sid in sorted({r['source_id']for r in rows}):
        rr=[r for r in rows if r['source_id']==sid]
        for m in ('continuation','running2','combined'):
            adds=[r for r in rr if out[r['id']][m]['recovered']]
            if adds:per_source.append(dict(source_id=sid,method=m,true_recovered=sum(r['truth']==r['raw_label']for r in adds),
                false_added=sum(r['truth']is not None and r['truth']!=r['raw_label']for r in adds),unknown=sum(r['truth']is None for r in adds)))
    save(OUT/'source-effects.json',per_source)
    diagnostics=['|跑动复核|真实补回窗|新增误报窗|未标注恢复窗|首次确认等待中位|','|---|---:|---:|---:|---:|']
    for m in methods:
        s=role['totals'][m];d=s['confirmation_delay_s'];delay='—'if d is None else f"{d['median']:.2f}秒"
        diagnostics.append(f"|{CN[m]}|{s['true_recovered']}|{s['added_false_positives']}|{s['unlabelled']}|{delay}|")
    c=role['totals']['continuation'];r=role['totals']['running2'];joint=role['totals']['combined']
    decision=f"续接合计补回{c['true_recovered']}个真实窗口、新增{c['added_false_positives']}个已知误报窗口；双窗口跑动补回{r['true_recovered']}个真实窗口、新增{r['added_false_positives']}个已知误报窗口。两项联合为{joint['true_recovered']}个真补回、{joint['added_false_positives']}个新增误报，存在重合，不能把两项直接相加。"
    (OUT/'analysis-report.md').write_text(f'''# 低分候选补报实验

{decision}

本轮只验证，不改正式模型、阈值、平台/Box代码、交付包或镜像。当前阈值仍为0.925。正式模型未使用10/30的85正例候选微调权重。本实验在原正式概率上因果回放，补算缺失的精确同人物体/骨架证据。

## 三档结果

下表单位是**人物动作窗口**，相邻窗口会重叠，不等于独立动作次数。误报包含other报动作及真动作报错类别；未标注恢复不计TP/FP，也不能视作正确。区间左闭右开，但0.65本身排除；0.925及以上沿用原规则。

{chr(10).join(tables)}

首次确认等待按本次发生确认的窗口置信度归档；完整候选链可跨档，未要求全链都在同一置信区间。跑动第一次候选不会回填成已识别结果，达到条件的当前窗口才输出。后续连续合格窗口不再额外等待。续接等待0仅指不等未来帧；不代表CPU/GPU处理和端到端延迟为0。

![分档补回与误报](figures/01-recovery-false-alarms.png)

## 数据范围与事件组

全流51,834窗口（旧4,234＋新九段47,600），有标签4,445窗：旧3,709＋新人工736（85正例、651other）。其余47,389窗保留用于时序回放/未知结果统计，未把预测other当真值。此前排除的实体、冲突及不确定窗口仍无评分标签。新九段均为历史反馈录像，旧train/val/test也已多轮查看；不是全新盲测。

{chr(10).join(separated)}

新多人事件组沿用上传审核ID；旧组按同源、人物/代次、同真值及相邻窗口≤0.65秒形成连续组。已被基线识别过的组新增正确窗口，只算补全时段，不算救回整段漏检。group any-trigger不等于完整保存时长，也不测片段边界。

## 固定规则与验证

正式alpha0.972、照度门35、动作阈值0.925、物体阈值0.581、挥手3帧、跑动速度0.35人体框高/秒保持。续接要求同人同代次、同最高分动作、最近正式高分锚点≤1.2秒、相邻输出≤0.65秒；other/换类/先验否决/断档清空，低分续接不能刷新高分锚点。当前候选重新满足原先验；在低亮、证据缺失或全歧义时不凭“跳过先验”主动补报。无既定先验的类别只用续接条件。

跑动主实验要求0.65<置信度<0.925、亮度≥35、可靠同人速度≥0.35，两个连续窗口且尾帧跨度≥0.5秒。失败/换类/高分原确认/断档重置。单窗口及三窗口≥1秒仅预声明诊断，不事后改选测试最优设置。

{chr(10).join(diagnostics)}

![跑动连续性与等待](figures/02-running-wait-tradeoff.png)

旧窗口使用当前推理对应的检测框和骨架，未使用外扩RGB裁剪框作速度分母。新窗口优先复用严格同源、人物、代次和帧号匹配的原像素特征；缺少的430窗从原片开头完整重放前端、校验人物/代次/尾框/32采样帧。物体按当前720增强裁剪流程，复用125份精确特征证据并补算101份；没有最近时间窗口替代。

旧4234个基线标签与历史正式结果差异0。另对736个人工多人窗口的10/30基线作交叉核对，见verification.json；若浮点/证据差异存在会逐条记录。{verification['checks']['causal_prefixes']}个前缀因果检查通过，11项状态机边界测试通过。未用未来帧、未跨人物/代次续接。

## 结论的边界

本轮测的是**固定历史流的决策与新增时间等待**，没有再次跑动作模型，也没有测新增物体计算接入实时系统后的排队、端到端推流延迟或算子吞吐。补算任务并行共享主机，因此其耗时不能当独占性能基准。未知标签的恢复数已单列，零已知误报不等于线上零风险；正例少时尤其不能作保证。

[逐窗结果](window-results.csv)、[分档表](summary.csv)、[事件组](event-groups.json)、[按源变化](source-effects.json)、[冻结协议](protocol.json)、[审计](verification.json)可复查。preliminary-invalid目录为已作废初稿，全部数字排除。
''')
    (OUT/'stats-appendix.md').write_text(f'''# 统计口径与限制

固定一次历史概率回放，无训练seed对照；bootstrap随机种子1101仅用于重采样。主要比较单位为同一人物同一时刻的配对窗口，三档边界见protocol.json。真值缺失不计入正确率/误报率；错报其他动作类计新增误报。

旧测试按{len(bysource)}个来源组作5000次配对cluster bootstrap；95%分位区间只描述固定模型在这些已见来源上的变化，不包含训练随机性或调参选择偏差：

```json
{json.dumps(stats,ensure_ascii=False,indent=2)}
```

不做基于独立窗口假设的显著性检验，不报告p值。相邻窗口重叠、同一动作多次触发、多人仅3个采集分组且曾参与校准，无法据此证明独立泛化。没有多重检验后挑选阈值；两个主方案、联合及单/三窗口诊断均预先声明。零误报增量的bootstrap区间退化为0只说明观测数据如此，不构成零风险上界。

延迟：续接无需未来帧；跑动从同一候选链首个合格低分窗口到首次满足数量/跨度条件。只在首次确认记录episode delay；其后连续输出wait=0。未确认候选无延迟值（不是0秒成功）。置信档以输出确认窗口为准。日志中的历史decision_age不是本实验实时延迟，未使用。

旧train/val/test分别报告；新736窗为反馈回归。85正例参与过未晋升的候选训练，当前正式模型未使用该权重，但该数据已用于历史调参，仍不称全新留出测试。
''')
    (OUT/'figure-catalog.md').write_text('''# 图表说明

- 01-recovery-false-alarms.png/pdf：两个主方案分别显示三档置信区间的真实补回、已知误报及未知恢复窗口。无误差条，柱高为本次全量计数。用于判断降低漏报是否伴随误报，不能把未知当正确或把窗口数当独立动作。
- 02-running-wait-tradeoff.png/pdf：单/双/三窗口速度复核的正确与误报窗口数，及每条首次确认链的时间等待（横线为中位数）。用于判断连续性要求的收益与代价；没有确认的组没有等待时间，不能解读为零延迟成功。单/三窗是预声明诊断，未据测试集改选。

图形使用英文避免CJK字体缺字；所有数值来自results.json与decisions.json，中文解释在analysis-report.md。图像不包含显著性结论。独立样本量限制见stats-appendix.md。
''')
    (OUT/'README.md').write_text('# 低分候选复核实验\n\n结果入口：[analysis-report.md](analysis-report.md)。\n\n复现顺序：prepare.py → launch_frontend.py 与 prepare_objects.py/launch_objects.py → test_policy.py → evaluate.py → verify.py → report.py。GPU步骤使用已有0.3.0镜像，只读挂载正式源码与模型；所有产物写本目录。已完成步骤可复用features/object-evidence。preliminary-invalid不参与分析。\n')
    print(decision)

if __name__=='__main__':main()
