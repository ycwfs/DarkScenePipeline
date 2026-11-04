from collections import Counter,defaultdict
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/other-action-review-matplotlib')
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES={'model2':'二窗纯连续','model3':'三窗纯连续',
       'prior2':'二窗连续＋动作先验','prior3':'三窗连续＋动作先验'}
ACTION_NAMES={'wave':'挥手','drink':'喝水','pick_up':'捡拾','throw':'抛物','fall':'摔倒',
    'talk':'交谈','fight':'打架','shake_hands':'握手'}
CLASSES=['wave','drink','pick_up','throw','fall','talk','fight','shake_hands']

def flat_stats(prefix, stats):
    delay=stats['delay_s']
    return {f'{prefix}_recovered_windows':stats['recovered_windows'],
        f'{prefix}_true_recovered':stats['true_recovered'],f'{prefix}_false_alarms':stats['false_alarms'],
        f'{prefix}_unknown':stats['unknown'],f'{prefix}_first_confirmations':stats['first_confirmations'],
        f'{prefix}_delay_median_s':None if delay is None else delay['median']}

def main():
    data=rows();dec=read(OUT/'decisions.json');result=read(OUT/'results.json');events=read(OUT/'event-results.json')
    partitions=['old_val','old_test','crowded_val','crowded_test'];table=['|数据组|动作|二窗：真实补回／误报／未标注|三窗：真实补回／误报／未标注|首次确认等待|','|---|---|---:|---:|---|']
    candidate_tables=[];bin_tables=[];methods=('model2','model3','prior2','prior3')
    all_candidate_counts=Counter(r['raw_label'] for r in data if r['bin'] is not None)
    nonrunning_counts={k:v for k,v in all_candidate_counts.items()if k!='chase'}
    primary_candidates=sum(result['partitions'][p]['classes'][a]['candidate_windows'] for p in partitions for a in CLASSES)
    overall={}
    for method in methods:
        overall[method]=dict(
            true=sum(result['partitions'][p]['totals'][method]['true_recovered']for p in partitions),
            false=sum(result['partitions'][p]['totals'][method]['false_alarms']for p in partitions),
            unknown=sum(result['partitions'][p]['totals'][method]['unknown']for p in partitions),
            positive_events=sum(result['events'][p][method]['newly_recovered_events']for p in partitions),
            false_events=sum(result['events'][p][method]['new_false_alarm_events']for p in partitions))
    for name in partitions:
        d=result['partitions'][name]
        for label in CLASSES:
            x=d['classes'][label]
            if not x['candidate_windows']:continue
            vals=[]
            for method in ('prior2','prior3'):
                s=x['decisions'][method];delay=s['delay_s'];
                vals.append((s['true_recovered'],s['false_alarms'],s['unknown'],None if delay is None else delay['median']))
            table.append(f"|{name}|{label}|{vals[0][0]}／{vals[0][1]}／{vals[0][2]}|{vals[1][0]}／{vals[1][1]}／{vals[1][2]}|{vals[0][3] if vals[0][3] is not None else '—'}秒／{vals[1][3] if vals[1][3] is not None else '—'}秒|")
            row=dict(partition=name,action=label,candidate_windows=x['candidate_windows'],known=x['known'],
                known_true=x['known_true'],known_other=x['known_other'])
            for method in methods:row.update(flat_stats(method,x['decisions'][method]))
            candidate_tables.append(row)
            band_labels={'0':'0.65-0.75','1':'0.75-0.85','2':'0.85-0.925'}
            for band,b in x['bins'].items():
                if not b['candidate_windows']:continue
                row=dict(partition=name,action=label,score_band=band_labels[band],candidate_windows=b['candidate_windows'],
                    known=b['known'],known_true=b['known_true'],known_other=b['known_other'])
                for method in methods:row.update(flat_stats(method,b['decisions'][method]))
                bin_tables.append(row)
    total_lines=['|数据组|规则|真实动作窗口补回|误报窗口|未标注窗口|新增检出动作组|新增误报组|确认延迟中位数|','|---|---|---:|---:|---:|---:|---:|---:|']
    for name in partitions:
        for method in ('model2','prior2','prior3'):
            s=result['partitions'][name]['totals'][method];e=result['events'][name][method];delay=s['delay_s']
            delay_text='—' if delay is None else f"{delay['median']:.2f}秒"
            total_lines.append(f"|{name}|{NAMES[method]}|{s['true_recovered']}|{s['false_alarms']}|{s['unknown']}|{e['newly_recovered_events']}|{e['new_false_alarm_events']}|{delay_text}|")
    save(OUT/'action-summary.json',dict(per_class=candidate_tables,partitions={p:result['partitions'][p]for p in partitions},events=result['events']))
    import csv
    with(OUT/'action-summary.csv').open('w')as f:
        w=csv.DictWriter(f,fieldnames=list(candidate_tables[0])if candidate_tables else ['partition','action']);w.writeheader();w.writerows(candidate_tables)
    with(OUT/'score-bin-summary.csv').open('w')as f:
        w=csv.DictWriter(f,fieldnames=list(bin_tables[0])if bin_tables else ['partition','action','score_band']);w.writeheader();w.writerows(bin_tables)
    fig,ax=plt.subplots(figsize=(11.5,5.2));labels=[];tp=[];fp=[]
    for name in partitions:
        for label in CLASSES:
            x=result['partitions'][name]['classes'][label]['decisions']['prior2']
            if x['recovered_windows']:
                labels.append(f'{name}\n{label}');tp.append(x['true_recovered']);fp.append(x['false_alarms'])
    x=range(len(labels));ax.bar([i-.19 for i in x],tp,.38,color='#0072B2',label='Correctly recovered windows')
    ax.bar([i+.19 for i in x],fp,.38,color='#D55E00',label='False action windows')
    ax.set_xticks(list(x),labels,rotation=55,ha='right');ax.set_ylabel('Windows');ax.set_title('Other-action review: prior-gated two-window consensus');ax.legend();ax.spines[['top','right']].set_visible(False);ax.margins(y=.2);fig.tight_layout()
    (OUT/'figures').mkdir(exist_ok=True)
    for ext in ('png','pdf'):fig.savefig(OUT/'figures'/f'01-other-action-consensus.{ext}',dpi=180)
    plt.close(fig)
    frozen=read(OUT/'protocol.json')
    overall_lines=[]
    for method in ('model2','prior2','prior3'):
        x=overall[method]
        overall_lines.append(f"- {NAMES[method]}：真动作窗口 {x['true']}、误报窗口 {x['false']}、未标注 {x['unknown']}；新增检出动作组 {x['positive_events']}、新增误报组 {x['false_events']}。")
    action_notes=[]
    for label in CLASSES:
        stats={m:dict(true=0,false=0)for m in ('model2','prior2','prior3')};n=0
        for p in partitions:
            x=result['partitions'][p]['classes'][label];n+=x['candidate_windows']
            for method in stats:
                stats[method]['true']+=x['decisions'][method]['true_recovered']
                stats[method]['false']+=x['decisions'][method]['false_alarms']
        if n:
            action_notes.append(f"- {ACTION_NAMES[label]}（{n}个留出候选）：纯二窗 {stats['model2']['true']}真／{stats['model2']['false']}误报；二窗加先验 {stats['prior2']['true']}真／{stats['prior2']['false']}误报；三窗加先验 {stats['prior3']['true']}真／{stats['prior3']['false']}误报。")
        else:
            action_notes.append(f"- {ACTION_NAMES[label]}：留出集中没有符合置信度范围的候选，无法评估。")
    plan=read(OUT/'candidate-plan.json');prior_audit=read(OUT/'prior-audit.json')
    held_ids={r['id']for r in data if r['scope']in ('onsite','online_supplement','new_multipeople')and r['split']in ('val','test')}
    object_support=sum(prior_audit.get(rid,{}).get('reason')=='single_frame_support'for rid in held_ids)
    (OUT/'analysis-report.md').write_text(f'''# 其他动作低置信度复核实验

## 结论

只接受“同类候选连续出现”并不足以作为低阈值动作复核。以下数字按动作窗口统计，事件组另列；未标注数据不当作真例或误报。

### 总体结果

主评估共有{primary_candidates}个旧／多人验证与测试候选窗口。

{chr(10).join(overall_lines)}

{chr(10).join(total_lines)}

二窗需要≥0.5秒，三窗需要≥1.0秒；等待从该连续候选第一次出现算到首次确认，主评估确认等待中位数约0.52秒。后续窗口不重复计新增延迟。独立候选必须每窗满足相应先验：挥手使用三帧腕高于双肩；喝水／捡拾使用当前物体模型、阈值0.581且遵守亮度门35。没有专属先验的动作只展示模型连续性对照，不据此建议启用。

### 按动作观察

{chr(10).join(action_notes)}

纯二窗规则补回的2个真动作窗口分别是一个旧测试喝水和一个多人验证挥手；加先验后保留挥手、否决了未获物体支持的真实喝水。多人测试仍剩2个挥手误报窗口；三窗规则将误报清零，但也没有补回真动作。喝水／捡拾门控大幅压制误报，但本轮不能证明保持召回，尤其真实喝水漏过这一例需要后续专门复核。

### 按动作和置信度

CSV [action-summary.csv](action-summary.csv) 给出逐组逐动作统计；[score-bin-summary.csv](score-bin-summary.csv) 再按0.65–0.75、0.75–0.85、0.85–0.925三档拆分真实补回、误报、未标注和候选覆盖。全量逐窗证据在 [window-results.csv](window-results.csv)。

![按动作比较二窗补回和误报](figures/01-other-action-consensus.png)

## 数据和验证范围

使用原正式模型的冻结融合概率，正式α=.972、照度门35、动作阈值.925、物体阈值.581、挥手3帧、跑动.35不变。全量共有{sum(all_candidate_counts.values())}个0.65–0.925非其他候选；排除跑动后为{sum(nonrunning_counts.values())}个，类别分布`{nonrunning_counts}`。本轮实际有留出候选的是挥手、喝水、捡拾、抛物和摔倒；交谈、打架、握手没有符合范围的留出候选，不能据此评价。置信度分档为(0.65,0.75)、[0.75,0.85)、[0.85,0.925)。主结果只纳入旧验证／测试和多人验证／测试；训练组只作诊断。多人13个窗口真值按用户确认更新。

低分喝水／捡拾候选序列中，held-out范围有{plan['object_sequence_windows']}个窗口需要精确物体证据，其中新补算{plan['object_inference_needed']}个、复用{plan['exact_object_evidence_reused']}个精确帧匹配结果；共{object_support}个窗口找到满足类别、置信度与人物ROI条件的单帧物体证据。新证据按同一人物、代次、源帧和720增强裁剪生成。挥手直接复用同窗精确骨架。没有用预测other、缺失先验或跨时刻近邻补标签。

所有旧/新split仍曾用于历史调参或人工回看，不是全新盲测；窗口相互重叠。样本不足的类别只报告候选/标签覆盖，不作有效性结论。该回放没有更改平台、Box、模型、参数、算子包或镜像，也未测试在线队列、端到端延迟或片段保存。

规则定义与审计见 [protocol.json](protocol.json)、[verification.json](verification.json)、[按动作汇总](action-summary.csv)、[按置信度分档汇总](score-bin-summary.csv)、[逐窗表](window-results.csv) 和 [动作组表](event-results.json)。
''')
    (OUT/'figure-catalog.md').write_text('''# 图表说明

- 01-other-action-consensus.png/pdf：按数据组与动作显示先验门控双窗口复核的真动作窗口和误报窗口。未标注窗口未绘入这两个柱。窗口不是独立事件；完整计数见CSV/JSON。

样本少、动作组相关，图中无置信区间或显著性标记；不能据此声称泛化。
''')
    print('report complete')

if __name__=='__main__':main()
