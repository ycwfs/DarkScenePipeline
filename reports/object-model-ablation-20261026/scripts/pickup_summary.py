"""Pair new pickup suppression with labeled historical FP/FN; keep unknown truth."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/darkpipe-mpl-feedback-20261026')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1]
MODELS = ('n', 's', 'm', 'm_finetuned')
VARIANTS = ('native_raw', 'native_gamma', 'small_raw', 'small_enhanced')


def main() -> None:
    path=OUT/'action-results.json'
    data=json.loads(path.read_text()); summary=[]; details=[]
    canonical=[r for r in data if r['model']=='n' and r['variant']=='native_raw'
               and r['split']=='feedback' and r['baseline']=='pick_up']
    assert len(canonical)==37
    tracks=defaultdict(list)
    for row in canonical:tracks[row['source_id'],row['track_id'],row['history_generation']].append(row)
    groups=[]
    for key,rows in sorted(tracks.items()):
        current=[]
        for row in sorted(rows,key=lambda r:r['timestamp']):
            if current and row['timestamp']-current[-1]['timestamp']>1.2:
                groups.append(current);current=[]
            current.append(row)
        if current:groups.append(current)
    assert sum(map(len,groups))==37 and len(groups)==18
    group_details=[]
    for model in MODELS:
        for variant in VARIANTS:
            subset=[r for r in data if r['model']==model and r['variant']==variant]
            new=[r for r in subset if r['split']=='feedback' and r['baseline']=='pick_up']
            assert len(new)==37 and len({r['id'] for r in new})==37
            retained=sum(r['decision']['label']=='pick_up' for r in new)
            row=dict(model=model,variant=variant,new_candidates=37,new_filtered=37-retained,
                     new_retained=retained,new_suppression_fraction=(37-retained)/37,
                     new_confirmed_fp=None,new_added_fn=None)
            for split in ('val','test'):
                pos=[r for r in subset if r['split']==split and r['truth']=='pick_up']
                neg=[r for r in subset if r['split']==split and r['truth']=='other'
                     and r['baseline']=='pick_up']
                assert len(pos)==4 and all(r['baseline']=='pick_up' for r in pos)
                tp=sum(r['decision']['label']=='pick_up' for r in pos)
                fp=sum(r['decision']['label']=='pick_up' for r in neg)
                row.update({f'{split}_positive_windows':len(pos),f'{split}_tp_retained':tp,
                            f'{split}_added_fn':len(pos)-tp,f'{split}_baseline_fp':len(neg),
                            f'{split}_fp_remaining':fp,f'{split}_fp_removed':len(neg)-fp})
            summary.append(row)
            for r in new:
                details.append(dict(model=model,variant=variant,id=r['id'],source_id=r['source_id'],
                    track_id=r['track_id'],history_generation=r['history_generation'],
                    timestamp=r['timestamp'],raw_luma=r['raw_luma'],truth=None,
                    prediction=r['decision']['label'],reason=r['decision']['reason'],
                    evidence_classes=' '.join(str(c) for c in sorted({e['class_id'] for e in r['decision']['evidence']}))))
            decisions={r['id']:r['decision']['label']=='pick_up' for r in subset}
            for index,group in enumerate(groups):
                group_details.append(dict(model=model,variant=variant,group_id=index,
                    source_id=group[0]['source_id'],track_id=group[0]['track_id'],
                    history_generation=group[0]['history_generation'],start=group[0]['timestamp'],
                    end=group[-1]['timestamp'],windows=len(group),
                    retained=any(decisions[r['id']] for r in group),
                    outcome='retained_candidate' if any(decisions[r['id']] for r in group) else 'suppressed_candidate',
                    event_ids=' '.join(r['id'] for r in group)))
    for name,rows in [('pickup-counts.csv',summary),('pickup-window-results.csv',details),
                      ('pickup-candidate-groups.csv',group_details)]:
        with (OUT/name).open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    table=['|模型|输入|新37窗过滤|新残留|触发减少比例|旧test真捡物保留/4|旧test误报过滤/3|',
           '|---|---|---:|---:|---:|---:|---:|']
    for r in summary:
        table.append(f"|{r['model']}|{r['variant']}|{r['new_filtered']}|{r['new_retained']}|{r['new_suppression_fraction']:.2%}|{r['test_tp_retained']}|{r['test_fp_removed']}|")
    group_summary=[]
    for model in MODELS:
        for variant in VARIANTS:
            rows=[r for r in group_details if r['model']==model and r['variant']==variant]
            retained=sum(r['retained'] for r in rows)
            group_summary.append(dict(model=model,variant=variant,candidate_groups=len(rows),
                suppressed_groups=len(rows)-retained,retained_groups=retained))
    with (OUT/'pickup-candidate-group-summary.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(group_summary[0]));writer.writeheader();writer.writerows(group_summary)
    group_table=['|模型|输入|压掉候选段/18|残留候选段|','|---|---|---:|---:|']
    for r in group_summary:
        if r['variant']=='native_raw' or r['model']=='m_finetuned' and r['variant']=='small_enhanced':
            group_table.append(f"|{r['model']}|{r['variant']}|{r['suppressed_groups']}|{r['retained_groups']}|")
    report=('# 捡拾物品：触发过滤收益与真实动作保留\n\n'
        '用户反馈“之前我看到的manypeople的数据集也基本全是误报”。记录为总体审核意见，'
        '未擅自把“基本”转换成37个窗口全部已确认负例。以下新数据统计过滤与残留；'
        '**若37窗全部为误报，过滤数及比例就等于误报减少数及比例**。旧val/test已有标签，直接统计误报过滤和新增漏报。\n\n'
        '固定动作阈值0.85、物体0.35、原图门30，沿用一帧同人ROI出现相容物体即支持规则；'
        '复用冻结的16组预测，没有重新推理、训练、调参或更换正式包。新捡物共37窗，来自01段14、03段11、07段10、08段2；'
        '全部原L>30。窗口计数不是保存视频数量。\n\n'+'\n'.join(table)+'\n\n'
        '## 同时看误报收益与漏报代价\n\n'
        '新数据主要为误报这一审核意见下，物体先验确实能大量压制捡物触发：'
        '高清n35/37、s36/37、微调m33/37；720增强微调m37/37。'
        '公开n/s/公开m对旧test的3个已知捡物误报全部过滤，但也过滤原先命中的4个真捡物；'
        '其中捡纸动作还涉及COCO没有纸类，不能归因于单纯网络大小。\n\n'
        '微调m高清在旧test保留4/4真捡物、过滤1/3误报；720增强保留3/4真捡物、过滤2/3误报。'
        '因此高清方案较有利于保留真实动作，720增强过滤更强。旧val高清仅保留2/4真捡物、过滤1/7误报；'
        '720增强为1/4、2/7，仍不应根据少量test正例宣称召回已可靠。\n\n'
        '## 高清微调m残留4窗的图像核对\n\n'
        '已查看各窗首/中/末原始ROI采样，见[对照图](examples/pickup-native-retained.jpg)。'
        '画面中的人物站立或行走，未见从地面捡起物体的动作；这是助手对采样帧的视觉核对，非用户逐窗最终标注，也不证明窗口以外没有动作。\n\n'
        '|窗口|人物/时间|放行物体预测|视觉观察|\n|---|---|---|---|\n'
        '|new-03_2854|T370，216.61s|手机0.378|人物站立，手在背后|\n'
        '|new-03_4507|T797，386.42s|纸0.685|人物站立，未见向地面取物|\n'
        '|new-07_294|T28，24.88s|瓶0.550|人物站立，手在背后|\n'
        '|new-07_3233|T559，254.24s|瓶最高0.788|瓶在背后手中，人物行走|\n\n'
        '这个残留机制很具体：检测到瓶/手机/纸只说明周围有物体，普通持物或旁人/背景物体也可能支持原先的捡物误判。'
        '本轮沿用用户选择的单帧规则，只记录该局限，没有擅自改为必须连续检测或新增姿态硬门。\n\n'
        '## 按人物与时间合并相邻窗口\n\n'
        '同原片/人物/generation且间隔≤1.2秒合并，37窗形成18段候选。只要段内还有窗口保留捡物，就算该段残留。\n\n'+
        '\n'.join(group_table)+'\n\n'
        '此候选分组不是实际保存视频片段；人物ID变化或业务的片段合并规则不同都会改变段数。\n\n'
        '## 统计范围\n\n'
        '每模型固定执行一次、重训次数0；新37窗相关且未逐窗确认，不报新误报率或漏报为0。'
        '旧test4正/3误报、val4正/7误报为既有辅助标签，样本少、同场景相关且多次查看；只给精确分子分母，不做显著性或泛化声明。'
        '公开模型支持瓶/杯/手机，七类模型还支持纸等，类别表差异明确保留。\n')
    (OUT/'pickup-tradeoff.md').write_text(report)
    selected=[r for r in summary if r['variant']=='native_raw' or r['model']=='m_finetuned' and r['variant']=='small_enhanced']
    fig,axes=plt.subplots(1,3,figsize=(14,4.3),gridspec_kw={'width_ratios':[1.4,1,1]})
    y=list(range(len(selected)));labels=[r['model']+' / '+r['variant'] for r in selected]
    for ax,good,bad,total,title in [
        (axes[0],'new_filtered','new_retained',37,'New: trigger suppression (truth incomplete)'),
        (axes[1],'test_tp_retained','test_added_fn',4,'Old test: true pickup retained / missed'),
        (axes[2],'test_fp_removed','test_fp_remaining',3,'Old test: false alarms removed / remaining')]:
        ax.barh(y,[r[good] for r in selected],color='#538b66')
        ax.barh(y,[r[bad] for r in selected],left=[r[good] for r in selected],color='#ba684e')
        for i,r in enumerate(selected):ax.text(total*.03,i,f'{r[good]} / {r[bad]}',va='center',color='white',fontsize=9)
        ax.set_xlim(0,total);ax.set_title(title,fontsize=9);ax.invert_yaxis();ax.set_xlabel('Windows')
        ax.set_yticks(y,labels if ax is axes[0] else ['']*len(y))
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUT/f'figures/04-pickup-tradeoff.{ext}',dpi=180)
    plt.close(fig)
    payload=dict(user_statement='之前我看到的manypeople的数据集也基本全是误报。',
        annotation_status='aggregate impression, not an explicit all-negative per-window confirmation',
        new_truth_preserved_as_unknown=True,metrics=summary,window_rows=len(details),
        candidate_groups=group_summary,
        source_counts=dict(Counter(r['source_id'] for r in details if r['model']=='n' and r['variant']=='native_raw')),
        input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),inference_rerun=False,production_changed=False)
    assert len(details)==37*16 and all(r['new_filtered']+r['new_retained']==37 for r in summary)
    (OUT/'pickup-summary.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    print('\n'.join(table));print('\n'.join(group_table))


if __name__=='__main__':main()
