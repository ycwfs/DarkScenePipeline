"""Overlay user-confirmed drinking labels on frozen predictions; no model reruns."""
from collections import defaultdict
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
ROOT = OUT.parents[1]
BASE = ROOT/'reports/object-prior-finetune-20261025'
MODELS = ('n', 's', 'm', 'm_finetuned')
VARIANTS = ('native_raw', 'native_gamma', 'small_raw', 'small_enhanced')
POSITIVE = 'new-01_1935'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name: str, data: object) -> None:
    (OUT/name).write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')


def save_csv(name: str, rows: list[dict]) -> None:
    with (OUT/name).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def metric(rows: list[dict], truth: dict[str, bool], model: str, variant: str, scope: str) -> dict:
    selected = [r for r in rows if r['id'] in truth]
    assert len(selected) == len(truth) and len({r['id'] for r in selected}) == len(truth)
    tp = sum(truth[r['id']] and r['decision']['label']=='drink' for r in selected)
    fp = sum(not truth[r['id']] and r['decision']['label']=='drink' for r in selected)
    positives = sum(truth.values()); negatives = len(truth)-positives
    return dict(scope=scope, model=model, variant=variant, windows=len(selected),
        baseline_tp=positives, baseline_fp=negatives, tp_retained=tp, added_fn=positives-tp,
        fp_remaining=fp, fp_removed=negatives-fp, fp_reduction=(negatives-fp)/negatives,
        positive_retention=tp/positives,
        fp_bypassed=sum(not truth[r['id']] and not r['decision']['applied']
                        and r['decision']['label']=='drink' for r in selected))


def main() -> None:
    original = json.loads((OUT/'action-results.json').read_text())
    canonical = [r for r in original if r['model']=='n' and r['variant']=='native_raw'
                 and r['split']=='feedback' and r['baseline']=='drink']
    assert len(canonical)==527 and len({r['id'] for r in canonical})==527
    by_id = {r['id']: r for r in canonical}
    review = json.loads((BASE/'review-candidates.json').read_text())
    reviewed = {}
    for image in review:
        event = image['event']
        if event['label']!='drink':
            continue  # "not drinking" is not a label for the separate pickup task.
        matches = [r for r in canonical if r['source_id']==f"new-{image['number']:02d}"
                   and r['track_id']==event['track_id']
                   and r['history_generation']==event['history_generation']
                   and abs(r['timestamp']-event['timestamp'])<1e-5]
        assert len(matches)==1
        reviewed[matches[0]['id']] = image['review_image']
    assert len(reviewed)==18 and POSITIVE in reviewed
    truth_all = {r['id']: r['id']==POSITIVE for r in canonical}
    truth_review = {key: key==POSITIVE for key in reviewed}
    scope_note = ('用户已明确确认：“全部9段的喝水触发，仅图中这段是真的”。'
                  '因此按当前0.85阈值筛出的527个喝水触发，标为1真、526非喝水误报；'
                  '同时单列18个已展示喝水审核样本，不将确认扩展为未触发窗口或捡物真值。')
    save('user-feedback-labels.json', dict(
        user_statement='别光统计增加的漏报啊，还得统计减少的误报啊，那之前跑的几段manypeople中真的在喝水的只有这一个样本，其余都不是的。',
        user_scope_confirmation='全部9段的喝水触发，仅图中这段是真的',
        date='2026-10-26', positive_id=POSITIVE, scope_note=scope_note,
        provided_image=str(BASE/'visual/01_source01_drink.jpg'),
        provided_image_sha256=sha(BASE/'visual/01_source01_drink.jpg'),
        label_semantics='binary is_drink; not_drink does not assert other for all action classes',
        original_predictions_sha256=sha(OUT/'action-results.json'),
        labels=[dict(id=r['id'], source_id=r['source_id'], event_index=r['event_index'],
            track_id=r['track_id'], history_generation=r['history_generation'], timestamp=r['timestamp'],
            is_drink=r['id']==POSITIVE, review_image=reviewed.get(r['id']),
            provenance='explicit user confirmation of all nine videos drinking triggers') for r in canonical]))
    summary=[]; window_rows=[]
    for model in MODELS:
        for variant in VARIANTS:
            rows=[r for r in original if r['model']==model and r['variant']==variant
                  and r['split']=='feedback' and r['baseline']=='drink']
            summary.append(metric(rows,truth_all,model,variant,'full9_user_confirmed'))
            summary.append(metric(rows,truth_review,model,variant,'reviewed18'))
            for r in rows:
                positive=truth_all[r['id']]; fired=r['decision']['label']=='drink'
                window_rows.append(dict(model=model,variant=variant,id=r['id'],
                    source_id=r['source_id'],track_id=r['track_id'],timestamp=r['timestamp'],
                    reviewed=r['id'] in reviewed,is_drink=positive,prediction=r['decision']['label'],
                    outcome=('TP' if fired else 'added_FN') if positive else ('remaining_FP' if fired else 'removed_FP'),
                    reason=r['decision']['reason']))
    # Fresh exhaustive grouping: the earlier audit segment table covers only 455/527 drink windows.
    # Keep every fixed candidate; do not call these groups saved clips or fully annotated actions.
    tracks=defaultdict(list)
    for r in canonical:tracks[r['source_id'],r['track_id'],r['history_generation']].append(r)
    groups=[]
    for key,rows in sorted(tracks.items()):
        current=[]
        for r in sorted(rows,key=lambda r:r['timestamp']):
            if current and r['timestamp']-current[-1]['timestamp']>1.2:
                groups.append(current);current=[]
            current.append(r)
        if current:groups.append(current)
    assert sum(map(len,groups))==527
    segment_rows=[]
    for model in MODELS:
        for variant in VARIANTS:
            results={r['id']:r['decision']['label'] for r in original
                     if r['model']==model and r['variant']==variant}
            for index,group in enumerate(groups):
                positive=any(r['id']==POSITIVE for r in group)
                retained=any(results[r['id']]=='drink' for r in group)
                segment_rows.append(dict(model=model,variant=variant,group_id=index,
                    source_id=group[0]['source_id'],track_id=group[0]['track_id'],
                    history_generation=group[0]['history_generation'],
                    start=group[0]['timestamp'],end=group[-1]['timestamp'],windows=len(group),
                    is_positive=positive,retained=retained,
                    outcome=('TP' if retained else 'added_FN') if positive else ('remaining_FP' if retained else 'removed_FP'),
                    event_ids=' '.join(r['id'] for r in group)))
    segment_summary=[]
    for model in MODELS:
        for variant in VARIANTS:
            rows=[r for r in segment_rows if r['model']==model and r['variant']==variant]
            segment_summary.append(dict(model=model,variant=variant,baseline_groups=len(rows),
                positive_groups=sum(r['is_positive'] for r in rows),
                negative_groups=sum(not r['is_positive'] for r in rows),
                tp_retained=sum(r['outcome']=='TP' for r in rows),
                added_fn=sum(r['outcome']=='added_FN' for r in rows),
                fp_remaining=sum(r['outcome']=='remaining_FP' for r in rows),
                fp_removed=sum(r['outcome']=='removed_FP' for r in rows)))
    save_csv('false-positive-tradeoff.csv',summary)
    save_csv('feedback-labeled-windows.csv',window_rows)
    save_csv('feedback-candidate-groups.csv',segment_rows)
    save_csv('feedback-candidate-group-summary.csv',segment_summary)
    save('false-positive-summary.json',dict(scope_note=scope_note,metrics=summary,
        group_gap_seconds=1.2,group_metrics=segment_summary,
        group_note='same source/track/generation, consecutive candidate timestamps gap<=1.2s; not actual saved files',
        known_positive=by_id[POSITIVE],new_pickup_truth='not supplied, excluded from FP claims',
        inference_rerun=False,model_selection=False))
    table=['|模型|输入|过滤误报/526|残留误报|误报减少比例|真喝水保留/1|新增漏报|',
           '|---|---|---:|---:|---:|---:|---:|']
    reviewed_table=['|模型|输入|过滤误报/17|残留误报|真喝水保留/1|新增漏报|',
                    '|---|---|---:|---:|---:|---:|']
    for r in summary:
        if r['scope']=='full9_user_confirmed':
            table.append(f"|{r['model']}|{r['variant']}|{r['fp_removed']}|{r['fp_remaining']}|{r['fp_reduction']:.2%}|{r['tp_retained']}|{r['added_fn']}|")
        else:
            reviewed_table.append(f"|{r['model']}|{r['variant']}|{r['fp_removed']}|{r['fp_remaining']}|{r['tp_retained']}|{r['added_fn']}|")
    group_table=['|模型|输入|过滤候选误报段/233|残留候选误报段|真喝水保留/1|',
                 '|---|---|---:|---:|---:|']
    for r in segment_summary:
        if r['variant']=='native_raw' or r['model']=='m_finetuned' and r['variant']=='small_enhanced':
            group_table.append(f"|{r['model']}|{r['variant']}|{r['fp_removed']}|{r['fp_remaining']}|{r['tp_retained']}|")
    text=('# 误报收益与漏报代价：用户审核后的补充统计\n\n'+scope_note+'\n\n'
          '固定动作阈值0.85、物体阈值0.35、原图门30，复用上一轮全部预测。这里只衡量喝水过滤；'
          '527为窗口数，非527段保存视频。用户所指正例是原片01/T317/G0/149.92s、event1935；'
          '其原动作置信度0.9447、原图亮度115.94。相邻150.44s的窗口置信度0.7532，'
          '不在0.85候选内，不重复计为第二个正例。\n\n'
          '## 用户确认的全9段喝水触发\n\n基线：真喝水1，误报526，尚未加入本次先验。\n\n'+
          '\n'.join(table)+'\n\n## 已展示的18个喝水审核样本\n\n'
          '25张审核图中有18张喝水候选（1真、17假），其余7张是捡物。'
          '“非喝水”不会自动变成“捡物是假”的标注。\n\n'+'\n'.join(reviewed_table)+'\n\n'
          '## 减少连续重复窗口的计数\n\n按原片＋人物ID＋history_generation分组，'
          '相邻喝水候选时间差≤1.2秒归为一段，527窗形成234个候选段；用户确认对应1真、233误报段。'
          '同一段只要还保留一个喝水窗口就算残留，不把部分过滤夸大为整段消除。'
          '这些是固定规则下的候选分组，不是实际保存文件数，也不保证跟踪ID跨遮挡不变化。\n\n'+
          '\n'.join(group_table)+'\n\n'
          '## 解读\n\n公开n/s确实过滤了大量误报，先前只强调漏检的结论不完整。'
          '高清原图下n过滤517/526（98.29%）、s过滤522/526（99.24%）；'
          '现场微调m过滤436/526（82.89%）。然而这一个用户确认的真喝水在所有16组合中也都被误过滤，'
          '所以当前证据支持“过滤很强，但真假区分仍不可靠”。\n\n'
          '现场微调m的高清输入比720增强裁剪在旧正样本上保留更多真实动作，'
          '但在本批用户确认的误报上残留90对49，多41个；不能笼统说高清输入整体更好。'
          '模型选择需要同时约束真实动作保留和误报数量，不能仅最大化过滤量或总体正确比例。\n\n'
          '新9段只确认一个正动作，不能由0/1推断总体召回恒为0；当前候选预先由动作阈值筛出，'
          '也无法从本表估计全视频未触发的真实动作或总体假阳性率。新增漏报指先验额外过滤的正例。'
          '低亮/绑定不可用而保留的误报同样计入残留，不隐藏弃权。\n\n'
          '本轮没有改模型、阈值、正式服务或提交包，也没有重新训练。'
          '标签作为单独覆盖层保存，原始预测和旧truth=null记录保留。\n')
    (OUT/'false-positive-tradeoff.md').write_text(text)
    selected=[r for r in summary if r['scope']=='full9_user_confirmed' and
              (r['variant']=='native_raw' or r['model']=='m_finetuned' and r['variant']=='small_enhanced')]
    labels=[r['model']+' / '+r['variant'] for r in selected]
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),gridspec_kw={'width_ratios':[2,1]})
    y=list(range(len(selected)))
    axes[0].barh(y,[r['fp_removed'] for r in selected],color='#538b66',label='Removed false alarms')
    axes[0].barh(y,[r['fp_remaining'] for r in selected],left=[r['fp_removed'] for r in selected],
                 color='#ba684e',label='Remaining false alarms')
    for i,r in enumerate(selected):
        axes[0].text(5,i,f"{r['fp_removed']} removed; {r['fp_remaining']} remain",va='center',color='white',fontsize=9)
    axes[0].set_yticks(y,labels);axes[0].invert_yaxis();axes[0].set_xlim(0,526)
    axes[0].set_xlabel('False drinking windows (baseline 526)');axes[0].legend(loc='lower left',bbox_to_anchor=(0,1),fontsize=8)
    axes[1].barh(y,[r['tp_retained'] for r in selected],color='#538b66',label='True drinking retained')
    axes[1].barh(y,[r['added_fn'] for r in selected],left=[r['tp_retained'] for r in selected],
                 color='#ba684e',label='True drinking rejected')
    for i in y:axes[1].text(.05,i,'0 retained; 1 rejected',va='center',color='white',fontsize=9)
    axes[1].set_yticks(y,['']*len(y));axes[1].invert_yaxis();axes[1].set_xlim(0,1)
    axes[1].set_xticks([0,1]);axes[1].set_xlabel('User-confirmed positive (baseline 1)')
    fig.suptitle('User-confirmed nine-video labels: fewer false alarms AND added missed drinking',fontsize=12)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUT/f'figures/03-false-positive-tradeoff.{ext}',dpi=180)
    plt.close(fig)
    assert all(r['tp_retained']+r['added_fn']==1 and
               r['fp_remaining']+r['fp_removed']==r['baseline_fp'] for r in summary)
    assert len(window_rows)==527*16
    save('feedback-label-verification.json',dict(passed=True,reviewed_drinking_windows=18,
        all_drinking_windows=527,known_positive=POSITIVE,user_confirmed_negative_windows=526,
        settings=16,labeled_rows=len(window_rows),candidate_groups=len(groups),
        original_predictions_sha256=sha(OUT/'action-results.json'),
        labels_sha256=sha(OUT/'user-feedback-labels.json')))
    print('\n'.join(table));print('Baseline candidate groups:',len(groups))


if __name__=='__main__':main()
