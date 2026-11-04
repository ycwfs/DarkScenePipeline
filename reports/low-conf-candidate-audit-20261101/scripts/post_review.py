"""Apply explicit user review of the 13 exported windows and audit labelled train data."""
from collections import Counter,defaultdict
import csv,json
from common import *

def main():
    rows,_=load();saved={r['id']:r for r in read(OUT/'replay-rows.json')};
    for r in rows:r.update({k:saved[r['id']][k] for k in ('baseline','support','prior_reason')})
    dec=read(OUT/'decisions.json');review=read(OUT/'unlabelled-combined-all-bins/review.json')['items']
    ids={x['id'] for x in review};assert len(ids)==13
    for r in rows:
        if r['id'] in ids:
            assert r['truth'] is None and dec[r['id']]['combined']['recovered']
            # User explicitly confirmed the displayed action prediction for every exported clip.
            r['truth']=r['raw_label'];r['truth_source']='user_confirmed_20261101'
        else:r['truth_source']='existing_label' if r['truth'] is not None else None
    def known(r):return r['truth'] is not None
    def action(r):return r['truth'] not in (None,'other')
    def metrics(rr,method=None):
        def pred(r):return r['baseline'] if method is None else (r['raw_label'] if dec[r['id']][method]['recovered'] else r['baseline'])
        pos=[r for r in rr if action(r)];neg=[r for r in rr if r['truth']=='other']
        misses=[r for r in pos if pred(r)!=r['truth']];wrong=[r for r in pos if pred(r) not in ('other',r['truth'])]
        fp=[r for r in neg if pred(r)!='other'];tp=[r for r in pos if pred(r)==r['truth']]
        return dict(n=len(rr),known=len([r for r in rr if known(r)]),positive=len(pos),other=len(neg),
                    tp=len(tp),missed_other=len([r for r in misses if pred(r)=='other']),wrong_action=len(wrong),
                    fp=len(fp),miss_ids=[r['id'] for r in misses],fp_ids=[r['id'] for r in fp],
                    truth_counts=dict(Counter(r['truth'] for r in rr)))
    roles={
      'old_train_all':[r for r in rows if r['scope']!='new_multipeople' and r['split']=='train'],
      'old_train_onsite':[r for r in rows if r['scope']=='onsite' and r['split']=='train'],
      'old_train_online_supplement':[r for r in rows if r['scope']=='online_supplement' and r['split']=='train'],
      'new_human_train':[r for r in rows if r['scope']=='new_multipeople' and r['split']=='train'],
      'all_labelled_old':[r for r in rows if r['scope']!='new_multipeople' and known(r)],
    }
    output={}
    source_names={v['id']:v['source'] for v in read(ROOT/'reports/wave-pose-prior-20261027/protocol.json')['old_sources']}
    source_names.update({r['source_id']:r['source'] for r in read(ROOT/'reports/manypeople-finetune-20261029/dataset.json')['records'] if r['scope']=='new'})
    for name,rr in roles.items():
        rr=[r for r in rr if known(r)]
        def event_metrics(method):
            ordered=sorted(rr,key=lambda r:(r['scope'],r['source_id'],r['track_id'],r['history_generation'],r['truth'],r['timestamp']))
            groups=[];current=[];last_key=None;last_time=None
            for r in ordered:
                key=(r['scope'],r['source_id'],r['track_id'],r['history_generation'],r['truth'])
                if key!=last_key or (last_time is not None and r['timestamp']-last_time>.650001):
                    if current:groups.append(current)
                    current=[]
                current.append(r);last_key=key;last_time=r['timestamp']
            if current:groups.append(current)
            positive=[g for g in groups if g[0]['truth']!='other'];negative=[g for g in groups if g[0]['truth']=='other']
            def prediction(r):
                return r['baseline'] if method=='baseline' else (r['raw_label'] if dec[r['id']][method]['recovered'] else r['baseline'])
            detected=sum(any(prediction(r)==g[0]['truth'] for r in g) for g in positive)
            false=sum(any(prediction(r)!='other' for r in g) for g in negative)
            return dict(positive_events=len(positive),detected_positive_events=detected,missed_positive_events=len(positive)-detected,
                        other_events=len(negative),false_alarm_events=false)
        output[name]=dict(baseline=metrics(rr),continuation=metrics(rr,'continuation'),running2=metrics(rr,'running2'),combined=metrics(rr,'combined'),
            events={m:event_metrics(m) for m in ('baseline','continuation','running2','combined')})
    # Candidate-window accounting after user adjudication.
    bins=[]
    for j,(lo,hi) in enumerate(BINS):
        rr=[r for r in rows if r['bin']==j]
        for method in ('continuation','running2','combined'):
            accepted=[r for r in rr if dec[r['id']][method]['recovered']]
            known_acc=[r for r in accepted if known(r)]
            bins.append(dict(bin=[lo,hi],method=method,recovered=len(accepted),true=sum(action(r) and r['truth']==r['raw_label'] for r in known_acc),
                false=sum(r['truth']=='other' or (action(r) and r['truth']!=r['raw_label']) for r in known_acc),
                unknown=sum(not known(r) for r in accepted)))
    # Export all changed and all train errors for direct inspection.
    inspect=[]
    for name,rr in roles.items():
        for r in rr:
            if not known(r):continue
            for method in (None,'continuation','running2','combined'):
                pred=r['baseline'] if method is None else (r['raw_label'] if dec[r['id']][method]['recovered'] else r['baseline'])
                if pred!=r['truth']:
                    inspect.append(dict(role=name,id=r['id'],source_id=r['source_id'],source=source_names.get(r['source_id']),source_scope=r['scope'],split=r['split'],timestamp=r['timestamp'],truth=r['truth'],truth_source=r['truth_source'],baseline=r['baseline'],method='baseline' if method is None else method,prediction=pred,raw_label=r['raw_label'],score=r['score'],bin=r['bin'],prior_reason=r['prior_reason']))
    dest=OUT/'post-review-20261101';dest.mkdir(exist_ok=True)
    save(dest/'results.json',dict(user_confirmed_ids=sorted(ids),user_confirmed_count=len(ids),roles=output,bins=bins,scope='User-confirmed exported windows; no production change'))
    with(dest/'train-errors.csv').open('w',newline='')as f:
        fields=list(inspect[0]);w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(inspect)
    # Focused summary for the 0.85-.925 bin and all three bins.
    def aggregate(j=None):
        rr=[r for r in rows if (j is None or r['bin']==j)];
        out={}
        for method in ('continuation','running2','combined'):
            aa=[r for r in rr if dec[r['id']][method]['recovered'] and known(r)]
            out[method]=dict(true=sum(action(r) and r['truth']==r['raw_label'] for r in aa),false=sum(r['truth']=='other' or (action(r) and r['truth']!=r['raw_label']) for r in aa),unknown=sum(dec[r['id']][method]['recovered'] and not known(r) for r in rr))
        return out
    save(dest/'summary.json',dict(high_bin=aggregate(2),all_bins=aggregate(),
        interpretation='High bin combined = 19 true and 4 known false after user confirmed the 10 high-bin unknowns; all-bin combined = 23 true and 7 known false after confirming all 13 unknowns.'))
    (dest/'analysis-report.md').write_text(f'''# 用户确认 13 个未标注窗口后的复核

用户确认导出的 13 个联合规则窗口预测全部正确，已将它们作为 `user_confirmed_20261101` 真值写入本隔离复核，不改变原始审核数据和生产结果。

## 联合补回数量

- **0.85–0.925 档**：原已标注真实 9 个 + 用户确认未标注 10 个 = **19 个真实窗口**；已知误报 **4 个**。
- **三档合计**：原已标注真实 10 个 + 用户确认未标注 13 个 = **23 个真实窗口**；已知误报 **7 个**。

这两个数字不能混用。13 个窗口中有 3 个位于 0.85 以下档位。

## 旧训练数据（已标注窗口）

|数据范围|方案|真实动作|漏报为other|错报其他动作|误报other|已标注窗口|
|---|---|---:|---:|---:|---:|---:|
'''+''.join(f"|{name}|{m}|{v[m]['tp']}|{v[m]['missed_other']}|{v[m]['wrong_action']}|{v[m]['fp']}|{v[m]['n']}|\n" for name,v in output.items() for m in ('baseline','continuation','running2','combined'))+f'''

窗口会重叠，按同源人物/代次、同一真值且间隔不超过0.65秒合并后，事件级结果如下：

|数据范围|方案|检出动作事件／动作事件数|漏掉动作事件|误报other事件／other事件数|
|---|---|---:|---:|---:|
'''+''.join(f"|{name}|{m}|{v['events'][m]['detected_positive_events']}/{v['events'][m]['positive_events']}|{v['events'][m]['missed_positive_events']}|{v['events'][m]['false_alarm_events']}/{v['events'][m]['other_events']}|\n" for name,v in output.items() for m in ('baseline','combined'))+f'''

`old_train_onsite` 是旧现场动作训练片段；`old_train_online_supplement` 是历史 other 补充集。`new_human_train` 是10月25日多人审核集的训练分组，85个正例中39个属于该组。基线指当前正式模型原阈值与先验；组合列是本轮续接＋双窗口跑动回放。漏报指真实动作被判为 other，误报指真实 other 被判为动作；错报动作类别另列。

旧现场训练部分按动作的基线漏报为：喝水31、捡拾15、挥手4、跑动1、抛物1、摔倒1；加入两条规则后为喝水29、捡拾13，其余这几类在该训练子集不再漏报。基线5个误报窗口为捡拾2、喝水2、跑动1；组合回放8个误报窗口为捡拾4、喝水2、跑动1、抛物1。误报增加的3个均由续接候选引入，提示续接可能延长原有假动作。

10月25日多人训练分组共320个已人工标注窗口（39个真动作、281个other）：正式模型基线漏报38个正例、误报2个other窗口；联合回放漏报37个、误报仍为2个。该数据参与过后续候选微调实验，因此只作为训练数据错误盘点，不是独立泛化评估。

逐条错误窗口见 [train-errors.csv](train-errors.csv)，用户确认窗口见 [confirmed-windows.json](confirmed-windows.json)。本报告是离线审计，未更新模型、阈值、算子包或镜像。
''')
    confirmed_review=[]
    for item in review:
        item=dict(item,truth=item['action'],review_status='confirmed',confirmation_source='user_2026-11-01')
        confirmed_review.append(item)
    save(dest/'confirmed-windows.json',confirmed_review)
    for folder in ('unlabelled-combined-all-bins','unlabelled-combined-085-0925'):
        gallery=OUT/folder
        review_path=gallery/'review.json'
        if not review_path.exists():continue
        package=read(review_path)
        for item in package['items']:
            item.update(truth=item['action'],review_status='confirmed',confirmation_source='user_2026-11-01')
        save(review_path,package)
        csv_path=gallery/'review.csv'
        if csv_path.exists():
            with csv_path.open(newline='') as stream:records=list(csv.DictReader(stream))
            actions={item['id']:item['action'] for item in package['items']}
            for row in records:
                row['action']=actions[row['id']];row['truth']=actions[row['id']]
                row['review_status']='confirmed'
            fields=list(records[0]) if records else []
            for name in ('action','truth'):
                if name not in fields:fields.append(name)
            with csv_path.open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(records)
        page=gallery/'index.html'
        if page.exists():
            text=page.read_text().replace('未标注补回窗口','用户确认的补回窗口')
            text=text.replace('未标注窗口','已确认正确的动作窗口').replace('不能直接视为真实动作或误报。请结合前后文审核。','用户已确认这些动作预测正确。视频保留前后文供复核。')
            text=text.replace('待人工确认：真实动作 / other / 不确定','用户已确认：预测动作正确')
            page.write_text(text)
    print(json.dumps(dict(high_bin=aggregate(2),all_bins=aggregate(),train=output),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
