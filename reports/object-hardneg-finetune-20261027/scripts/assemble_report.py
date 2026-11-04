"""Build a complete, descriptive report from frozen outputs and explicit case review."""
import csv
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from build_dataset import OUT,NAMES,save


def main():
    summaries=[json.loads((OUT/'long-videos'/f'new-{n:02}'/'summary.json').read_text())for n in [1,7,9]]
    cases=json.loads((OUT/'long-review-cases.json').read_text())
    annotations=json.loads((OUT/'long-review-labels.json').read_text())
    for c in cases:
        assert c['id']in annotations,c['id']
        c.update(annotations[c['id']])
    short_cases=json.loads((OUT/'short-review-cases.json').read_text())
    short_labels=json.loads((OUT/'short-review-labels.json').read_text())
    for c in short_cases:c.update(short_labels[c['id']])
    short_summaries=[json.loads((OUT/'short-regression'/f'new-{n:02}'/'summary.json').read_text())for n in [1,7,9]]
    context=json.loads((OUT/'object-context-evaluation.json').read_text())
    objects=json.loads((OUT/'object-evaluation.json').read_text())
    old=json.loads((OUT/'cached-old-window-evaluation.json').read_text())
    selected=json.loads((OUT/'selection.json').read_text())['selected']
    dataset=json.loads((OUT/'dataset-manifest.json').read_text())
    selection=json.loads((OUT/'selection.json').read_text())['rows']
    frames=sum(s['frames']for s in summaries);seconds=sum(s['seconds']for s in summaries)
    windows=sum(s['paired_checks']for s in summaries)
    table=[]
    for s in summaries:
        n=s['source']['number'];matches=[c for c in cases if c['number']==n]
        row=dict(source=f'new-{n:02}',seconds=s['seconds'],frames=s['frames'],windows=s['paired_checks'])
        for variant in ('baseline','candidate'):
            row[f'{variant}_drink_windows']=s[variant]['actions'].get('drink',0)
            row[f'{variant}_pickup_windows']=s[variant]['actions'].get('pick_up',0)
            row[f'{variant}_clips']=s[variant]['clips']['clips_saved']
            row[f'{variant}_false_interaction_windows']=sum(c[f'{variant}_windows']for c in matches if c['truth']=='false_action')
            row[f'{variant}_false_interaction_groups']=sum(c[f'{variant}_windows']>0 for c in matches if c['truth']=='false_action')
        table.append(row)
    save(OUT/'final-summary.json',dict(long_videos=table,cases=cases,frames=frames,seconds=seconds,paired_windows=windows,
        object_context=context['summary'],old_object_test=objects['summary']['old_test'],old_cached=old['summary'],
        selected=selected,production_replaced=False,short_regression=short_summaries,short_review=short_cases))
    with (OUT/'long-video-summary.csv').open('w',newline='')as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    names=['phone','camera','microphone'];x=np.arange(3)
    for i,mode in enumerate(['baseline','candidate']):
        y=[context['summary']['test']['models'][mode]['confused_as_container'][n]['errors']for n in names]
        bars=axes[0].bar(x+(i-.5)*.34,y,.34,label=mode,color=['#536d8f','#d08333'][i]);axes[0].bar_label(bars,padding=3)
    axes[0].set_xticks(x,[f'{n}\n(n={context["summary"]["test"]["models"]["baseline"]["confused_as_container"][n]["total"]})'for n in names])
    axes[0].set_ylabel('Boxes mistaken for bottle/cup');axes[0].set_title('Heldout reviewed deployment ROIs');axes[0].legend()
    for i,mode in enumerate(['baseline','candidate']):
        y=[r[f'{mode}_false_interaction_windows']for r in table]
        bars=axes[1].bar(x+(i-.5)*.34,y,.34,label=mode,color=['#536d8f','#d08333'][i]);axes[1].bar_label(bars,padding=3)
    axes[1].set_xticks(x,[r['source']for r in table]);axes[1].set_ylabel('Reviewed false interaction windows')
    axes[1].set_title('Complete videos, matched upstream tasks');axes[1].legend()
    for ax in axes:ax.set_ylim(0,max(3,ax.get_ylim()[1]+1));ax.yaxis.get_major_locator().set_params(integer=True)
    fig.savefig(figdir/'01-errors.png',dpi=180);fig.savefig(figdir/'01-errors.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    epochs=[int(Path(r['path']).stem[5:])+1 for r in selection]
    axes[0].plot(epochs,[r['score']for r in selection],marker='o',color='#536d8f');axes[0].axvline(9,color='#d08333',linestyle='--',label='selected epoch 9')
    axes[0].set(xlabel='Epoch',ylabel='Validation mAP50-95',title='Frozen one-to-one validation selection');axes[0].legend()
    modes=['baseline','candidate'];y=[objects['summary']['old_test']['models'][m]['bottle']['tp']for m in modes]
    z=[objects['summary']['new_test']['models'][m]['bottle']['tp']for m in modes]
    x=np.arange(2)
    axes[1].bar(x-.17,y,.34,label='Old bottle GT (n=12)',color='#536d8f')
    axes[1].bar(x+.17,z,.34,label='New bottle GT (n=4)',color='#d08333')
    axes[1].set_xticks(x,modes);axes[1].set(ylabel='True bottles detected',title='Recall check on tight reviewed crops',ylim=(0,5));axes[1].legend()
    fig.savefig(figdir/'02-selection-recall.png',dpi=180);fig.savefig(figdir/'02-selection-recall.pdf');plt.close(fig)
    headings='| 视频 | 时长(s) | 人物动作窗 | 喝水触发 前→后 | 捡物触发 前→后 | 已审核交互误报窗 前→后 | 自动片段 前→后 |\n|---|---:|---:|---:|---:|---:|---:|\n'
    mdtable=headings+'\n'.join(f'| {r["source"]} | {r["seconds"]:.2f} | {r["windows"]} | {r["baseline_drink_windows"]}→{r["candidate_drink_windows"]} | {r["baseline_pickup_windows"]}→{r["candidate_pickup_windows"]} | {r["baseline_false_interaction_windows"]}→{r["candidate_false_interaction_windows"]} | {r["baseline_clips"]}→{r["candidate_clips"]} |'for r in table)
    lines=['# 困难物体微调与完整长视频对照','',
        '本轮完成手机、相机、话筒的视觉辅助标注和八类YOLO26m微调。存在局部物体误检改善，但仍有手机被当成瓶子；不满足“已经消除这些误报”的结论。候选权重独立保留，正式默认模型和交付包没有替换。','',
        f'完整处理01/07/09三段原视频，共{seconds:.2f}秒（{seconds/60:.2f}分钟）、{frames}原帧、{windows}个人物动作窗。两路共享完全相同的增强/人体/姿态/轨迹/RGB/ST-GCN输入，只分别执行正式物体先验；每路均保存完整视频和动作片段。','',mdtable,'',
        '触发窗存在重叠，不能视为独立动作或独立统计样本；自动片段也可能按15秒上限拆分。仅喝水/捡物保留或改变的触发组逐项审核，其余动作未做全量人工GT，不报告总体准确率。未触发时段没有穷尽审核，因此不报告整段真动作召回。','',
        '## 上次同输入演示回归','',
        '另外严格复用上次三个短片输入（01的140–165秒、07的10–40秒、09的94–124秒，共85秒）：01喝水触发0→0；07手机通话造成的喝水误报1→1；09手持相机拍摄造成的喝水误报1→0。相机错误瓶框.589消失；手机错误瓶框.597→.728。这组能证明之前相机那个具体例子得到改善，但不能替代完整时间轴结果。','',
        '长视频与短片的轨迹初始化及动作窗口相位不同，短片09有一次误报而完整09没有该触发，并非只替换检测器能改变上游结果；每组内部都严格共享同一上游结果进行配对。短片基线复现了上轮展示的触发数。','',
        '## 物体与召回的权衡','',
        '部署上下文的32张保留审核图：瓶类误框3→1，手机被认作容器1/8→1/8，相机2/3→0/3，话筒0/1→0/1。注意：这些手机/相机/话筒正确类别在该测试集均未达到.581，减少误认通常是降低错误瓶子分数，不等于已经正确识别物品。四个真实瓶框在该部署裁剪集均0/4，没有改善。','',
        '更紧的人工有效区域裁剪对照揭示尺度敏感：新增真瓶1/4→0/4；原47张旧test图中瓶3/12→3/12、纸6/12→6/12、展品4/7→4/7。不能把不同裁剪上下文的数字混用，也不能隐去真瓶漏检。','',
        '训练来源44张部署ROI仅作拟合诊断：话筒被误认瓶2/10→0/10，正确话筒0/10→2/10，但新增3个未匹配话筒预测；不能称为独立话筒测试成功。旧55个冻结候选辅助回归：test真交互6/15→6/15、other误报1→1；val真交互11/14→12/14、other误报4→4。该辅助回放使用旧动作候选，不等于当前配置全量重跑。','',
        '## 已知真实动作','',
        '01号149.92秒/T317此前用户确认的真喝水，动作分数.94443通过阈值，但两个检测器都无兼容容器支持，最终均other，本次未修复。09号约54.8秒红衣人物持透明瓶靠嘴，邻近人物窗口在动作阈值阶段已经other，并未调用物体先验。参见 known-positive-audit.json 和网页定位按钮。历史“已触发窗只有一个真例”不能扩展到全视频时刻；本轮未以这些测试结果重新选epoch或阈值。','',
        '## 数据与可复现性','',
        '训练413图/308唯一视图，验证41图，test79图。新增审核含50个训练视图（含6个旧相机补充）、10个val、32个test；原图/增强图成对，不等于独立样本。标注主要为手机74、相机9、话筒20训练框，旧瓶115框用于保留原能力。来源组和像素跨split检查通过。','',
        '从当前七类m继续微调，新增microphone索引7，旧索引保持。seed27、20轮、640、batch12、AdamW lr1e-4、freeze10；部署one-to-one头验证mAP50–95选择第9轮.36516，10图PT/ONNX校验通过。完整方法见REPRODUCE.md，checkpoint/模型哈希见protocol.json与export-verification.json。','',
        '只有一个seed，测试来自3个相关现场原视频，稀少类别框且由助手视觉辅助标注；此前这些视频已被调参查看。本报告为描述性对照，不作统计显著性、跨场景泛化或“零误报保证”。杯无标注，不能评价。','',
        '## 采用建议','',
        '暂不替换正式物体权重。优先补充真实手持相机、不同姿态打电话及不同话筒的独立场景，同时保留充足真实瓶/杯；新增样本应与最终验证来源隔离。一帧支持规则使单帧高置信错误仍可放行整个动作，该问题在保持算法不变时不能仅靠本次少量微调保证消除。','',
        '本轮未更改阈值、生产代码、镜像或正式打包文件；候选模型仅用于复现实验。完整离线流程经过原0.3.0镜像执行，不代表平台无限流或NFS稳定性已重验。']
    (OUT/'analysis-report.md').write_text('\n'.join(lines)+'\n')
    (OUT/'README.md').write_text('# 物体困难负例微调\n\n打开 [可视化总览](index.html)，查看三段完整对照、自动片段和逐项误检证据。\n\n结论和数值：[analysis-report.md](analysis-report.md)。训练与复现：[REPRODUCE.md](REPRODUCE.md)。\n\n本轮候选尚未替换生产包；仍有手机误认瓶子，且紧裁剪真瓶出现退步。\n')
    (OUT/'stats-appendix.md').write_text('''# 统计附录

分析单位分开：物体框/审核图、来源组、动作窗、触发组、实际片段。三段完整长视频内的时间重叠窗高度相关，不把上万个窗当独立样本。主要对照是同帧同ROI同动作输入的检测器替换，不是两次任意不同跟踪的比较。

单seed27，不存在seed间均值±标准差。仅3个相关新来源，且错误挖掘过、历史阈值校准过；人工辅助框而非独立盲审。无法提供可靠总体95%置信区间或泛化显著性检验，因此不做p值、功效或多重比较的确认性声明。图不画伪造误差条，所有误差变化为固定本批的计数差。相机/话筒样本极少，零误认不代表类别召回充分。

物体TP阈值conf≥.581，同类别IoU≥.5，按置信度贪心一对一匹配。误认手机/相机/话筒为容器另用IoU≥.3诊断，不能与TP的IoU口径混为一谈。原部署ROI仅在已审核区域内中心预测计分，区域外没有GT不计误报。紧裁剪图另列，不能覆盖部署上下文表。

选模20个epoch按固定val mAP50–95，未按test再次挑epoch；没有独立模型超参数搜索。两个训练预检查失败均保留，正式训练重新从固定原权重开始。未把新增测试失败再次加入训练。

动作false_action标注见long-review-labels.json，保留组所有代表采样可查看。其他动作及没有候选时段未穷尽标注，因此总体动作precision/recall不可计算。历史“仅一处喝水”反馈仅用于原有候选范围，不扩展到新时间窗。
''')
    (OUT/'figure-catalog.md').write_text('''# 图表说明

`figures/01-errors.png/.pdf`：左图是保留审核ROI中物体误认容器数，右图为完整视频中审核确认的交互误报窗。无误差条，固定本批描述计数。应注意：局部相机误框减少是否对应最终动作收益，手机残留是否保持；不能把物体错误消失等同正确分类，不能把窗数视为独立事件。

`figures/02-selection-recall.png/.pdf`：左图显示固定val one-to-one选模曲线，第9轮预定标准最高；右图列出紧裁剪旧/新真瓶TP，防止只展示误报下降。原部署ROI结果另见正文，不能混用两种上下文。无多seed标准差和CI；本图用于说明选择依据与召回代价。

`context-comparisons/*.jpg`：完全相同720增强ROI下微调前后检测，白色是人工审核框；彩色仅≥.581的模型输出。训练来源明确标train，未检出不能解释为正确类别识别。

`long-review/*.jpg`：完整视频的全部保留/改变交互触发组证据。上排基线，下排候选，原始像素用于人工判读，框来自增强图检测并还原坐标。代表帧无法替代完整动作视频，网页同时提供完整前后视频与片段。
''')
    cards=[]
    for r in table:
        n=int(r['source'][-2:]);base=f'long-videos/{r["source"]}'
        cliplinks=[]
        for mode in ['baseline','candidate']:
            for f in sorted((OUT/base/f'{mode}-clips').rglob('*.mp4')):
                cliplinks.append(f'<li><a href="{html.escape(str(f.relative_to(OUT)))}">{mode} · {html.escape(f.name)}</a></li>')
        marks=([149.92]if n==1 else [21.32]if n==7 else [54.8,102.56,121.7])
        buttons=' '.join(f'<button onclick="seekPair({n},{t})">{t:g}s</button>'for t in marks)
        cards.append(f'<section><h2>{r["source"]} · {r["seconds"]:.2f}秒</h2><p>{buttons} <button onclick="playPair({n})">两路同时播放</button></p><div class="pair"><div>微调前<video id="b{n}" controls preload="metadata" src="{base}/baseline.mp4"></video></div><div>微调后（候选）<video id="c{n}" controls preload="metadata" src="{base}/candidate.mp4"></video></div></div><p>喝水窗 {r["baseline_drink_windows"]} → {r["candidate_drink_windows"]}；捡物窗 {r["baseline_pickup_windows"]} → {r["candidate_pickup_windows"]}；已审交互误报窗 {r["baseline_false_interaction_windows"]} → {r["candidate_false_interaction_windows"]}</p><details><summary>自动保存片段</summary><ul>{"".join(cliplinks)}</ul></details></section>')
    review=''.join(f'<article><h3>{c["id"]} · {c["start"]:.2f}s · {html.escape(c.get("description",""))}</h3><a href="{c["image"]}"><img loading="lazy" src="{c["image"]}"></a></article>'for c in cases)
    short_html='<h2>上次同输入短片：相机误报消失，手机误报保留</h2><p>与完整视频分开统计；85秒、2125输入帧，截取后重启轨迹/窗口。</p>'
    for s in short_summaries:
        n=s['source']['number'];base=f'short-regression/new-{n:02}';identity=100+n
        short_html+=f'<section><h3>new-{n:02} 原片{s["source"]["source_offset"]}秒起 · 喝水窗 {s["baseline"]["actions"].get("drink",0)}→{s["candidate"]["actions"].get("drink",0)}</h3><button onclick="playPair({identity})">同时播放</button><div class="pair"><div>微调前<video id="b{identity}" controls preload="metadata" src="{base}/baseline.mp4"></video></div><div>微调后<video id="c{identity}" controls preload="metadata" src="{base}/candidate.mp4"></video></div></div></section>'
    short_html+=''.join(f'<article><h3>{html.escape(c["description"])}</h3><img loading="lazy" src="{c["image"]}"></article>'for c in short_cases)
    examples=''.join(f'<article><p>#{i:03}：{title}</p><a href="context-comparisons/{i:03}.jpg"><img loading="lazy" src="context-comparisons/{i:03}.jpg"></a></article>'for i,title in [(58,'话筒训练来源，拟合诊断'),(67,'话筒训练来源，拟合诊断'),(82,'测试手机，完整视频仍可能因另一个ROI放行'),(96,'测试手机：微调后新增瓶子误框'),(116,'测试相机'),(117,'测试相机')])
    label_gallery='<details><summary>标注审核图（绿框为人工辅助框，白框为有效审核区域）</summary>'+''.join(f'<a href="review/{f.name}"><img loading="lazy" src="review/{f.name}"></a>'for f in sorted((OUT/'review').glob('verified-*.jpg')))+'</details>'
    doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>物体困难负例微调对照</title><style>
body{font:16px/1.65 system-ui,sans-serif;max-width:1500px;margin:auto;padding:24px;background:#f5f6f8;color:#19212d}section,article{background:white;padding:20px;margin:24px 0;border:1px solid #d5dae1;border-radius:10px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}video{display:block;width:100%;background:#111}img{width:100%;height:auto}button{padding:8px 14px;cursor:pointer}a{color:#235eab}.note{padding:16px;background:#fff1d9}h1{font-size:28px}@media(max-width:850px){.pair{grid-template-columns:1fr}}
</style><h1>手机／相机／话筒标注微调与长视频对照</h1><p class="note">这是实验候选，未替换正式包。仍有手机误认瓶子的喝水误报；物体误框减少不等于正确识别物品，真实瓶子也存在漏检。</p><p><a href="analysis-report.md">完整报告</a> · <a href="REPRODUCE.md">标注与复现</a> · <a href="long-video-summary.csv">数值表</a> · <a href="models/MODEL_CARD.md">候选模型说明</a></p>'''+''.join(cards)+'<h2>全部保留／改变的交互触发组</h2>'+review+short_html+'<h2>同像素物体对照</h2>'+examples+'<h2>统计图</h2><img src="figures/01-errors.png"><img src="figures/02-selection-recall.png">'+'''<script>
function seekPair(n,t){for(const p of ['b','c']){const v=document.getElementById(p+n);v.pause();v.currentTime=t;}}
function playPair(n){const a=document.getElementById('b'+n),b=document.getElementById('c'+n);b.currentTime=a.currentTime;a.play();b.play();}
</script></html>'''
    doc=doc.replace('<h2>统计图</h2>',label_gallery+'<p><a href="object-hardneg-candidate-20261027.zip">候选模型与标注数据包（不是平台正式包）</a></p><h2>统计图</h2>')
    (OUT/'index.html').write_text(doc)


if __name__=='__main__':main()
