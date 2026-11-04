"""Assemble descriptive counts, paired review videos and source-level caveats."""
from collections import Counter
import csv
import html
import json
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/wave-prior-mpl')
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from common import OUT, save


def load(name):return json.loads((OUT/name).read_text())


def main():
    verification=load('verification.json');assert verification['passed']
    cases=load('wave-cases.json');labels={r['id']:r for r in load('review-labels.json')}
    assert all(c['id']in labels for c in cases),'Unreviewed wave groups remain'
    assert all(labels[c['id']].get('reviewed_end',float('-inf'))>=c['end'] for c in cases if c['scope']!='old-videos'), 'Review does not cover the whole group'
    old=load('old-window-summary.json');videos=[]
    for scope in ['long-videos','short-regression','wave-shorts','old-videos']:
        for f in sorted((OUT/scope).glob('*/summary.json')):
            videos.append(dict(scope=scope,directory=str(f.parent.relative_to(OUT)),**json.loads(f.read_text())))
    grid=[]
    for scope in ['long-videos','short-regression','wave-shorts','old-videos']:
        cc=[c for c in cases if c['scope']==scope]
        for setting in ['baseline','1_total','3_total','5_total','8_total','3_consecutive','5_consecutive','8_consecutive']:
            sums=Counter()
            for c in cc:
                count=c['baseline_windows']if setting=='baseline'else c['variants'][setting]
                sums[labels[c['id']]['truth']]+=count
                sums['total']+=count
            grid.append(dict(scope=scope,setting=setting,**{k:sums[k]for k in ['wave','not_wave','uncertain','total']}))
    with (OUT/'wave-grid.csv').open('w',newline='')as f:
        writer=csv.DictWriter(f,fieldnames=list(grid[0]));writer.writeheader();writer.writerows(grid)
    save(OUT/'final-summary.json',dict(primary_min_frames=3,object_model='official7',old=old,
        videos=videos,reviewed_grid=grid,labels=list(labels.values()),verification=verification,
        production_updated=False,integrated_candidate=load('integrated-candidate-check.json'),
        integrated_bright_candidate=load('integrated-bright-candidate-check.json')))
    figures=OUT/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.right':False,'axes.spines.top':False})
    current=[r for r in videos if r['scope']=='long-videos']
    fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    xx=np.arange(len(current))
    for shift,variant,title,color in [(-.18,'baseline','Current pipeline','#5d83a3'),(.18,'candidate','3-frame pose prior','#d08d44')]:
        values=[r[variant]['actions'].get('wave',0)for r in current]
        axes[0].bar(xx+shift,values,.36,label=title,color=color)
        for x,y in zip(xx+shift,values):axes[0].text(x,y+.3,str(y),ha='center')
    axes[0].set_xticks(xx,[f"Video {r['source']['number']:02}"for r in current]);axes[0].set_ylabel('Wave windows (not all are false alarms)');axes[0].legend(fontsize=8)
    old_all=old['summary']['onsite']['all']
    keys=['baseline','3_total','5_total','8_total','8_consecutive']
    axes[1].plot(range(len(keys)),[old_all[k]['true_wave_retained']for k in keys],'o-',color='#437961',label='True waves retained / 142')
    axes[1].set_xticks(range(len(keys)),['Baseline','3 total','5 total','8 total','8 consecutive'],rotation=15)
    axes[1].set_ylim(135,143);axes[1].set_ylabel('Historical annotated wave windows');axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(figures/'01-wave-tradeoff.png',dpi=170);fig.savefig(figures/'01-wave-tradeoff.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.4));settings=['baseline','1_total','3_total','5_total','8_total','3_consecutive','5_consecutive','8_consecutive']
    rr=[next(r for r in grid if r['scope']=='long-videos'and r['setting']==s)for s in settings];bottom=np.zeros(len(rr))
    for key,label,color in [('not_wave','Reviewed false wave','#ad645b'),('wave','Reviewed true wave','#4b8167'),('uncertain','Uncertain gesture','#b9a461')]:
        yy=np.array([r[key]for r in rr]);ax.bar(range(len(rr)),yy,bottom=bottom,label=label,color=color);bottom+=yy
    ax.set_xticks(range(len(rr)),['Base','1 total','3 total','5 total','8 total','3 consec.','5 consec.','8 consec.'],rotation=15)
    ax.set_ylabel('Remaining wave windows in three long videos');ax.legend();fig.tight_layout()
    fig.savefig(figures/'02-count-grid.png',dpi=170);fig.savefig(figures/'02-count-grid.pdf');plt.close(fig)
    table='| 集合 | 时长（秒） | 挥手窗口 原→3帧 | 保存片段 原→3帧 |\n|---|---:|---:|---:|\n'
    for scope in ['long-videos','short-regression','wave-shorts','old-videos']:
        for r in [v for v in videos if v['scope']==scope]:
            name=Path(r['source'].get('source',r['directory'])).name
            table+=f"| {scope}/{name} | {r['seconds']:.2f} | {r['baseline']['actions'].get('wave',0)}→{r['candidate']['actions'].get('wave',0)} | {r['baseline']['clips']['clips_saved']}→{r['candidate']['clips']['clips_saved']} |\n"
    oldtable='| 条件 | 旧现场真挥手保留/142 | 旧现场已知误挥手 | 新长片已审误挥手 | 新长片待确认 |\n|---|---:|---:|---:|---:|\n'
    for setting in settings:
        oo=old_all[setting];gg=next(r for r in grid if r['scope']=='long-videos'and r['setting']==setting)
        oldtable+=f"| {setting} | {oo['true_wave_retained']} | {oo['false_wave']} | {gg['not_wave']} | {gg['uncertain']} |\n"
    new_base=next(r for r in grid if r['scope']=='long-videos'and r['setting']=='baseline')
    new_after=next(r for r in grid if r['scope']=='long-videos'and r['setting']=='3_total')
    false_groups = sum(c['scope']=='long-videos' and labels[c['id']]['truth']=='not_wave' for c in cases)
    removed_false_groups = sum(c['scope']=='long-videos' and labels[c['id']]['truth']=='not_wave' and c['candidate_windows']==0 for c in cases)
    report=f'''# 挥手：同一腕高于双肩的骨架先验

沿用当前正式现场7类物体检测器，不使用80类归并或8类话筒候选。所有六模型及alpha=.972、原图门35、动作阈值.925、物体阈值.581、720增强裁剪保持。新增的是挥手的几何条件，未重新训练模型。

建议先采用用户提出的**同一只腕、累计至少3个不同源帧**。3帧是本轮事前主方案；1/5/8帧及连续帧为同证据诊断，不把这组小样本称为跨场景最佳值。旧val各档均保留6/9真挥手、已知误挥手0，缺乏支持提高帧数的验证证据；旧全量3/5/8累计都保持139/142，连续8帧降为138/142。

本轮由助手对三段长片的原图时序逐组复核，判为非挥手的窗口：{new_base['not_wave']}→{new_after['not_wave']}（{false_groups}个相邻触发组中{removed_false_groups}组过滤）；另有待确认窗口{new_base['uncertain']}→{new_after['uncertain']}，不算作纠错或真实召回。逐组依据见review-labels.json与审核页，仍可由用户纠正标注。邻近窗口强相关，窗口数不是独立事件次数。

## 规则及实际入口

- 在现有1秒、32采样帧的同一人物窗口里，分别统计左腕和右腕；不将两只手或不同人物的帧凑数。
- 图像y轴向下，要求某腕y严格小于两肩y的较小值。腕与双肩三个关键点置信度均至少.5，坐标有效；源帧ID去重。
- 原图L<35、原图亮度未知、明确关闭骨架分支时跳过。亮场若姿态缺失/遮挡导致证据不足，则由wave改为other，可能增加漏检。
- 只过滤已通过动作阈值的wave，不主动制造wave；不改变RGB/ST-GCN概率，不影响喝水/捡物物体先验。显示、事件和保存均使用过滤后的label。
- 不增加姿态/动作模型前向、不补点或平滑、不额外等待未来帧。25fps下3帧约为0.12秒帧量，首尾间隔通常0.08秒；实际帧率不同，持续时间不同。

隔离候选已真正接入ActionEngine、PipelineConfig和CLI/清单：`--wave_pose_min_frames 3`，0关闭，范围0–32。代码在operator/dark-demo-v2，ckpts本机链接到正式权重；这不是已更新的平台上传归档。未替换正式包/Box/镜像。

## 长短视频

{table}

完整新01/07/09沿用原完整输入；原三段短片重启跟踪和窗口相位，单独统计。旧7段选择依据既有标注和原结果，包含val/test/train；暗场挥手跳过先验，不能计为新条件验证通过。所有对照共享完全相同的增强/人体/跟踪/姿态/RGB/ST-GCN和7类物体结果，只改变挥手决策。

另外补充4段各12秒的挥手触发附近原视频，包括儿童摆臂、低手位候选、站立候选和抬手手势；完整短片均重跑。这些是看到长片触发后选取的针对性诊断，同时展示保留和过滤例，不视为独立样本或选参依据。

## 帧数对照

{oldtable}

total是在同一秒内同一只腕累计；consecutive为相邻可靠观测连续满足，间隔超过120ms断开。除3累计外，其余只做计数回放，未分别重跑整段编码或声称实时最优。

旧现场及历史线上共535源、4234窗的冻结缓存保持原来源拆分；239个原挥手窗逐帧匹配同一个track和精确检测框，7648个含重复采样的姿态输入完成核验。旧现场3个已知误挥手仍在：other225有2窗原L<35，规则跳过；喝水34有1窗腕持续高于双肩（21帧），符合几何条件。举手、喝水、摸头等仍可能通过，手低于肩的真实招手也可能被过滤。

139个被保留的旧真挥手窗口中，80个因低亮跳过，59个实际经过几何条件并通过。3累计过滤的另一个旧窗口缺少可靠真值，不能算作已纠正误报。未新增已知真挥手损失仅适用于这批回归资料，不代表所有姿态遮挡场景都不受影响。

旧train/test和新长片均历史查看过，本轮为工程/场景回归；未标注的时段、漏检人物、未触发动作未穷尽审核，不报告整体准确率或全场景动作召回率。

## 验证

17项规则回归通过；参数/清单合约通过。原镜像下实际候选service.run_offline处理《挥手3》，22事件的人物/源帧/最终类别/先验与配对实现一致，概率差在2e-4内。另跑12秒亮场站立短片，86事件的人物/帧/类别/先验均一致，实际否决2个wave窗口且不保存挥手片段。

亮场独立加载模型的RGB概率最大差0.004866、融合概率最大差0.004729，骨架概率差0；原先2e-4数值复现断言未通过，日志保留，不能称逐概率复现成功。后续单独核验确认全部动作类别、原动作类别和先验证据完全相同。主实验的前后对照共享同次上游结果，概率逐项完全一致；这一跨次概率差不混入主实验效果比较。

{verification['source_count']}组配对输入，{verification['video_count']}个输出/片段全部完整FFmpeg解码通过；片段写入失败、强制释放、未完成、丢帧、弃段均0。原视频结束时缺少未来后卷是正常输入边界。有限离线测试，不是实时吞吐、平台NFS或无限流稳定性验收。

完整对比入口index.html；实现implementation/wave_prior.py；协议protocol.json；复现REPRODUCE.md；正式交付未变的哈希检查production-preserved.json。
'''
    (OUT/'analysis-report.md').write_text(report)
    (OUT/'stats-appendix.md').write_text('''# 统计口径

固定模型的一次确定性规则对照，没有新训练/随机种子重复。主规则3累计在看结果前冻结；旧验证集各档同分，1/5/8及连续条件只作诊断，不用新长片或历史test隐式选择更好的阈值。

报告明确的配对计数和差值，不把相邻窗口/采样帧/重启的短片当独立重复；仅3个长视频、同一现场人物高度相关，无法从本轮估计跨现场不确定性，不给伪精确p值或独立样本置信区间。人工审核由助手辅助完成，儿童摆臂等不确定项单列；旧逐人标注继承历史审核结果。长短视频重叠、不相加计算独立误报率。

长片FP/TP只针对已经触发wave的审核子集，无法估计未触发的真实挥手或未检出人物；规则只会删除wave，不能修复原有漏报。旧缓存是真实同轨迹关键点和固定概率回放，不是本轮重新执行全部模型。主输出及保存另有真实完整推理验证。
''')
    (OUT/'figure-catalog.md').write_text('''# 图表说明

01-wave-tradeoff：左图为三长片挥手触发量前后对比，触发减少不自动等于误报减少；右图为旧已标注真挥手保留数，连续8帧造成额外损失。数据见final-summary.json；无误差线，描述性计数。

02-count-grid：相同长片证据下，不同帧数条件保留的已审误挥手/已审真挥手/待确认窗口。必须把待确认与错误分开；不能用更少触发量直接宣布更优。只有3累计有实际完整输出，其余是同证据决策回放。
''')
    cards=[]
    for r in videos:
        folder=r['directory'];cc=[c for c in cases if c['directory']==folder]
        title=html.escape(r['source'].get('source',folder))
        buttons='<button onclick="const v=this.closest(\'section\').querySelectorAll(\'video\');v[1].currentTime=v[0].currentTime;v.forEach(x=>x.play())">同步播放</button> <button onclick="this.closest(\'section\').querySelectorAll(\'video\').forEach(x=>x.pause())">暂停</button>'
        jumps=' '.join(f'<button onclick="this.closest(\'section\').querySelectorAll(\'video\').forEach(x=>{{x.currentTime={max(0,c["start"]-1.5):.2f};x.play()}})">T{c["track_id"]} {c["start"]:.2f}秒</button>'for c in cc)
        clips=''.join(f'<li><a href="{html.escape(str(f.relative_to(OUT)))}">{mode}: {html.escape(f.name)}</a></li>'for mode in ['baseline','candidate']for f in sorted((OUT/folder/f'{mode}-clips').rglob('*.mp4')))
        cards.append(f'<section><h2>{title}</h2><p>{r["seconds"]:.2f}秒；挥手窗 {r["baseline"]["actions"].get("wave",0)} → {r["candidate"]["actions"].get("wave",0)}</p><p>{buttons}</p><p>{jumps}</p><div class="pair">'+''.join(f'<div><h3>{label}</h3><video controls preload="metadata" src="{folder}/{mode}.mp4"></video></div>'for mode,label in [('baseline','原方案'),('candidate','增加3帧骨架先验')])+f'</div><details><summary>自动保存片段</summary><ul>{clips}</ul></details></section>')
    reviews=''.join(f'<details><summary>{html.escape(c["id"])} — {html.escape(labels[c["id"]]["assessment"])}</summary><img loading="lazy" src="{c["image"]}">'+(f'<img loading="lazy" src="wave-review/{c["id"]}-context.jpg">'if (OUT/'wave-review'/f'{c["id"]}-context.jpg').exists()else '')+'</details>'for c in cases)
    (OUT/'index.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>挥手骨架先验对照</title><style>body{font:17px/1.6 system-ui;background:#f3f5f7;color:#172330;max-width:1500px;margin:30px auto;padding:0 20px}section,details{background:white;padding:16px;margin:16px 0;border-radius:8px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}video,img{width:100%}a{color:#1264ab}button,summary{cursor:pointer}button{margin:4px;padding:7px}h1,h2,h3{line-height:1.35}@media(max-width:800px){.pair{grid-template-columns:1fr}}</style><h1>挥手：同一腕高于双肩，累计至少3帧</h1><p>物体检测沿用正式7类；六模型及原四参数不变。仅在亮度门35以上检查同一人的可靠骨架，源帧去重。静止举手/喝水可能通过，不确定动作单列。候选已实现，正式包未替换。</p><p><a href="analysis-report.md">详细结果</a> · <a href="wave-grid.csv">帧数对照</a> · <a href="REPRODUCE.md">复现</a> · <a href="operator/dark-demo-v2/README.md">候选入口及参数</a></p><p>下面均为完整配对输出。低亮旧片会跳过先验；同输入短片与完整长片不能重复计为独立样本。</p><img src="figures/01-wave-tradeoff.png"><img src="figures/02-count-grid.png">'''+''.join(cards)+'<h2>全部挥手组审核（保留、过滤、待确认均展示）</h2>'+reviews+'</html>')
    (OUT/'README.md').write_text('''# 挥手骨架先验验证

[视频对照](index.html)、[详细报告](analysis-report.md)、[复现](REPRODUCE.md)。沿用正式7类物体模型，主条件为同一腕高于双肩至少3个实际源帧，关键点置信度至少.5，原照度门35约束。

隔离候选operator/dark-demo-v2已接入真实ActionEngine与参数wave_pose_min_frames（默认3，0关闭）。正式平台/Box/镜像未替换；候选权重目录仅本机链接。长短视频、动作片段、全部wave审核、旧窗口回归和1/3/5/8帧诊断均保留。不要把减少的所有触发都叫作纠正误报。
''')
    print('Report, figures and complete review page generated')


if __name__=='__main__':main()
