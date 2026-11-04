"""Generate descriptive evidence report, class tables, figures and local video review."""
import csv
import html
import json
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/darkpipe-coco80-mpl')
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from common import OUT, save


def load(name):return json.loads((OUT/name).read_text())


def main():
    names=load('classes.json');selection=load('selection.json');chosen=selection['selected']
    a=load('original80-coco-test.json')['coco'];b=load('candidate80-coco-test.json')['coco']
    objects=load('object-evaluation.json')['summary'];old=load('cached-old-window-evaluation.json')['summary']
    robot=load('robot-probe.json');verify=load('verification.json');data=load('dataset-manifest.json')
    class_rows=[dict(class_id=c,name=names[c],original_ap=a['ap_by_class'][str(c)],candidate_ap=b['ap_by_class'][str(c)],
        delta_pp=100*(b['ap_by_class'][str(c)]-a['ap_by_class'][str(c)]))for c in range(80)]
    with (OUT/'coco80-per-class.csv').open('w',newline='')as f:
        writer=csv.DictWriter(f,fieldnames=list(class_rows[0]));writer.writeheader();writer.writerows(class_rows)
    videos=[]
    for scope in ['long-videos','short-regression']:
        for number in [1,7,9]:
            s=load(f'{scope}/new-{number:02}/summary.json');videos.append(dict(scope=scope,number=number,**s))
    long=[v for v in videos if v['scope']=='long-videos'];short=[v for v in videos if v['scope']=='short-regression']
    long_cases=load('long-review-cases.json');short_cases=load('short-review-cases.json')
    labels=load('review-labels.json');anchors=load('known-positive-audit.json')
    final=dict(classes=80,selected=dict(family=chosen['family'],epoch=chosen['epoch']),
        coco_test=dict(images=4000,original=a['metrics'],candidate=b['metrics']),class_rows=class_rows,
        object_summary=objects,old_window_regression=old,robot_probe=robot,
        videos=videos,review_labels=labels,known_positives=anchors,verification=verify,
        production_updated=False,recommendation='Experimental only; generic average largely retained, but class-specific/onsite recall and robot exclusion are not preserved.')
    save(OUT/'final-summary.json',final)
    figures=OUT/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    cs=[0,39,41,65,67,73];x=np.arange(7)
    av=[a['metrics']['metrics/mAP50-95(B)']]+[a['ap_by_class'][str(c)]for c in cs]
    bv=[b['metrics']['metrics/mAP50-95(B)']]+[b['ap_by_class'][str(c)]for c in cs]
    fig,ax=plt.subplots(figsize=(10,4.5));ax.bar(x-.18,np.array(av)*100,.36,label='Original COCO80',color='#5e84a4')
    ax.bar(x+.18,np.array(bv)*100,.36,label='Site-adapted COCO80',color='#de8e4a')
    ax.set_xticks(x,['All 80']+[names[c]for c in cs]);ax.set_ylabel('AP50:95 (%)');ax.set_ylim(0,75);ax.legend()
    ax.set_title('4,000 held-out COCO images: average retention hides class losses')
    for xx,aa,bb in zip(x,av,bv):ax.text(xx,max(aa,bb)*100+2,f'{(bb-aa)*100:+.2f} pp',ha='center',fontsize=9)
    fig.tight_layout();fig.savefig(figures/'01-generic-retention.png',dpi=170);fig.savefig(figures/'01-generic-retention.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    context=objects['deployment_roi_test']['models'];crop=objects['review_crop_test']['models']
    vals=[[context[k]['classes']['bottle']['fp']for k in ['current7','candidate80']],
          [crop[k]['classes']['bottle']['tp']for k in ['current7','candidate80']],
          [old['test']['models'][k]['true_actions_retained']for k in ['baseline','candidate']]]
    for j,(title,v)in enumerate(zip(['False bottle boxes\n32 deployment ROIs','True bottle hits\n16 reviewed boxes','True interaction windows\n15 old positive windows'],vals)):
        axes[0].bar(j-.17,v[0],.34,color='#5e84a4',label='Current seven-class'if j==0 else None)
        axes[0].bar(j+.17,v[1],.34,color='#de8e4a',label='Candidate COCO80'if j==0 else None)
        axes[0].text(j-.17,v[0]+.15,str(v[0]),ha='center');axes[0].text(j+.17,v[1]+.15,str(v[1]),ha='center')
    axes[0].set_xticks(range(3),['False bottle\n32 deployment ROIs','True bottle\n16 reviewed boxes','True interactions\n15 old windows'])
    axes[0].set_ylim(0,8);axes[0].set_ylabel('Count (different cohorts; do not sum)');axes[0].legend(fontsize=8)
    for family,color in [('broad','#ae5651'),('retention','#4e8567')]:
        rows=[r for r in selection['rows']if r['family']==family]
        axes[1].plot([r['epoch']for r in rows],[100*r['coco']['metrics']['metrics/mAP50-95(B)']for r in rows],label=family,color=color)
    baseline=selection['baseline']['coco']['metrics']['metrics/mAP50-95(B)']*100
    axes[1].axhline(baseline,color='#333',linestyle='--',label='Original validation')
    axes[1].axhline(baseline-1,color='#999',linestyle=':',label='Acceptance floor')
    axes[1].set_xlabel('Epoch');axes[1].set_ylabel('COCO1,000 validation AP50:95 (%)');axes[1].legend(fontsize=8)
    fig.suptitle('Fewer container false positives do not imply better action recognition')
    fig.tight_layout();fig.savefig(figures/'02-site-tradeoff-and-training.png',dpi=170);fig.savefig(figures/'02-site-tradeoff-and-training.pdf');plt.close(fig)
    table='| 原类别 | 原80类AP | 微调AP | 变化（百分点） |\n|---|---:|---:|---:|\n'
    table+=f"| 80类平均 | {av[0]*100:.2f} | {bv[0]*100:.2f} | {(bv[0]-av[0])*100:+.2f} |\n"
    for c in cs:
        r=class_rows[c];table+=f"| {r['name']} | {r['original_ap']*100:.2f} | {r['candidate_ap']*100:.2f} | {r['delta_pp']:+.2f} |\n"
    def video_table(rows):
        t='| 原视频 | 秒数 | 喝水窗 前→后 | 捡物窗 前→后 | 保存片段 前→后 |\n|---|---:|---:|---:|---:|\n'
        for r in rows:
            aa,bb=r['baseline'],r['candidate']
            t+=f"| {r['number']:02} | {r['seconds']:.2f} | {aa['actions'].get('drink',0)}→{bb['actions'].get('drink',0)} | {aa['actions'].get('pick_up',0)}→{bb['actions'].get('pick_up',0)} | {aa['clips']['clips_saved']}→{bb['clips']['clips_saved']} |\n"
        return t
    review_text='\n'.join(f"- {r['id']}: {r['assessment']}"for r in labels)
    report=f'''# 保留COCO80类的现场归并微调与验证

完成两组各20轮训练。当前实验候选保留原80类、原类别ID，不新增类别；相机/话筒按用户要求归cell phone，纸张归book，机器人/展品及无法明确对应的手持小物件作为背景。这些是项目扩展语义，不能称为COCO原生相机/话筒识别。

结论：通用平均指标大体保持，但手机等类别明显退化；现场瓶误框减少，真实瓶/旧动作召回下降；机器人仍被当人。因此保留实验候选，未替换正式算子或人体入口。

## 通用检测能力

2400张COCO训练回放覆盖80类（每类≥30框），1000张选模，剩余4000张仅锁定后测试。单seed，2组验证选择，不报告显著性。以下都是AP50–95。

{table}
74个未调整类别的分类权重、共享特征、定位参数与原模型逐张量相同；排序竞争仍可能引起少量AP变化。保留输出类别不等于每类性能无损。手机合并语义有实际代价，不能用平均下降小掩盖。这里按原COCO定义评分：项目要求识别成手机的相机/话筒，可能在原COCO真值中被计为手机假阳性；未逐例复审全部COCO预测，不能把AP下降全解释为真实手机漏检，也不能假定全部由语义差异造成。

## 现场物体与旧动作

固定物体阈值.581。32张新保留部署ROI中，当前7类→候选80类：瓶误框3→0，真实瓶0/4→0/4。原始未微调80类在这组也没有瓶误框，不能把相对7类的全部改善归因于本次训练。合并手机类正确0/12→1/12（命中的是话筒），另有手机类误框0→3，不能称普遍准确识别。
79张较紧审核裁剪：真瓶4/16→0/16，纸张/书本6/12→6/12，合并手机0/15→0/15；与部署ROI不是同一裁剪，不混合。

锁定后补充阈值诊断（未据此调参）：候选瓶阈值.35仍0/16，.1时3/16且2误框；当前7类在默认.581为4/16且1误框。降阈值不能在这组数据上自动取代原7类能力，详见bottle-threshold-diagnostic.json。这不是新阈值的完整动作流水线实验。

旧55个历史冻结动作候选仅辅助回归：test25窗真交互6/15→2/15，other误报1→1；val30窗真交互11/14→9/14，other误报4→5。它们使用历史动作候选，不是当前参数全量流水线成绩。当前7类与原COCO80起点不同，不能把所有差异归因于本次训练；报告同时保留原80类对照。

## 完整长视频

01/07/09共{sum(r['seconds']for r in long):.2f}秒、{sum(r['frames']for r in long)}原帧、{sum(r['paired_checks']for r in long)}配对人物窗。原镜像中正式增强/人体/跟踪/姿态/RGB/ST-GCN各执行一次，两个物体prior用同一任务，逐事件断言概率/帧/几何一致。阈值保持alpha=.972、门35、动作.925、物体.581。

{video_table(long)}
另复用之前三段短片的原输入，跟踪和窗口重新初始化，须分表，不与整段数字混合：

{video_table(short)}
已审核交互组：

{review_text}

已确认的01号149.92秒/T317真喝水：本轮融合分数.94423，超过动作阈值.925，但两版均未检测到兼容容器，最终仍为other。09号54.8秒附近另有人持瓶靠嘴，匹配到的相邻人物窗口已在动作阈值阶段变为other，物体先验未执行；这是辅助视觉观察，非新增独立动作标注。

所有非交互动作未全量人工标注；未触发时段也未穷尽审核，不报告整体准确率/动作召回率。用户此前“只有一例真喝水”的反馈范围是历史触发集合，不能扩展到全视频任意时刻。上述逐窗证据见known-positive-audit.json。

## 机器人限制

机器人标注已作为背景训练。但3号录像3张完整帧中，当前人体检测器、原80类和本次80类均3/3把机器人检测成人（人体阈值.35、评分匹配IoU.4）。物体先验位于动作计算之后，本轮双路对照没有替换上游人体检测器，所以“机器人不进入动作识别”尚未实现。没有通过固定坐标遮罩隐藏这一失败。

## 方法、验证与产物

现场717训练图/41val/79test；包含413物体图和304已清理展品的亮暗人体图，训练再混合COCO。审核训练框含瓶115、书本/纸张64、合并手机103、真人346；现场杯子仍无审核正例。现场训练补1288原模型保留预测，仅弱标签，与人工审核目标框分开记录，未用于测试GT。现场图重复3次，共4551训练条目；原图/增强图及重复不计独立样本。源/group/像素跨split检查通过，但现场视频已历史查看/调参，非独立盲测。

首组freeze10常规训练全部未过通用验证下降≤1pp门槛。随后在未看test时新增保守组：只训练原6类person/bottle/cup/remote/cell phone/book的最终分类行，冻结定位/特征/BN和其余74行，EMA保持冻结参数。固定验证规则选第{chosen['epoch']}轮；现场val AP由14.18到{chosen['score']*100:.2f}，不作为独立测试收益。

10图PT/ONNX一致性及80类权重继承/冻结核验通过。{verify['video_count']}个输出/片段视频完整FFmpeg解码通过，源帧数对齐、配对上游输入一致；片段写入失败/强制释放/未完成/丢帧均0。这是有限离线实验，不是实时吞吐、平台NFS或无限流稳定性验收。

正式模型、参数、平台包、Box包及原0.3.0镜像未更新。候选模型及映射适配器在本目录，需适配COCO ID，不能直接覆盖正式7类权重。完整视频/动作片段见index.html；复现见REPRODUCE.md。
'''
    (OUT/'analysis-report.md').write_text(report)
    (OUT/'stats-appendix.md').write_text('''# 统计口径

一随机种子，两种在验证集上比较的训练策略；各20epoch。候选按现场val AP最大、COCO1000 val AP下降≤.01约束选择；最终COCO4000测试只在锁定后打开。单seed不能估计训练随机性，不给均值±标准差或显著性p值。报告有限基准的确定性配对差值（百分点）和计数，不把相邻帧/窗口、原增强成对图当独立重复，也不伪造置信区间。

COCO官方标注按非crowd转换为YOLO80；本地缺一张训练图从另一完整同名副本修复，未删类。COCO val历史参与预训练模型开发，属于原任务能力保持回归。原80类与微调80类的名称/ID相同，但项目相机/话筒/纸张使用用户批准的扩展语义；现场按映射标签计分，COCO按原始标签计分。部分原任务FP可能是项目扩展语义预测，未全量逐例复审，不作单一因果解释。

现场是助手视觉辅助审核及既往接受的人体框，非独立专家GT；新01/07/09不训练，但早前看过/校准，且同场景人物有关联。现场val/test只覆盖审核目标类别，其他通用类未完整标注，不将其未匹配预测算为误报。原图与增强裁剪尺度不同的诊断必须分开。

完整视频共享上游概率/轨迹/ROI；对照当前7类与候选80类时，类别词表和对应先验类别集合适配了新语义，不是单一参数或单一数据因素的纯消融。辅助55旧窗沿用旧候选选择，不能并入当前视频精度。机器人探针只3个相关帧，失败足以否定已经排除，不足以估算全部场景错误率。
''')
    (OUT/'figure-catalog.md').write_text('''# 图表说明

01-generic-retention：4000张COCO锁定后测试；柱为原80类与候选AP50–95，文字为百分点差，无误差线。需要注意整体平均接近而手机/遥控器下降；不能据此称每类均保留。

02-site-tradeoff-and-training：左图是三个不同审核集合的计数，不能相加；瓶误框减少同时瓶和真动作召回下降。右图是两个训练策略的固定COCO验证曲线，说明首组被排除的依据；它不是独立test曲线，不宣称显著性。
''')
    cards=[]
    for r in videos:
        folder=f'{r["scope"]}/new-{r["number"]:02}'
        clips=[]
        for mode in ['baseline','candidate']:
            for f in sorted((OUT/folder/f'{mode}-clips').rglob('*.mp4')):
                clips.append(f'<li><a href="{html.escape(str(f.relative_to(OUT)))}">{mode}: {html.escape(f.name)}</a></li>')
        controls='<p><button onclick="const v=this.closest(\'section\').querySelectorAll(\'video\');const t=v[0].currentTime;v.forEach(x=>{x.currentTime=t;x.play()})">同步播放</button> <button onclick="this.closest(\'section\').querySelectorAll(\'video\').forEach(x=>x.pause())">暂停两侧</button></p>'
        anchors_for_video=[(c['start'],c['id'])for c in (long_cases if r['scope']=='long-videos' else short_cases)if c['number']==r['number']]
        if r['scope']=='long-videos'and r['number']==1:
            anchors_for_video.append((149.92,'已确认真喝水：两版均漏报'))
        if r['scope']=='long-videos'and r['number']==9:
            anchors_for_video.append((54.8,'持瓶靠嘴：动作阈值阶段未通过'))
        for timestamp,label in anchors_for_video:
            controls+=f'<p><button onclick="this.closest(\'section\').querySelectorAll(\'video\').forEach(x=>{{x.currentTime={max(0,timestamp-2):.2f};x.play()}})">{html.escape(label)}（{timestamp:.2f}秒）</button></p>'
        cards.append(f'<section><h2>{r["scope"]} / {r["number"]:02} — {r["seconds"]:.2f} 秒</h2>'+controls+'<div class="pair">'+
            ''.join(f'<div><h3>{title}</h3><video controls preload="metadata" src="{folder}/{mode}.mp4"></video><p><a href="{folder}/{mode}.mp4">打开完整视频</a></p></div>'for mode,title in [('baseline','当前7类'),('candidate','候选80类')])+f'</div><details><summary>自动保存片段</summary><ul>{"".join(clips)}</ul></details></section>')
    cases=''.join(f'<section><h3>{html.escape(c["id"])} — {html.escape(next((r["assessment"]for r in labels if r["id"]==c["id"]),"待审核"))}</h3><img src="{c["image"]}"></section>'for c in long_cases+short_cases)
    (OUT/'index.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>COCO80现场微调验证</title><style>body{max-width:1500px;margin:30px auto;padding:0 20px;font:17px/1.65 system-ui;background:#f3f5f7;color:#172330}section{background:white;padding:20px;margin:20px 0;border-radius:8px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}video,img{width:100%;height:auto}a{color:#1264ab}summary{cursor:pointer}h1,h2,h3{line-height:1.3}@media(max-width:800px){.pair{grid-template-columns:1fr}}</style>
<h1>保留COCO80类：现场归并微调对照</h1><p>相机/话筒归手机，纸张归书本，机器人作背景。候选保持80类；通用平均AP近似保持，但手机类别及现场真动作召回下降，机器人仍会进入人体检测，未替换正式交付。</p>
<p><a href="analysis-report.md">完整结果报告</a> · <a href="coco80-per-class.csv">80类逐类结果</a> · <a href="REPRODUCE.md">复现说明</a> · <a href="models/yolo26m_coco80_candidate.pt">候选PT</a> · <a href="models/yolo26m_coco80_candidate.onnx">候选ONNX</a></p>
<p>左右是同一上游输入的完整输出。先验只在通过动作阈值的喝水/捡物窗口按需运行，物体框不是每帧全画面检测；本次没有重测实时吞吐。</p>
<p>绿色框为独立人体跟踪结果。青色框为80类物体检测，可能包含重复的person框；青色T编号表示提供该裁剪的人物轨迹，不是每个青色框中人的身份。</p>
<h2>训练标签映射示例</h2><img src="mapping-examples.jpg">
<img src="figures/01-generic-retention.png"><img src="figures/02-site-tradeoff-and-training.png">'''+cases+''.join(cards)+
        '<h2>机器人入口检查</h2>'+''.join(f'<img src="{f.relative_to(OUT)}">'for f in sorted((OUT/'robot-review').glob('probe-*.jpg')))+
        '<h2>部署裁剪对照（原80类 / 当前7类 / 候选80类）</h2>'+''.join(f'<details><summary>{f.stem}</summary><img loading="lazy" src="{f.relative_to(OUT)}"></details>'for f in sorted((OUT/'comparisons').glob('*.jpg')))+ '</html>')
    (OUT/'README.md').write_text('''# COCO80现场语义归并微调

入口[index.html](index.html)，详细[analysis-report.md](analysis-report.md)。保持原80类输出，未更新正式包；通用平均指标接近原模型，但手机等类别下降、现场真动作召回下降、机器人排除未解决。

候选models/yolo26m_coco80_candidate.pt/.onnx；完整视频long-videos/，同输入短片short-regression/，片段在各自baseline-clips与candidate-clips；逐类CSV、标注与复现脚本均在本目录。原/增强成对图和重复条目不视为独立样本。
''')
    print('Report and review page assembled')


if __name__=='__main__':main()
