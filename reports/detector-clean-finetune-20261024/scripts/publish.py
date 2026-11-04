"""Produce exact comparison tables, descriptive uncertainty and review artifacts."""
from collections import defaultdict
import html
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/detector-clean-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from compare import OUT, ROOT, OLD, NAMES, TITLES, BRIGHT, save

DISPLAY = dict(zip(NAMES, ['原始模型', '上轮暗场微调', '本轮清理混合微调']))


def load(name: str) -> dict:
    return json.loads((OUT / name).read_text())


def main() -> None:
    evaluation = load('evaluation.json')
    exhibit = load('exhibit-audit.json')
    selection = load('selection.json')
    benchmark = load('benchmark.json')
    replay = load('replay-summary.json')
    protocol = load('protocol.json')
    metrics = evaluation['summary']
    epoch = int(Path(selection['selected']['path']).stem[5:]) + 1
    table = '| 检查项 | 原始模型 | 上轮暗场微调 | 本轮清理混合微调 |\n|---|---:|---:|---:|\n'
    for domain, title in [('dark', '暗场辅助框匹配'), ('bright', '亮场真人框匹配')]:
        table += f'| {title}（IoU≥0.5） | ' + ' | '.join(f'{metrics[domain]["0.5"][n]["matched"]}/{metrics[domain]["0.5"][n]["gt"]}' for n in NAMES) + ' |\n'
        table += f'| {title}（IoU≥0.3） | ' + ' | '.join(f'{metrics[domain]["0.3"][n]["matched"]}/{metrics[domain]["0.3"][n]["gt"]}' for n in NAMES) + ' |\n'
        table += f'| {title}（IoU≥0.7） | ' + ' | '.join(f'{metrics[domain]["0.7"][n]["matched"]}/{metrics[domain]["0.7"][n]["gt"]}' for n in NAMES) + ' |\n'
    table += '| 展品误框帧数（越少越好） | ' + ' | '.join(f'{exhibit["totals"][n]["detector_frames"]}/{exhibit["frames"]}' for n in NAMES) + ' |\n'
    table += '| 亮场核对帧未匹配框（含定位偏差） | ' + ' | '.join(str(metrics['bright']['0.5'][n]['unmatched']) for n in NAMES) + ' |\n'
    table += '| 暗场核对帧未匹配辅助框 | ' + ' | '.join(str(metrics['dark']['0.5'][n]['unmatched']) for n in NAMES) + ' |\n'
    table += '| 亮场真人关联≥10点骨架 | ' + ' | '.join(str(metrics['bright']['0.5'][n]['associated_pose10']) for n in NAMES) + ' |\n'
    table += '| 检测器耗时中位数（ms） | ' + ' | '.join(f'{benchmark["summary"][n]["median"]:.2f}' for n in NAMES) + ' |\n'
    (OUT / 'numeric-summary.md').write_text(table)
    figures = OUT / 'figures'
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    colors = ['#6f7f91', '#cf6c45', '#257c75']
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), constrained_layout=True)
    for ax, domain, title in zip(axes[:2], ['dark', 'bright'], ['Dark: 44 auxiliary boxes', 'Bright: 47 reviewed people']):
        values = [metrics[domain]['0.5'][n]['matched'] / metrics[domain]['0.5'][n]['gt'] * 100 for n in NAMES]
        ax.bar(range(3), values, color=colors, width=.65)
        ax.set(ylim=(0, 110), ylabel='Matched boxes (%)', title=title)
        ax.set_xticks(range(3), ['Original', 'Dark-only', 'Clean + mixed'])
        for i, (n, value) in enumerate(zip(NAMES, values)):
            m = metrics[domain]['0.5'][n]
            ax.text(i, value+2, f'{m["matched"]}/{m["gt"]}', ha='center')
    ax = axes[2]
    vals = [exhibit['totals'][n]['detector_frames'] for n in NAMES]
    ax.bar(range(3), vals, color=colors, width=.65)
    ax.set(ylim=(0, 1060), title='Exhibit: 955 fixed-camera frames', ylabel='False-positive frames (lower is better)')
    ax.set_xticks(range(3), ['Original', 'Dark-only', 'Clean + mixed'])
    for i, value in enumerate(vals):
        ax.text(i, value+15, str(value), ha='center')
    fig.suptitle('Same detector threshold 0.35; IoU 0.5; no runtime exhibit mask', fontsize=12)
    for ext in ('png', 'pdf'):
        fig.savefig(figures / f'01-regression.{ext}', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3.8), constrained_layout=True)
    xs = np.arange(1, 13)
    for domain, color in [('dark', '#435d8a'), ('bright', '#d2852e')]:
        ys = [r['metrics'][domain]['metrics/mAP50-95(B)'] for r in selection['rows'][2:]]
        ax.plot(xs, ys, 'o-', color=color, label=f'{domain.title()} validation')
    ax.plot(xs, [r['score'] for r in selection['rows'][2:]], 'o-', color='#257c75', label='Equal domain mean')
    ax.axhline(selection['rows'][0]['score'], color='#6f7f91', ls=':', label='Original: domain mean')
    ax.axhline(selection['rows'][1]['score'], color='#cf6c45', ls=':', label='Previous: domain mean')
    ax.axvline(epoch, color='#666666', ls='--', alpha=.6, label=f'Selected epoch {epoch}')
    ax.set(xlabel='Training epoch', ylabel='mAP50-95', xticks=xs, title='Checkpoint selection uses validation only; one-to-one head')
    ax.legend(ncol=2, fontsize=8, loc='best')
    for ext in ('png', 'pdf'):
        fig.savefig(figures / f'02-selection.{ext}', dpi=180)
    plt.close(fig)
    # Paired source-group bootstrap describes variability of THIS regression set.
    dark_groups = {s['video_id']: s['group'] for s in json.loads((OLD/'manifest.json').read_text())['samples']}
    bright_groups = {v['id']: v['frozen_split']['group'] for v in json.loads((BRIGHT/'protocol.json').read_text())['videos']}
    rng = np.random.default_rng(24)
    contrasts = []
    for domain in ('dark', 'bright'):
        grouped = defaultdict(list)
        for row in evaluation['audits']:
            if row['domain'] == domain:
                group = (dark_groups if domain == 'dark' else bright_groups)[row['video_id']]
                grouped[group].append(row)
        for base in ('original', 'previous'):
            arrays = np.array([[sum(r['by_iou']['0.5']['cleaned']['matched'] for r in rows),
                                sum(r['by_iou']['0.5'][base]['matched'] for r in rows),
                                sum(r['by_iou']['0.5']['cleaned']['gt'] for r in rows)] for rows in grouped.values()])
            draws = rng.integers(0, len(arrays), (10000, len(arrays)))
            summed = arrays[draws].sum(axis=1)
            changes = 100*(summed[:, 0]-summed[:, 1])/summed[:, 2]
            total = arrays.sum(axis=0)
            contrasts.append(dict(domain=domain, baseline=base, groups=len(arrays),
                change_percentage_points=float(100*(total[0]-total[1])/total[2]),
                descriptive_paired_group_bootstrap95=np.percentile(changes, [2.5, 97.5]).tolist()))
    save(OUT/'descriptive-contrasts.json', dict(contrasts=contrasts, resamples=10000, seed=24,
        caveat='Only source-group sampling variability; one training seed, auxiliary labels, same cameras. No population significance claims.'))
    change_table = '| 域 | 对照 | 来源组数 | 本轮匹配率变化（百分点） | 来源组重采样95%区间 |\n|---|---|---:|---:|---|\n'
    for c in contrasts:
        lo, hi = c['descriptive_paired_group_bootstrap95']
        change_table += f'| {c["domain"]} | {DISPLAY[c["baseline"]]} | {c["groups"]} | {c["change_percentage_points"]:+.2f} | [{lo:+.2f}, {hi:+.2f}] |\n'
    (OUT/'stats-appendix.md').write_text('# 统计口径与限制\n\n'+change_table+'''

匹配指标是同图、同阈值、同一标注的配对描述。训练只有seed24一个种子，不能估计训练随机性的标准差或置信区间，也不作显著性检验及p值声明。上表10,000次来源组配对bootstrap仅描述这批数据的组成变化，不包含标注误差、相机差异和随机训练方差；暗场只有5个有效标注来源组，区间很不稳定。

亮场41帧47个真人框是助手目视核对，另有一个遮挡忽略区域；暗场44帧44框是此前筛选过的辅助标签，非穷尽真值，未匹配框不可一律当误检。IoU0.3/0.7仅作框位置敏感性分析，不参与选模。展品955帧来自9段同相机视频，邻帧高度相关，不能按955个独立实验做二项检验。

四个匹配率对照均为探索性描述，不进行多重显著性筛选；没有基于显著性挑选结果。骨架关联≥10个可信点是覆盖指标，不是关键点定位精度。速度是41张图各重复5次、每模型205次，均值/标准差/中位/p95见benchmark.json；包含检测预后处理，不代表整链实时性。
''')
    (OUT/'figure-catalog.md').write_text('''# 图表说明

- `figures/01-regression.png/.pdf`：固定测试回归的三项结果，来源evaluation.json和exhibit-audit.json。前两项IoU≥0.5框匹配越高越好，第三项展品误框帧数越低越好。显示原始整数；无误差棒，因为不是重复训练均值。用于判断是否在抑制展品时损伤真人；辅助标签与同相机相关性限制必须保留。
- `figures/02-selection.png/.pdf`：12轮暗、亮验证mAP50-95及等权平均，来源selection.json；明确标出所选轮和两基线。用于检查选模是否兼顾两域。非test曲线；单种子，不代表重训稳定性。
- `posters/*.jpg`及`videos/*.mp4`：同增强帧上三检测器、同一固定RTMO姿态输出的对照；无平滑、无推理区域屏蔽。暗场强制展示姿态仅作诊断，不表示生产照度门已改变。P编号是本帧框序号，不能当身份追踪。
''')
    val_table = '| 模型 | 暗val mAP50-95 | 亮val mAP50-95 | 等权平均 |\n|---|---:|---:|---:|\n'
    for name, row in zip(NAMES, [selection['rows'][0], selection['rows'][1], selection['selected']]):
        val_table += f'| {DISPLAY[name]} | {row["metrics"]["dark"]["metrics/mAP50-95(B)"]:.4f} | {row["metrics"]["bright"]["metrics/mAP50-95(B)"]:.4f} | {row["score"]:.4f} |\n'
    differences = []
    for row in evaluation['audits']:
        m = row['by_iou']['0.5']
        if len({m[n]['matched'] for n in NAMES}) > 1:
            differences.append(f'| {row["source"]} F{row["frame"]} | '+ ' | '.join(f'{m[n]["matched"]}/{m[n]["gt"]}' for n in NAMES)+' |')
    (OUT/'per-frame-differences.md').write_text('| 帧 | 原始 | 上轮 | 本轮 |\n|---|---:|---:|---:|\n'+'\n'.join(differences)+'\n')
    details = f'''# 清理人体框标注后的检测器微调对比

本轮完成12轮单种子微调，按暗/亮验证mAP50-95等权平均选出第{epoch}轮；测试集不参与选轮。

**清理标注并补入亮场背景有效：本批展品误框从上轮944/955降到0/955；暗场IoU0.5匹配保持43/44，亮场从41/47恢复到45/47。** 本轮更适合作为后续逐人方案的检测候选，但还不能称为完全优于原始模型：亮场总数虽追平原模型，漏检对象并不完全相同；更严格IoU0.7下暗场33/44低于上轮36/44，亮场40/47低于原始41/47，框定位仍有代价。

{table}

## 数据和训练

此前暗场训练100个来源代表帧的目视检查未发现亮场固定展品大量混入，不能把前轮退化直接解释为“展品被大量当作正例”。本轮排除other238中人物/框不可靠的5帧，保留1,626个暗场原始帧、1,741个框；额外从冻结train补入72个亮帧、89个真人框。96张新增train/val图逐图核对，剔除原始模型草稿中的16个展品框和4个柱面人像框；上一轮候选在60张相同图中均提出展品框，均未加入标签。补标3个验证帧中远处坐着的真人。背景像素保留供模型学习，不作固定区域屏蔽。

训练使用raw与Retinexformer增强两视图；亮场8倍采样。3,396张唯一视图，4,404个加权训练条目，不能将重复项当新增独立样本。验证为暗69帧+亮24帧，按域等权选模。冻结split及来源组无交叉；旧数据和原标签未改。亮场回归是前轮已查看过的测试集，不是新的独立泛化验证。

从原始YOLO26m PT重新初始化；12轮、seed24、batch12、imgsz640、AdamW、LR1e-4→2e-5、freeze10、无mosaic。保持80类头，但本轮只训练/评估person，其他物体类别能力不作保证。显式nms=False评估部署one-to-one头。姿态、增强、动作权重与阈值固定。单RTX3090 GPU3。

## 验证选模

{val_table}

原始/前次候选同样进入验证比较，本轮候选独立保存；最终是否采用要结合真人漏检与展品误检，不能只看平均mAP。全部epoch分域分数见selection.json。

本轮验证等权mAP0.8778高于上轮0.7674，但原始模型仍为0.9064；严格按该验证指标总体推荐者仍是原始模型。因此这里只保存候选和对照，不自动换装。

## 回放与限制

41段亮场3,163帧+8段暗场722帧，全量回放共3,885帧。框匹配仅使用预先固定的41亮帧/47真人框、44暗帧/44辅助框；5个暗场来源有合格辅助框，另外3段只作无真值诊断。所有对照共用暗场微调RTMO，关联骨架数变化由检测框改变引起，不能当作新姿态模型精度。显示无平滑；暗场骨架仅诊断展示，生产照度门75未改。

《喝水26》F58中后方被展板遮挡的人，本轮仍会漏检。该区域在上一轮就因全身范围不确定列为ignore，不计入47个人体框分母；不能用表中匹配率掩盖这个可见问题。远处、遮挡和极暗姿态仍需逐片段查看。

具体看IoU0.5：other95为原始1/2→本轮2/2（原始在IoU0.3为2/2，主要是框位置差异）；挥手28从上轮1/3恢复3/3；挥手29从1/3恢复2/3，另一个框定位仍不够准；追逐25仍为1/2，原始是2/2。11段三栏视频共996帧，包含这些问题而非只展示成功案例。RTMO原始输出仍会给展品姿态，本轮显示减少来自检测框与骨架关联，并非姿态模型学会拒绝展品。未重测下游动作分类，不能把框指标当动作识别精度。

误检区域仅用于评估计数，不用于模型输入或后处理。此实验同时清理少量标注并补入亮场背景，不能视为“只删除展品框”的单因素消融。只有一个训练种子，辅助标注非独立人工GT；对新相机、新展品的泛化仍需额外验证。详见stats-appendix.md和per-frame-differences.md。

## 文件

- [对照视频与逐帧预览](index.html)
- [数值摘要](numeric-summary.md)、[逐帧差异](per-frame-differences.md)、[统计说明](stats-appendix.md)
- 模型：models/yolo26m_clean_candidate.pt、models/yolo26m_clean_candidate.onnx；哈希见model-manifest.json。
- 重现：scripts/build_dataset.py → scripts/train.py → scripts/select_export.py → scripts/compare.py → scripts/publish.py → scripts/verify.py。
- Python：项目.venv；训练额外PYTHONPATH=reports/dark-finetune-20261024/vendor-yolo；CUDA库环境与前轮一致，详见RUNBOOK.md。

当前仅生成独立候选和对比材料，生产权重、算子包、盒子包及镜像未替换。
'''
    (OUT/'analysis-report.md').write_text(details)
    (OUT/'README.md').write_text(details)
    cards = []
    for v in replay['videos']:
        if not v['video']:
            continue
        title = html.escape(Path(v['source']).name)
        cards.append(f'<article><h2>{title} · {"暗场" if v["domain"]=="dark" else "亮场"}</h2>'
            f'<video controls preload="metadata" poster="posters/{v["id"]}.jpg" src="{v["video"]}"></video>'
            f'<p>{v["processed_frames"]}帧 · {v["fps"]:g}fps · <a href="{v["video"]}" download>下载MP4</a></p></article>')
    gallery = ''.join(f'<figure><img loading="lazy" src="posters/{v["id"]}.jpg" alt="{html.escape(v["source"])}">'
                      f'<figcaption>{html.escape(v["source"])} · {v["domain"]}</figcaption></figure>' for v in replay['videos'])
    header = '<tr><th>检查项</th>' + ''.join(f'<th>{DISPLAY[n]}</th>' for n in NAMES) + '</tr>'
    lines = [line.strip('|').split('|') for line in table.splitlines()[2:]]
    htmltable = '<table>'+header+''.join('<tr>'+''.join(f'<td>{html.escape(c.strip())}</td>' for c in row)+'</tr>' for row in lines)+'</table>'
    (OUT/'index.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>清理人体标注后的检测器对比</title><style>body{margin:0;background:#f4f6f8;color:#172333;font:16px/1.6 system-ui,sans-serif}main{max-width:1320px;margin:auto;padding:24px}h1{font-size:30px}h2{font-size:20px}article,figure,.panel{background:white;border:1px solid #d9e1e8;border-radius:9px;padding:16px;margin:20px 0}video,img{width:100%;height:auto}table{width:100%;border-collapse:collapse;font-size:14px}td,th{text-align:left;padding:8px;border-bottom:1px solid #ddd}a{color:#126d78}.scroll{overflow-x:auto}figcaption{overflow-wrap:anywhere}.muted{color:#526170}</style><main>
<h1>清理人体框标注后的检测器对比</h1><p>左：原始模型　中：上轮暗场微调　右：本轮清理标注＋亮暗混合微调。</p>
<p class="muted">三栏共用同一个姿态模型；检测阈值0.35；无平滑、无推理展品屏蔽。暗场骨架仅诊断展示，照度门75未改。P是当前帧编号。</p>
<div class="panel scroll">'''+htmltable+'''</div><p><a href="README.md">完整说明</a> · <a href="analysis-report.md">分析报告</a> · <a href="stats-appendix.md">统计口径</a></p>
<div class="panel"><img src="figures/01-regression.png" alt="三模型回归指标"><img src="figures/02-selection.png" alt="验证集选模曲线"></div>
'''+''.join(cards)+'<h2>全部49段固定采样帧</h2>'+gallery+'</main></html>')
    print('Published', len(cards), 'videos; selected epoch', epoch)


if __name__ == '__main__':
    main()
