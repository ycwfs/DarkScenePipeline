"""Build a portable review UI and an explicitly unlabelled dataset staging index."""
from html import escape
from pathlib import Path
import json

from collect import OUT, NAMES, read, save, sha


def main():
    groups=read(OUT/'review-items.json');sources=read(OUT/'sources.json');summary=read(OUT/'summary.json')
    manifest=sha(OUT/'review-items.json')
    data=dict(items=groups,sources=sources,names=NAMES,summary=summary,manifest=manifest)
    for g in groups:
        p=OUT/'render-metadata'/f'{g["id"]}.json'
        if g['category']=='videomamba' and p.exists():
            m=read(p)
            g['media_start']=m.get('source_start_time',m['start_frame']/25)
    template=(OUT/'scripts/review.html').read_text()
    packed=json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    (OUT/'index.html').write_text(template.replace('__DATA__',packed))
    style='<meta charset="utf-8"><style>body{font:16px/1.7 system-ui;margin:30px;background:#f4f7f9;color:#17283a}a{color:#1266a8}table{border-collapse:collapse;background:white}td,th{padding:8px 14px;border:1px solid #d7e1e8}video{max-width:95%;width:960px}article{background:white;padding:18px;margin:16px 0}small{color:#667}</style>'
    clips=read(OUT/'original-clips.json')
    lines=[style,'<title>原始454个动作片段</title><h1>初次推理的全部454个保存片段</h1><p><a href="index.html">返回逐人审核</a> · 此处保留原始模型叠字；标签尚未人工确认。</p><table><tr><th>编号</th><th>原片/人物</th><th>预测</th><th>原片秒</th><th>播放</th><th>对应审核编号</th></tr>']
    for c in clips:
        lines.append(f'<tr><td>{c["id"]}</td><td>{c["number"]:02d} / {escape(c["identity"])}</td><td>{c["label_zh"]}</td><td>{c["start"]:.2f}–{c["end"]:.2f}</td><td><a href="{c["video"]}" target="_blank">播放原保存片段</a></td><td>{", ".join(c["review_ids"])}</td></tr>')
    (OUT/'original-clips.html').write_text(''.join(lines)+'</table>')
    lines=[style,'<title>九段完整原片</title><h1>九段完整原片</h1><a href="index.html">返回审核</a><p>仅将 AVI 无损封装为浏览器可播放的 MP4，没有重编码或增强。</p>']
    for s in sources:
        lines.append(f'<article><h2>原片 {s["number"]:02d} · {s["duration"]:.2f} 秒</h2><p>{escape(Path(s["absolute_path"]).name)}</p><video controls preload="none" src="sources/{s["number"]:02d}.mp4"></video></article>')
    (OUT/'sources.html').write_text(''.join(lines))
    annotations=dict(schema_version=1,manifest_sha256=manifest,annotations=[])
    save('annotations-empty.json',annotations)
    split_groups={1:'recording_20261005_1300_multiview',2:'recording_20261005_1300_multiview',3:'recording_20261005_1300_multiview',
                  4:'recording_20261001',6:'recording_20261001',8:'recording_20261001',
                  5:'recording_20261016_14xx_multiview',7:'recording_20261016_14xx_multiview',9:'recording_20261016_14xx_multiview'}
    save('dataset-staging.json',dict(schema_version=1,status='awaiting_user_review',training_ready=False,
        model_training_started=False,review_manifest_sha256=manifest,reviewed_positive_samples=0,
        reviewed_other_samples=0,other_candidate_windows=summary['other_candidates'],
        source_windows='base-windows.jsonl',review_items='review-items.json',negative_pool='other-candidates.csv',
        split_status='unassigned; group by recording session/date across views, not random overlapping clips',
        recording_groups=split_groups,
        constraints=['Human action labels must come from review; model predictions are not GT.',
                    'Robot/exhibit/wrong-person detections are excluded from action training, not labelled human other.',
                    'Not-drink feedback is not a blanket other label; label actual actions individually.',
                    'XCLIP proposals are a separate optional review pool; use matched original people for future training.',
                    'RGB crops and skeleton sequences must use the same person and original source frames.',
                    'Preview pixels contain overlays and resampling: never use them as training input.',
                    'All nine recordings have already influenced threshold calibration; future holdout is not blind.']))
    rows='\n'.join(f'| {k} | {v} |'for k,v in summary['by_label'].items())
    source_rows='\n'.join(f'| {k} | {sum(v.values())} | '+ '、'.join(f'{a}{b}'for a,b in v.items())+' |'for k,v in summary['by_source'].items())
    text=f'''# 10-25 多人现场动作审核（2026-10-29）

主入口：[逐人视频审核页](index.html)。共 **{summary['main_review']}条 VideoMamba＋骨架候选**，覆盖原始994次非other人物窗、**全部454个自动保存文件**。新生成原图上下文＋黄色人物框＋同人放大短片；原来的保存文件也全部保留。

| 历史预测 | 审核记录 |
|---|---:|
{rows}

| 原片 | 审核条数 | 预测分布 |
|---|---:|---|
{source_rows}

## 无桌面Ubuntu／VS Code Remote访问

HTML Preview插件的Webview可能不能加载本地MP4。推荐通过HTTP服务和VS Code端口转发，用本机Chrome或Edge打开；服务器无需桌面，也无需下载全部视频。

在远程终端进入本目录，服务未运行时执行：

```bash
python3 scripts/serve.py --port 8766
```

VS Code底部“端口／PORTS”面板 → “转发端口” → 输入8766 → 点击该行“在浏览器中打开”，在**本机外部浏览器**访问 `http://127.0.0.1:8766/index.html`。如VS Code分配了其他本地端口，以“本地地址”为准。找不到面板时，命令面板搜索“Ports: Focus on Ports View”。

服务仅监听服务器127.0.0.1，支持MP4 Range请求和原片拖动定位，无额外Python依赖。保持服务进程及Remote SSH连接运行即可；前台服务用Ctrl+C停止。不要在HTML Preview或Simple Browser中打开视频。

标注保存在**本机浏览器**，请使用“导出审核JSON”备份。插件、本地文件和HTTP地址的浏览器存储相互独立；若之前已有标签，请先导出JSON再在新页面导入。

## 如何审核

1. 打开 index.html，默认是主审核473条。按动作或原片筛选，查看黄色框及右侧同人放大。
2. 真正发生预测动作，点“预测正确”；没有目标动作，点“应为其他”；其他真实动作请选择正确类别。机器人、人形展品、错框或换人点“非真人／人物框错”；看不清点“不确定”。
3. 动作起止使用**原片秒数**。初值是触发窗口覆盖范围，不是已验证的动作边界；前后上下文不自动标为动作。可暂停并用当前时间设定开始和结束。
4. 浏览器自动保存，完成一部分即可“导出审核 JSON”；JSON 可以重新导入继续。也可编辑 review.csv，或直接反馈“审核编号＋正确动作／其他／排除＋必要的起止秒数”。
5. 一条只标黄色框对应人物；同画面中的其他人物需要自己的标签。T/G 是本次轨迹编号，非永久身份。

首次可从先前真实喝水开始：搜索 **M01-014**（原片01 / 149.92秒附近）；页面会提示既往反馈。该提示不代替本轮的逐人起止审核。

## 汇总范围与去重

核对{summary['historical_runs']}份完整离线推理／选定阈值回放记录，含10-25全部9段、后续物体模型/挥手/跑动实验、正式包与Box演示复核，以及独立XCLIP实验。清单和原始日志SHA见runs.json。未把搜索网格中数百万个未选参数都作为新预测；中断诊断和拼接实时/故障重复录像不用于精确原片标注，见excluded-runs.json。

主审核按原片＋人物轨迹/代次＋预测动作分组，触发间隔≤1.2秒、每组触发跨度≤10秒。不同人物不会仅因时间重叠而合并。不同实验的人物ID不直接等同；只有时间差≤0.28秒、框IoU≥0.75且竞争身份有≥0.10间隔才作候选关联，关系保留供人工查验。

主审核473条均包含最初低阈值推理触发；后续已完成离线实验没有新增独立动作位置。454是保存文件数，473是审核候选数，差异主要来自保存器把较近的触发保存在一个文件、而审核按更短触发间隔拆分。对应关系见original-clips.csv；未删除任何原始片段。

**XCLIP补充页有{summary['xclip_review']}条候选**，是未采用的独立零样本实验，默认不混入主审核。补充条目直接定位同目录sources中的原片，通过浏览器绘制目标框和同人放大；无需反复存储相同原片。不能把这些预测直接当标签。

审核短片保留原始光照，约12.5fps（源03为24.999/2fps），播放时长不加速；黄色框只显示附近±0.60秒的同轨迹历史框，无可靠附近框时不画。没有重新运行模型或人工补造骨架。完整原片MP4只无损换封装，原文件/模型/算子包均未修改。

## 数据集准备状态

已保留47,600个原始人物窗及其源帧号、人物/代次、原始人体框。**{summary['other_candidates']}个other窗口是待用负样本候选，不是确认标签**；其余{summary['held_windows']}窗暂挂审核。为防止候选人物关联误差污染负样本，这里保守排除了所有人非other候选的时间区间及前后1秒（包括XCLIP候选）。这只是初始候选池，审核后可以恢复更多负样本。

人工标签目前为空，training_ready=false，没有启动训练。审核确认的动作可形成正样本，误报中的真人可形成困难other负样本；机器人/展品与不确定条目排除。RGB与骨架必须在相同人物、同一原始时间窗上提取，**不要使用带框/文字的审核视频作为训练输入**。

后续按采集日期/时段把跨视角原片绑定成组：01/02/03，04/06/08，05/07/09；不随机拆分重叠片段。具体train/val/test尚未分配。这9段已经参与过阈值校准，后续留出成绩也不能称全新盲测。

## 文件

- index.html：可筛选、播放、保存/导出标签的审核页；sources.html：全部9段原片。
- clips/、posters/：473条主审核短片和封面；original-clips/：454个历史保存文件的本地硬链接。
- review.csv：可编辑空白审核表；review-items.json：完整候选及可追溯映射；observations.json：7049条跨实验非other记录（含重复）。
- original-clips.csv / original-clips.html：454个历史文件索引和播放入口。
- base-windows.jsonl、other-candidates.csv、dataset-staging.json：待标注数据底稿，未自动认定为真值。
- scripts/collect.py、render.py、media.py、build.py：汇总与复现；verify.py：视频/完整性核验；import_labels.py：验证并整理审核JSON或CSV。

下载portable-review.zip后**先完整解压，再打开index.html**；支持离线使用。审核进度在浏览器，请导出JSON另存。所有路径均相对，视频可随文件夹搬运。
'''
    (OUT/'README.md').write_text(text)
    print(json.dumps(dict(review_manifest_sha256=manifest,main=summary['main_review'],xclip=summary['xclip_review']),ensure_ascii=False))


if __name__=='__main__':main()
