"""Offline, portable visual brightness review page and unmodified-pixel contact sheets."""
import csv
import html
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from build_samples import ROOT, OUT, FONT, BINS, save, sha


def build_contact(samples: list, path: Path, key: str) -> None:
    canvas = Image.new('RGB', (1440, 2010), '#171b1b')
    d = ImageDraw.Draw(canvas)
    title = '原图亮度分档 · 每档3个真实现场样本' if key == 'raw' else '相同样本增强后 · 分档仍按原图L'
    d.text((20, 15), title, font=ImageFont.truetype(FONT, 30), fill='#f4f0e5')
    d.text((20, 61), 'L是原图全画面平均亮度，含原有黑边；图片仅等比缩小，未额外调亮。',
           font=ImageFont.truetype(FONT, 19), fill='#b9c8c1')
    for i, row in enumerate(samples):
        x, y = (i % 3)*480, 112 + (i//3)*312
        image = Image.open(OUT/row['files'][key]).convert('RGB')
        image.thumbnail((480, 270))
        canvas.paste(image, (x+(480-image.width)//2, y+(270-image.height)//2))
        d.text((x+10, y+275), f'{row["sample_id"]}  L={row["poster_luma"]:.2f}  {Path(row["source"]).name}',
               font=ImageFont.truetype(FONT, 20), fill='#f4f0e5')
    canvas.save(path)


def main() -> None:
    samples = json.loads((OUT/'samples.json').read_text())['samples']
    protocol = json.loads((OUT/'protocol.json').read_text())
    assert len(samples) == 18
    build_contact(samples, OUT/'overview-raw.jpg', 'raw')
    build_contact(samples, OUT/'overview-enhanced.jpg', 'enhanced')
    card_groups = []
    for bi, (lo, hi) in enumerate(BINS):
        band = f'{lo} ≤ L < {hi}' if hi<256 else 'L ≥ 75'
        cards = []
        for row in (r for r in samples if r['bin_index'] == bi):
            key = row['sample_id']
            note = ' '.join(row['notes'])
            if row['crossing_thresholds']:
                note += ' 本片段部分帧跨过'+ '/'.join(str(t) for t in row['crossing_thresholds'])+'，分档依据预览关键帧。'
            title = html.escape(Path(row['source']).name)
            files = row['files']
            cards.append(f'''<article id="sample-{key}" data-sample="{key}">
<div class="meta"><span class="sample-number">S{key}</span><h3>{title}</h3>
<p class="luma-value">{row['poster_luma']:.2f}<small>预览帧原图 L</small></p>
<dl><div><dt>片段均值</dt><dd>{row['mean_luma']:.2f}</dd></div>
<div><dt>最暗 — 最亮</dt><dd>{row['min_luma']:.2f} — {row['max_luma']:.2f}</dd></div>
<div><dt>源视频时间</dt><dd>{row['source_start_seconds']:.2f} — {row['source_end_seconds']:.2f}s</dd></div>
<div><dt>预览帧</dt><dd>F{row['poster_frame']:04d}</dd></div></dl>
<p class="gate-state" aria-live="polite"></p><p class="sample-note">{html.escape(note)}</p></div>
<div class="media"><a href="{files['poster']}" target="_blank" class="poster-link" title="打开完整对照图"><img loading="lazy" width="1440" height="498" src="{files['poster']}" alt="{title}，左侧原图右侧增强，原图亮度{row['poster_luma']:.2f}"></a>
<details><summary>播放同帧对照视频 · {row['seconds']:.2f} 秒</summary><video controls preload="none" playsinline poster="{files['poster']}" src="{files['video']}" aria-label="{title}对照视频"></video></details>
<div class="links"><a href="{files['raw']}" target="_blank">原图 PNG</a><a href="{files['enhanced']}" target="_blank">增强 PNG</a><a href="{files['video']}" download>下载视频</a></div></div></article>''')
        count = protocol['bins'][bi]
        card_groups.append(f'<section data-bin="{bi}" id="band-{bi}"><div class="band-heading"><h2>{band}</h2><p>全库 {count["frames"]:,} 帧落在此区间 · 此处展示3段</p></div>'+''.join(cards)+'</section>')
    buttons = '<button type="button" class="selected" data-filter="all" aria-pressed="true">全部18段</button>'
    for bi, (lo, hi) in enumerate(BINS):
        title = f'{lo}–{hi}' if hi<256 else '≥75'
        buttons += f'<button type="button" data-filter="{bi}" aria-pressed="false">{title}</button>'
    thresholds = ''.join(f'<button type="button" data-threshold="{t}" aria-pressed="{str(t==75).lower()}" class="{"selected" if t==75 else ""}">{t if t else "关闭"}</button>' for t in (75,60,45,30,15,0))
    browser_rows = [{k:r[k] for k in ('sample_id','bin_index','min_luma','max_luma','mean_luma')} for r in samples]
    (OUT/'index.html').write_text('''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>原图亮度样本册 · DarkScene</title><link rel="stylesheet" href="style.css"></head>
<body><main><header><div><p class="eyebrow">DARKSCENE / 现场样本册 / 2026.10.24</p><h1>先看亮度，<br>再选分流点。</h1><p class="intro">18段真实现场视频，按原图平均亮度分为6档。<br>每张对照图：左边原图，右边同帧增强。</p></div>
<aside><span class="large-number">18<span>段</span></span><p>全部来自训练划分原片<br>无人工压暗 · 无额外调色</p><a href="luma-samples-review.zip" download>下载完整预览包 ↗</a><a href="overview-raw.jpg" target="_blank">打开18张原图总览 ↗</a></aside></header>
<div class="reading-note"><strong>看左边选亮度，看右边判断增强后是否看得清。</strong><p>L 为0–255像素亮度，不是 lux。它统计全画面，包括黑边、墙面和屏幕；同一个L下，人体可见程度也可能不同。图片可点开放大，视频可展开播放。</p></div>
<div class="controls"><div><label>显示亮度档位</label><nav aria-label="筛选亮度区间">'''+buttons+'''</nav></div>
<div><label>预览门限 <span id="threshold-label">T = 75</span></label><div class="thresholds" role="group" aria-label="预览亮度门限">'''+thresholds+'''</div></div>
<p class="control-note">门限按钮仅标记片段与T的关系，不修改实际算子。实际分流还取决于识别窗口和骨架质量；“关闭”不是亮度为0的档位。</p><p id="visible-count" aria-live="polite">正在显示18段</p></div>
'''+''.join(card_groups)+'''<footer><h2>边界附近的数据覆盖</h2><p>现有413段去重原片、42,193帧。75～100区间只有《喝水18》和《挥手21》两段、118帧；本页选了其中一段。70以上至75附近及75上下的覆盖稀疏，不能假装每个阈值附近都有充足样本。</p>
<p>这里只帮助你选择亮度范围，未展示人体检测或骨架预测，也未改变默认门限75。确定候选门限后，再核对动作误报、漏报和速度。</p>
<a href="README.md">样本清单与说明</a> · <a href="samples.csv" download>下载CSV</a> · <a href="overview-enhanced.jpg" target="_blank">增强图总览</a></footer></main>
<script id="sample-data" type="application/json">'''+json.dumps(browser_rows, ensure_ascii=False).replace('<','\\u003c')+'''</script><script src="review.js"></script></body></html>''')
    fields = ['sample_id','source','split','poster_frame','poster_luma','mean_luma','min_luma','max_luma','seconds','clip_frames','source_start_seconds','source_end_seconds']
    with (OUT/'samples.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in samples)
    table = '| 编号 | 原图亮度档 | 视频 | 预览帧L | 片段均值L | 帧内范围 |\n|---|---|---|---:|---:|---|\n'
    for r in samples:
        lo,hi=BINS[r['bin_index']];band=f'{lo}–{hi}' if hi<256 else '≥75'
        table+=f'| {r["sample_id"]} | {band} | {Path(r["source"]).name} | {r["poster_luma"]:.2f} | {r["mean_luma"]:.2f} | {r["min_luma"]:.2f}–{r["max_luma"]:.2f} |\n'
    (OUT/'README.md').write_text('''# 原图亮度分档样本

已准备18段真实现场样本，每档3段，均来自冻结train划分。只展示亮度与同帧增强，不训练、调参或更换模型。

- [打开本地交互预览](index.html)：按区间筛选、切换预览门限、点图放大、展开播放视频。
- [18张原图总览](overview-raw.jpg)、[同帧增强总览](overview-enhanced.jpg)。
- [完整可搬运预览包](luma-samples-review.zip)：解压后直接打开index.html，所有图和视频均在包内。
- [精确样本表](samples.csv)、[全库亮度分布及协议](protocol.json)、[逐帧数据](samples.json)。

'''+table+'''

分档按预览关键帧原图L，视频片段保留原fps，最多4秒；少量片段跨区间，页面显示最小/最大值和提示。L为原始帧按长边720等比缩小（宽高取偶数）后`0.114 B均值+0.587 G均值+0.299 R均值`，与算法定义一致。不是lux；不裁黑边、不做人为压暗、不按增强图分档。PNG保留缩小后的实际像素；MP4为H264有损预览，精确L按编码前原始像素计算。

《喝水8》原片自带左右黑边，低L部分来自黑边，不能将它等同于无黑边画面的照明条件。《捡东西6》有展台遮挡，同样说明全局亮度不代表人体可见性。75～100只有两段原片共118帧（喝水18、挥手21），边界75附近覆盖不足。本页门限按钮只是视觉参考，关闭不是L=0；窗口平均分流与逐帧亮度也不完全相同。

增强固定使用原NTIRE.pth，没有人体框、姿态或动作预测，避免将画面可见程度误当识别准确率。选取依据是亮度与场景多样性，不是识别成功与否。用户选定候选门限后再验证动作表现；当前默认门限仍75。

复现：项目.venv，CUDA及cudnn/cublas/cuda_runtime库路径沿用现有项目；先检查GPU占用。`scripts/build_samples.py`生成帧与视频，`scripts/publish.py`生成页面/总览/ZIP，`scripts/verify.py`核验图像及浏览器播放。来源、增强模型及各产物SHA256在protocol.json、samples.json和verification.json。
''')
    paths = [OUT/n for n in ['index.html','style.css','review.js','README.md','overview-raw.jpg','overview-enhanced.jpg','samples.csv','samples.json','protocol.json']]
    for row in samples:
        paths.extend(OUT/name for name in row['files'].values())
    archive = OUT/'luma-samples-review.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as z:
        for path in paths:
            z.write(path, path.relative_to(OUT))
    save(OUT/'bundle.json', dict(path=archive.name, bytes=archive.stat().st_size, sha256=sha(archive), members=len(paths)))
    print('Published 18 samples;', sum(r['clip_frames'] for r in samples), 'frames;', round(archive.stat().st_size/2**20,1), 'MiB ZIP')


if __name__ == '__main__':
    main()
