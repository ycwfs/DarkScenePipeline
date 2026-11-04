"""Browse independent per-person results and the actual crop sequences side by side."""
import html
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]
rows = json.loads((OUT/'sample-results.json').read_text())
cards = []
for row in rows:
    count = row['predictions']
    caption = ('线上已增强切片，裁掉旧底栏；原图亮度不可用。' if row['processed'] else
               '保留原图照度门；低于75时使用逐人RGB。')
    crops = ''.join(f'<a href="crops/{html.escape(p.name)}" target="_blank">{html.escape(p.stem.split("_T")[-1])}</a> '
                    for p in sorted((OUT/'crops').glob(row['name']+'_T*.jpg')))
    cards.append(f'''<article><h2>{html.escape(row['name'])}</h2>
<video controls loop preload="metadata" poster="{html.escape(row['still'])}" src="{html.escape(row['video'])}"></video>
<div class="tools"><button onclick="this.closest('article').querySelector('video').playbackRate=0.5">0.5倍慢放</button>
<button onclick="this.closest('article').querySelector('video').playbackRate=1">原速</button>
<a href="{html.escape(row['video'])}" download>下载</a></div>
<p>{caption}<br>{row['frames']}帧 · 最多{row['max_detected_people']}个检测候选 · {count}次逐人判断</p>
<details><summary>查看各人物实际输入裁剪（轨迹编号）</summary><p>{crops}</p>
<p>每张图为同一轨迹窗口内4个采样时刻，识别实际使用32帧；填边保持人体比例。</p></details>
<details><summary>分支与最终分类统计</summary><pre>{html.escape(json.dumps(dict(labels=row['labels'],modes=row['modes']),ensure_ascii=False,indent=2))}</pre>
<a href="events/{html.escape(row['name'])}.json">查看逐人概率与源帧</a></details></article>''')
page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>逐人动作识别 · RGB与骨架融合</title><style>
body{margin:0;background:#101923;color:#eef4fa;font:16px/1.65 system-ui,sans-serif}main{max-width:1480px;margin:auto;padding:28px}
h1{font-size:30px;margin:0 0 12px}h2{font-size:23px;margin:0 0 12px}p{color:#c2d0dd}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}
article{padding:18px;background:#1b2735;border:1px solid #38495d;border-radius:10px}video{width:100%;background:#000}a{color:#a5cfff}
.notice{padding:16px;background:#272b33;border-left:4px solid #f7c866;margin:20px 0}.tools{display:flex;gap:12px;align-items:center;margin-top:10px}
button{background:#35495f;color:white;border:1px solid #60758a;border-radius:5px;padding:7px 12px;font:inherit;cursor:pointer}
summary{cursor:pointer;color:#b9cbe1}details{margin:12px 0}pre{white-space:pre-wrap;font-size:13px}
@media(max-width:800px){.grid{grid-template-columns:1fr}main{padding:15px}}</style><main>
<h1>每个人的动作，由自己的视频与骨架共同判断</h1>
<p>人体检测与短期跟踪 → 逐人视频裁剪 → 该人的RGB分类 + 该人的骨架分类 → 独立融合。
每个人的框下显示融合后的类别；已移除全画面的结果底栏。</p>
<p>绿框：其他（只显示文字）；橙框：达到阈值的动作及概率；灰框：识别中。低于阈值的动作统一显示“其他”。
T编号是短期跟踪编号。新出现的人约1秒后首次分类，
拥挤位置的文字通过连线对应人物框。</p>
<div class="notice">本轮是独立演示，沿用现有模型，尚未用逐人裁剪数据训练。挥手样例中可见未挥手者被判为喝水，
拥挤负样本也有逐人误报；人形展品和背景仍可能被当作人。结果按模型原样展示。</div>
<p>α=0.7，最终阈值0.75，原图照度门75。每段可展开查看实际裁剪输入与逐人概率。
<a href="README.md">完整说明与验证</a></p><div class="grid">''' + ''.join(cards) + '</div></main></html>'
(OUT/'index.html').write_text(page)
print(f'Gallery: {len(rows)} videos')
