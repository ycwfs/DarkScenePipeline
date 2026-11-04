"""Verify every output frame and create a self-contained browser gallery."""
from collections import Counter
import html
import json
from pathlib import Path
import subprocess

from common import OUT

rows = json.loads((OUT / 'sample-results.json').read_text())
assert len(rows) == 6
media, cards = [], []
person_classes, selected_classes = Counter(), Counter()
for row in rows:
    path = OUT / row['video']
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-count_frames', '-show_entries', 'stream=codec_name,width,height,avg_frame_rate,nb_read_frames,duration',
        '-of', 'json', str(path)]))['streams'][0]
    decoded = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-f', 'null', '-'],
                             check=True, capture_output=True, text=True, timeout=120)
    assert not decoded.stderr.strip(), (path, decoded.stderr)
    assert probe['codec_name'] == 'h264' and int(probe['nb_read_frames']) == row['frames']
    assert [probe['width'], probe['height']] == row['output_shape']
    media.append(dict(video=row['video'], full_decode=True, **probe))
    events = json.loads((OUT / 'events' / f"{row['name']}.json").read_text())
    for event in events:
        person_classes.update(a['label'] for a in event['actor_poses'])
        selected_classes.update(a['label'] for a in event['actor_poses'] if a['selected_for_fusion'])
    seq = row['name'].rsplit('_', 1)[-1]
    clock = row['name'][9:15]
    clock = ':'.join([clock[:2], clock[2:4], clock[4:]])
    badge = ' · 你指定的片段' if row['requested'] else ''
    cards.append(f'''<article><h2>{clock} · #{seq}{badge}</h2>
<video controls loop preload="metadata" poster="{html.escape(row['still'])}" src="{html.escape(row['video'])}"></video>
<div class="buttons"><button onclick="this.closest('article').querySelector('video').playbackRate=0.5">0.5倍慢放</button>
<button onclick="this.closest('article').querySelector('video').playbackRate=1">原速</button>
<a href="{html.escape(row['video'])}" download>下载视频</a></div>
<p>{row['duration']:.2f}秒 · {row['frames']}帧 · {row['events']}次识别，底栏均为其他<br>
最多同时显示{row['max_people']}个骨架 · {row['frames_with_multiple_people']}帧含多人骨架</p>
<details><summary>源文件与逐事件结果</summary><p class="filename">{html.escape(row['source'])}</p>
<a href="events/{html.escape(row['name'])}.json">查看概率与选中人物</a></details></article>''')

page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>线上多人负样本 · 2026-10-23</title><style>
body{margin:0;background:#101923;color:#ecf3fa;font:16px/1.7 system-ui,sans-serif}main{max-width:1480px;margin:auto;padding:26px}
h1{font-size:30px;margin-bottom:8px}h2{font-size:22px;margin:0 0 12px}.intro{max-width:1100px;color:#cbd7e3}
.notice{padding:14px 18px;border-left:4px solid #f4d05b;background:#202b39;margin:20px 0}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px}article{padding:18px;background:#1b2735;border:1px solid #38495d;border-radius:10px}
video{width:100%;background:#000}a{color:#9bcaff}p{color:#cbd7e3}.buttons{display:flex;gap:12px;align-items:center;margin-top:12px}
button{background:#34475d;color:#fff;border:1px solid #60758a;border-radius:5px;padding:7px 12px;cursor:pointer;font:inherit}
.filename{overflow-wrap:anywhere;font-size:13px}details{color:#aebed0}summary{cursor:pointer}
@media(max-width:800px){.grid{grid-template-columns:1fr}main{padding:15px}}</style><main>
<h1>线上多人负样本 · 6段完整回放</h1>
<p class="intro">当前模型逐人显示骨架类别；黄色与“*”表示选中参与融合的骨架，绿色为其余有效骨架。
底栏使用0.7×RGB + 0.3×所选骨架，最终阈值0.75。6段共20次识别，底栏全部为“其他”。</p>
<div class="notice">当前每个识别窗口最多6个候选，拥挤画面中的所有人不一定都会显示。人形机器人和机器狗也出现了人体误检；
人物边上的喝水、挥手、握手等是骨架模型原始分类，仍可能错误，底栏为其他不表示逐人识别全对。</div>
<p class="intro">输入是已增强的线上录像：已裁掉旧底栏，不重复增强。原始照度无法还原，此处不复算照度门。
每段开头约1秒积累窗口，短片可选择0.5倍慢放。<a href="README.md">查看处理口径及结果说明</a></p>
<div class="grid">''' + ''.join(cards) + '</div></main></html>'
(OUT / 'index.html').write_text(page)
(OUT / 'media-verification.json').write_text(json.dumps(media, ensure_ascii=False, indent=2) + '\n')
summary = dict(videos=len(rows), frames=sum(r['frames'] for r in rows),
               events=sum(r['events'] for r in rows), fired_events=sum(r['fired_events'] for r in rows),
               frames_with_multiple_people=sum(r['frames_with_multiple_people'] for r in rows),
               max_displayed_people=max(r['max_people'] for r in rows),
               person_classes_per_event=dict(person_classes), selected_classes=dict(selected_classes),
               full_decode_passed=True, all_fusion_checks_passed=all(r['fusion_verified'] for r in rows),
               raw_luma_available=False)
(OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(summary, ensure_ascii=False, indent=2))
