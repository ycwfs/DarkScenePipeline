"""Decode-check previews and build an offline gallery without external assets."""
import html
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
groups = [('', '默认照度门75'), ('default-multi', '默认照度门75：含人形展品'),
          ('gate-off-demo', '双人功能展示：照度门关闭')]
rows, media = [], []
for subdir, title in groups:
    base = ROOT / subdir
    for row in json.loads((base / 'sample-results.json').read_text()):
        video = base / row['video']
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-count_frames', '-show_entries', 'stream=codec_name,width,height,nb_read_frames,duration',
            '-of', 'json', str(video)]))['streams'][0]
        subprocess.run(['ffmpeg', '-v', 'error', '-i', str(video), '-f', 'null', '-'],
                       check=True, capture_output=True, timeout=120)
        assert probe['codec_name'] == 'h264' and int(probe['nb_read_frames']) == row['frames']
        row = dict(row, group=title, subdir=subdir, video=str(video.relative_to(ROOT)),
                   still=str((base/'stills'/f"{row['name']}.jpg").relative_to(ROOT)))
        rows.append(row)
        media.append(dict(video=row['video'], full_decode=True, **probe))

order = [('gate-off-demo', '挥手3'), ('gate-off-demo', '挥手4'),
         ('default-multi', '追逐24'), ('default-multi', '追逐25'), ('', '喝水16'), ('', 'other80')]
cards = []
for group, name in order:
    row = next(r for r in rows if r['subdir']==group and r['name']==name)
    note = ('本片源图低于默认亮度门，仅此展示显式设lum_gate=0；左上与右下为真实同框两人。'
            if group=='gate-off-demo' else '左侧固定人形展品被RTMO当成人，本片展示真实模型局限，不能把它计作真实人员。'
            if group=='default-multi' else '人物边上为骨架模型自己的类别，底栏为融合后的结果，二者可以不同。')
    cards.append(f'''<article><h2>{html.escape(name)}</h2><p class="mode">{html.escape(row['group'])}</p>
<video controls preload="metadata" poster="{html.escape(row['still'])}" src="{html.escape(row['video'])}"></video>
<p>{note}</p><p>{row['frames']}帧 · 同时显示多个骨架 {row['frames_with_multiple_people']}帧 · 全部{row['events']}条事件融合公式核验通过</p>
<a href="{html.escape(row['video'])}" download>下载视频</a></article>''')
all_links=''.join(f'<li><a href="{html.escape(r["video"])}">{html.escape(r["name"])} · {html.escape(r["group"])}</a></li>' for r in rows)
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>多人骨架与动作 · 2026-10-22</title><style>
body{margin:0;background:#10171f;color:#e8eef3;font:16px/1.65 system-ui,sans-serif}main{max-width:1400px;margin:auto;padding:28px}
h1{font-size:30px}h2{margin:0;font-size:22px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px}
article{padding:18px;background:#1b2632;border:1px solid #3a4858;border-radius:10px}video{width:100%;background:black}
p{color:#c4d0dc}.mode{color:#ffda6a}a{color:#9acaff}ul{columns:2}@media(max-width:750px){.grid{grid-template-columns:1fr}ul{columns:1}}
</style><main><h1>多人骨架与动作 · 2026-10-22</h1>
<p>逐人显示骨架分类；黄色骨架与“*”表示最高分人物，其完整概率向量参与 0.7×RGB + 0.3×骨架融合。
绿色表示其他有效人物。人物标签含“其他”，底栏使用融合概率与0.75阈值。</p>
<p>视频开头约1秒积累窗口。关闭照度门的两段展示与默认75的样例已分开标注；生产默认值保持75。
模型未重训，类别、姿态和展品误检仍可能出错。<a href="README.md">查看完整说明</a></p><div class="grid">'''+''.join(cards)+'''</div><h2>全部测试视频</h2><ul>'''+all_links+'''</ul></main></html>'''
(ROOT/'index.html').write_text(page)
(ROOT/'media-verification.json').write_text(json.dumps(media,ensure_ascii=False,indent=2)+'\n')
(ROOT/'combined-summary.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(videos=len(rows),frames=sum(r['frames'] for r in rows),
    events=sum(r['events'] for r in rows),multiperson_frames=sum(r['frames_with_multiple_people'] for r in rows)),indent=2))
