"""Verify rendered videos and build a standalone local review page/contact sheet."""
import html
import json
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/evidence')
rows = json.loads((ROOT/'sample-results.json').read_text())
assert len(rows) == 10
font_path = '/repo/platform/dark-demo-v2/darkpipe/assets/NotoSansCJK-subset.ttf'
font = ImageFont.truetype(font_path, 24)
small = ImageFont.truetype(font_path, 17)
cell_height = 370
sheet = Image.new('RGB', (1000, 5*cell_height), '#121820')
draw = ImageDraw.Draw(sheet)
cards, media = [], []
labels = {'Drinking water':'喝水', 'Waving':'挥手', 'Picking up object':'捡拾物品',
          'Chasing':'追逐', 'Other':'其他'}
for i, row in enumerate(rows):
    video = ROOT/row['video']
    probe = json.loads(subprocess.run(['ffprobe','-v','error','-show_entries',
        'stream=codec_name,width,height,nb_frames,avg_frame_rate,duration','-of','json',str(video)],
        capture_output=True,text=True,check=True).stdout)['streams'][0]
    decode = subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],
                            capture_output=True,text=True,timeout=60)
    assert decode.returncode == 0, decode.stderr
    assert probe['codec_name'] == 'h264' and int(probe['nb_frames']) == row['frames']
    media.append(dict(file=row['video'], fully_decoded=True, **probe))
    stem = Path(row['source']).stem
    image_path = f"stills/{row['number']:02d}_{stem}_pose.jpg"
    im = Image.open(ROOT/image_path).convert('RGB')
    im.thumbnail((488, 296))
    x, y = (i % 2)*500+6, (i//2)*cell_height
    sheet.paste(im,(x,y+42))
    display_stem = stem.replace('捡东西', '捡拾物品')
    draw.text((x+5,y+4),f"{row['number']:02d}  {display_stem}",font=font,fill='#f3f7f9')
    result = ' / '.join(f"{labels.get(k,k)} {v}" for k,v in row['event_labels'].items())
    draw.text((x+5,y+339),f"events: {sum(row['event_labels'].values())}  pose frames: {row['frames_with_skeleton']}/{row['frames']}",
              font=small,fill='#bbc9cf')
    note = ('照度门关闭骨架分支，只显示动作结果。' if row['number']==9 else
            '最终结果为其他，不绘制动作人物。' if row['number']==10 else
            '快速移动时，跟随不可靠的骨架会暂时隐藏。' if row['number']==8 else
            '包含低于 0.75 后回到其他的时刻。' if 'Other' in row['event_labels'] else
            '绿线为骨架，黄点为关键点。')
    cards.append(f'''<article><h2>{row['number']:02d} · {html.escape(stem)}</h2>
<video controls playsinline preload="metadata" poster="{html.escape(image_path)}" src="{html.escape(row['video'])}"></video>
<p class="result">{html.escape(result)} <span>（事件数）</span></p>
<p>{row['duration']:.2f} 秒 · {row['frames']} 帧 · 有骨架 {row['frames_with_skeleton']} 帧</p>
<p>{note}</p><a href="{html.escape(row['video'])}" download>下载视频</a></article>''')
sheet.save(ROOT/'overview.jpg',quality=95)
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>动作骨架 · 10 个本地样本</title><style>
*{box-sizing:border-box}body{margin:0;background:#10161d;color:#edf3f5;font:16px/1.6 system-ui,sans-serif}
main{max-width:1280px;margin:auto;padding:36px 24px}h1{font-size:32px;margin:8px 0}h2{font-size:21px;margin:0 0 14px}
.intro{max-width:880px;color:#bac9d2}.tag{color:#82deae;font-size:14px;letter-spacing:1px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px;margin-top:30px}
article{background:#1b2530;border:1px solid #344250;border-radius:14px;padding:18px}video{display:block;width:100%;border-radius:8px;background:#000}
p{margin:8px 0;color:#bfccd5}.result{color:#9be4b6}span{color:#a2b0bc;font-size:14px}a{color:#9acaff}footer{margin-top:28px;color:#aabac6}
@media(max-width:700px){.grid{grid-template-columns:1fr}main{padding:20px 14px}h1{font-size:26px}}
</style><main><div class="tag">LOCAL PREVIEW · 2026-10-19</div><h1>动作骨架 · 10 个本地样本</h1>
<p class="intro">默认阈值 0.75；处理长边 720；照度门 75；RGB 与骨架融合权重 0.7。骨架只绘制在分支参与且与最终动作一致的候选人物上。视频开头约 1 秒积累识别窗口，此时没有骨架。</p>
<p class="intro">这是绘制功能预览，包含训练内样本，不能作为泛化精度或实时性能验收。骨架候选不等于可靠的人物身份定位。<a href="README.md">查看说明</a> · <a href="overview.jpg">截图总览</a></p>
<div class="grid">'''+''.join(cards)+'''</div><footer>10 段均为 H.264，保留原视频帧率与完整帧数。页面无外部依赖，解压预览包后直接打开 index.html 即可。</footer></main></html>'''
(ROOT/'index.html').write_text(page)
(ROOT/'media-verification.json').write_text(json.dumps(media,ensure_ascii=False,indent=2)+'\n')
print('verified',len(media),'videos;',sum(r['frames'] for r in rows),'frames; gallery and sheet saved')
