"""Create a standalone video gallery and portable preview bundle (no training data)."""
import html
import json
from pathlib import Path
from urllib.parse import quote
import zipfile

import cv2

from scan import OUT, save
from build_review import gamma, overlay


def main() -> None:
    catalog=json.loads((OUT/'catalog.json').read_text());cards=[]
    posters=OUT/'posters';posters.mkdir(exist_ok=True)
    for v in catalog['videos']:
        d=json.loads((OUT/'data'/v['id']/'annotations.json').read_text())
        f=d['frames'][len(d['frames'])//2]
        image=overlay(gamma(cv2.imread(str(OUT/f['image']))),f,d['video'])
        image=cv2.resize(image,(480,round(image.shape[0]*480/image.shape[1])))
        poster=posters/(v['id']+'.jpg');cv2.imwrite(str(poster),image,[cv2.IMWRITE_JPEG_QUALITY,88])
        cards.append(f'<article data-name="{html.escape(v["source"],quote=True)}"><h2>{html.escape(v["source"])}</h2>'
            f'<p>{v["dark_frames"]} 暗帧 · {v["no_box_frames"]} 无框帧 · {v["id"][:8]}</p>'
            f'<video controls preload="none" poster="{quote(poster.relative_to(OUT).as_posix())}" src="{quote(v["video"])}"></video>'
            f'<p><a href="{quote(v["video"])}" download>下载本段</a>　<span>报告问题请提供文件名＋底栏 F 帧号＋人物 ID。</span></p></article>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>暗场人体与骨架 · 对照视频</title>
<style>body{margin:0;background:#f4f3ee;color:#192c30;font:14px "Noto Sans CJK SC","Microsoft YaHei",sans-serif}header{padding:28px 4vw;border-top:5px solid #116b5d;border-bottom:1px solid #cad7ce}h1{font-size:27px;font-weight:500}header p{line-height:1.9;max-width:1050px}input{font:inherit;padding:10px;border:1px solid #cbd5ce;width:min(500px,85vw)}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(480px,100%),1fr));gap:24px;padding:26px 4vw}article{background:#fffefa;border:1px solid #d8ded7;padding:18px}h2{font-size:15px;overflow-wrap:anywhere;font-weight:500}p{color:#62716f;font-size:12px}video{width:100%;background:#152027}a{color:#116b5d}article[hidden]{display:none}</style>
<header><h1>暗场人体框与骨架 · 167 段对照视频</h1><p>左侧原图，右侧 Gamma 提亮及待审核标注。全部 19,892 个 L&lt;75 暗帧纳入；视频也保留亮帧上下文。无框不等于无人，缺少骨架点不代表该关节不存在。仅有两个示例框获用户确认；其他标注尚待审核，暂未训练。</p><input id="search" placeholder="搜索视频文件名或目录" aria-label="搜索视频"><span id="count"></span></header><main>'''+''.join(cards)+'''</main><script>const input=document.getElementById('search');input.oninput=()=>{let n=0;for(const c of document.querySelectorAll('article')){c.hidden=!c.dataset.name.toLowerCase().includes(input.value.toLowerCase());if(!c.hidden)n++;}document.getElementById('count').textContent='　'+n+' 段';};input.oninput();document.querySelectorAll('video').forEach(v=>v.addEventListener('play',()=>document.querySelectorAll('video').forEach(o=>{if(o!==v)o.pause()})));</script></html>'''
    (OUT/'videos.html').write_text(page)
    info='视频审核预览包：解压后打开 videos.html。包含全部 167 段原图/标注对照视频；标注仍为待审核草稿，不能作为已确认训练真值。\n请用“源文件名＋F帧号或时间＋人物ID＋问题”反馈。完整可编辑审核页位于原工作目录的 index.html，依赖 data/，不包含在这个视频预览包中。\n'
    (OUT/'视频预览说明.txt').write_text(info)
    archive=OUT/'dark-annotations-video-review.zip'
    paths=[OUT/'videos.html',OUT/'视频预览说明.txt',OUT/'review-index.csv',OUT/'summary.json']
    paths+=sorted((OUT/'videos').glob('*.mp4'))+sorted(posters.glob('*.jpg'))
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        for path in paths:z.write(path,path.relative_to(OUT).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert len([n for n in z.namelist() if n.endswith('.mp4')])==167
    save(OUT/'preview-bundle.json',dict(path=archive.name,bytes=archive.stat().st_size,videos=167,
        zip_crc_verified=True,contains_editable_frame_assets=False))
    print(archive,archive.stat().st_size)


if __name__=='__main__':main()
