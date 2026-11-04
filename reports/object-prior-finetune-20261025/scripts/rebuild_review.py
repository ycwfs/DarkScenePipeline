"""Render final accepted labels from the frozen manifest, without reranking or renumbering."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from build_dataset import OUT, save


def main() -> None:
    path=OUT/'reviewed-labels.json'
    rows=[r for r in json.loads(path.read_text())['records'] if r['training_allowed']]
    dataset=json.loads((OUT/'dataset-manifest.json').read_text())['records']
    retained={r['id'] for r in dataset}
    folder=OUT/'annotations-final';folder.mkdir(exist_ok=True)
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',16)
    pages=[]
    for start in range(0,len(rows),12):
        sheet=Image.new('RGB',(1600,1290),(24,30,39));draw=ImageDraw.Draw(sheet)
        for j,r in enumerate(rows[start:start+12]):
            im=Image.open(r['image_view']).convert('RGB');scale=min(390/im.width,350/im.height)
            w,h=round(im.width*scale),round(im.height*scale);left=(400-w)//2;top=75+(350-h)//2
            x,y=j%4*400,j//4*430;sheet.paste(im.resize((w,h)),(x+left,y+top))
            draw.text((x+5,y+3),f'{r["review_kind"]}:{r["tile_index"]} {r["split"]}',font=font,fill='white')
            draw.text((x+5,y+26),Path(r['source']).name[:36],font=font,fill='white')
            draw.text((x+5,y+49),'进入数据集' if r['id']in retained else '重复展品限量排除',font=font,fill='#b6d4ee')
            for b in r['boxes']:
                q=np.asarray(b['bbox'])*np.array([w/im.width,h/im.height]*2)+np.array([x+left,y+top]*2)
                draw.rectangle(q.tolist(),outline='#ffe077',width=2)
                draw.text((q[0],q[1]),str(b['class_id']),font=font,fill='#ffe077')
        file=f'page-{start//12:03d}.jpg';sheet.save(folder/file,quality=94)
        pages.append(dict(page=file,ids=[r['id']for r in rows[start:start+12]]))
    save(folder/'manifest.json',dict(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        accepted_views=len(rows),pages=pages,note='Final accepted auxiliary labels; generated from frozen review, no model pseudo-labels.'))
    gallery='\n'.join(f'<a href="annotations-final/{p["page"]}"><img loading="lazy" width="1200" src="annotations-final/{p["page"]}" alt="标注页 {i+1}"></a>'for i,p in enumerate(pages))
    (OUT/'annotations.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>最终物体标注</title>'
        '<style>body{font-family:system-ui;margin:30px;background:#eef2f5;color:#142a3c}img{display:block;max-width:100%;height:auto;margin:20px 0}</style>'
        '<h1>407 个已接受物体标注视图</h1><p>0 瓶 · 1 杯 · 2 手机 · 3 相机 · 4 纸张 · 5 可移动小物 · 6 机器人/展品。'
        '这是助手视觉辅助标注，仍可人工修正；不是逐帧密集标注。无框视图是已审核背景。</p>'
        '<p>535 个历史来源按帧抽样审核，另 6 段旧 AVI 确认为空景并隔离。新 9 段不进入训练。'
        '<a href="index.html">返回实验结果</a></p>'+gallery+'</html>')
    print(len(rows),'accepted views;',len(pages),'pages')


if __name__=='__main__':
    main()
