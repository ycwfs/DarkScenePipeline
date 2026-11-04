"""Show unreviewed proposal geometry and IDs, retaining direct links to raw pixels."""
import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parents[1]
COLORS=['#ff4444','#44dd55','#55ccff','#ff66ff','#ffff55','#ff9900','#ff9966']


def main():
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',16)
    dest=OUT/'draft-review';dest.mkdir(exist_ok=True)
    index=[]
    for path in sorted((OUT/'drafts').glob('*/proposals.json')):
        source=json.loads(path.read_text());views=source['records']
        tile_w,tile_h=400,420
        sheet=Image.new('RGB',(tile_w*min(4,len(views)),tile_h*((len(views)+3)//4)+48),(25,25,25))
        draw=ImageDraw.Draw(sheet)
        draw.text((8,6),f'{source["video"]["source"]} | {source["split"]} | 未审核框草稿',font=font,fill='white')
        for i,row in enumerate(views):
            im=Image.open(path.parent/row['image']).convert('RGB')
            scale=min((tile_w-10)/im.width,(tile_h-55)/im.height)
            nw,nh=round(im.width*scale),round(im.height*scale)
            x,y=(i%4)*tile_w,(i//4)*tile_h+48
            left,top=x+(tile_w-nw)//2,y+30
            sheet.paste(im.resize((nw,nh)),(left,top))
            draw.text((x+5,y),f'f{row["frame"]} {row["view"]} gamma{row["brightness_gamma"]}',font=font,fill='white')
            for j,p in enumerate(row['proposals']):
                box=[left+p['bbox'][0]*scale,top+p['bbox'][1]*scale,left+p['bbox'][2]*scale,top+p['bbox'][3]*scale]
                color=COLORS[p['class_id']]
                draw.rectangle(box,outline=color,width=2)
                text=f'{j}:{p["class_name"]} {p["confidence"]:.2f}'
                tx,ty=box[0],max(top,box[1]-17)
                draw.rectangle([tx,ty,tx+len(text)*8+2,ty+18],fill='black')
                draw.text((tx,ty),text,font=font,fill=color)
        filename=f'{source["video"]["id"]}.jpg';sheet.save(dest/filename,quality=94)
        index.append(dict(video_id=source['video']['id'],source=source['video']['source'],split=source['split'],
                          image=f'draft-review/{filename}',views=len(views),proposals=sum(len(v['proposals'])for v in views)))
    (OUT/'draft-review-index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n')
    print('Prepared',len(index),'source draft sheets')


if __name__=='__main__':main()
