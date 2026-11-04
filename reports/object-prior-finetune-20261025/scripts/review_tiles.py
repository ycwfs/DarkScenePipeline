"""Freeze coordinate-preserving numbered tiles for manual object review."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['actions','others'],default='actions');args=p.parse_args()
    records=[]
    for path in (OUT/'drafts').glob('*/proposals.json'):
        x=json.loads(path.read_text())
        if (x['video']['video_label'] in ('drink','pick_up')) != (args.kind=='actions'):continue
        for row in x['records']:
            if row['view']=='full': continue
            records.append(dict(row,video_id=x['video']['id'],source=x['video']['source'],split=x['split'],
                                image_path=str(path.parent/row['image']),group=x['group']))
    records.sort(key=lambda r:(r['source'],r['frame'],r['view']))
    target=OUT/f'tiles-{args.kind}';target.mkdir(exist_ok=True)
    manifest=target/'manifest.json'
    if manifest.exists():
        old=json.loads(manifest.read_text())
        assert [r['id']for r in old]==[r['id']for r in records], 'Tile index frozen; wait for drafts to finish before first build'
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',16)
    for start in range(0,len(records),12):
        sheet=Image.new('RGB',(1600,1260),(25,25,25));draw=ImageDraw.Draw(sheet)
        for j,row in enumerate(records[start:start+12]):
            im=Image.open(row['image_path']).convert('RGB')
            # Preserve full ROI. Draw sparse coordinate guides for precise manual boxes.
            scale=min(390/im.width,360/im.height)
            nw,nh=round(im.width*scale),round(im.height*scale)
            left,top=(400-nw)//2,(360-nh)//2
            row['tile_index']=start+j;row['tile_transform']=dict(scale_x=nw/im.width,scale_y=nh/im.height,left=left,top=top)
            x,y=j%4*400,j//4*420
            sheet.paste(im.resize((nw,nh)),(x+left,y+60+top))
            draw.text((x+5,y+2),f'{start+j:04d} '+Path(row['source']).name,font=font,fill='white')
            draw.text((x+5,y+22),f'f{row["frame"]} {row["view"]} {row["split"]}',font=font,fill='white')
            for n in range(0,400,50):
                draw.text((x+n+1,y+43),str(n),font=font,fill='#aaaaaa')
                draw.line([(x+n,y+60),(x+n,y+64)],fill='#aaaaaa')
            for n in range(0,360,50):
                draw.text((x+1,y+60+n),str(n),font=font,fill='#aaaaaa')
        sheet.save(target/f'page-{start//12:03d}.jpg',quality=95)
    manifest.write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
    print(args.kind,len(records),'views')


if __name__=='__main__':main()
