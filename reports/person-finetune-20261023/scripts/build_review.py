"""Contact sheets for visual annotation; model probabilities are deliberately hidden."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'reports/person-action-20261023/scripts'))
from person_core import _cjk_font


def fit(image: Image.Image, width: int, height: int) -> Image.Image:
    tile=Image.new('RGB',(width,height),(230,230,230))
    image=image.copy()
    image.thumbnail((width,height))
    tile.paste(image,((width-image.width)//2,(height-image.height)//2))
    return tile


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--label',required=True)
    parser.add_argument('--rows',type=int,default=6)
    args=parser.parse_args()
    videos=json.loads((OUT/'source-inventory.json').read_text())['videos']
    videos=sorted([item for item in videos if item['video_label']==args.label],key=lambda item:item['source'])
    rows=[]
    for item in videos:
        path=OUT/'data'/item['id']/'manifest.json'
        if not path.exists():
            continue
        manifest=json.loads(path.read_text())
        tids=sorted({record['track_id'] for record in manifest['windows']})
        for tid in tids:
            samples=manifest['review'][str(tid)]
            rows.append(dict(video_id=item['id'],source=item['source'],track_id=tid,
                             video_label=item['video_label'],samples=samples,
                             duration=item['seconds'],windows=sum(w['track_id']==tid for w in manifest['windows'])))
    folder=OUT/'review'
    folder.mkdir(exist_ok=True)
    full_font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    font=ImageFont.truetype(str(full_font),17) if full_font.exists() else _cjk_font(17)
    small=ImageFont.truetype('DejaVuSans.ttf',14)
    pages=[]
    for start in range(0,len(rows),args.rows):
        subset=rows[start:start+args.rows]
        canvas=Image.new('RGB',(1360,190*len(subset)),(250,250,250))
        draw=ImageDraw.Draw(canvas)
        for index,row in enumerate(subset):
            y=index*190
            title=f"{row['video_id'][:8]}  {Path(row['source']).name}  T{row['track_id']:02d}  ({row['windows']} windows)"
            draw.text((4,y+2),title,fill='black',font=font)
            scene=Image.open(OUT/'data'/row['video_id']/'review'/'scene.jpg')
            canvas.paste(fit(scene,200,135),(0,y+30))
            samples=row['samples']
            for col,k in enumerate(np.linspace(0,len(samples)-1,12).round().astype(int)):
                sample=samples[k]
                tile=fit(Image.open(OUT/sample['image']),94,135)
                x=208+col*96
                canvas.paste(tile,(x,y+28))
                draw.text((x,y+166),f"{sample['timestamp']:.2f}s",fill='black',font=small)
        name=f'{args.label}-{start//args.rows:02d}.jpg'
        canvas.save(folder/name,quality=94)
        pages.append(dict(image=name,rows=[{k:v for k,v in row.items() if k!='samples'} for row in subset]))
    (folder/f'{args.label}-index.json').write_text(json.dumps(pages,ensure_ascii=False,indent=2)+'\n')
    print(f'{args.label}: {len(rows)} tracks, {len(pages)} pages, {len(videos)} source videos')


if __name__=='__main__':
    main()
