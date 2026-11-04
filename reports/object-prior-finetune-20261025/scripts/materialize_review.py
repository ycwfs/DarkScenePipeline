"""Convert explicit visual-review coordinates into auditable native-pixel labels."""
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

OUT=Path(__file__).resolve().parents[1]


def inverse(box, transform, width, height):
    return np.clip((np.asarray(box,float)-[transform['left'],transform['top']]*2)/
                   ([transform['scale_x'],transform['scale_y']]*2),[0]*4,[width,height]*2)


def main():
    manifest=[]
    for kind in ('actions','others','supplement','extra'):
        p=OUT/f'review-{kind}.json'
        if not p.exists():continue
        reviewed=json.loads(p.read_text())
        views=json.loads((OUT/f'tiles-{kind}/manifest.json').read_text())
        for key,annotation in reviewed['records'].items():
            row=views[int(key)];record=dict(row,review=annotation,review_kind=kind)
            if 'exclude'in annotation:
                record['training_allowed']=False;manifest.append(record);continue
            source_image=Path(row['image_path']);image=cv2.imread(str(source_image))
            height,width=image.shape[:2];transform=row['tile_transform']
            region=annotation.get('valid_region',[0,0,400,360])
            valid=inverse(region,transform,width,height)
            x1,y1=np.floor(valid[:2]).astype(int);x2,y2=np.ceil(valid[2:]).astype(int)
            assert x2>x1 and y2>y1
            crop=image[y1:y2,x1:x2]
            boxes=[]
            for cls,*bounds in annotation['boxes']:
                box=inverse(bounds,transform,width,height)
                box-=np.array([x1,y1]*2)
                box=np.clip(box,[0]*4,[x2-x1,y2-y1]*2)
                assert np.all(box[2:]-box[:2]>=2),(kind,key,box)
                boxes.append(dict(class_id=cls,bbox=box.tolist(),
                                  source='assistant manually drawn from original/gamma crop',
                                  certainty='auxiliary visual review, not independent human GT'))
            raw=cv2.imread(str(source_image.parent/row['raw_file']))
            rx,ry,_,_=row['crop_xyxy_raw']
            raw_crop=raw[ry+y1:ry+y2,rx+x1:rx+x2]
            assert raw_crop.shape==crop.shape
            dest=OUT/'reviewed-images'/kind;dest.mkdir(parents=True,exist_ok=True)
            stem=f'{int(key):04d}_{row["id"]}'
            cv2.imwrite(str(dest/(stem+'_raw.png')),raw_crop)
            cv2.imwrite(str(dest/(stem+'_view.png')),crop)
            record.update(training_allowed=True,image_raw=str(dest/(stem+'_raw.png')),
                image_view=str(dest/(stem+'_view.png')),width=crop.shape[1],height=crop.shape[0],boxes=boxes,
                reviewed_crop_xyxy_raw=list(map(int,[rx+x1,ry+y1,rx+x2,ry+y2])))
            manifest.append(record)
    (OUT/'reviewed-labels.json').write_text(json.dumps(dict(records=manifest,
        counts=dict(reviewed=len(manifest),usable=sum(r['training_allowed']for r in manifest),
                    classes=dict(Counter(b['class_id']for r in manifest for b in r.get('boxes',[]))))),
                    ensure_ascii=False,indent=2)+'\n')
    # Render all accepted boxes to verify coordinate conversion and catch mistakes.
    usable=[r for r in manifest if r['training_allowed']]
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',15)
    for start in range(0,len(usable),12):
        sheet=Image.new('RGB',(1600,1260),(24,24,24));draw=ImageDraw.Draw(sheet)
        for j,row in enumerate(usable[start:start+12]):
            im=Image.open(row['image_view']).convert('RGB');scale=min(390/im.width,360/im.height)
            nw,nh=round(im.width*scale),round(im.height*scale);x=j%4*400+(400-nw)//2;y=j//4*420+50
            sheet.paste(im.resize((nw,nh)),(x,y))
            draw.text((j%4*400+4,j//4*420+4),f'{row["review_kind"]}:{row["tile_index"]} '+Path(row['source']).name,font=font,fill='white')
            for b in row['boxes']:
                q=np.array(b['bbox'])*([nw/im.width,nh/im.height]*2)+np.array([x,y]*2)
                draw.rectangle(q.tolist(),outline='#ffcc00',width=2)
                draw.text((q[0],q[1]),str(b['class_id']),font=font,fill='#ffcc00')
        folder=OUT/'label-quality';folder.mkdir(exist_ok=True)
        sheet.save(folder/f'page-{start//12:03d}.jpg',quality=94)
    print('Materialized',len(manifest),'reviews;',len(usable),'usable views')


if __name__=='__main__':main()
