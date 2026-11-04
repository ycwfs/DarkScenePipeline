"""Diagnostic mining only; no model boxes are accepted as ground truth."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parents[1]
BASE=ROOT/'reports/object-model-ablation-20261026'
data=json.loads((BASE/'inputs.json').read_text());pred=json.loads((BASE/'predictions/m_finetuned.json').read_text())['predictions']
sources=json.loads((ROOT/'reports/manypeople-inference-20261025/input-manifest.json').read_text())
nums={r['absolute_path']:r['number']for r in sources};candidates=defaultdict(list)
for key,r in data['images'].items():
 num=nums.get(r.get('source'))
 if num is None:continue
 area=(r['rect'][2]-r['rect'][0])*(r['rect'][3]-r['rect'][1])
 if area<12000:continue
 score=max([b[4]for b in pred[key]['small_enhanced'] if b[5] in [0,1,2,3,5]]+[0])
 candidates[num].append((score,area,key,r))
selected=[]
for num,rows in sorted(candidates.items()):
 chosen=[]
 for score,area,key,r in sorted(rows,key=lambda v:(-v[0],-v[1])):
  if any(abs(r['frame']-v[3]['frame'])<100 for v in chosen):continue
  chosen.append((score,area,key,r))
  if len(chosen)>=20:break
 for score,area,key,r in sorted(chosen,key=lambda v:v[3]['frame']):
  idx=len(selected);split='test' if num in [1,7,9]else 'val'if num==3 else 'train'
  selected.append(dict(index=idx,number=num,split=split,key=key,frame=r['frame'],seconds=r['frame']/25,
    source=r['source'],rect=r['rect'],variants=r['variants'],mining_score=score))
for start in range(0,len(selected),16):
 sheet=Image.new('RGB',(1600,1600),(24,24,24));d=ImageDraw.Draw(sheet)
 for j,r in enumerate(selected[start:start+16]):
  im=Image.open(BASE/r['variants']['native_raw']['path']).convert('RGB');w,h=im.size
  scale=min(400/w,360/h);nw,nh=round(w*scale),round(h*scale);left=(400-nw)//2;top=40+(360-nh)//2
  r['tile_transform']=dict(scale_x=nw/w,scale_y=nh/h,left=left,top=top,width=w,height=h)
  tile=Image.new('RGB',(400,400),(24,24,24));td=ImageDraw.Draw(tile)
  tile.paste(im.resize((nw,nh)),(left,top));td.text((5,4),f"#{r['index']:03d} src{r['number']:02d} {r['seconds']:.1f}s {r['split']}",fill='white')
  td.text((5,20),f"native {w}x{h} | proposal only",fill='white')
  for x in range(50,400,50):td.text((x,386),str(x),fill='#aaa')
  for y in range(100,400,50):td.text((1,y),str(y),fill='#aaa')
  sheet.paste(tile,(j%4*400,j//4*400))
  (OUT/'review/tiles').mkdir(exist_ok=True);tile.save(OUT/'review/tiles'/f'{r["index"]:03d}.png')
 sheet.save(OUT/'review'/f'labels-{start//16:02d}.jpg',quality=96)
(OUT/'label-views.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2)+'\n')
print('views',len(selected),[(n,len([r for r in selected if r['number']==n]))for n in range(1,10)])
