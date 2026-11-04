import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parents[1]
rows=json.loads((ROOT/'reports/object-prior-finetune-20261025/review-candidates.json').read_text())
for start in range(0,len(rows),12):
 sheet=Image.new('RGB',(1600,1140),(25,25,25));d=ImageDraw.Draw(sheet)
 for j,r in enumerate(rows[start:start+12]):
  source=ROOT/'reports/object-prior-finetune-20261025'/r['review_image'];im=Image.open(source)
  im=im.crop((600,85,1260,423));im.thumbnail((390,320))
  x=j%4*400;y=j//4*380;sheet.paste(im,(x,y+45));d.text((x+5,y+8),f"C{start+j+1} src{r['number']:02d} {r['event']['timestamp']:.2f}s",fill='white')
 sheet.save(OUT/'review'/f'prior-candidates-{start//12}.jpg')
sources=json.loads((ROOT/'reports/manypeople-inference-20261025/input-manifest.json').read_text())
for r in sources:
 cap=cv2.VideoCapture(r['absolute_path']);sheet=Image.new('RGB',(1600,1040),(24,24,24));d=ImageDraw.Draw(sheet)
 for j,t in enumerate(np.linspace(8,r['duration']-8,8)):
  cap.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,img=cap.read();assert ok
  x=j%2*800;y=j//2*260
  sheet.paste(Image.fromarray(cv2.cvtColor(cv2.resize(img,(460,259)),cv2.COLOR_BGR2RGB)),(x,y))
  # Right image top half retains more detail of people in distance.
  roi=img[:720,480:1440];roi=cv2.resize(roi,(338,253))
  sheet.paste(Image.fromarray(cv2.cvtColor(roi,cv2.COLOR_BGR2RGB)),(x+460,y))
  d.rectangle((x,y,x+130,y+20),fill='black');d.text((x+2,y+2),f'src{r["number"]:02d} t={t:.1f}',fill='white')
 cap.release();sheet.save(OUT/'review'/f'source-{r["number"]:02d}.jpg',quality=93)
