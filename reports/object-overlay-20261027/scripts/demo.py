"""Real six-model production inference; wrapper only records visualization metadata."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import cv2
p=argparse.ArgumentParser();p.add_argument('--group',choices=['old','new']);p.add_argument('--ids',default='');p.add_argument('--operator',default='/app');p.add_argument('--repo',default='/repo');p.add_argument('--out',default='/evidence');a=p.parse_args()
sys.path.insert(0,a.operator)
from darkpipe.config import PipelineConfig
from darkpipe.person import service
OUT=Path(a.out); inputs=json.loads((OUT/'inputs.json').read_text()); Base=service.Output
for item in inputs:
 if a.group and item['group']!=a.group:continue
 if a.ids and item['id'] not in a.ids.split(','):continue
 dest=OUT/'demos'/item['id'];dest.mkdir(parents=True,exist_ok=True)
 overlays=(dest/'frames.jsonl').open('w'); peak=[0]; best=[-1]; classes=Counter()
 class Observe(Base):
  def __init__(self,engine,callback,writer=None):
   def observe(shown,clean,record):
    boxes=record.get('object_overlays',[])
    peak[0]=max(peak[0],record['people'])
    classes.update(b['class_id']for b in boxes)
    overlays.write(json.dumps({k:record[k]for k in ['frame','timestamp','people','raw_luma','object_overlays']})+'\n')
    score=sum(not b['followed']for b in boxes)*100+len(boxes)*10+record['people']
    if score>best[0]:
     best[0]=score;cv2.imwrite(str(dest/'preview.jpg'),shown)
    callback(shown,clean,record)
   super().__init__(engine,observe,writer)
 service.Output=Observe
 cfg=PipelineConfig(person_mode=True,recognize='behavior',gpus='0',device='cuda:0',
  input=str(Path(a.repo)/item['input']),output=str(dest/'output.mp4'),events_json=str(dest/'events.json'),
  ckpt_dir=str(Path(a.operator)/'ckpts'),proc_max_side=720,sr='bicubic',sr_scale=2,reco_span_sec=1.,
  clip_dir=str(dest/'clips'),clip_session='review')
 rows=service.run_offline(cfg);overlays.close()
 result=dict(input=item,stats=cfg.stats,people_max=peak[0],object_classes=dict(classes),
  labels=dict(Counter(r['label']for r in rows)),prior_reasons=dict(Counter(r['object_prior']['reason']for r in rows)))
 (dest/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(result,ensure_ascii=False),flush=True)
