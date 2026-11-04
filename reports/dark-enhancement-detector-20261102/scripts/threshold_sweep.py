import json,sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run import ROOT, OUT, Detector, greedy_match
rows=json.loads((OUT/'records.json').read_text())
# Keep the same images used by the main experiment; raw720 is stored on disk.
det=Detector(ROOT/'platform/dark-demo-v2/ckpts/yolo26m_person_20261024.onnx')
thresholds=[.10,.15,.20,.25,.30,.35,.40,.50]
summary={}
for condition in ['raw720','ntire720','ft720']:
  summary[condition]={}
  for tau in thresholds:
    gt=matched=pred=unmatched=0
    per=[]
    for r in rows:
      image=cv2.imread(str(OUT/'frames'/f"{r['id']:03d}_{condition}.png"))
      truth=np.asarray(r['conditions'][condition]['shape']) # unused
      scale=np.array([image.shape[1]/r['native_shape'][1], image.shape[0]/r['native_shape'][0]],np.float32)
      gt_boxes=np.asarray(r['gt_boxes_native'],np.float32)*np.tile(scale,2)
      p=det(image,tau)
      m=greedy_match(gt_boxes,p[:,:4])
      gt+=len(gt_boxes);matched+=len(m);pred+=len(p);unmatched+=len(p)-len(m)
      per.append((len(m),len(p)-len(m)))
    summary[condition][str(tau)]={'gt':gt,'matched':matched,'recall':matched/gt,'predictions':pred,'unmatched_predictions':unmatched}
(OUT/'threshold-sweep.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
