"""Training/validation-only proposal images for explicit non-person removal review."""
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parents[1]
OLD=ROOT/'reports/dark-finetune-20261024'
sys.path.insert(0,str(OLD/'scripts'))
from inference import session,YoloOnnxDetector
from demos import downscale
from prepare import save,iou
from person_core import person_boxes
from darkpipe.stages.enhance_retinexformer import RetinexformerStage

TRAIN=['other94','other96','other98','other99','other102','other105','other107','other108',
       '喝水25','喝水28','挥手31','挥手32','追逐26','追逐27','other101','other113',
       'other124','other138','other162','other187','喝水32','喝水37','捡东西24','挥手43']
VAL=['other104','other110','喝水27','挥手30','other130','other164','捡东西22','挥手48']


def main() -> None:
    cv2.setNumThreads(1);torch.set_num_threads(2)
    inventory=json.loads((ROOT/'reports/dark-annotations-20261023/inventory.json').read_text())['raw_videos']
    selected=[]
    for split,names in [('train',TRAIN),('val',VAL)]:
        for name in names:
            video=next(v for v in inventory if v['source'].startswith('real-actiondata-all/') and Path(v['source']).stem==name)
            assert video['frozen_split']['split']==split,(name,video['frozen_split'])
            scan=json.loads((ROOT/'reports/dark-annotations-20261023/data'/video['id']/'scan.json').read_text())
            indices=[f['frame'] for f in scan['frames'] if f['luma']>=75]
            selected.append(dict(**video,sample_frames=[indices[round((len(indices)-1)*q)] for q in [.2,.5,.8]]))
    save(OUT/'bright-source-protocol.json',dict(videos=selected,
        selection='train/val frozen splits only; 3 deterministic bright quantiles .2/.5/.8; no test use',
        thresholds='proposals only person=.25; runtime stays .35',label_status='pending visual review'))
    op=ROOT/'platform/dark-demo-v2/ckpts';models={}
    for name,path in [('original',op/'yolo26m_detect.onnx'),('previous',OLD/'models/yolo26m_dark_candidate.onnx')]:
        models[name]=YoloOnnxDetector(str(path.parent),'cuda:0');models[name].sess=session(path)
    enhancer=RetinexformerStage(str(op),chunk=1);enhancer.load('cuda:0')
    rows=[];panels=[];font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',18)
    for v in selected:
        cap=cv2.VideoCapture(str(ROOT/v['source']));pending=set(v['sample_frames']);frame=0
        while pending:
            ok,native=cap.read();assert ok,(v['source'],frame)
            if frame not in pending:frame+=1;continue
            pending.remove(frame);raw=downscale(native);enh=enhancer([raw])[0]
            boxes={k:person_boxes(m._frames_to_dets([enh])[0],confidence=.25).tolist() for k,m in models.items()}
            index=len(rows);dest=OUT/'data/bright'/f'{index:03d}';dest.mkdir(parents=True,exist_ok=True)
            cv2.imwrite(str(dest/'raw.png'),raw);cv2.imwrite(str(dest/'enhanced.png'),enh)
            row=dict(index=index,source=v['source'],video_id=v['id'],sha256=v['sha256'],split=v['frozen_split']['split'],
                group=v['frozen_split']['group'],frame=frame,width=raw.shape[1],height=raw.shape[0],proposals=boxes)
            rows.append(row)
            display=enh.copy()
            for j,b in enumerate(boxes['original']):
                bb=np.rint(b[:4]).astype(int);cv2.rectangle(display,tuple(bb[:2]),tuple(bb[2:]),(0,240,60),2)
                cv2.putText(display,f'O{j}',tuple(bb[:2]+[0,12]),0,.45,(0,240,60),1)
            for j,b in enumerate(boxes['previous']):
                if any(iou(b[:4],o[:4])>.4 for o in boxes['original']):continue
                bb=np.rint(b[:4]).astype(int);cv2.rectangle(display,tuple(bb[:2]),tuple(bb[2:]),(0,70,255),2)
                cv2.putText(display,f'N{j}',tuple(bb[:2]+[0,12]),0,.45,(0,70,255),1)
            panel=np.full((438,720,3),24,np.uint8);panel[34:]=display
            pil=Image.fromarray(cv2.cvtColor(panel,cv2.COLOR_BGR2RGB));ImageDraw.Draw(pil).text((5,3),
                f'{index:02d} {v["frozen_split"]["split"]} {Path(v["source"]).name} F{frame}',font=font,fill='white')
            panels.append(np.array(pil));frame+=1
        cap.release();save(OUT/'bright-proposals.json',dict(records=rows));print('[bright]',v['source'],flush=True)
    for start in range(0,len(panels),4):
        ps=panels[start:start+4]
        while len(ps)<4:ps.append(np.zeros_like(panels[0]))
        Image.fromarray(np.vstack([np.hstack(ps[:2]),np.hstack(ps[2:])])).save(OUT/'audit'/f'bright-{start//4:02d}.jpg')
    enhancer.close()


if __name__=='__main__':main()
