"""Portable candidate detector probe. Does not change action labels or person tracking."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


NAMES = ['bottle', 'cup', 'phone', 'camera', 'paper', 'portable_object', 'robot_or_exhibit']


def predict(model: Path, image: np.ndarray, device: str, confidence: float) -> tuple:
    options=ort.SessionOptions(); options.intra_op_num_threads=2; options.inter_op_num_threads=1
    providers=['CPUExecutionProvider'] if device=='cpu' else [('CUDAExecutionProvider', {'device_id':int(device)})]
    runner=ort.InferenceSession(str(model),sess_options=options,providers=providers)
    assert runner.get_providers()[0]==('CPUExecutionProvider' if device=='cpu' else 'CUDAExecutionProvider')
    h,w=image.shape[:2];scale=min(640/w,640/h);nw,nh=round(w*scale),round(h*scale)
    left,top=(640-nw)//2,(640-nh)//2
    canvas=np.full((640,640,3),114,np.uint8)
    canvas[top:top+nh,left:left+nw]=cv2.resize(image,(nw,nh))
    x=np.ascontiguousarray(canvas[:,:,::-1].transpose(2,0,1)[None],dtype=np.float32)/255
    pred=runner.run(None,{runner.get_inputs()[0].name:x})[0][0].copy()
    assert pred.shape==(300,6)
    pred[:,[0,2]]=(pred[:,[0,2]]-left)/scale
    pred[:,[1,3]]=(pred[:,[1,3]]-top)/scale
    pred[:,:4]=np.clip(pred[:,:4],[0]*4,[w,h]*2)
    return pred[pred[:,4]>=confidence], runner.get_providers()


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('image',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',type=Path,default=Path(__file__).resolve().parents[1]/'models/yolo26m_objects_candidate.onnx')
    p.add_argument('--device',default='cpu',help='cpu or CUDA visible device index')
    p.add_argument('--confidence',type=float,default=.35)
    p.add_argument('--gamma',action='store_true',help='Apply the raw-ROI brightness transform. Omit for prepared dataset images.')
    a=p.parse_args();assert 0<=a.confidence<=1
    cv2.setNumThreads(1);im=cv2.imread(str(a.image));assert im is not None
    gamma=1.
    if a.gamma:
        luma=float(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).mean())
        gamma=1. if luma>=75 else .5 if luma>=15 else .32
        im=cv2.LUT(im,np.round((np.arange(256)/255.)**gamma*255).astype(np.uint8))
    rows,providers=predict(a.model,im,a.device,a.confidence)
    a.output.mkdir(exist_ok=True,parents=True)
    for row in rows:
        x1,y1,x2,y2=map(int,row[:4]);cv2.rectangle(im,(x1,y1),(x2,y2),(0,220,255),2)
        cv2.putText(im,f'{NAMES[int(row[5])]} {row[4]:.2f}',(x1,max(12,y1-3)),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,220,255),1)
    assert cv2.imwrite(str(a.output/'objects.jpg'),im)
    (a.output/'objects.json').write_text(json.dumps(dict(image=str(a.image),model=str(a.model),
        detections=rows.tolist(),providers=providers,gamma=gamma,classes=NAMES,
        status='experimental object predictions, not action truth'),indent=2)+'\n')


if __name__=='__main__':
    main()
