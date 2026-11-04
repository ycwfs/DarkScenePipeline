"""Real-image confident output parity, original source weights and exported candidate."""
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment

from inference import OUT,ROOT,session,YoloOnnxDetector
from prepare import iou,save


def main():
    torch.set_num_threads(2);cv2.setNumThreads(1);rows=[]
    for name,pt,onnx in [('original',ROOT.parent/'yolo26m-detect.pt',ROOT/'platform/dark-demo-v2/ckpts/yolo26m_detect.onnx'),
                        ('candidate',OUT/'models/yolo26m_dark_candidate.pt',OUT/'models/yolo26m_dark_candidate.onnx')]:
        model=YOLO(str(pt)).model.float().eval();model.end2end=True
        runner=session(onnx,False)
        for path in sorted((OUT/'dataset/images/val').glob('*.png'))[::12]:
            frame=cv2.imread(str(path));canvas,_,_,_=YoloOnnxDetector._letterbox(frame)
            x=np.ascontiguousarray(canvas[:,:,::-1].transpose(2,0,1)[None].astype(np.float32)/255)
            with torch.no_grad():pred=model(torch.from_numpy(x))[0].numpy()[0]
            actual=runner.run(None,{runner.get_inputs()[0].name:x})[0][0]
            a=actual[(actual[:,4]>=.35)&(actual[:,5]==0)]
            b=pred[(pred[:,4]>=.35)&(pred[:,5]==0)]
            assert len(a)==len(b),(name,path,len(a),len(b))
            error=0.
            if len(a):
                cost=np.array([[1-iou(aa[:4],bb[:4]) for bb in b] for aa in a]);ii,jj=linear_sum_assignment(cost)
                error=float(np.max(np.abs(a[ii]-b[jj])))
                assert error<.1,(name,path,error)
            rows.append(dict(model=name,image=path.name,people=len(a),max_absolute_error=error))
    save(OUT/'detector-runtime-parity.json',dict(rows=rows,passed=True));print(rows)


if __name__=='__main__':main()
