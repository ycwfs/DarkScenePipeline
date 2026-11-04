"""Compare evaluation pose preprocessing to the existing RTMO runtime wrapper."""
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from rtmlib import RTMO
from evaluate import ROOT, OUT, MODEL_PATHS, Frontend, save, sha


def main() -> None:
    cv2.setNumThreads(1)
    evaluator=Frontend(MODEL_PATHS['detector'],MODEL_PATHS['pose'])
    runtime=RTMO(str(MODEL_PATHS['pose']),model_input_size=(640,640),backend='onnxruntime',device='cuda:0',score_thr=.45)
    results=[]
    for p in sorted((ROOT/'reports/luma-samples-20261024/samples').glob('*/enhanced.png')):
        frame=cv2.imread(str(p));_,xy,cf,_=evaluator.poses(frame)
        rx,rc=runtime(frame)
        valid=rc.max(1)>0;rx=rx[valid];rc=rc[valid]
        assert len(xy)==len(rx),(p,len(xy),len(rx))
        maximum=0.
        if len(xy):
            cost=np.linalg.norm(xy[:,None]-rx[None,:],axis=3).mean(2)
            rr,cc=linear_sum_assignment(cost)
            maximum=float(np.max(np.abs(xy[rr]-rx[cc])))
            assert maximum<.01 and np.max(np.abs(cf[rr]-rc[cc]))<1e-4,(p,maximum)
        results.append(dict(sample=p.parent.name,people=len(xy),max_coordinate_error=maximum))
    save(OUT/'pose-runtime-parity.json',dict(passed=True,model_sha256=sha(MODEL_PATHS['pose']),records=results))
    print('RTMO runtime parity passed:',len(results),'frames',flush=True)


if __name__=='__main__':main()
