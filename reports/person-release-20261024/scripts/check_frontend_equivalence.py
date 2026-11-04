import json,sys
from pathlib import Path
sys.path.insert(0,'/app')
import cv2,numpy as np,torch
from darkpipe.config import PipelineConfig
from darkpipe.person.frontend_process import FrontendProcess
from darkpipe.person.runtime import Frontend,prepare_stages
from darkpipe.stages import build_stages

def main():
    torch.set_num_threads(2);cv2.setNumThreads(2)
    cfg=PipelineConfig(device='cuda:0',gpus='0',ckpt_dir='/app/ckpts',proc_max_side=720,enhance_chunk=1,recognize='off')
    stages,_=build_stages(cfg)
    stages=[s for s in stages if not s.post_recognition]
    for s in stages:s.load(cfg.device)
    front=Frontend(cfg);worker=FrontendProcess(cfg);rows=[]
    cap=cv2.VideoCapture('/inputs/挥手.mp4')
    try:
        for i in range(3):
            ok,raw=cap.read();assert ok
            a,la=prepare_stages(stages,cfg,raw);ba,pa=front(a,True)
            b,lb,bb,pb,_,_=worker(raw)
            assert np.array_equal(a,b) and la==lb
            np.testing.assert_allclose(ba,bb,rtol=0,atol=1e-5)
            for x,y in zip(pa,pb):
                assert (x is None)==(y is None)
                if x is not None:np.testing.assert_allclose(x,y,rtol=0,atol=1e-5)
            rows.append(dict(frame=i,pixels_equal=True,boxes=len(ba),pose_equal=True))
    finally:
        cap.release();worker.close()
        for s in stages:s.close()
    print(json.dumps(rows))
if __name__=='__main__':main()
