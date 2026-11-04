"""Robot detections at the frontend boundary, separate from final action filtering."""
import copy
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import onnxruntime as ort
import torch

from common import OUT, PREVIOUS, ROOT, save


def main():
    sys.path.insert(0,'/app')
    from darkpipe.config import PipelineConfig
    from darkpipe.person.runtime import PersonEngine,prepare_stages
    from darkpipe.person.core import iou,person_boxes
    torch.set_num_threads(2);cv2.setNumThreads(1)
    cfg=PipelineConfig(person_mode=True,recognize='behavior',gpus='0',device='cuda:0',
        ckpt_dir='/app/ckpts',proc_max_side=720,skel_fusion_alpha=.972,lum_gate=35.,
        reco_min_conf=.925,object_min_conf=.581,person_object_prior=True)
    engine=PersonEngine(cfg,0,lambda row:None,realtime=False)
    models={'current_person':engine.front.detector}
    for key,file in [('original80','yolo26m_coco80_original.onnx'),('candidate80','yolo26m_coco80_candidate.onnx')]:
        det=copy.copy(engine.front.detector);opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
        det.sess=ort.InferenceSession(str(OUT/'models'/file),sess_options=opts,providers=['CUDAExecutionProvider'])
        assert det.sess.get_providers()[0]=='CUDAExecutionProvider';models[key]=det
    rows=[]
    annotations=json.loads((PREVIOUS/'reviewed-labels.json').read_text())['records']
    for r in annotations:
        if not r.get('included')or not any(b['class_id']==6 for b in r['reviewed_global_boxes']):continue
        cap=cv2.VideoCapture(r['source']);cap.set(cv2.CAP_PROP_POS_FRAMES,r['frame']);ok,raw=cap.read();cap.release();assert ok
        image,_=prepare_stages(engine.stages,cfg,raw)
        scale=np.array([raw.shape[1]/image.shape[1],raw.shape[0]/image.shape[0]]*2)
        truth=[b['bbox']for b in r['reviewed_global_boxes']if b['class_id']==6]
        variants={}
        for name,det in models.items():
            predictions=person_boxes(det._frames_to_dets([image])[0])
            variants[name]=dict(person_boxes=[(np.array(b[:4])*scale).tolist()+[float(b[4])]for b in predictions],
                robot_as_person=sum(any(iou(np.array(b[:4])*scale,np.array(g))>=.4 for b in predictions)for g in truth))
        rows.append(dict(index=r['index'],source=r['source'],frame=r['frame'],split=r['split'],robot_boxes=truth,variants=variants))
        (OUT/'robot-review').mkdir(exist_ok=True)
        tiles=[]
        for name in models:
            im=raw.copy()
            for b in truth:cv2.rectangle(im,tuple(map(int,b[:2])),tuple(map(int,b[2:])),(255,255,255),2)
            for b in variants[name]['person_boxes']:
                cv2.rectangle(im,tuple(map(int,b[:2])),tuple(map(int,b[2:4])),(0,80,255),2)
            im=cv2.resize(im,(720,404));cv2.putText(im,name,(10,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,255),2);tiles.append(im)
        cv2.imwrite(str(OUT/'robot-review'/f'probe-{r["index"]:03}.jpg'),np.concatenate(tiles,axis=1))
    engine.close()
    save(OUT/'robot-probe.json',dict(records=rows,threshold=.35,matching_iou=.4,
        note='Full-frame same enhancement. Current_person is the active upstream detector; candidate80 remains object-prior-only in paired long runs. No static region is used to suppress inference.'))
    print([(r['index'],{k:v['robot_as_person']for k,v in r['variants'].items()})for r in rows])


if __name__=='__main__':main()
