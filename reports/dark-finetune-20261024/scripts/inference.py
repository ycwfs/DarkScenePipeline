"""Deployment-shaped ONNX inference and optional causal skeleton stabilization."""
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'platform/dark-demo-v2'))
sys.path.insert(0,str(ROOT/'reports/person-action-20261023/scripts'))
from darkpipe.stages.object_prior import YoloOnnxDetector
from person_core import PersonTracker, associate_poses, person_boxes


def session(path: Path, cuda=True):
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    providers=[('CUDAExecutionProvider',{'device_id':0})] if cuda else ['CPUExecutionProvider']
    obj=ort.InferenceSession(str(path),sess_options=options,providers=providers)
    if cuda and obj.get_providers()[0]!='CUDAExecutionProvider':raise RuntimeError('GPU fallback')
    return obj


class Frontend:
    def __init__(self,detector: Path,pose: Path,cuda=True):
        self.detector=YoloOnnxDetector(str(detector.parent),'cuda:0')
        self.detector.sess=session(detector,cuda)
        self.pose_session=session(pose,cuda)
        self.tracker=PersonTracker(max_gap=.12)

    def poses(self,image):
        h,w=image.shape[:2];ratio=640/max(h,w)
        canvas=np.full((640,640,3),114,np.uint8)
        resized=cv2.resize(image,(int(w*ratio),int(h*ratio)),interpolation=cv2.INTER_LINEAR)
        canvas[:resized.shape[0],:resized.shape[1]]=resized
        x=canvas.transpose(2,0,1)[None].astype(np.float32)
        dets,points=self.pose_session.run(None,{self.pose_session.get_inputs()[0].name:x})
        dets=dets[0];points=points[0]
        xywh=dets[:,:4].copy();xywh[:,2:]-=xywh[:,:2]
        keep=np.asarray(cv2.dnn.NMSBoxes(xywh.tolist(),dets[:,4].tolist(),.45,.45),int).reshape(-1)
        return dets[keep,:4]/ratio,points[keep,:,:2]/ratio,points[keep,:,2],dets[keep,4]

    def __call__(self,image,timestamp):
        boxes=person_boxes(self.detector._frames_to_dets([image])[0],confidence=.35)
        pose_boxes,xy,cf,scores=self.poses(image)
        assigned=associate_poses(boxes,xy,cf)
        tracks=self.tracker.update(boxes,assigned,image,timestamp)
        return tracks,dict(boxes=boxes.tolist(),pose_boxes=pose_boxes.tolist(),
            pose_xy=xy.tolist(),pose_scores=cf.tolist(),pose_conf=scores.tolist())


class CausalPoseFilter:
    """No future frames or missing-joint filling. Boxes remain immediate.

    Each detector track is confirmed for 0.4s; each joint must survive 0.16s.
    One-Euro smoothing adapts to motion; >0.12s gaps restart confirmation.
    """
    def __init__(self,min_track=.4,min_joint=.16,max_gap=.12,min_cutoff=2.5,beta=5.):
        self.min_track=min_track;self.min_joint=min_joint;self.max_gap=max_gap
        self.min_cutoff=min_cutoff;self.beta=beta;self.states={}

    @staticmethod
    def alpha(cutoff,dt):return 1/(1+1/(2*np.pi*cutoff*dt))

    def update(self,tracks,timestamp):
        self.states={k:v for k,v in self.states.items() if timestamp-v['last']<=self.max_gap}
        output={}
        for t in tracks:
            box=np.asarray(t.box);center=(box[:2]+box[2:])/2;diag=max(np.linalg.norm(box[2:]-box[:2]),1)
            state=self.states.get(t.tid)
            if state is None:
                state=dict(start=timestamp,last=timestamp,joints=[None]*17);self.states[t.tid]=state
            dt=max(timestamp-state['last'],1/120);state['last']=timestamp
            out=np.zeros((17,3),np.float32)
            if t.pose is None:
                state['joints']=[None]*17;output[t.tid]=out;continue
            for j,(x,y,score) in enumerate(t.pose):
                previous=state['joints'][j]
                if score<.3 or not np.isfinite([x,y,score]).all():
                    state['joints'][j]=None;continue
                relative=(np.array([x,y])-center)/diag
                if previous is None:
                    previous=dict(start=timestamp,raw=relative,filtered=relative,velocity=np.zeros(2))
                    state['joints'][j]=previous
                else:
                    velocity=(relative-previous['raw'])/dt
                    if np.linalg.norm(velocity)>8.:
                        state['joints'][j]=None;continue
                    a=self.alpha(1.,dt);derivative=a*velocity+(1-a)*previous['velocity']
                    cutoff=self.min_cutoff+self.beta*np.linalg.norm(derivative)
                    a=self.alpha(cutoff,dt)
                    previous.update(filtered=a*relative+(1-a)*previous['filtered'],raw=relative,velocity=derivative)
                if timestamp-state['start']>=self.min_track-1e-6 and timestamp-previous['start']>=self.min_joint-1e-6:
                    out[j,:2]=previous['filtered']*diag+center;out[j,2]=score
            output[t.tid]=out
        return output
