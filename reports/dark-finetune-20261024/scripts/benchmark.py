"""Isolated detector/pose latency only, not end-to-end streaming FPS."""
import time
import cv2
import numpy as np
from inference import Frontend,OUT,ROOT
from prepare import save


def main():
    cv2.setNumThreads(1)
    paths=sorted((OUT/'dataset/images/val').glob('*.png'))[::7]
    images=[cv2.imread(str(p)) for p in paths]
    op=ROOT/'platform/dark-demo-v2/ckpts';results={}
    for name,det,pose in [('original',op/'yolo26m_detect.onnx',op/'rtmo-s_end2end.onnx'),
                         ('candidate',OUT/'models/yolo26m_dark_candidate.onnx',OUT/'models/rtmo-s_dark_candidate.onnx')]:
        model=Frontend(det,pose)
        for image in images[:3]:model.detector._frames_to_dets([image]);model.poses(image)
        detector=[];skeleton=[]
        for repeat in range(5):
            for image in images:
                start=time.perf_counter();model.detector._frames_to_dets([image]);detector.append((time.perf_counter()-start)*1000)
                start=time.perf_counter();model.poses(image);skeleton.append((time.perf_counter()-start)*1000)
        results[name]={key:dict(n=len(values),median_ms=float(np.median(values)),p95_ms=float(np.percentile(values,95)))
                       for key,values in [('detector',detector),('pose',skeleton)]}
        del model
    save(OUT/'benchmark.json',dict(results=results,
        scope='One idle RTX3090; ONNX preprocessing+inference+NMS; excludes enhancement/actions/network/encoder',
        thresholds=dict(detector=.35,pose=.45),not_full_pipeline_fps=True))
    print(results,flush=True)


if __name__=='__main__':main()
