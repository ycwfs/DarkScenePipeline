"""Export the validation-selected YOLO candidate without touching original files."""
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from prepare import OUT,save
from inference import session


if __name__=='__main__':
    torch.set_num_threads(2)
    result=json.loads((OUT/'detector-training-result.json').read_text())
    target=OUT/'models/yolo26m_dark_candidate.pt'
    shutil.copy2(result['best'],target)
    model=YOLO(str(target))
    model.model.end2end=True
    exported=model.export(format='onnx',imgsz=640,opset=17,simplify=False,dynamic=False,
                          batch=1,device='cpu',half=False,nms=False)
    onnx=session(Path(exported),False)
    x=np.full((1,3,640,640),.3,np.float32)
    with torch.inference_mode():expected=model.model(torch.from_numpy(x))
    if isinstance(expected,tuple):expected=expected[0]
    actual=onnx.run(None,{onnx.get_inputs()[0].name:x})[0]
    assert actual.shape==(1,300,6),actual.shape
    error=float(np.max(np.abs(actual-expected.numpy())))
    # The sorted top-k can exchange near-tied empty-image rows; the deployed
    # confidence range is what must agree, not sub-threshold ranking.
    mask=actual[0,:,4]>=.35
    confident_error=float(np.max(np.abs(actual[0,mask]-expected.numpy()[0,mask]))) if mask.any() else 0.
    assert confident_error<.1,confident_error
    checkpoint=torch.load(target,map_location='cpu',weights_only=False)
    save(OUT/'models/yolo26m_dark_candidate.verification.json',dict(
        selected_by='Ultralytics validation mAP50-95 fitness',best_checkpoint=result['best'],
        output_shape=list(actual.shape),blank_confident_rows=int(mask.sum()),
        max_absolute_difference=error,confident_max_absolute_difference=confident_error,
        warning='Blank-image check only; real-frame paired evaluation follows'))
    print(exported,flush=True)
