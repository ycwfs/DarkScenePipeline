"""Optional FP16 export and real-frame frontend check, keeping FP32 I/O."""
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.transformers.float16 import convert_float_to_float16

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'NVIDIA-Box/runtime'))
from darkpipe.stages.object_prior import YoloOnnxDetector
models=['yolo26m_person_20261024.onnx','rtmo-s_person_20261024.onnx','yolo26m_objects_20261026.onnx']
destination=OUT/'fp16';destination.mkdir(exist_ok=True)
cap=cv2.VideoCapture(str(ROOT/'reports/full-prior-release-20261027/inputs/camera.mkv'))
ok,frame=cap.read();assert ok;cap.release()
results={}
for name in models:
    source=ROOT/'NVIDIA-Box/weights'/name
    output=destination/name
    if not output.exists():
        model=convert_float_to_float16(onnx.load(str(source)),keep_io_types=True)
        onnx.save(model,str(output))
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    samples={};values={}
    for mode,path in [('fp32',source),('fp16',output)]:
        try:
            sess=ort.InferenceSession(str(path),sess_options=options,providers=['CUDAExecutionProvider'])
            if name.startswith('rtmo'):
                x=np.full((1,3,640,640),114,np.float32)
                image=cv2.resize(frame,(640,360))
                x[0,:,:360,:]=image.transpose(2,0,1)
            else:
                image,*_=YoloOnnxDetector._letterbox(frame)
                x=np.ascontiguousarray(image[:,:,::-1].transpose(2,0,1)[None],dtype=np.float32)/255.
            feed={sess.get_inputs()[0].name:x}
            for _ in range(3):sess.run(None,feed)
            durations=[]
            for _ in range(30):
                start=time.perf_counter();value=sess.run(None,feed);durations.append(time.perf_counter()-start)
            samples[mode]=dict(mean_ms=np.mean(durations)*1000,p95_ms=np.percentile(durations,95)*1000)
            values[mode]=value
        except Exception as exc:
            samples[mode]=dict(error=f'{type(exc).__name__}: {exc}')
    results[name]=samples
    print(name,samples,flush=True)
    (OUT/'onnx-probe.json').write_text(json.dumps(results,indent=2)+'\n')
