"""Portable ONNX of the unchanged NTIRE enhancement weights, fixed default Box shape."""
import hashlib
import json
from pathlib import Path
import sys
import torch
import onnx

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'NVIDIA-Box/runtime'))
from darkpipe.stages.enhance_retinexformer import RetinexformerStage
stage=RetinexformerStage(str(ROOT/'NVIDIA-Box/weights'));stage.load('cuda:0')
out=ROOT/'NVIDIA-Box/weights/retinexformer_ntire_720_box.onnx'
with torch.inference_mode():
    torch.onnx.export(stage.net,torch.zeros((1,3,404,720),device='cuda',dtype=torch.float16),
        str(out),input_names=['image'],output_names=['enhanced'],opset_version=17,
        dynamo=False)
onnx.checker.check_model(str(out))
record=dict(derived_from=stage.ckpt,source_sha256=hashlib.sha256(Path(stage.ckpt).read_bytes()).hexdigest(),
    output=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),input_shape=[1,3,404,720],
    precision='float16',training='No retraining; same NTIRE weights; exported for Box TensorRT')
Path(__file__).resolve().parents[1].joinpath('enhance-export.json').write_text(json.dumps(record,indent=2)+'\n')
print(record,flush=True)
