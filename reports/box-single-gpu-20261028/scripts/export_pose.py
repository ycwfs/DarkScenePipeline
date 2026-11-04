"""Equivalent rank-1 ArgMax rewrite for the TensorRT 10 ONNX parser."""
import hashlib
import json
from pathlib import Path
import onnx
from onnx import helper, numpy_helper
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
source=ROOT/'NVIDIA-Box/weights/rtmo-s_person_20261024.onnx'
model=onnx.load(str(source));nodes=[];rewrites=[]
opset=next(x.version for x in model.opset_import if not x.domain)
for node in model.graph.node:
    if node.op_type!='ArgMax':
        nodes.append(node);continue
    attrs={a.name:a.i for a in node.attribute}
    assert attrs['axis']==0 and attrs['keepdims']==0
    original_output=node.output[0];expanded=node.input[0]+'_trt_2d';reduced=original_output+'_trt_1d'
    if opset>=13:
        axes=node.name+'_trt_axes'
        model.graph.initializer.append(numpy_helper.from_array(np.array([0],np.int64),axes))
        before=helper.make_node('Unsqueeze',[node.input[0],axes],[expanded],name=node.name+'_expand')
        after=helper.make_node('Squeeze',[reduced,axes],[original_output],name=node.name+'_restore')
    else:
        before=helper.make_node('Unsqueeze',[node.input[0]],[expanded],axes=[0],name=node.name+'_expand')
        after=helper.make_node('Squeeze',[reduced],[original_output],axes=[0],name=node.name+'_restore')
    for attr in node.attribute:
        if attr.name=='axis':attr.i=1
    node.input[0]=expanded;node.output[0]=reduced
    nodes.extend([before,node,after]);rewrites.append(node.name)
assert rewrites
del model.graph.node[:];model.graph.node.extend(nodes)
onnx.checker.check_model(model)
output=source.with_name('rtmo-s_person_20261024_box.onnx');onnx.save(model,str(output))
record=dict(source=source.name,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    output=output.name,sha256=hashlib.sha256(output.read_bytes()).hexdigest(),rewrites=rewrites,
    change='Unsqueeze rank1 -> ArgMax axis1 -> Squeeze; original indices, values and external shapes preserved')
Path(__file__).resolve().parents[1].joinpath('pose-export.json').write_text(json.dumps(record,indent=2)+'\n')
print(record)
