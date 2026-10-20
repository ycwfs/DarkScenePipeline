"""Diagnostic launcher: bound CPU threads; leave the archived package untouched."""
import runpy
import sys
import cv2
import torch
import onnxruntime as ort

torch.set_num_threads(4)
torch.set_num_interop_threads(1)
cv2.setNumThreads(2)
original_session = ort.InferenceSession

def limited_session(*args, **kwargs):
    options = kwargs.pop('sess_options', None) or ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.add_session_config_entry('session.intra_op.allow_spinning', '0')
    options.add_session_config_entry('session.inter_op.allow_spinning', '0')
    kwargs['sess_options'] = options
    return original_session(*args, **kwargs)

ort.InferenceSession = limited_session
sys.path.insert(0, '/opt/darkpipe/op')
sys.argv[0] = '/opt/darkpipe/op/main.py'
print('[audit] Diagnostic thread limits: torch=4, interop=1, cv2=2, ORT=1/no-spin', flush=True)
runpy.run_path('/opt/darkpipe/op/main.py', run_name='__main__')
