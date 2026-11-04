"""Probe native PyTorch compiler on the unmodified enhancement network."""
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'NVIDIA-Box/runtime'))
from darkpipe.stages.enhance_retinexformer import RetinexformerStage
from darkpipe.utils import bgr_batch_to_tensor,tensor_to_bgr_list
torch.set_num_threads(2)
stage=RetinexformerStage(str(ROOT/'NVIDIA-Box/weights'));stage.load('cuda:0')
results={}
with torch.inference_mode():
    raw=np.random.default_rng(42).integers(0,256,(404,720,3),dtype=np.uint8)
    x=bgr_batch_to_tensor([raw],'cuda:0',torch.float16)
    expected=stage.net(x)
    for mode in ['eager','compile']:
        net=stage.net if mode=='eager' else torch.compile(stage.net,mode='reduce-overhead',fullgraph=True)
        start=time.perf_counter()
        for _ in range(3):net(x)
        torch.cuda.synchronize();startup=time.perf_counter()-start
        times=[]
        for _ in range(30):
            start=time.perf_counter();output=net(x);tensor_to_bgr_list(output)
            times.append(time.perf_counter()-start)
        results[mode]=dict(ms=np.mean(times)*1000,startup_seconds=startup,
            max_difference=float((expected-output).abs().max()))
        print(mode,results[mode],flush=True)
        (Path(__file__).resolve().parents[1]/'compile-probe.json').write_text(json.dumps(results,indent=2)+'\n')
