import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'NVIDIA-Box/runtime'))
from darkpipe.stages.enhance_retinexformer import RetinexformerStage
from darkpipe.utils import bgr_batch_to_tensor, tensor_to_bgr_list
torch.set_num_threads(2)
stage=RetinexformerStage(str(ROOT/'NVIDIA-Box/weights'));stage.load('cuda:0')
results={}
with torch.inference_mode():
    raw=np.random.default_rng(42).integers(0,256,(404,720,3),dtype=np.uint8)
    x=bgr_batch_to_tensor([raw],'cuda:0',torch.float16)
    expected=stage.net(x)
    for layout in ['original','contiguous']:
        a=x if layout=='original' else x.contiguous()
        for graph_mode in [False,True]:
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3): stage.net(a)
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            if graph_mode:
                with torch.cuda.graph(graph): output=stage.net(a)
            times=[]
            for _ in range(20):
                torch.cuda.synchronize();start=time.perf_counter()
                if graph_mode: graph.replay()
                else: output=stage.net(a)
                shown=tensor_to_bgr_list(output)
                times.append(time.perf_counter()-start)
            results[f'{layout}_{graph_mode}']=dict(ms=np.mean(times)*1000,
                max_diff=float((expected-output).abs().max()),stride=list(a.stride()))
            print(layout,graph_mode,results[f'{layout}_{graph_mode}'],flush=True)
    (Path(__file__).resolve().parents[1]/'layout-probe.json').write_text(json.dumps(results,indent=2)+'\n')
