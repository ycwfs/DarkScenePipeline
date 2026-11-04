"""Measure captured execution of existing trained networks with changed-input checks."""
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'NVIDIA-Box/runtime'))
from darkpipe.stages.enhance_retinexformer import RetinexformerStage
from darkpipe.stages.recognize import BehaviorRecognizer

torch.set_num_threads(2)
torch.cuda.set_device(0)
results = {}
with torch.inference_mode():
    stage = RetinexformerStage(str(ROOT/'NVIDIA-Box/weights'))
    stage.load('cuda:0')
    rgb = BehaviorRecognizer(str(ROOT/'NVIDIA-Box/weights'), skel_fusion_alpha=1.)
    rgb.ckpt = str(ROOT/'NVIDIA-Box/weights/videomamba_t_behavior_32f_person_roi_20261023.pth')
    rgb.load('cuda:0')
    for name, net, shape, dtype in [('enhance',stage.net,(1,3,404,720),torch.float16),
                                   ('rgb1',rgb.net,(1,3,32,224,400),torch.float32),
                                   ('rgb2',rgb.net,(2,3,32,224,400),torch.float32)]:
        x = torch.rand(shape, device='cuda', dtype=dtype)
        context = torch.autocast('cuda',dtype=torch.float16)
        with context:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    net(x)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_y = net(x)
            durations = {}
            for mode in ['eager', 'graph']:
                samples=[]
                for _ in range(30):
                    started=time.perf_counter()
                    if mode=='eager': net(x)
                    else: graph.replay()
                    torch.cuda.synchronize()
                    samples.append(time.perf_counter()-started)
                durations[mode] = dict(mean_ms=float(np.mean(samples))*1000,
                    p95_ms=float(np.percentile(samples,95))*1000, n=len(samples))
            differences=[]
            for _ in range(3):
                x.uniform_()
                expected=net(x)
                graph.replay()
                differences.append(float((expected-static_y).abs().max()))
            results[name]=dict(timing=durations,maximum_absolute_difference=max(differences))
            print(name, results[name],flush=True)
        del graph, x, static_y
    Path(__file__).resolve().parents[1].joinpath('graph-probe.json').write_text(json.dumps(results,indent=2)+'\n')
