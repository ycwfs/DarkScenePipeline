"""One discarded train-only batch verifies CUDA kernels, AMP, gradients and memory."""
import copy
import os
import sys
from common import *
sys.path.insert(0,str(OP))
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from train_models import Windows
from darkpipe.stages.recognize import BehaviorRecognizer

torch.set_num_threads(3)
rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local_rank=int(os.environ.get('LOCAL_RANK',0))
if world>1:
    torch.cuda.set_device(local_rank);torch.distributed.init_process_group('nccl')
    os.environ['TRITON_CACHE_DIR']=f'/tmp/ft29-smoke-triton-rank{rank}'
rows=read(OUT/'dataset-plan.json')['records']
chosen=[];labels=set()
for i,r in enumerate(rows):
    if r['split']=='train' and Path(r['path']).exists() and r['label']not in labels:
        chosen.append(i);labels.add(r['label'])
for i,r in enumerate(rows):
    if len(chosen)>=8:break
    if r['split']=='train' and Path(r['path']).exists() and i not in chosen:chosen.append(i)
assert len(chosen)==8
rec=BehaviorRecognizer(str(OP/'ckpts'),skel_fusion_alpha=1.)
rec.ckpt=str(OP/'ckpts/videomamba_t_behavior_32f_person_roi_20261023.pth');rec.load(f'cuda:{local_rank}')
net=rec.net;teacher=copy.deepcopy(net).eval().requires_grad_(False)
if world>1:net=torch.nn.parallel.DistributedDataParallel(net,device_ids=[local_rank],broadcast_buffers=False)
x,y,w,old,_=next(iter(DataLoader(Windows(rows,chosen[rank::world],'rgb',None),batch_size=8//world)))
x=x.cuda();y=y.cuda();net.train();torch.cuda.reset_peak_memory_stats()
with torch.autocast('cuda',dtype=torch.float16):
    logits=net(x)
    with torch.no_grad():target=teacher(x)
    loss=F.cross_entropy(logits,y,label_smoothing=.02)+.5*F.kl_div(F.log_softmax(logits.float()/2,dim=1),F.softmax(target.float()/2,dim=1),reduction='batchmean')*4
scaler=torch.amp.GradScaler('cuda',init_scale=512.)
scaler.scale(loss).backward()
assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all()for p in net.parameters())
optimizer=torch.optim.AdamW(net.parameters(),lr=1e-6)
scaler.unscale_(optimizer);scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
with torch.autocast('cuda',dtype=torch.float16):second=F.cross_entropy(net(x),y)
scaler.scale(second).backward()
assert torch.isfinite(second) and all(p.grad is None or torch.isfinite(p.grad).all()for p in net.parameters())
result=dict(train_only=True,discarded_weights=True,batch=8,labels=[rows[i]['label']for i in chosen],loss=float(loss),
    peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,finite_gradients=True)
result.update(gpu_count=torch.cuda.device_count(),distributed_processes=world,rank=rank,steps=2,second_loss=float(second))
save(OUT/f'training-smoke-ddp{world}-2step-rank{rank}.json',result);print(result,flush=True)
if world>1:torch.distributed.barrier();torch.distributed.destroy_process_group()
