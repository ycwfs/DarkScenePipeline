"""Conservative branch adaptation with old-source replay and old-teacher distillation."""
import argparse
from collections import Counter
import copy
import faulthandler
import math
import os
import random
import sys
import time
from common import *
sys.path.insert(0, str(OP))
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader,Dataset,WeightedRandomSampler,Sampler
from darkpipe.constants import IN_MEAN,IN_STD
from darkpipe.stages.recognize import BehaviorRecognizer
from darkpipe.stages.skeleton_aux import STGCN


from evaluation_metrics import metrics


def evaluation_indices(rows, branch, valid):
    return [i for i,r in enumerate(rows) if r['split']=='val' and r['label_kind']!='pseudo'
        and (r['scope']=='new' or r.get('current_index') is not None)
        and (branch=='rgb' or valid[i])]


def score(rows, ids, probs):
    pred=probs.argmax(1);pred[(pred!=9)&(probs.max(1)<.75)]=9
    return {domain:metrics([rows[i]for j,i in enumerate(ids)if rows[i]['scope']==domain],
        [pred[j]for j,i in enumerate(ids)if rows[i]['scope']==domain])for domain in ('old','new')}


def select_key(result, baseline):
    accepted=all(result[d]['positive_macro_recall']>=baseline[d]['positive_macro_recall']-delta-1e-9
        for d,delta in [('old',.05),('new',.10)])
    fpr=sum(result[d]['other_fpr']for d in ('old','new'))/2
    return (accepted,-fpr,sum(result[d]['macro_f1']for d in ('old','new'))/2,
            sum(result[d]['positive_tp']for d in ('old','new')))


class Windows(Dataset):
    def __init__(self, rows, ids, branch, features, augment=False):
        self.rows,self.ids,self.branch,self.features,self.augment=rows,ids,branch,features,augment

    def __len__(self):return len(self.ids)

    def __getitem__(self,k):
        i=self.ids[k];r=self.rows[i]
        if self.branch=='rgb':
            with np.load(r['path'])as a:x=a['rgb']
            if self.augment and random.random()<.5:x=x[:,:,::-1]
            x=(np.ascontiguousarray(x).astype(np.float32)/255-IN_MEAN)/IN_STD
            x=torch.from_numpy(x).permute(3,0,1,2).contiguous()
        else:x=torch.from_numpy(self.features[i])
        return x,LABELS.index(r['label']),r['weight'],r['scope']=='old',i


def weights(rows,ids):
    counts=Counter((rows[i]['label'],rows[i]['scope'],rows[i]['source_id'])for i in ids)
    sources=Counter((c,d)for c,d,s in counts)
    domains={c:{d for cc,d in sources if cc==c}for c in LABELS}
    output=[]
    for i in ids:
        r=rows[i];c,d,s=r['label'],r['scope'],r['source_id']
        mass=(.7 if d=='old' else .3) if len(domains[c])==2 else 1.
        output.append(mass/(sources[c,d]*counts[c,d,s]))
    return output


class ShardedWeightedSampler(Sampler):
    """The same global weighted draw, partitioned without overlap across ranks."""
    def __init__(self, values, count, rank, world, seed):
        self.values=torch.as_tensor(values,dtype=torch.double)
        self.count=count//8*8;self.rank=rank;self.world=world;self.seed=seed;self.epoch=0
    def __len__(self):return self.count//self.world
    def __iter__(self):
        generator=torch.Generator().manual_seed(self.seed+self.epoch)
        drawn=torch.multinomial(self.values,self.count,True,generator=generator).tolist()
        return iter(drawn[self.rank::self.world])


@torch.inference_mode()
def infer(net,loader,branch):
    net.eval();parts=[]
    for x,*_ in loader:
        with torch.autocast('cuda',dtype=torch.float16,enabled=branch=='rgb'):
            p=net(x.cuda(non_blocking=True)).softmax(1).float().cpu().numpy()
        parts.append(p)
    return np.concatenate(parts) if parts else np.zeros((0,10),np.float32)


def main():
    p=argparse.ArgumentParser();p.add_argument('branch',choices=['rgb','skeleton']);a=p.parse_args()
    protocol=read(OUT/'protocol.json');cfg=protocol['training'];data=read(OUT/'dataset.json');assert data['ready']
    assert data['protocol_sha256']==sha(OUT/'protocol.json')
    branch=a.branch;dest=OUT/branch;dest.mkdir(exist_ok=True)
    faulthandler.enable();faulthandler.dump_traceback_later(180,repeat=True)
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local_rank=int(os.environ.get('LOCAL_RANK',0))
    distributed=world>1
    if distributed:
        assert branch=='rgb'
        torch.cuda.set_device(local_rank);dist.init_process_group('nccl')
        os.environ['TRITON_CACHE_DIR']=f'/tmp/ft29-triton-rank{rank}'
    assert not (dest/'history.json').exists(),'Preserve existing run; choose an explicit new experiment for retries'
    torch.set_num_threads(3);random.seed(cfg['seed']+rank);np.random.seed(cfg['seed']+rank);torch.manual_seed(cfg['seed']+rank)
    rows=data['records'];cache=np.load(OUT/'skeleton-inputs.npz');gi=GATES.index(35.)
    features=cache['features'][gi];valid=cache['valid'][gi]
    train=[i for i,r in enumerate(rows)if r['split']=='train' and (branch=='rgb' or valid[i])]
    val=evaluation_indices(rows,branch,valid)
    init=Path(protocol['models'][branch]['path']);assert sha(init)==protocol['models'][branch]['sha256']
    metadata=torch.load(init,map_location='cpu',weights_only=False)
    if branch=='rgb':
        rec=BehaviorRecognizer(str(OP/'ckpts'),skel_fusion_alpha=1.);rec.ckpt=str(init);rec.load(f'cuda:{local_rank}');net=rec.net
    else:
        net=STGCN(num_classes=10);net.load_state_dict(metadata.get('model',metadata));net=net.cuda()
    teacher=copy.deepcopy(net).eval().requires_grad_(False)
    # Triton autotuners are not thread-safe: isolate each GPU in a DDP process.
    if distributed:net=DistributedDataParallel(net,device_ids=[local_rank],broadcast_buffers=False)
    batch=cfg['rgb_batch'] if branch=='rgb' else 64
    epochs=cfg['rgb_epochs'] if branch=='rgb' else cfg['skeleton_epochs']
    sampler=(ShardedWeightedSampler(weights(rows,train),len(train),rank,world,cfg['seed'])if distributed else
        WeightedRandomSampler(weights(rows,train),len(train),replacement=True,generator=torch.Generator().manual_seed(cfg['seed'])))
    worker_options=dict(multiprocessing_context='spawn',persistent_workers=True)if branch=='rgb'else {}
    loader=DataLoader(Windows(rows,train,branch,features,True),batch_size=batch//world,sampler=sampler,
        num_workers=3 if branch=='rgb' else 0,pin_memory=True,drop_last=True,**worker_options)
    validation=DataLoader(Windows(rows,val,branch,features),batch_size=batch,shuffle=False,
        num_workers=2 if branch=='rgb' else 0,pin_memory=True,**worker_options)
    if branch=='rgb':
        # Continue from the same declared initialization across all six fixed epochs.
        optimizer=torch.optim.AdamW([dict(params=[p for n,p in net.named_parameters()if not n.removeprefix('module.').startswith('head.')],lr=cfg['backbone_lr']),
            dict(params=[p for n,p in net.named_parameters()if n.removeprefix('module.').startswith('head.')],lr=cfg['head_lr'])],weight_decay=.05)
    else:optimizer=torch.optim.AdamW(net.parameters(),lr=cfg['skeleton_lr'],weight_decay=.001)
    steps=max(1,epochs*len(loader));warm=max(1,len(loader)//2)
    schedule=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda k:(k+1)/warm if k<warm else .5*(1+math.cos(math.pi*min(1.,(k-warm)/max(steps-warm,1)))))
    scaler=torch.amp.GradScaler('cuda',enabled=branch=='rgb',init_scale=512.)
    baseline=None;best=None;best_epoch=0;history=[]
    for epoch in range(epochs+1):
        start=time.monotonic();losses=[];skipped=0
        if epoch:
            if distributed:sampler.epoch=epoch
            net.train()
            for step,(x,y,w,old,_)in enumerate(loader,1):
                x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);w=w.cuda().float();old=old.cuda()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16,enabled=branch=='rgb'):
                    logits=net(x);loss=(F.cross_entropy(logits,y,reduction='none',label_smoothing=.02)*w).mean()
                    old_count=old.sum().float()
                    if distributed:dist.all_reduce(old_count)
                    if old.any():
                        with torch.no_grad():target=teacher(x[old])
                        temperature=cfg['distillation_temperature']
                        kd=F.kl_div(F.log_softmax(logits[old].float()/temperature,dim=1),F.softmax(target.float()/temperature,dim=1),reduction='sum')*temperature**2*world/old_count.clamp_min(1)
                        loss=loss+cfg['distillation_old']*kd
                if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite {branch} epoch{epoch} step{step}')
                scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(net.parameters(),5.)
                before=scaler.get_scale();scaler.step(optimizer);scaler.update()
                if scaler.get_scale()>=before:schedule.step()
                else:skipped+=1
                losses.append(float(loss.detach()))
                if step==1 or step%10==0:
                    faulthandler.cancel_dump_traceback_later();faulthandler.dump_traceback_later(180,repeat=True)
                    if rank==0:print(branch,'EPOCH',epoch,'STEP',step,len(loader),'loss',float(np.mean(losses)),'seconds',time.monotonic()-start,flush=True)
            assert skipped<len(loader)
        average_loss=float(np.mean(losses))if losses else None
        if distributed and losses:
            total_loss=torch.tensor([sum(losses),len(losses)],device=f'cuda:{local_rank}',dtype=torch.float64)
            dist.all_reduce(total_loss);average_loss=float(total_loss[0]/total_loss[1])
        if distributed:
            dist.barrier()
            if rank!=0:
                dist.barrier()
                continue
        bare=net.module if distributed else net
        probs=infer(bare,validation,branch);result=score(rows,val,probs)
        if baseline is None:baseline=result
        key=select_key(result,baseline)
        row=dict(epoch=epoch,loss=average_loss,skipped=skipped,seconds=time.monotonic()-start,metrics=result,key=key)
        history.append(row);np.savez_compressed(dest/f'val-{epoch:02d}.npz',indices=np.array(val),probs=probs)
        if epoch:
            saved={k:v for k,v in metadata.items()if k not in ('model','training','training_metrics','validation')}
            saved.update(model={k:v.detach().cpu()for k,v in bare.state_dict().items()},epoch=epoch,training=protocol,
                source='old person ROI replay + user-reviewed crowded person windows + weak other; no XCLIP',
                dataset_sha256=sha(OUT/'dataset.json'),validation=result,class_order=LABELS,
                seen=[c for c in LABELS if any(rows[i]['label']==c for i in train)],
                training_amendment_sha256=sha(OUT/'protocol-amendment-before-training.json'),
                framework=dict(torch=torch.__version__,cuda=torch.version.cuda,world_size=world))
            if branch=='rgb':saved.update(preprocess='fullfov224x400',input_scope='tracked_person_roi',person_preprocessing=dict(margin=.15,pad=114,span_sec=1.,frames=32,proc_max_side=720))
            torch.save(saved,dest/f'epoch-{epoch:02d}.pth')
        if best is None or key>best:best=key;best_epoch=epoch
        save(dest/'history.json',history)
        print(branch,'VALIDATION',json.dumps(row),flush=True)
        if distributed:dist.barrier()
    if rank==0:
        chosen=init if best_epoch==0 else dest/f'epoch-{best_epoch:02d}.pth'
        save(dest/'selection.json',dict(epoch=best_epoch,path=str(chosen),sha256=sha(chosen),key=best,
            train_windows=len(train),validation_windows=len(val),test_used=False,protocol_sha256=sha(OUT/'protocol.json'),
            distributed_processes=world,global_batch=batch))
    if distributed:dist.barrier();dist.destroy_process_group()
    faulthandler.cancel_dump_traceback_later()


if __name__=='__main__':main()
