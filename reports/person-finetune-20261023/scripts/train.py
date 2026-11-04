"""Fine-tune VideoMamba on reviewed person windows; select only on frozen validation."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'reports/person-action-20261023/scripts'))
from person_core import OP, BEHAVIORS, fuse_person
from darkpipe.constants import IN_MEAN, IN_STD
from darkpipe.stages.recognize import BehaviorRecognizer, reject_probs


def write_json(path: Path, value: object) -> None:
    pending=path.with_suffix(path.suffix+'.tmp')
    pending.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    pending.replace(path)


class Windows(Dataset):
    def __init__(self, records: list, augment: bool=False):
        self.records,self.augment=records,augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple:
        row=self.records[index]
        with np.load(OUT/row['path']) as archive:
            x=archive['rgb']
        if x.shape!=(32,224,400,3) or x.dtype!=np.uint8:
            raise ValueError(f'Unexpected person input: {row["path"]}: {x.shape}, {x.dtype}')
        if self.augment and random.random()<.5:
            x=x[:,:,::-1]
        x=(np.ascontiguousarray(x).astype(np.float32)/255.-IN_MEAN)/IN_STD
        return torch.from_numpy(x).permute(3,0,1,2).contiguous(), BEHAVIORS.index(row['label']),index


def sampling_weights(records: list) -> list:
    counts=Counter((row['label'],row['video_id']) for row in records)
    videos=Counter(label for label,_ in counts)
    return [1./(videos[row['label']]*counts[row['label'],row['video_id']]) for row in records]


def metrics(truth: list, predicted: list) -> dict:
    cm=np.zeros((10,10),np.int64)
    for actual,guess in zip(truth,predicted):
        cm[actual,guess]+=1
    support=cm.sum(1)
    f1=2*np.diag(cm)/np.maximum(cm.sum(0)+support,1)
    other=BEHAVIORS.index('other')
    return dict(n=len(truth),correct=int(np.trace(cm)),
                top1=float(np.trace(cm)/max(len(truth),1)),
                macro_f1=float(f1[support>0].mean()) if np.any(support) else 0.,
                other_count=int(support[other]),other_false_positives=int(support[other]-cm[other,other]),
                other_fpr=float((support[other]-cm[other,other])/max(support[other],1)),
                classes={label:dict(n=int(support[i]),recall=float(cm[i,i]/max(support[i],1)),
                                   f1=float(f1[i])) for i,label in enumerate(BEHAVIORS)},
                confusion_matrix=cm.tolist())


@torch.inference_mode()
def evaluate(net: torch.nn.Module, records: list, batch: int=8) -> tuple:
    net.eval()
    loader=DataLoader(Windows(records),batch_size=batch,shuffle=False,num_workers=2,pin_memory=True)
    outputs=[]
    for x,_,indices in loader:
        with torch.autocast('cuda',dtype=torch.float16):
            probabilities=net(x.cuda(non_blocking=True)).softmax(1).float().cpu().numpy()
        for index,p in zip(indices.tolist(),probabilities):
            row=records[index]
            sk=None if row['skeleton_probs'] is None else np.array(row['skeleton_probs'],np.float32)
            final=fuse_person(p,sk,row['lumas'])
            outputs.append(dict(id=row['id'],video_id=row['video_id'],group=row['group'],
                track_id=row['track_id'],source=row['source'],start=row['start'],end=row['end'],
                truth=row['label'],sealed_feedback_holdout=row['sealed_feedback_holdout'],
                rgb_probs=p.tolist(),rgb_top1=BEHAVIORS[int(p.argmax())],
                rgb_label=BEHAVIORS[int(reject_probs(p,BEHAVIORS,.75).argmax())],
                fused_label=final['label'],fused_probs=final['fused_probs'],mode=final['mode'],
                skeleton_probs=row['skeleton_probs'],
                skeleton_label=None if sk is None else BEHAVIORS[int(reject_probs(sk,BEHAVIORS,.75).argmax())]))
    truth=[BEHAVIORS.index(row['truth']) for row in outputs]
    result={mode:metrics(truth,[BEHAVIORS.index(row[key]) for row in outputs])
            for mode,key in [('rgb','rgb_label'),('fused','fused_label'),('rgb_argmax','rgb_top1')]}
    valid=[row for row in outputs if row['skeleton_label'] is not None]
    result['skeleton']=metrics([BEHAVIORS.index(row['truth']) for row in valid],
                               [BEHAVIORS.index(row['skeleton_label']) for row in valid])
    result['videos']=len({row['video_id'] for row in records})
    return result,outputs


def selection_key(result: dict) -> tuple:
    return (result['fused']['macro_f1'],result['rgb']['macro_f1'],-result['fused']['other_fpr'])


def main() -> None:
    protocol=json.loads((OUT/'protocol.json').read_text())
    config=protocol['candidate']
    dataset=json.loads((OUT/'training-data.json').read_text())
    assert dataset['ready_for_training']
    assert hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()==dataset['audit']['protocol_sha256']
    assert hashlib.sha256((OUT/'annotations.json').read_bytes()).hexdigest()==dataset['audit']['annotations_sha256']
    records=dataset['records']
    train=[row for row in records if row['split']=='train']
    val=[row for row in records if row['split']=='val']
    assert train and val
    assert not ({row['group'] for row in train}&{row['group'] for row in val})
    assert not any(row['sealed_feedback_holdout'] for row in train+val)
    torch.set_num_threads(4)
    torch.manual_seed(config['seed']); np.random.seed(config['seed']); random.seed(config['seed'])
    ckpt=OP/'ckpts'/protocol['init']
    assert hashlib.sha256(ckpt.read_bytes()).hexdigest()==protocol['initial_sha256']
    rec=BehaviorRecognizer(str(ckpt.parent),skel_fusion_alpha=1.)
    rec.ckpt=str(ckpt); rec.load('cuda:0')
    net=rec.net
    meta=torch.load(ckpt,map_location='cpu',weights_only=False)
    checkpoints=OUT/'checkpoints'; checkpoints.mkdir(exist_ok=True)
    evaluation=OUT/'evaluation'; evaluation.mkdir(exist_ok=True)
    if (OUT/'training-history.json').exists() or any(checkpoints.glob('epoch-*.pth')):
        raise RuntimeError('Existing training run; do not silently overwrite experiment evidence')
    sampler=WeightedRandomSampler(sampling_weights(train),len(train),replacement=True,
                                  generator=torch.Generator().manual_seed(config['seed']))
    loader=DataLoader(Windows(train,True),batch_size=config['batch'],sampler=sampler,
                      num_workers=2,pin_memory=True,drop_last=True,persistent_workers=True)
    if not len(loader):
        raise ValueError('No complete training batch')
    optimizer=torch.optim.AdamW([
        dict(params=[p for name,p in net.named_parameters() if not name.startswith('head.')],lr=config['backbone_lr']),
        dict(params=[p for name,p in net.named_parameters() if name.startswith('head.')],lr=config['head_lr'])],
        weight_decay=config['weight_decay'])
    total=config['epochs']*len(loader)
    warm=max(1,len(loader)//2)
    schedule=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:
        (step+1)/warm if step<warm else .5*(1+math.cos(math.pi*min(1.,(step-warm)/max(total-warm,1)))))
    scaler=torch.amp.GradScaler('cuda',init_scale=config['amp_initial_scale'])
    result,outputs=evaluate(net,val)
    write_json(evaluation/'val-epoch-00.json',dict(metrics=result,records=outputs))
    hist=[dict(epoch=0,validation=result)]
    write_json(OUT/'training-history.json',hist)
    best,best_epoch=selection_key(result),0
    print('[baseline]',json.dumps(result),flush=True)
    for epoch in range(1,config['epochs']+1):
        net.train(); started=time.monotonic(); sums=Counter()
        for step,(x,y,_) in enumerate(loader,1):
            x=x.cuda(non_blocking=True); y=y.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                logits=net(x)
                loss=F.cross_entropy(logits,y,label_smoothing=config['label_smoothing'])
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss: epoch={epoch}, step={step}')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(),5.)
            old_scale=scaler.get_scale()
            scaler.step(optimizer); scaler.update()
            if scaler.get_scale()>=old_scale:
                schedule.step()
            else:
                sums['skipped_updates']+=1
            sums.update(loss=float(loss)*len(y),examples=len(y),correct=int((logits.argmax(1)==y).sum()))
            if step==1 or step%50==0 or step==len(loader):
                print(f'[train] epoch={epoch} step={step}/{len(loader)} loss={sums["loss"]/sums["examples"]:.4f} '
                      f'acc={sums["correct"]/sums["examples"]:.4f} elapsed={time.monotonic()-started:.1f}s',flush=True)
        result,outputs=evaluate(net,val)
        record=dict(epoch=epoch,loss=sums['loss']/sums['examples'],
                    train_top1=sums['correct']/sums['examples'],examples=sums['examples'],
                    skipped_updates=sums['skipped_updates'],seconds=time.monotonic()-started,validation=result)
        if sums['skipped_updates']==len(loader):
            raise RuntimeError('All optimizer updates skipped; refusing to save a trained checkpoint')
        save={key:value for key,value in meta.items() if key not in
              ['model','training','training_metrics','realact_top1','realact_macro_f1','arid_top1','arid_macro_f1',
               'seen','archive_sha256']}
        save.update(model={key:value.detach().cpu() for key,value in net.state_dict().items()},
                    epoch=epoch,init=ckpt.name,source='reviewed local tracked-person ROI data',
                    preprocess='fullfov224x400',input_scope='tracked_person_roi',
                    person_preprocessing=dict(margin=.15,pad=114,span_sec=1.,frames=32,proc_max_side=720),
                    seen=[label for label in BEHAVIORS if any(row['label']==label for row in train)],
                    training=protocol,training_metrics=record,lwf=0.,lwf_scope='disabled for person-domain adaptation',
                    training_data_sha256=hashlib.sha256((OUT/'training-data.json').read_bytes()).hexdigest())
        torch.save(save,checkpoints/f'epoch-{epoch:02d}.pth')
        write_json(evaluation/f'val-epoch-{epoch:02d}.json',dict(metrics=result,records=outputs))
        hist.append(record); write_json(OUT/'training-history.json',hist)
        if selection_key(result)>best:
            best,best_epoch=selection_key(result),epoch
        print(f'[epoch] {epoch}: val RGB F1={result["rgb"]["macro_f1"]:.4f}, '
              f'fused F1={result["fused"]["macro_f1"]:.4f}, best={best_epoch}',flush=True)
    selected=ckpt if best_epoch==0 else checkpoints/f'epoch-{best_epoch:02d}.pth'
    write_json(OUT/'selection.json',dict(epoch=best_epoch,path=str(selected),
        sha256=hashlib.sha256(selected.read_bytes()).hexdigest(),key=best,
        selected_on='validation only',test_evaluated=False))
    rec.close()


if __name__=='__main__':
    main()
