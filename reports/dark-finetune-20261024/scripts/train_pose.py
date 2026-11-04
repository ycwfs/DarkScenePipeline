"""RTMO-s fine-tuning using official losses and masked temporal auxiliary labels."""
import rtmo_support  # isolated numpy/MMCV before importing torch or numpy
import argparse
from collections import defaultdict
import json
import random
import time

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset

from prepare import OUT, ROOT, iou, save

FLIP=np.array([0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15])


def letterbox(image: np.ndarray) -> tuple:
    h,w=image.shape[:2];ratio=640/max(h,w)
    resized=cv2.resize(image,(int(w*ratio),int(h*ratio)),interpolation=cv2.INTER_LINEAR)
    padded=np.full((640,640,3),114,np.uint8)
    padded[:resized.shape[0],:resized.shape[1]]=resized
    return torch.from_numpy(padded.transpose(2,0,1).copy()),ratio


class PoseDataset(Dataset):
    def __init__(self,split: str, training=False):
        self.training=training
        samples=json.loads((OUT/'manifest.json').read_text())['samples']
        self.items=[]
        for s in samples:
            if s['split']!=split:continue
            # Do not turn a partially supervised person's body into background.
            if training and not all(p['pose_accepted'] for p in s['people']):continue
            for view in (['raw','enhanced'] if training else ['enhanced']):
                self.items.append((s,view))

    def __len__(self):return len(self.items)

    def __getitem__(self,index):
        s,view=self.items[index]
        path=OUT/f'dataset/images/{s["split"]}/{s["video_id"]}_{s["frame"]:06d}_{view}.png'
        image=cv2.imread(str(path))
        if image is None:raise IOError(path)
        h,w=image.shape[:2];scale=np.array([w/s['width'],h/s['height']])
        boxes=np.array([p['bbox_xyxy'] for p in s['people']],np.float32)*np.tile(scale,2)
        xy=np.array([p['keypoints_xy'] for p in s['people']],np.float32)*scale
        mask=np.array([p['keypoint_mask'] for p in s['people']],np.float32)
        if self.training and random.random()<.5:
            image=image[:,::-1].copy();boxes[:,[0,2]]=w-boxes[:,[2,0]]
            xy[:,:,0]=w-xy[:,:,0];xy=xy[:,FLIP];mask=mask[:,FLIP]
        tensor,ratio=letterbox(image)
        return tensor,dict(boxes=(boxes*ratio).astype(np.float32),
            xy=(xy*ratio).astype(np.float32),mask=mask,
            sample_id=s['id'],video_id=s['video_id'],frame=s['frame'],ratio=ratio)


def collate(batch):return torch.stack([x for x,y in batch]),[y for x,y in batch]


def data_samples(rows,device):
    from mmpose.structures import PoseDataSample
    from mmengine.structures import InstanceData
    result=[]
    for row in rows:
        b=torch.as_tensor(row['boxes'],device=device)
        sample=PoseDataSample(metainfo=dict(input_size=(640,640)))
        sample.gt_instance_labels=InstanceData(bboxes=b,labels=torch.zeros(len(b),dtype=torch.long,device=device),
            keypoints=torch.as_tensor(row['xy'],device=device),
            keypoints_visible=torch.as_tensor(row['mask'],device=device),
            keypoints_visible_weights=torch.as_tensor(row['mask'],device=device),
            areas=(b[:,2]-b[:,0])*(b[:,3]-b[:,1]))
        result.append(sample)
    return result


@torch.inference_mode()
def evaluate(model,loader,device='cuda:0'):
    model.eval();records=[]
    for images,rows in loader:
        images=images.to(device).float()
        result=model.head.predict(model.extract_feat(images),data_samples(rows,device),
                                  test_cfg=dict(input_size=(640,640),score_thr=.45,nms_thr=.45))
        for row,pred in zip(rows,result):
            boxes=np.asarray(pred.bboxes);truth=row['boxes']
            mat=np.array([[iou(a,b) for b in boxes] for a in truth])
            matches={}
            if len(boxes):
                ii,jj=linear_sum_assignment(1-mat)
                matches={int(i):int(j) for i,j in zip(ii,jj) if mat[i,j]>=.5}
            correct=0;labelled=int(row['mask'].sum());errors=[]
            for i,j in matches.items():
                valid=row['mask'][i]>0
                distance=np.linalg.norm(pred.keypoints[j]-row['xy'][i],axis=1)/max(np.linalg.norm(truth[i,2:]-truth[i,:2]),1)
                ok=(distance<=.1)&(pred.keypoint_scores[j]>=.3)&valid
                correct+=int(ok.sum());errors.extend(distance[valid].tolist())
            records.append(dict(sample_id=row['sample_id'],video_id=row['video_id'],frame=row['frame'],
                gt_boxes=len(truth),pred_boxes=len(boxes),matched=len(matches),
                labelled_joints=labelled,correct_joints=correct,normalized_errors=errors,
                boxes=boxes.tolist(),keypoints=np.asarray(pred.keypoints).tolist(),
                keypoint_scores=np.asarray(pred.keypoint_scores).tolist(),scores=np.asarray(pred.scores).tolist()))
    labelled=sum(x['labelled_joints'] for x in records);gt=sum(x['gt_boxes'] for x in records)
    matched=sum(x['matched'] for x in records);pred=sum(x['pred_boxes'] for x in records)
    summary=dict(images=len(records),labelled_joints=labelled,
        pck10_including_missed=sum(x['correct_joints'] for x in records)/max(labelled,1),
        auxiliary_box_recall=matched/max(gt,1),unmatched_predictions=pred-matched,
        label_status='filtered auxiliary labels, not independent human GT')
    return summary,records


def train(args):
    torch.set_num_threads(2);cv2.setNumThreads(1)
    random.seed(24);np.random.seed(24);torch.manual_seed(24);torch.cuda.manual_seed_all(24)
    model=rtmo_support.load_model(OUT/'models/rtmo-s_body7_original.pth','cuda:0')
    model.head.assigner.vis_weight=0.  # unknown joints are not visibility evidence
    for name,param in model.named_parameters():
        if name.startswith('backbone.') or 'out_kpt_vis' in name:param.requires_grad=False
    groups=[dict(params=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('neck.')],lr=2e-5),
            dict(params=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('head.')],lr=5e-5)]
    optimizer=torch.optim.AdamW(groups,weight_decay=.01)
    trainset=PoseDataset('train',True);valset=PoseDataset('val')
    loader=DataLoader(trainset,batch_size=8,shuffle=True,num_workers=2,pin_memory=True,collate_fn=collate)
    val_loader=DataLoader(valset,batch_size=8,num_workers=2,collate_fn=collate)
    config=dict(epochs=args.epochs,train_images=len(trainset),val_images=len(valset),
        lr_neck=2e-5,lr_head=5e-5,freeze='backbone and visibility output; all BatchNorm running statistics',
        official_loss=True,unknown_visibility_loss_weight=0.,seed=24,batch=8,amp=False)
    save(OUT/'pose-training-config.json',config)
    summary,records=evaluate(model,val_loader)
    save(OUT/'evaluation/pose-val-epoch00.json',dict(summary=summary,records=records))
    best=summary['pck10_including_missed'];best_epoch=0;history=[dict(epoch=0,**summary)]
    destination=OUT/'models/rtmo-s_dark_candidate.pth'
    torch.save(dict(state_dict=model.cpu().state_dict(),epoch=0,config=config),destination)
    model.cuda();print('[pose baseline]',summary,flush=True)
    for epoch in range(1,args.epochs+1):
        model.train();model.backbone.eval()
        for module in model.modules():
            if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()
        totals=defaultdict(float);start=time.monotonic()
        for step,(images,rows) in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            losses=model(images.cuda(non_blocking=True).float(),data_samples(rows,'cuda:0'),mode='loss')
            total=sum(v.mean() for k,v in losses.items() if k.startswith('loss'))
            if not torch.isfinite(total):raise RuntimeError(f'nonfinite loss epoch={epoch} step={step}')
            total.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise RuntimeError(f'nonfinite gradient {epoch} {step}')
            optimizer.step()
            for k,v in losses.items():
                if k.startswith('loss'):totals[k]+=float(v.detach().mean())
            if step%50==0:print(f'[pose] epoch={epoch} step={step}/{len(loader)} loss={float(total):.5f}',flush=True)
            if args.smoke:break
        summary,records=evaluate(model,val_loader)
        row=dict(epoch=epoch,**summary,losses={k:v/(step+1) for k,v in totals.items()},seconds=time.monotonic()-start)
        history.append(row)
        save(OUT/f'evaluation/pose-val-epoch{epoch:02d}.json',dict(summary=row,records=records))
        if summary['pck10_including_missed']>best:
            best=summary['pck10_including_missed'];best_epoch=epoch
            torch.save(dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()},epoch=epoch,config=config),destination)
        save(OUT/'pose-training-history.json',dict(history=history,best_epoch=best_epoch))
        print('[pose epoch]',row,'best',best_epoch,flush=True)
        if args.smoke:break
    torch.save(dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()},epoch=epoch,config=config),OUT/'models/rtmo-s_dark_last.pth')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--smoke',action='store_true');args=parser.parse_args();train(args)
