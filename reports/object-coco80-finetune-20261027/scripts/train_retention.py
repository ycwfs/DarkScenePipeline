"""Conservative validation-driven alternative: update six existing classifier rows only."""
import json
import os
os.environ['YOLO_AUTOINSTALL']='false'

import cv2
import torch
from ultralytics import YOLO

from common import BASE, OUT, save
from train import LocalTrainer

TARGETS=(0,39,41,65,67,73)


class RetentionTrainer(LocalTrainer):
    def optimizer_step(self):
        super().optimizer_step()
        # EMA must not introduce rounding drift into the explicitly frozen model.
        if self.ema:
            source=self.model.state_dict();dest=self.ema.ema.state_dict()
            protected=[c for c in range(80)if c not in TARGETS]
            for name,tensor in dest.items():
                if name in self.retention_heads:
                    tensor[protected]=source[name][protected].to(tensor.dtype)
                else:
                    tensor.copy_(source[name])


def main():
    if (OUT/'runs/retention').exists():raise FileExistsError('No implicit overwrite')
    torch.set_num_threads(2);cv2.setNumThreads(1)
    model=YOLO(str(BASE));names=json.loads((OUT/'classes.json').read_text())
    original={k:v.detach().cpu().float().clone()for k,v in model.model.state_dict().items()}
    def configure(trainer):
        sd=trainer.model.state_dict()
        assert all(torch.equal(sd[k].detach().cpu().float(),v)for k,v in original.items())
        assert list(trainer.model.names.values())==names
        heads=[]
        for name,p in trainer.model.named_parameters():
            selected=('.cv3.'in name or'.one2one_cv3.'in name)and p.ndim and p.shape[0]==80
            p.requires_grad_(bool(selected))
            if selected:
                heads.append(name)
                mask=torch.zeros_like(p);mask[list(TARGETS)]=1
                p.register_hook(lambda grad,mask=mask:grad*mask)
        assert len(heads)==12,heads
        trainer.retention_heads=heads
        for group in trainer.optimizer.param_groups:group['weight_decay']=0.
        save(OUT/'retention-initialization.json',dict(passed=True,target_rows=TARGETS,
            classifier_tensors=heads,shared_features_and_boxes_frozen=True,other74_rows_frozen=True,
            batchnorm_stats_frozen=True,weight_decay=0,ema_frozen_tensors_restored_each_step=True))
    def freeze_norm(trainer):
        for module in trainer.model.modules():
            if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()
    model.add_callback('on_pretrain_routine_end',configure)
    model.add_callback('on_train_batch_start',freeze_norm)
    save(OUT/'retention-protocol.json',dict(reason='Initial broad adaptation failed generic validation guard; no test results inspected.',
        baseline_coco_val=.5251037728611181,broad_first2_val=[.449,.4525],
        strategy='Update only original class logits for person,bottle,cup,remote,cell phone,book; freeze all features, localization, BN and other rows.',
        targets=TARGETS,epochs=20,seed=27,lr0=.0002,weight_decay=0,data='Same frozen dataset as broad adaptation',
        selection='Same generic validation guard and site val objective. Final compare both training families using validation only.'))
    result=model.train(trainer=RetentionTrainer,data=str(OUT/'objects.yaml'),epochs=20,imgsz=640,batch=16,
        device=0,workers=4,project=str(OUT/'runs'),name='retention',exist_ok=False,
        seed=27,deterministic=True,optimizer='AdamW',lr0=.0002,lrf=.2,
        weight_decay=0.,warmup_epochs=1,freeze=10,patience=20,amp=True,
        cache=False,mosaic=0.,mixup=0.,copy_paste=0.,degrees=0.,translate=.03,
        scale=.1,fliplr=.5,hsv_h=.005,hsv_s=.1,hsv_v=.1,close_mosaic=0,
        plots=False,save_period=1,val=True,pretrained=True,nms=False)
    save(OUT/'retention-training-result.json',dict(metrics=result.results_dict,last=str(model.trainer.last)))


if __name__=='__main__':main()
