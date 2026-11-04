"""Keep all 80 original classifier rows and train with complete COCO replay."""
import csv
import json
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ['YOLO_AUTOINSTALL'] = 'false'

import cv2
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer

from common import BASE, OUT, save


class LocalTrainer(DetectionTrainer):
    def read_results_csv(self):
        with self.csv.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        return {k.strip(): [float(r[k]) for r in rows] for k in rows[0]} if rows else {}


def main():
    if (OUT/'runs/coco80').exists():
        raise FileExistsError('No implicit retraining over existing run')
    torch.set_num_threads(2); cv2.setNumThreads(1)
    model = YOLO(str(BASE))
    original = {k: v.detach().cpu().float().clone() for k,v in model.model.state_dict().items()}
    names = json.loads((OUT/'classes.json').read_text())
    assert list(model.names.values()) == names and len(names) == 80
    def verify(trainer):
        assert list(trainer.model.names.values()) == names
        loaded = trainer.model.state_dict()
        assert loaded.keys() == original.keys()
        mismatches = [k for k in original if not torch.equal(loaded[k].detach().cpu().float(), original[k])]
        assert not mismatches, mismatches
        heads = [k for k,v in loaded.items() if ('.cv3.' in k or '.one2one_cv3.' in k) and v.ndim and v.shape[0]==80]
        assert len(heads) == 12, heads
        save(OUT/'initialization-check.json',dict(passed=True,all_tensors=len(original),preserved_80class_outputs=heads,
            phone_row=67,bottle_row=39,cup_row=41,all_names_identical=True))
    model.add_callback('on_pretrain_routine_end',verify)
    result = model.train(trainer=LocalTrainer,data=str(OUT/'objects.yaml'),epochs=20,imgsz=640,batch=16,
        device=0,workers=4,project=str(OUT/'runs'),name='coco80',exist_ok=False,
        seed=27,deterministic=True,optimizer='AdamW',lr0=.0001,lrf=.2,
        weight_decay=.0005,warmup_epochs=1,freeze=10,patience=20,amp=True,
        cache=False,mosaic=0.,mixup=0.,copy_paste=0.,degrees=0.,translate=.03,
        scale=.1,fliplr=.5,hsv_h=.005,hsv_s=.1,hsv_v=.1,close_mosaic=0,
        plots=False,save_period=1,val=True,pretrained=True,nms=False)
    save(OUT/'training-result.json',dict(metrics=result.results_dict,automatic_best=str(model.trainer.best),
        last=str(model.trainer.last),selection='Separate fixed validation selection; test excluded'))


if __name__=='__main__':
    main()
