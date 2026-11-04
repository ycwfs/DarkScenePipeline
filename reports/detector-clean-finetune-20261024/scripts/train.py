"""Fixed single training run; original initialization and previous hyperparameters."""
import cv2
import torch
from ultralytics import YOLO
from prepare_bright import ROOT,OUT,save


def main() -> None:
    torch.set_num_threads(2);cv2.setNumThreads(1)
    if (OUT/'runs/detector').exists():
        raise FileExistsError('Completed/running experiment exists; use a separate output directory for a new run.')
    model=YOLO(str(ROOT.parent/'yolo26m-detect.pt'))
    result=model.train(data=str(OUT/'detector.yaml'),epochs=12,imgsz=640,batch=12,device=0,
        workers=2,project=str(OUT/'runs'),name='detector',exist_ok=False,seed=24,
        deterministic=True,optimizer='AdamW',lr0=.0001,lrf=.2,weight_decay=.0005,
        warmup_epochs=1,freeze=10,patience=12,amp=True,cache=False,
        mosaic=0.,mixup=0.,copy_paste=0.,degrees=0.,translate=.03,scale=.1,
        fliplr=.5,hsv_h=.005,hsv_s=.1,hsv_v=.1,close_mosaic=0,
        plots=False,save_period=1,val=True,pretrained=True,nms=False)
    save(OUT/'training-result.json',dict(metrics=result.results_dict,
        auto_best=str(model.trainer.best),last=str(model.trainer.last),
        selection='Separate one-to-one head validation required; do not use auto_best for test.'))


if __name__=='__main__':main()
