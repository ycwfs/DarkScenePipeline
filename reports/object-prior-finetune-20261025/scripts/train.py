"""One frozen 20-epoch object-head adaptation, without touching production weights."""
import cv2
import torch
from ultralytics import YOLO
from build_dataset import OUT, ROOT, save


def main() -> None:
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    if (OUT / 'runs/objects').exists():
        raise FileExistsError('Existing training run; resume explicitly or use a new experiment.')
    model = YOLO(str(ROOT.parent / 'yolo26m-detect.pt'))
    result = model.train(data=str(OUT / 'objects.yaml'), epochs=20, imgsz=640, batch=12,
        device=0, workers=2, project=str(OUT / 'runs'), name='objects', exist_ok=False,
        seed=25, deterministic=True, optimizer='AdamW', lr0=.0001, lrf=.2,
        weight_decay=.0005, warmup_epochs=1, freeze=10, patience=20, amp=True,
        cache=False, mosaic=0., mixup=0., copy_paste=0., degrees=0., translate=.03,
        scale=.1, fliplr=.5, hsv_h=.005, hsv_s=.1, hsv_v=.1, close_mosaic=0,
        plots=False, save_period=1, val=True, pretrained=True, nms=False)
    save(OUT / 'training-result.json', dict(metrics=result.results_dict,
        automatic_best=str(model.trainer.best), last=str(model.trainer.last),
        note='Final selection separately evaluates every epoch with the deployed one-to-one head.'))


if __name__ == '__main__':
    main()
