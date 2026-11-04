"""Frozen 8-class adaptation of the current object detector, with head-row preservation."""
import copy
import csv
import hashlib
import os

os.environ.setdefault('OMP_NUM_THREADS', '2')
import cv2
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.models.yolo.detect.train import DetectionTrainer
from build_dataset import OUT, OLD, NAMES, save


class LocalTrainer(DetectionTrainer):
    def read_results_csv(self):
        # The installed Ultralytics optional CSV dependency (polars) is absent.
        # Checkpoint metadata only needs the small numeric history as column lists.
        with self.csv.open(newline='') as stream:
            rows=list(csv.DictReader(stream))
        return {key.strip():[float(r[key]) for r in rows] for key in rows[0]} if rows else {}


def main() -> None:
    torch.set_num_threads(2); cv2.setNumThreads(1)
    baseline = OLD/'models/yolo26m_objects_candidate.pt'
    if (OUT/'runs/hardneg').exists():
        raise FileExistsError('Existing training run; no implicit overwrite')
    protocol = dict(
        baseline=str(baseline), baseline_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),
        dataset_sha256=hashlib.sha256((OUT/'dataset-manifest.json').read_bytes()).hexdigest(),
        train_new_sources=[2,4,5,6,8],val_new_sources=[3],test_new_sources=[1,7,9],
        note='New training/val roles explicitly supersede prior feedback-only roles; original source grouping retained. '
             'All nine videos were previously inspected/calibrated, so test is held out from this training only. '
             'Same site and some same visitors across sources; no independent scene generalization claim.',
        classes=NAMES, seed=27, epochs=20, imgsz=640, batch=12, optimizer='AdamW', lr0=.0001,
        freeze=10, augmentation='No mosaic/mixup; translation .03, scale .1, flip .5; mild HSV',
        selection='Maximum validation one-to-one mAP50-95 across 20 saved epochs, earliest breaks ties; no test-based selection',
        evaluation=dict(confidence=.581,iou=.5,rgb_alpha=.972,raw_luma_gate=35,action_threshold=.925,
            primary='phone/camera/microphone mistaken for bottle/cup; final action decisions on complete new01/07/09',
            secondary='true-container recall, old heldout object regression',
            mic_prior='New class 7 is not added to prior support classes; fixed policy isolates detector change'),
        limits=['Assistant visual auxiliary boxes; not independently adjudicated',
                'Single training seed; sparse camera/microphone heldout support',
                'Validation contains no camera/microphone; no class-specific epoch tuning',
                'No cup annotations; cannot estimate cup performance',
                'Mined hard cases, not representative exhaustive object annotations'])
    save(OUT/'protocol.json',protocol)
    torch.manual_seed(27)
    old = YOLO(str(baseline))
    expanded = DetectionModel(copy.deepcopy(old.model.yaml), nc=8, verbose=False)
    src=old.model.float().state_dict(); dst=expanded.state_dict(); preserved=[]
    for k,v in dst.items():
        if src[k].shape==v.shape:
            v.copy_(src[k])
        else:
            assert ('.cv3.' in k or '.one2one_cv3.' in k) and src[k].shape[0]==7 and v.shape[0]==8,(k,v.shape)
            v[:7].copy_(src[k]); preserved.append(k)
    assert len(preserved)==12
    expanded.load_state_dict(dst);expanded.names=dict(enumerate(NAMES));expanded.args=old.model.args
    old.model=expanded
    initial=OUT/'models/initialized-eight-class.pt';initial.parent.mkdir(exist_ok=True)
    old.save(str(initial))
    model=YOLO(str(initial))
    for k in preserved:assert torch.equal(model.model.float().state_dict()[k][:7],src[k]),k
    # The rebuilt trainer must preserve the 8-class initialization, including old class rows.
    def verify_trainer(trainer):
        sd=trainer.model.float().state_dict()
        for k in preserved:assert torch.equal(sd[k][:7].cpu(),src[k]),f'Lost pretrained class rows: {k}'
        save(OUT/'initialization-check.json',dict(passed=True,preserved_outputs=preserved))
    model.add_callback('on_pretrain_routine_end',verify_trainer)
    result=model.train(trainer=LocalTrainer,data=str(OUT/'objects.yaml'),epochs=20,imgsz=640,batch=12,
        device=0,workers=2,project=str(OUT/'runs'),name='hardneg',exist_ok=False,
        seed=27,deterministic=True,optimizer='AdamW',lr0=.0001,lrf=.2,
        weight_decay=.0005,warmup_epochs=1,freeze=10,patience=20,amp=True,
        cache=False,mosaic=0.,mixup=0.,copy_paste=0.,degrees=0.,translate=.03,
        scale=.1,fliplr=.5,hsv_h=.005,hsv_s=.1,hsv_v=.1,close_mosaic=0,
        plots=False,save_period=1,val=True,pretrained=True,nms=False)
    save(OUT/'training-result.json',dict(metrics=result.results_dict,
        automatic_best=str(model.trainer.best),last=str(model.trainer.last)))


if __name__=='__main__':main()
