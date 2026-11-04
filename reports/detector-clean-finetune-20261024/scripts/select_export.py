"""Select fixed-run checkpoints on equal-weight dark/bright validation, then export."""
import shutil
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from prepare_bright import ROOT,OUT,OLD,save,session,YoloOnnxDetector,iou


def main() -> None:
    torch.set_num_threads(2);cv2.setNumThreads(1)
    checkpoints=[ROOT.parent/'yolo26m-detect.pt',OLD/'models/yolo26m_dark_candidate.pt']
    checkpoints+=sorted((OUT/'runs/detector/weights').glob('epoch*.pt'),key=lambda p:int(p.stem[5:]))
    assert len(checkpoints)==14,len(checkpoints)
    rows=[]
    for path in checkpoints:
        metrics={}
        for domain in ['dark','bright']:
            model=YOLO(str(path))
            result=model.val(data=str(OUT/f'detector-{domain}-val.yaml'),split='val',imgsz=640,
                batch=12,device=0,workers=2,nms=False,plots=False,save_json=False,
                project=str(OUT/'runs'),name='selection',exist_ok=True)
            metrics[domain]=result.results_dict
        score=np.mean([m['metrics/mAP50-95(B)'] for m in metrics.values()])
        row=dict(path=str(path),metrics=metrics,score=float(score))
        rows.append(row);save(OUT/'selection.json',dict(rows=rows,
            criterion='mean(dark_val_mAP50-95, bright_val_mAP50-95), deployed one-to-one head nms=False; test never used'))
        print('[selection]',path.name,score,flush=True)
    selected=max(rows[2:],key=lambda r:r['score'])
    recommended=max(rows,key=lambda r:r['score'])
    save(OUT/'selection.json',dict(rows=rows,selected=selected,recommended=recommended,
        criterion='Equal domain mean validation mAP50-95; earliest checkpoint breaks ties; no test-based selection'))
    target=OUT/'models/yolo26m_clean_candidate.pt';target.parent.mkdir(exist_ok=True)
    shutil.copy2(selected['path'],target)
    model=YOLO(str(target));model.model.end2end=True
    exported=model.export(format='onnx',imgsz=640,opset=17,simplify=False,dynamic=False,
        batch=1,device='cpu',half=False,nms=False)
    runner=session(type(target)(exported),False);model.model.float().eval();checks=[]
    image_paths=sorted((OUT/'dataset/images/val').glob('*.png'))
    for i in np.linspace(0,len(image_paths)-1,10).round().astype(int):
        path=image_paths[i];frame=cv2.imread(str(path));canvas,_,_,_=YoloOnnxDetector._letterbox(frame)
        x=np.ascontiguousarray(canvas[:,:,::-1].transpose(2,0,1)[None].astype(np.float32)/255)
        with torch.no_grad():pred=model.model(torch.from_numpy(x))[0].numpy()[0]
        actual=runner.run(None,{runner.get_inputs()[0].name:x})[0][0]
        assert actual.shape==(300,6)
        a=actual[(actual[:,4]>=.35)&(actual[:,5]==0)];b=pred[(pred[:,4]>=.35)&(pred[:,5]==0)]
        assert len(a)==len(b),(path,len(a),len(b))
        error=0.
        if len(a):
            cost=np.array([[1-iou(aa[:4],bb[:4]) for bb in b] for aa in a]);ii,jj=linear_sum_assignment(cost)
            error=float(np.max(np.abs(a[ii]-b[jj])));assert error<.1,(path,error)
        checks.append(dict(image=path.name,people=len(a),max_absolute_error=error))
    save(OUT/'export-verification.json',dict(passed=True,checks=checks,shape=[1,300,6],head='one-to-one'))
    print('[exported]',exported,flush=True)


if __name__=='__main__':main()
