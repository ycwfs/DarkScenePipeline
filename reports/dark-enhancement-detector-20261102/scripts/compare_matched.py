"""Matched CPU-FP32 enhancement; production postprocessing; resolution ablations."""
import hashlib
import json
import time
import cv2
import numpy as np
import torch
from run import (ROOT, OUT, PIPE, Detector, downscale, enhance, make_enhancer,
                 greedy_match, person_boxes, save, source_frames)

torch.set_num_threads(8)
torch.set_num_interop_threads(2)
cv2.setNumThreads(1)

def resize_long(image, size):
    h,w=image.shape[:2]; s=min(size/max(h,w),1.)
    return cv2.resize(image,(round(w*s)//2*2,round(h*s)//2*2), interpolation=cv2.INTER_AREA)

def detect_tiled(detector, image, confidence=.35):
    # Fixed 2x2 tiles, 20% overlap relative to tile width/height.
    # Four detector calls, merge at full-frame coordinates using production NMS.
    h,w=image.shape[:2]; th,tw=round(h/1.8),round(w/1.8)
    outputs=[]
    for top in [0,h-th]:
        for left in [0,w-tw]:
            pred=detector(image[top:top+th,left:left+tw], confidence).copy()
            pred[:,[0,2]]+=left; pred[:,[1,3]]+=top
            outputs.append(pred)
    return person_boxes(np.concatenate(outputs),confidence)

def evaluate(gt,pred,shape,native_shape):
    s=np.array([shape[1]/native_shape[1],shape[0]/native_shape[0]]*2)
    native_pred=pred.copy();native_pred[:,:4]/=s
    match=greedy_match(gt,native_pred[:,:4])
    return {'pred_boxes_native':native_pred[:,:4].tolist(), 'pred_scores':pred[:,4].tolist(),
            'matched':len(match), 'misses':len(gt)-len(match),
            'unmatched':len(pred)-len(match), 'ious':[m[2] for m in match],
            'matched_gt_indices':[m[0] for m in match],
            'matched_pred_indices':[m[1] for m in match]}

def main():
    protocol=json.loads((OUT/'protocol.json').read_text())
    protocol.update({'precision':'both enhancers CPU FP32; ORT CPU FP32',
        'person_postprocess':'import actual darkpipe.person.core.person_boxes: confidence .35, width>8 height>20, NMS .75',
        'matching':'maximum-cardinality one-to-one IoU>=.5; IoU maximised second',
        'resolution':'720/1280 long-side pre-enhancement; all full-frame detector inputs fixed640; optional 4-tile640 inputs',
        'diagnostic_only':True,'source_footage':'historical; no new platform failure video supplied'})
    save(OUT/'matched-protocol.json',protocol)
    hard=json.loads((OUT/'hard/labels.json').read_text())['samples']
    manifest=json.loads((ROOT/'reports/dark-finetune-20261024/manifest.json').read_text())
    original=[dict(s,id=f'A{s["id"]:04d}',gt_boxes_native=[p['bbox_xyxy'] for p in s['people']])
              for s in manifest['samples'] if s['split']=='test']
    for row in original: row['cohort']='auxiliary44'
    for row in hard: row['cohort']='quartile24'
    outdir=OUT/'matched/frames';outdir.mkdir(parents=True,exist_ok=True)
    detectors={k:Detector(PIPE/'ckpts'/v) for k,v in {
        'formal':'yolo26m_person_20261024.onnx','original':'yolo26m_detect.onnx'}.items()}
    enhancers={'ntire':make_enhancer(PIPE/'ckpts/NTIRE.pth'),
               'ft':make_enhancer(ROOT/'ckpts_realdeploy_ft/NTIRE.pth')}
    records=[];started=time.time()
    for n,(sample,native) in enumerate(source_frames(original+hard),1):
        rid=sample['id']; is_hard=sample['cohort']=='quartile24'
        cache=OUT/f'matched/{rid}.json'
        if cache.exists():
            cached=json.loads(cache.read_text())
            assert cached['model_hashes']==protocol['models']
            records.append(cached)
            continue
        raw=downscale(native); images={'raw_native':native,'raw720':raw}
        row={k:sample[k] for k in ['id','source','frame','luma','cohort','gt_boxes_native']}
        row.update(native_shape=list(native.shape[:2]),conditions={},model_hashes=protocol['models'])
        for name,model in enhancers.items():
            imgpath=outdir/f'{rid}_{name}720.png'
            t=time.perf_counter()
            if imgpath.exists(): images[f'{name}720']=cv2.imread(str(imgpath))
            else:
                images[f'{name}720']=enhance(model,raw)
                cv2.imwrite(str(imgpath),images[f'{name}720'])
            row[f'{name}_enhance_seconds']=time.perf_counter()-t
        if is_hard:
            images['raw1280']=resize_long(native,1280)
            # A true resolution control: enhance at1280 BEFORE detector640.
            for name,model in enhancers.items():
                imgpath=outdir/f'{rid}_{name}1280.png';t=time.perf_counter()
                if imgpath.exists(): images[f'{name}1280']=cv2.imread(str(imgpath))
                else:
                    images[f'{name}1280']=enhance(model,images['raw1280'])
                    cv2.imwrite(str(imgpath),images[f'{name}1280'])
                row[f'{name}_enhance1280_seconds']=time.perf_counter()-t
        for name,image in images.items():
            if name!='raw_native':cv2.imwrite(str(outdir/f'{rid}_{name}.png'),image)
            result={'shape':list(image.shape[:2])}
            gt=np.asarray(row['gt_boxes_native'],np.float32).reshape(-1,4)
            for detname,detector in detectors.items():
                # Retain low-score results so threshold diagnostics don't rerun ONNX.
                t=time.perf_counter();pred=detector(image,.1)
                selected=person_boxes(pred,.35)
                result[detname]=evaluate(gt,selected,image.shape,native.shape)
                result[detname]['low_conf_boxes_native']=(pred[:,:4]/np.array(
                    [image.shape[1]/native.shape[1],image.shape[0]/native.shape[0]]*2)).tolist()
                result[detname]['low_conf_scores']=pred[:,4].tolist()
                result[detname]['detect_seconds']=time.perf_counter()-t
            row['conditions'][name]=result
        if is_hard:
            for name in ['raw1280','ntire1280','ft1280']:
                image=images[name];t=time.perf_counter()
                pred=detect_tiled(detectors['formal'],image)
                row['conditions'][name+'_tiles']={'shape':list(image.shape[:2]),
                    'formal':evaluate(gt,pred,image.shape,native.shape)}
                row['conditions'][name+'_tiles']['formal']['detect_seconds']=time.perf_counter()-t
        if not is_hard:
            oldid=int(rid[1:]);cached=cv2.imread(str(OUT/f'frames/{oldid:03d}_ntire720.png'))
            assert cached.shape==images['ntire720'].shape
            delta=images['ntire720'].astype(float)-cached.astype(float)
            row['cached_ntire_vs_cpu_mae']=float(np.abs(delta).mean())
            row['cached_ntire_vs_cpu_psnr']=float(10*np.log10(255**2/max(np.mean(delta**2),1e-10)))
        save(cache,row);records.append(row)
        save(OUT/'matched/records.json',records)
        print(f'[{n}/68] {rid} {sample["source"]} F{sample["frame"]} '
              f'raw/NTIRE/FT hits '+str([row['conditions'][k]['formal']['matched'] for k in ['raw720','ntire720','ft720']]),flush=True)
    save(OUT/'matched/records.json',records)
    save(OUT/'matched/complete.json',{'records':len(records),'seconds':time.time()-started,
          'cohorts':{'auxiliary44':44,'quartile24':24},'models':protocol['models']})

if __name__=='__main__':main()
