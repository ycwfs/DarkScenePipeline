"""Oracle ROI scale probe; uses reviewed reference boxes only for diagnosis."""
import json
from pathlib import Path
import cv2,numpy as np
from run import OUT,Detector,person_boxes,greedy_match,save

def main():
    labels=json.loads((OUT/'hard/labels.json').read_text())['samples']
    detector=Detector(OUT.parents[1]/'platform/dark-demo-v2/ckpts/yolo26m_person_20261024.onnx')
    records=json.loads((OUT/'matched/records.json').read_text())
    by={r['id']:r for r in records}; out=[]
    for s in labels:
        row=by[s['id']]; gt=np.asarray(row['gt_boxes_native'],np.float32)
        result={'id':s['id'],'conditions':{}}
        for name in ['raw720','ntire720','ft720']:
            image=cv2.imread(str(OUT/f'matched/frames/{s["id"]}_{name}.png'))
            scale=np.array([image.shape[1]/row['native_shape'][1],image.shape[0]/row['native_shape'][0]])
            g=gt*np.tile(scale,2); box=g[0]; size=box[2:]-box[:2]; margin=np.tile(size*.5,2)
            lo=np.maximum(0,np.floor(box-margin)).astype(int); hi=np.minimum([image.shape[1],image.shape[0]]*2,np.ceil(box+margin)).astype(int)
            crop=image[lo[1]:hi[1],lo[0]:hi[0]]
            pred=person_boxes(detector(crop,.1),.35)
            if len(pred): pred[:,:4]+=[lo[0],lo[1],lo[0],lo[1]]
            m=greedy_match(g,pred[:,:4])
            result['conditions'][name]={'crop_shape':list(crop.shape[:2]),'model_person_height':float(640*(box[3]-box[1])/max(crop.shape[:2])),
                'matched':len(m),'predictions':len(pred),'unmatched':len(pred)-len(m),'ious':[x[2] for x in m],
                'pred_boxes_720':pred[:,:4].tolist(),'pred_scores':pred[:,4].tolist()}
        out.append(result)
    save(OUT/'roi-probe.json',{'protocol':'reviewed box crop, 50% box margin; oracle diagnostic, not deployable recall','records':out})
    print('roi probe',len(out))
if __name__=='__main__':main()
