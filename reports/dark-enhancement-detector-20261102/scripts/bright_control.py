"""Predeclared small bright regression control: no model selection."""
import json,time
import cv2,numpy as np,torch
from run import ROOT,OUT,PIPE,Detector,make_enhancer,enhance,save
from compare_matched import evaluate

def main():
    torch.set_num_threads(8)
    refs=json.loads((ROOT/'reports/bright-regression-20261024/visual-audit-labels.json').read_text())
    selected=[r for r in refs['labels'] if r['index'] in [1,27,29,30,37]]
    save(OUT/'bright-protocol.json',{'selection':'existing bright audit indices 1,27,29,30,37; exhibits and multiple real people',
          'no_model_selection':True,'samples':selected})
    detector=Detector(PIPE/'ckpts/yolo26m_person_20261024.onnx')
    models={'ntire720':make_enhancer(PIPE/'ckpts/NTIRE.pth'),
            'ft720':make_enhancer(ROOT/'ckpts_realdeploy_ft/NTIRE.pth')}
    rows=[]
    for row in selected:
        raw=cv2.imread(str(ROOT/f'reports/bright-regression-20261024/data/{row["video_id"]}/raw.png'))
        assert raw.shape[:2]==(404,720),raw.shape
        images={'raw720':raw,**{k:enhance(m,raw) for k,m in models.items()}}
        record={**row,'conditions':{}}
        for name,image in images.items():
            pred=detector(image)
            record['conditions'][name]=evaluate(np.array(row['person_boxes']),pred,image.shape,image.shape)
            dest=OUT/f'bright/{row["index"]}_{name}.png';dest.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(dest),image)
        rows.append(record)
    save(OUT/'bright/records.json',rows)
    print('bright control complete',len(rows))

if __name__=='__main__':main()
