"""Render paired raw/720/enhanced detector overlays for the missed-frame audit."""
import json, sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import ROOT, OUT, source_frames


def draw(image, gt, pred, title):
    image = image.copy()
    for b in gt:
        b = np.asarray(b, np.int32)
        cv2.rectangle(image, tuple(b[:2]), tuple(b[2:]), (60, 220, 60), 2)
    for b in pred:
        b = np.asarray(b, np.int32)
        cv2.rectangle(image, tuple(b[:2]), tuple(b[2:]), (40, 80, 235), 2)
    cv2.rectangle(image, (0, 0), (min(image.shape[1], 500), 32), (20, 20, 20), -1)
    cv2.putText(image, title, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, .65,
                (240, 240, 240), 1, cv2.LINE_AA)
    return cv2.resize(image, (720, round(image.shape[0] * 720 / image.shape[1])))


def main():
    rows = json.loads((OUT / 'records.json').read_text())
    targets = [r for r in rows if r['conditions']['raw_native']['formal']['unmatched_gt']
               or r['conditions']['raw720']['formal']['unmatched_gt']]
    dest = OUT / 'visual' / 'failure-comparisons'; dest.mkdir(parents=True, exist_ok=True)
    wanted = {(r['source'], r['frame']): r for r in targets}
    for row, native in source_frames([r for r in rows if (r['source'], r['frame']) in wanted]):
        panels=[]
        for c,label in [('raw_native','raw native'),('raw720','raw 720'),('ntire720','NTIRE 720'),('ft720','onsite FT 720')]:
            image = native if c == 'raw_native' else cv2.imread(str(OUT/'frames'/f"{row['id']:03d}_{c}.png"))
            scale=np.array([image.shape[1]/native.shape[1],image.shape[0]/native.shape[0]],np.float32)
            gt=(np.asarray(row['gt_boxes_native'],np.float32)*np.tile(scale,2)).tolist()
            pred=row['conditions'][c]['formal']['pred_boxes']
            panels.append(draw(image,gt,pred,f"{label} | GT green / pred red"))
        h=max(x.shape[0] for x in panels); panels=[cv2.copyMakeBorder(x,0,h-x.shape[0],0,0,cv2.BORDER_CONSTANT,value=(25,25,25)) for x in panels]
        sheet=np.hstack(panels)
        name=f"{row['source'].split('/')[-1].rsplit('.',1)[0]}_F{row['frame']:04d}_id{row['id']:04d}.png"
        cv2.imwrite(str(dest/name),sheet)
    print('rendered',len(list(dest.glob('*.png'))),'comparisons')


if __name__ == '__main__': main()
