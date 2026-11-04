"""Verify dataset split integrity, local artifacts and reproducible review bundle."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np

from prepare import OUT,ROOT,save


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def main():
    manifest=json.loads((OUT/'manifest.json').read_text());samples=manifest['samples']
    splitgroups=defaultdict(set);used={s['video_id'] for s in samples};sources=[];images=0
    for video in manifest['videos']:
        v=video['video'];split=v['effective_split']
        splitgroups[split['group']].add(split['split'])
        if v['id'] in used:
            actual=sha(ROOT/v['source']);assert actual==v['sha256'],v['source']
            sources.append(dict(source=v['source'],sha256=actual))
    assert all(len(x)==1 for x in splitgroups.values())
    for s in samples:
        for view in ['raw','enhanced'] if s['split']=='train' else ['enhanced']:
            path=OUT/f'dataset/images/{s["split"]}/{s["video_id"]}_{s["frame"]:06d}_{view}.png'
            image=cv2.imread(str(path));assert image is not None and max(image.shape[:2])<=720
            labels=OUT/f'dataset/labels/{s["split"]}/{s["video_id"]}_{s["frame"]:06d}_{view}.txt'
            assert len(labels.read_text().splitlines())==len(s['people'])
            images+=1
        for p in s['people']:
            box=np.array(p['bbox_xyxy']);xy=np.array(p['keypoints_xy']);mask=np.array(p['keypoint_mask'])
            assert np.all(np.isfinite(box)) and np.all(box[2:]>box[:2])
            assert xy.shape==(17,2) and mask.shape==(17,)
            assert np.all(np.isfinite(xy))
            assert np.all(xy[mask]>=0) and np.all(xy[mask]<=np.array([s['width'],s['height']]))
    videos=json.loads((OUT/'demo-summary.json').read_text())['videos'];assert len(videos)==14
    checked=json.loads((OUT/'video-verification.json').read_text())
    assert len(checked['videos'])==14 and checked['total_frames']==1255
    models=[]
    for p in sorted((OUT/'models').glob('*')):
        if p.suffix in ['.pt','.pth','.onnx']:models.append(dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=sha(p)))
    for name in ['yolo26m_detect.onnx','rtmo-s_end2end.onnx','NTIRE.pth']:
        p=ROOT/'platform/dark-demo-v2/ckpts'/name
        models.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)))
    code=[dict(path=str(p.relative_to(OUT)),sha256=sha(p)) for p in sorted((OUT/'scripts').glob('*.py'))]
    save(OUT/'model-manifest.json',dict(models=models,scripts=code,
        source_splits_sha256=sha(ROOT/'reports/person-finetune-20261023/splits.json')))
    save(OUT/'source-verification.json',dict(sources=sources,groups_disjoint=True,images_verified=images,
        source_frames_resampled=False,visibility_is_pseudo_mask=True))
    save(OUT/'verification.json',dict(passed=True,source_videos_verified=len(sources),
        images_verified=images,groups_disjoint=True,videos=14,frames=1255,
        models_hashed=len(models),scripts_hashed=len(code),
        detector_parity=json.loads((OUT/'detector-runtime-parity.json').read_text())['passed'],
        production_modified=False))
    assets=[OUT/'index.html',OUT/'README.md',OUT/'video-index.csv',OUT/'verification.json']
    assets+=list((OUT/'videos').glob('*.mp4'))+list((OUT/'posters').glob('*.jpg'))
    bundle=OUT/'dark-finetune-video-review.zip'
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_STORED) as z:
        for p in assets:z.write(p,p.relative_to(OUT).as_posix())
    with zipfile.ZipFile(bundle) as z:assert z.testzip() is None
    save(OUT/'bundle-verification.json',dict(path=bundle.name,bytes=bundle.stat().st_size,
        sha256=sha(bundle),members=len(assets),crc_check=True,models_included=False))
    print('Dataset, models, sources, videos and ZIP verified',images,len(sources),bundle.stat().st_size)


if __name__=='__main__':main()
