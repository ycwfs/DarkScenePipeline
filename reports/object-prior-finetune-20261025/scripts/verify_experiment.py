"""Check frozen artifact consistency, portable ONNX probe and complete demo decoding."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

from build_dataset import OUT, ROOT, save
from infer_image import predict
from prior_rules import decide


class Links(HTMLParser):
    def __init__(self):
        super().__init__();self.links=[]

    def handle_starttag(self, tag, attrs):
        for key,value in attrs:
            if key in ('href','src','poster') and value and not value.startswith(('http:', 'https:', '#')):
                self.links.append(value)


def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def decode(row: dict) -> dict:
    path=OUT/row['video']
    probe=subprocess.run(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
        '-show_entries','stream=codec_name,width,height,nb_read_frames','-of','json',str(path)],
        check=True,capture_output=True,text=True)
    stream=json.loads(probe.stdout)['streams'][0]
    assert stream['codec_name']=='h264' and int(stream['nb_read_frames'])==row['frames']
    result=subprocess.run(['ffmpeg','-v','error','-xerror','-err_detect','explode','-i',str(path),
        '-f','null','-'],capture_output=True,text=True)
    assert result.returncode==0 and not result.stderr, (str(path),result.stderr)
    return dict(video=row['video'],frames=row['frames'],sha256=digest(path),full_decode_passed=True,stream=stream)


def main() -> None:
    cv2.setNumThreads(1)
    lock=json.loads((OUT/'prior-lock.json').read_text())
    assert digest(OUT/'scripts/prior_rules.py')==lock['rules_sha256']
    assert digest(OUT/'models/yolo26m_objects_candidate.onnx')==lock['model_sha256']
    rows=json.loads((OUT/'dataset-manifest.json').read_text())['records']
    groups={};hashes={}
    for r in rows:
        assert r['split'] in ('train','val','test') and 'data-10-25-manypeople' not in r['source']
        assert groups.setdefault(r['group'],r['split'])==r['split']
        h=digest(Path(r['dataset_image']));assert h==r['image_sha256']
        assert hashes.setdefault(h,r['split'])==r['split']
        label=Path(r['dataset_label']).read_text().splitlines()
        assert len(label)==len(r['boxes'])
    count=0
    for f in (OUT/'action-evaluation').glob('*.json'):
        for r in json.loads(f.read_text())['records']:
            recomputed=decide(r['baseline'],r['samples'])
            for key in ('conservative','strict','reason','evidence'):assert recomputed[key]==r[key],(f,key)
            count+=1
    assert count==619
    evaluation=json.loads((OUT/'object-evaluation.json').read_text())
    checks=[]
    cases=[next(r for r in evaluation['records'] if r['split']=='test' and r['variants']['finetuned_roi']['counts']['0']['tp']>0),
           next(r for r in evaluation['records'] if r['split']=='test' and any(b['class_id']==2 for b in r['boxes']))]
    for r in cases:
        expected=np.asarray(r['variants']['finetuned_roi']['detections'],dtype=float).reshape(-1,6)
        expected=expected[expected[:,4]>=.35]
        actual,providers=predict(OUT/'models/yolo26m_objects_candidate.onnx',cv2.imread(r['image']),'cpu',.35)
        assert len(expected)==len(actual)
        errors=[]
        for a in actual:
            dist=np.max(np.abs(expected-a),axis=1);i=int(dist.argmin())
            assert dist[i]<.15 and int(expected[i,5])==int(a[5])
            errors.append(float(dist[i]));expected=np.delete(expected,i,axis=0)
        checks.append(dict(id=r['id'],predictions=len(actual),max_abs_error=max(errors,default=0.),providers=providers))
    demos=json.loads((OUT/'demo-manifest.json').read_text())
    with ThreadPoolExecutor(max_workers=2) as pool:media=list(pool.map(decode,demos))
    links=0
    for name in ('index.html','annotations.html'):
        parser=Links();parser.feed((OUT/name).read_text())
        for target in parser.links:assert (OUT/target).exists(),(name,target)
        links+=len(parser.links)
    save(OUT/'final-verification.json',dict(passed=True,dataset_images=len(rows),group_split_checks=len(groups),
        locked_rule_and_model_unchanged=True,action_rule_replay_matched=count,portable_cpu_probe=checks,
        demo_videos=media,html_local_links=links,production_prior_enabled=False,
        limitation='No new end-to-end streaming/performance test; existing release engineering checks remain separately documented.'))
    print('Verified',len(rows),'dataset images;',count,'rule decisions;',len(media),'fully decoded videos;',links,'local links')


if __name__=='__main__':
    main()
