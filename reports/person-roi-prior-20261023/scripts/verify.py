"""Replay actual evidence through unchanged production rules and verify artifacts."""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
import numpy as np

OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT/'platform/dark-demo-v2'))
from darkpipe.stages.object_prior import PriorGate


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__();self.links=[]

    def handle_starttag(self, tag: str, attrs: list) -> None:
        self.links.extend(v for k,v in attrs if k in {'href','src'} and v)


def main() -> None:
    protocol=json.loads((OUT/'protocol.json').read_text())
    provenance=json.loads((OUT/'model-provenance.json').read_text())
    assert sha(OUT/'protocol.json')==provenance['protocol_sha256']
    assert sha(OUT/'scripts/probe.py')==provenance['probe_sha256']
    for name,digest in provenance['model_sha256'].items():
        assert sha(ROOT/'platform/dark-demo-v2/ckpts'/name)==digest,name
    previous=ROOT/'reports/person-finetune-20261023'
    assert sha(previous/'annotations.json')==protocol['annotation_sha256']
    assert sha(previous/'splits.json')==protocol['splits_sha256']
    gate=PriorGate('', 'cpu');checks=0;people=0;frames=0
    for item in protocol['cases']:
        result=json.loads((OUT/'data'/item['id']/'results.json').read_text())
        assert result['item']==item
        assert sha(previous/'data'/item['id']/'manifest.json')==result['cache_manifest_sha256']
        expected=np.linspace(0,item['frames']-1,min(6,item['frames'])).round().astype(int).tolist()
        assert [r['frame']for r in result['frames']]==expected
        shapes={r['frame']:r for r in result['frames']}
        frames+=len(expected);people+=len(result['records'])
        for r in result['records']:
            meta=shapes[r['frame']];h,w=meta['shape'];nh,nw=meta['native_shape']
            roi=np.array(r['roi']);nr=np.array(r['native_roi'])
            assert (roi[2:]>roi[:2]).all() and (nr[2:]>nr[:2]).all()
            assert (roi[:2]>=0).all() and (roi[2:]<=[w,h]).all()
            mapped=nr/np.array([nw/w,nh/h,nw/w,nh/h])
            assert np.allclose(mapped,roi,atol=1), (item['source'], r['frame'])
            for variant,e in r['variants'].items():
                det=np.array(e['detections'],np.float32).reshape(-1,6)
                assert np.isfinite(det).all()
                assert (det[:,:4]>=-1e-5).all() and (det[:,:4]<=np.array([w,h,w,h])+1e-4).all()
                if not e['pose_usable']:continue
                pose=np.array(r['pose'],np.float32);pose[:,:2]/=np.array([w,h]);sk=pose[None,None]
                selected=det[det[:,4]>.25];image=np.zeros((h,w,3),np.uint8)
                assert gate._drink(sk,[True],[0],[image],[selected],h)[0]==(e['drink_score']>.25)
                assert gate._pickup(sk,[True],[0],[image],[selected],h)[0]==(e['pickup_score']>.25)
                checks+=2
    completion=json.loads((OUT/'completion.json').read_text())
    assert completion['frames']==frames and completion['person_frames']==people
    parser=Links();parser.feed((OUT/'index.html').read_text())
    for link in parser.links:assert (OUT/link).is_file(),link
    review=json.loads((OUT/'visual-review.json').read_text())
    assert len(review)==40
    files=list((OUT/'scripts').glob('*.py'))+list(OUT.glob('*.json'))+list(OUT.glob('*.md'))
    files+=list(OUT.glob('*.csv'))+list(OUT.glob('*.html'))+list((OUT/'figures').glob('*'))
    files+=list((OUT/'visual').glob('*.jpg'))+list((OUT/'data').glob('*/results.json'))
    files=[f for f in files if f.name not in {'artifact-manifest.json','verification.json'}]
    manifest={str(f.relative_to(OUT)):dict(bytes=f.stat().st_size,sha256=sha(f))for f in sorted(files)}
    (OUT/'artifact-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    result=dict(videos=len(protocol['cases']),frames=frames,person_frames=people,
                production_geometry_checks=checks,coordinates_verified=True,
                visual_cases_reviewed=len(review),html_links_verified=len(parser.links),
                model_hashes_unchanged=True,source_annotation_and_split_unchanged=True,
                artifact_files=len(manifest),complete=True)
    (OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
