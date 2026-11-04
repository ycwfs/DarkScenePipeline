"""Package the experimental model and reviewed labels; never overwrite release aliases."""
import hashlib
import json
from pathlib import Path
import zipfile
from build_dataset import OUT,save


def digest(path):
    h=hashlib.sha256()
    with path.open('rb')as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    assert json.loads((OUT/'verification.json').read_text())['passed']
    files=[]
    for directory in ['dataset','scripts','figures','context-comparisons','long-review','short-review']:
        files.extend(p for p in (OUT/directory).rglob('*')if p.is_file()and '__pycache__'not in p.parts and p.suffix!='.cache')
    files.extend((OUT/'review').glob('verified-*.jpg'))
    for name in ['models/yolo26m_objects_candidate.pt','models/yolo26m_objects_candidate.onnx','models/MODEL_CARD.md',
        'README.md','analysis-report.md','stats-appendix.md','figure-catalog.md','REPRODUCE.md','protocol.json',
        'label-views.json','extra-views.json','reviewed-labels.json','dataset-manifest.json','objects.yaml',
        'selection.json','training-result.json','initialization-check.json','export-verification.json',
        'object-evaluation.json','object-context-evaluation.json','cached-old-window-evaluation.json',
        'long-review-cases.json','long-review-labels.json','known-positive-audit.json','final-summary.json',
        'long-video-summary.csv','verification.json','short-review-cases.json','short-review-labels.json','production-preserved.json']:
        p=OUT/name;assert p.exists(),p;files.append(p)
    for n in [1,7,9]:
        files.append(OUT/'long-videos'/f'new-{n:02}'/'summary.json')
        files.append(OUT/'short-regression'/f'new-{n:02}'/'summary.json')
    manifest={str(p.relative_to(OUT)):dict(bytes=p.stat().st_size,sha256=digest(p))for p in sorted(set(files))}
    archive=OUT/'object-hardneg-candidate-20261027.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4)as z:
        for p in sorted(set(files)):z.write(p,str(p.relative_to(OUT)))
        z.writestr('artifact-manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
        z.writestr('CANDIDATE_ONLY.txt','Experimental 8-class object detector and auxiliary reviewed YOLO data. Not the platform operator or NVIDIA-Box release. Production remains unchanged. Full videos stay in the experiment directory.\n')
    with zipfile.ZipFile(archive)as z:
        assert z.testzip()is None
        for name,r in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==r['sha256']
    archive.with_suffix('.zip.sha256').write_text(f'{digest(archive)}  {archive.name}\n')
    save(OUT/'artifact-manifest.json',dict(package=archive.name,sha256=digest(archive),bytes=archive.stat().st_size,
        candidate_only=True,production_aliases_updated=False,files=manifest))
    print(archive,archive.stat().st_size)


if __name__=='__main__':main()
