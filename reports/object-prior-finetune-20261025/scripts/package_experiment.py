"""Package only frozen reviewed data and completed evidence; never production code."""
import hashlib
import json
from pathlib import Path
import zipfile

from build_dataset import OUT, save


def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def archive(name: str, files: set) -> dict:
    prefix=name.removesuffix('.zip')
    rows=[]
    target=OUT/name
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for path in sorted(files):
            relative=path.relative_to(OUT).as_posix()
            assert not relative.startswith(('..','/')) and path.is_file()
            z.write(path,prefix+'/'+relative)
            rows.append(dict(path=relative,bytes=path.stat().st_size,sha256=digest(path)))
        z.writestr(prefix+'/CONTENTS.json',json.dumps(dict(files=rows),ensure_ascii=False,indent=2)+'\n')
    sha=digest(target)
    (OUT/(name+'.sha256')).write_text(f'{sha}  {name}\n')
    manifest=dict(archive=name,sha256=sha,bytes=target.stat().st_size,files=rows,
        production_operator=False,object_prior_promoted=False)
    save(OUT/(name+'.manifest.json'),manifest)
    print(name,len(rows),'files',round(target.stat().st_size/1024**2,1),'MiB',sha,flush=True)
    return {k:v for k,v in manifest.items()if k!='files'}


def main() -> None:
    assert json.loads((OUT/'final-verification.json').read_text())['passed']
    datasets=json.loads((OUT/'dataset-manifest.json').read_text())['records']
    reviewed=[r for r in json.loads((OUT/'reviewed-labels.json').read_text())['records'] if r['training_allowed']]
    portable=[]
    for r in datasets:
        portable.append(dict(id=r['id'],split=r['split'],source=r['source'],group=r['group'],
            image=Path(r['dataset_image']).relative_to(OUT).as_posix(),
            label=Path(r['dataset_label']).relative_to(OUT).as_posix(),
            boxes=r['boxes'],image_sha256=r['image_sha256']))
    save(OUT/'dataset-portable.json',dict(records=portable,classes=json.loads((OUT/'protocol.json').read_text())['classes'],
        note='Resolve image and label paths relative to this JSON. Original manifests preserve historical absolute paths.'))
    common={OUT/name for name in ['README.md','analysis-report.md','stats-appendix.md','figure-catalog.md',
        'REPRODUCE.md','DELIVERY.md','RUN_STATE.md','index.html','annotations.html','demo-manifest.json','demo-review.json',
        'object-counts.csv','action-windows.csv','feedback-counts.csv','clip-coverage-454.csv','clip-coverage-454.json',
        'action-summary.json','statistics.json','protocol.json','prior-lock.json','environment.json','final-verification.json']}
    for folder in ('demos','figures','annotations-final'):
        common.update(p for p in (OUT/folder).iterdir()if p.is_file())
    review=archive('object-prior-review_20261026.zip',common)
    candidate=set(common)
    for pattern in ('*.json','*.csv','*.yaml'):
        candidate.update(p for p in OUT.glob(pattern)if '.manifest.' not in p.name and p.name not in
            ('delivery-verification.json','experiment-delivery.json'))
    for folder in ('scripts','models','action-evaluation'):
        candidate.update(p for p in (OUT/folder).iterdir()if p.is_file())
    for path in OUT.glob('tiles-*/manifest.json'):candidate.add(path)
    for r in datasets:
        candidate.update((Path(r['dataset_image']),Path(r['dataset_label'])))
    for r in reviewed:
        candidate.update((Path(r['image_view']),Path(r['image_raw'])))
    for name in ('train.log','select-export.log','evaluate-actions-old.log','evaluate-actions-new.log',
                 'evaluate-objects.log','rules-regression.log','final-verification.log',
                 'runs/objects/results.csv','runs/objects/args.yaml'):
        candidate.add(OUT/name)
    result=archive('object-prior-candidate_20261026.zip',candidate)
    save(OUT/'experiment-delivery.json',dict(review=review,candidate=result,
        production_release='dark-demo-v2_person_tau085_20261025; no object prior; unchanged image'))


if __name__=='__main__':
    main()
