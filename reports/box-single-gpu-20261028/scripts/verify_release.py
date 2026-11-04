"""Verify final delivery against the independently tested archive, then refresh the alias."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parents[1]
archive=ROOT/'NVIDIA-Box_Orin4GB_candidate_20261028.zip'
review=OUT/'NVIDIA-Box_review.zip'

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):
            h.update(block)
    return h.hexdigest()

with zipfile.ZipFile(archive)as final,zipfile.ZipFile(review)as tested:
    names=set(final.namelist())
    assert names==set(tested.namelist())
    assert not any(n.endswith('.engine')or '/.cache/'in n or 'retinexformer_ntire_720_box' in n for n in names)
    manifest=json.loads(final.read('NVIDIA-Box/manifest.json'))
    previous=json.loads(tested.read('NVIDIA-Box/manifest.json'))
    differences=[]
    for rel,record in manifest['files'].items():
        raw=final.read('NVIDIA-Box/'+rel)
        assert len(raw)==record['bytes']and hashlib.sha256(raw).hexdigest()==record['sha256'],rel
        if previous['files'][rel]['sha256']!=record['sha256']:
            differences.append(rel)
    assert set(differences)=={'README.md','VERIFICATION.md'},differences
    assert all(manifest['files'][n]['sha256']==previous['files'][n]['sha256']for n in manifest['files'] if n.startswith(('runtime/','weights/')))
sidecar=json.loads(archive.with_suffix('.zip.manifest.json').read_text())
archive_hash=digest(archive)
assert archive_hash==sidecar['sha256']
assert len(names)==99
suite=json.loads((OUT/'final-suite.json').read_text())
assert len(suite['results'])==5 and all(r['returncode']==0 for r in suite['results'])
for name in ('live-final-verification.json','demos-verification.json','fault-verification.json','platform-unchanged.json'):
    assert json.loads((OUT/name).read_text())['passed']
assert '107 passed' in (OUT/'cpu-archive.log').read_text()
alias=ROOT/'NVIDIA-Box.zip'
if alias.exists() and digest(alias)!=digest(ROOT/'NVIDIA-Box_full_priors_20261027.zip'):
    if digest(alias)!=archive_hash:
        backup=OUT/'previous-default-box.zip'
        assert not backup.exists()
        shutil.copyfile(alias,backup)
stage=alias.with_name('.NVIDIA-Box.zip.next')
shutil.copyfile(archive,stage)
assert digest(stage)==archive_hash
stage.replace(alias)
(alias.with_suffix('.zip.sha256')).write_text(f'{archive_hash}  {alias.name}\n')
sidecar['archive']=alias.name
alias.with_suffix('.zip.manifest.json').write_text(json.dumps(sidecar,ensure_ascii=False,indent=2)+'\n')
result=dict(passed=True,archive=str(archive.relative_to(ROOT)),sha256=archive_hash,
    bytes=archive.stat().st_size,archive_files=len(names),manifest_payload_files=len(manifest['files']),
    alias=str(alias.relative_to(ROOT)),alias_sha256=digest(alias),
    compared_with_tested_archive=str(review.relative_to(ROOT)),
    final_vs_tested_changes=differences,executables_and_weights_identical=True,
    engines_in_archive=False,cpu_tests=107,suite=suite['results'],
    physical_orin_tested=False,platform_unchanged=json.loads((OUT/'platform-unchanged.json').read_text()))
(OUT/'release-verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items()if k not in ('suite','platform_unchanged')},ensure_ascii=False,indent=2))
