"""Bundle reviewed demo media, source code and compact verification records."""
import hashlib
import json
from pathlib import Path
import zipfile

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


manifest = json.loads((ROOT/'platform/dark-demo-v2_multiperson_20261022.tar.gz.manifest.json').read_text())
for rel, record in manifest['files'].items():
    assert digest(ROOT/'platform/dark-demo-v2'/rel) == record['sha256'], rel
for relative, alias in [('platform/dark-demo-v2_multiperson_20261022.tar.gz','platform/dark-demo-v2.tar.gz'),
                        ('platform/dark-demo-v2_multiperson_20261022.zip','platform/dark-demo-v2.zip'),
                        ('NVIDIA-Box_20261022.zip','NVIDIA-Box.zip')]:
    record = json.loads((ROOT/(relative+'.manifest.json')).read_text())
    assert digest(ROOT/relative) == digest(ROOT/alias) == record['sha256']
provenance = json.loads((OUT/'provenance.json').read_text())
provenance['production_source_and_all_release_archives_verified_unchanged'] = True
(OUT/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
files = {OUT/name for name in ('README.md','index.html','selection.json','sample-results.json',
                               'verification.json','provenance.json','regression.xml')}
for subdir,pattern in [('scripts','*.py'),('videos','*.mp4'),('stills','*.jpg'),('crops','*.jpg'),('events','*.json')]:
    files.update((OUT/subdir).glob(pattern))
archive = OUT/'逐人动作融合演示_20261023.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for path in sorted(files):
        z.write(path,path.relative_to(OUT).as_posix())
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
result = dict(archive=archive.name, bytes=archive.stat().st_size, sha256=digest(archive),
              videos=7, files=len(files))
(OUT/'demo-archive.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
