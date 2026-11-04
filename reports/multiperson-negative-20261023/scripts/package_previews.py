"""Verify deployment provenance and bundle the six videos with their audit evidence."""
import hashlib
import json
from pathlib import Path
import zipfile

from common import ROOT, OUT, OP


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


manifest = ROOT / 'platform/dark-demo-v2_multiperson_20261022.tar.gz.manifest.json'
record = json.loads(manifest.read_text())
for rel, meta in record['files'].items():
    assert digest(OP / rel) == meta['sha256'], rel
assert digest(ROOT / 'platform/dark-demo-v2_multiperson_20261022.tar.gz') == record['sha256']
source = dict(operator_archive=record['archive'], archive_sha256=record['sha256'],
              all_source_files_match_delivery=True, files_verified=len(record['files']),
              model=record['files']['ckpts/videomamba_t_behavior_32f_realactall_full_pn2.pth'],
              image='darkpipe-operator:0.3.0', physical_gpu=1,
              weights_and_production_code_changed=False)
(OUT / 'source-verification.json').write_text(json.dumps(source, ensure_ascii=False, indent=2) + '\n')
files = {OUT / name for name in ('README.md', 'index.html', 'selection.json', 'candidates.json',
    'sample-results.json', 'summary.json', 'media-verification.json', 'source-verification.json')}
for subdir, pattern in [('scripts', '*.py'), ('events', '*.json'), ('stills', '*.jpg'), ('videos', '*.mp4')]:
    files.update((OUT / subdir).glob(pattern))
archive = OUT / '多人负样本可视化_20261023.zip'
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
    for path in sorted(files):
        z.write(path, path.relative_to(OUT).as_posix())
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
result = dict(archive=archive.name, bytes=archive.stat().st_size,
              sha256=digest(archive), files=len(files), videos=6)
(OUT / 'preview-archive.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(result, ensure_ascii=False, indent=2))
