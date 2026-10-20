"""Build a reproducible standalone ZIP and verify it against the GPU-tested code."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[3]
BOX = ROOT / 'NVIDIA-Box'
OUT = ROOT / 'reports/nvidia-box-verification-20261018'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    tested = OUT / 'extracted/NVIDIA-Box'
    paths = sorted(p for p in BOX.rglob('*') if p.is_file() and
                   '__pycache__' not in p.parts and p.name != 'manifest.json')
    for p in paths:
        if p.suffix in {'.py', '.pth', '.onnx'} or p.name == 'requirements.txt':
            reference = tested / p.relative_to(BOX)
            assert reference.is_file() and sha(p) == sha(reference), f'Untested runtime change: {p}'
    manifest = {
        'version': '20261018',
        'default_config': {'proc_max_side': 720, 'skel_alpha': .7, 'lum_gate': 75,
                           'reco_min_conf': .65, 'fuse_mode': 'alpha'},
        'files': {str(p.relative_to(BOX)): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in paths},
    }
    (BOX / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    paths.append(BOX / 'manifest.json')
    archive = ROOT / 'NVIDIA-Box_20261018.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(paths):
            name = 'NVIDIA-Box/' + p.relative_to(BOX).as_posix()
            info = zipfile.ZipInfo(name, (2026,10,18,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            z.writestr(info, p.read_bytes())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert len(z.namelist()) == len(paths)
        for rel, meta in manifest['files'].items():
            assert hashlib.sha256(z.read('NVIDIA-Box/'+rel)).hexdigest() == meta['sha256'], rel
    shutil.copy2(archive, ROOT / 'NVIDIA-Box.zip')
    release = {'archive': archive.name, 'bytes': archive.stat().st_size, 'sha256': sha(archive),
               'md5': hashlib.md5(archive.read_bytes()).hexdigest(), 'file_count': len(paths),
               'runtime_matches_gpu_tested_extraction': True, 'zip_crc_and_hashes': 'passed',
               'default_zip_identical': sha(archive) == sha(ROOT/'NVIDIA-Box.zip'),
               'weights': json.loads((OUT/'evidence/before-update.json').read_text())['weights']}
    (OUT/'evidence/release.json').write_text(json.dumps(release, ensure_ascii=False, indent=2)+'\n')
    (ROOT/'NVIDIA-Box_20261018.zip.manifest.json').write_text(json.dumps(
        {'release': release, 'contents': manifest}, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(release, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
