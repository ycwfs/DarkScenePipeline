"""Verify actual release bytes and extract isolated copies for entry-point checks."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parents[1]
UNPACKED = REPORT / 'unpacked'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def check_file(data, record, source, target):
    assert hashlib.sha256(data).hexdigest() == record['sha256'], target
    assert len(data) == record.get('size', record.get('bytes')), target
    assert data == source.read_bytes(), source
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)


def main():
    releases = [
        ('platform/dark-demo-v2_multiperson_20261022.tar.gz', 'platform/dark-demo-v2.tar.gz'),
        ('platform/dark-demo-v2_multiperson_20261022.zip', 'platform/dark-demo-v2.zip'),
        ('NVIDIA-Box_20261022.zip', 'NVIDIA-Box.zip'),
    ]
    verified = []
    for relative, alias_relative in releases:
        archive, alias = ROOT / relative, ROOT / alias_relative
        manifest = json.loads(Path(str(archive) + '.manifest.json').read_text())
        archive_hash = digest(archive)
        assert archive_hash == manifest['sha256']
        assert archive.stat().st_size == manifest['size']
        box = relative.startswith('NVIDIA')
        records = manifest['contents']['files'] if box else manifest['files']
        source = ROOT / ('NVIDIA-Box' if box else 'platform/dark-demo-v2')
        if relative.endswith('.tar.gz'):
            dest = UNPACKED / 'platform-tar/dark-demo-v2'
            with tarfile.open(archive) as tar:
                assert set(tar.getnames()) == {'dark-demo-v2/' + r for r in records}
                for rel, record in records.items():
                    check_file(tar.extractfile('dark-demo-v2/' + rel).read(), record,
                               source / rel, dest / rel)
        else:
            prefix = 'NVIDIA-Box/' if box else ''
            dest = UNPACKED / ('box/NVIDIA-Box' if box else 'platform-zip')
            with zipfile.ZipFile(archive) as z:
                assert z.testzip() is None
                expected = {prefix + r for r in records}
                if box:
                    expected.add(prefix + 'manifest.json')
                    assert json.loads(z.read(prefix + 'manifest.json')) == manifest['contents']
                assert set(z.namelist()) == expected
                for rel, record in records.items():
                    check_file(z.read(prefix + rel), record, source / rel, dest / rel)
                    if 'platform_source' in record:
                        assert record['sha256'] == digest(ROOT / 'platform/dark-demo-v2' /
                                                          record['platform_source'])
                if box:
                    (dest / 'manifest.json').write_bytes(z.read(prefix + 'manifest.json'))
        weight = dest / ('weights' if box else 'ckpts') / 'videomamba_t_behavior_32f_realactall_full_pn2.pth'
        assert digest(weight) == 'd30b36c4c1ca9ab276442b4d0544b3d705f4ead68c7dcb5a721b696f9ee5f362'
        shutil.copyfile(archive, alias)
        alias_manifest = dict(manifest, archive=alias.name)
        Path(str(alias) + '.manifest.json').write_text(json.dumps(alias_manifest, ensure_ascii=False, indent=2) + '\n')
        Path(str(archive) + '.sha256').write_text(f'{archive_hash}  {archive.name}\n')
        Path(str(alias) + '.sha256').write_text(f'{archive_hash}  {alias.name}\n')
        assert digest(alias) == archive_hash
        verified.append(dict(archive=relative, alias=alias_relative, sha256=archive_hash,
                             bytes=archive.stat().st_size, files_verified=len(records),
                             all_files_match_source=True, alias_byte_identical=True,
                             extracted=str(dest)))
        print(json.dumps(verified[-1], ensure_ascii=False), flush=True)
    (REPORT / 'archive-verification.json').write_text(json.dumps(verified, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
