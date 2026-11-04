"""Check every archive payload and old/new model bytes; publish matching local aliases."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import zipfile

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def tar_hashes(path):
    with tarfile.open(path) as archive:
        return {str(Path(member.name).relative_to('dark-demo-v2')):digest(archive.extractfile(member).read())
                for member in archive if member.isfile()}


def main():
    tar=ROOT/'platform/dark-demo-v2_person_object_20261026.tar.gz'
    zip_path=ROOT/'platform/dark-demo-v2_person_object_20261026.zip'
    box=ROOT/'NVIDIA-Box_person_object_20261026.zip'
    hashes=tar_hashes(tar)
    with zipfile.ZipFile(zip_path) as archive:
        zh={name:digest(archive.read(name)) for name in archive.namelist() if not name.endswith('/')}
    assert hashes==zh
    for rel,sha in hashes.items():
        assert digest((ROOT/'platform/dark-demo-v2'/rel).read_bytes())==sha,rel
    old=tar_hashes(ROOT/'platform/dark-demo-v2_person_tau075_20261024.tar.gz')
    weight_files=[p for p in hashes if p.startswith('ckpts/')]
    assert all(old[p]==hashes[p]for p in weight_files if p in old)
    assert set(weight_files)-set(old)=={'ckpts/yolo26m_objects_20261026.onnx'}
    assert hashes['ckpts/yolo26m_objects_20261026.onnx']=='7c9bc11befb2bc74890767fe769e0efb692f061b379473286fe8bf3b8e5848b5'
    with zipfile.ZipFile(box) as archive:
        manifest=json.loads(archive.read('NVIDIA-Box/manifest.json'))
        assert manifest['default_config']['reco_min_conf']==.925
        assert manifest['default_config']['object_min_conf']==.581
        assert manifest['default_config']['person_object_prior'] is True
        for rel,record in manifest['files'].items():
            data=archive.read('NVIDIA-Box/'+rel)
            assert len(data)==record['bytes'] and digest(data)==record['sha256'],rel
            assert digest((ROOT/'NVIDIA-Box'/rel).read_bytes())==record['sha256'],rel
            if 'platform_source'in record:
                assert record['sha256']==hashes[record['platform_source']],rel
    extracted=OUT/'release-extracted';extracted.mkdir(exist_ok=True)
    with tarfile.open(tar) as archive:archive.extractall(extracted,filter='data')
    with zipfile.ZipFile(box) as archive:archive.extractall(extracted)
    published=[]
    import sys
    publish='--publish' in sys.argv
    for source,dest in [(tar,ROOT/'platform/dark-demo-v2.tar.gz'),
                        (zip_path,ROOT/'platform/dark-demo-v2.zip'),(box,ROOT/'NVIDIA-Box.zip')]:
        if not publish:
            published.append(dict(path=str(source.relative_to(ROOT)),sha256=digest(source.read_bytes()),bytes=source.stat().st_size))
            continue
        shutil.copyfile(source,dest)
        sha=digest(dest.read_bytes());assert sha==digest(source.read_bytes())
        dest.with_suffix(dest.suffix+'.sha256').write_text(f'{sha}  {dest.name}\n')
        side=json.loads(source.with_suffix(source.suffix+'.manifest.json').read_text())
        side['archive']=dest.name
        dest.with_suffix(dest.suffix+'.manifest.json').write_text(json.dumps(side,ensure_ascii=False,indent=2)+'\n')
        published.append(dict(path=str(source.relative_to(ROOT)),alias=str(dest.relative_to(ROOT)),sha256=sha,bytes=source.stat().st_size))
    result=dict(passed=True,platform_payload_files=len(hashes),box_payload_files=len(manifest['files']),
                unchanged_model_files=len(weight_files)-1, added_model_files=1,changed_from_tau075=[p for p in hashes if old.get(p)!=hashes[p]],
                archives=published,model_training=False,object_prior=True)
    (OUT/'archive-verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
