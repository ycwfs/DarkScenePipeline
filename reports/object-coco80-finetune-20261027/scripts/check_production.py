"""Confirm no experiment changed official release payloads, archives or image identity."""
import json
import subprocess

from common import OUT, ROOT, save, sha


def main():
    rows=[]
    for file,folder,key in [('platform/dark-demo-v2.tar.gz.manifest.json','platform/dark-demo-v2','files'),
                            ('NVIDIA-Box.zip.manifest.json','NVIDIA-Box','contents')]:
        d=json.loads((ROOT/file).read_text());files=d[key]if key=='files'else d[key]['files']
        mismatches=[name for name,record in files.items()if sha(ROOT/folder/name)!=record['sha256']]
        archive=ROOT/('platform/dark-demo-v2.tar.gz'if folder.startswith('platform')else'NVIDIA-Box.zip')
        digest=sha(archive)
        assert not mismatches and digest==d['sha256'],(file,mismatches)
        rows.append(dict(manifest=file,files=len(files),mismatches=mismatches,archive_sha256=digest,archive_unchanged=True))
    result=subprocess.run(['docker','image','inspect','--format','{{.Id}}','darkpipe-operator:0.3.0'],capture_output=True,text=True,check=True)
    identity=result.stdout.strip();assert identity=='sha256:d359716620d7eb757cd7c36733e25642469450f1c801e0367fc99248f0a2f212'
    save(OUT/'production-preserved.json',dict(payloads=rows,image_id=identity,passed=True))
    print('Official platform and Box payloads/archives and original image ID unchanged')


if __name__=='__main__':main()
