import hashlib,json,shutil,sys,tarfile,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'platform'));import package_v2
sys.path.insert(0,str(ROOT/'scripts'));import package_nvidia_box

def save(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def main():
 records=[]
 for suffix in ('.tar.gz','.zip'):
  out=ROOT/'platform'/('dark-demo-v2_person_20261024'+suffix)
  records.append(package_v2.build(ROOT/'platform/dark-demo-v2',out))
  out.with_name(out.name+'.sha256').write_text(f'{records[-1]["sha256"]}  {out.name}\n')
  print(out.name,records[-1]['sha256'],flush=True)
 box=package_nvidia_box.build(ROOT/'NVIDIA-Box',ROOT/'platform/dark-demo-v2',ROOT/'NVIDIA-Box_20261024.zip',True)
 print('box',box,flush=True)
 dest=OUT/'release-extracted';dest.mkdir(exist_ok=True)
 with tarfile.open(ROOT/'platform/dark-demo-v2_person_20261024.tar.gz')as tar:tar.extractall(dest)
 with zipfile.ZipFile(ROOT/'NVIDIA-Box_20261024.zip')as z:z.extractall(dest)
 platform=dest/'dark-demo-v2';nbox=dest/'NVIDIA-Box'
 for name,info in records[0]['files'].items():
  assert hashlib.sha256((platform/name).read_bytes()).hexdigest()==info['sha256']
  assert (platform/name).read_bytes()==(ROOT/'platform/dark-demo-v2'/name).read_bytes()
 for out,source in package_nvidia_box.runtime_sources(platform).items():
  assert (nbox/out).read_bytes()==(platform/source).read_bytes()
 manifest=json.loads((nbox/'manifest.json').read_text())
 for name,info in manifest['files'].items():assert hashlib.sha256((nbox/name).read_bytes()).hexdigest()==info['sha256']
 with zipfile.ZipFile(ROOT/'platform/dark-demo-v2_person_20261024.zip')as z:
  assert set(z.namelist())==set(records[0]['files'])
  for name,info in records[0]['files'].items():assert hashlib.sha256(z.read(name)).hexdigest()==info['sha256']
 save(OUT/'archive-verification.json',dict(platform_files=len(records[0]['files']),box_files=len(manifest['files']),
      tar_zip_payload_equal=True,source_equal=True,box_platform_equal=True,archives=records,box=box))
 print('archive validation complete',flush=True)
if __name__=='__main__':main()
