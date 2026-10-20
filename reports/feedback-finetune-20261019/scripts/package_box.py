"""Build a versioned standalone model package with an auditable content manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    old=json.loads((args.source/'manifest.json').read_text())
    names=set(old['files']) | {'weights/videomamba_t_behavior_32f_realactall_full_pn2.pth'}
    files={}
    for name in sorted(names):
        path=args.source/name
        assert path.is_file(),path
        if path.suffix=='.py':compile(path.read_bytes(),str(path),'exec')
        files[name]=dict(bytes=path.stat().st_size,sha256=sha(path))
    manifest=dict(version='20261019-pn2',
                  default_config=dict(proc_max_side=720,skel_alpha=.7,lum_gate=75,reco_min_conf=.75,fuse_mode='alpha'),
                  model='weights/videomamba_t_behavior_32f_realactall_full_pn2.pth',files=files)
    (args.source/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    names.add('manifest.json')
    with zipfile.ZipFile(args.output,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in sorted(names):
            info=zipfile.ZipInfo('NVIDIA-Box/'+name,(2026,10,19,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o100644<<16
            z.writestr(info,(args.source/name).read_bytes())
    with zipfile.ZipFile(args.output) as z:
        assert z.testzip() is None
        for name,meta in files.items():
            assert hashlib.sha256(z.read('NVIDIA-Box/'+name)).hexdigest()==meta['sha256'],name
    report=dict(archive=args.output.name,sha256=sha(args.output),size=args.output.stat().st_size,contents=manifest)
    Path(str(args.output)+'.manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('[box-package]',args.output,report['size'],report['sha256'])


if __name__=='__main__':main()
