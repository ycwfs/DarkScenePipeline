"""Verify final archive contents, then atomically install verified local delivery files."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import zipfile

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'reports/feedback-finetune-20261019'
NAME='videomamba_t_behavior_32f_realactall_full_pn2.pth'
SHA='d30b36c4c1ca9ab276442b4d0544b3d705f4ead68c7dcb5a721b696f9ee5f362'

def sha(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def atomic_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True,exist_ok=True)
    tmp=dst.with_name('.'+dst.name+'.pn2-tmp')
    shutil.copy2(src,tmp)
    tmp.replace(dst)


def main() -> None:
    deliveries=[(ROOT/'platform/dark-demo-v2_pn2_20261019.tar.gz','dark-demo-v2'),
                (ROOT/'platform/dark-demo-v2_pn2_20261019.zip','dark-demo-v2'),
                (ROOT/'NVIDIA-Box_20261019.zip','NVIDIA-Box')]
    reports=[]
    for path,tree in deliveries:
        meta=json.loads(Path(str(path)+'.manifest.json').read_text())
        assert sha(path)==meta['sha256'],path
        files=meta.get('files') or meta['contents']['files']
        if path.name.endswith('.tar.gz'):
            with tarfile.open(path) as tar:
                assert set(tar.getnames())=={tree+'/'+n for n in files}
                data={n:tar.extractfile(tree+'/'+n).read() for n in files}
        else:
            with zipfile.ZipFile(path) as z:
                prefix='NVIDIA-Box/' if tree=='NVIDIA-Box' else ''
                expected={prefix+n for n in files}
                if tree=='NVIDIA-Box':expected.add(prefix+'manifest.json')
                assert set(z.namelist())==expected
                assert z.testzip() is None
                data={n:z.read(prefix+n) for n in files}
                if tree=='NVIDIA-Box':assert json.loads(z.read(prefix+'manifest.json'))==meta['contents']
        for name,info in files.items():
            digest=hashlib.sha256(data[name]).hexdigest()
            assert digest==info['sha256']==sha(OUT/'staging'/tree/name),name
            # Only documentation changed after the independently extracted runtime tests.
            if not name.endswith('.md') and name!='manifest.json':
                assert digest==sha(OUT/'unpacked'/tree/name),('untested runtime change',name)
        weights={n:v for n,v in files.items() if n.startswith(('ckpts/','weights/'))}
        expected=8 if tree=='dark-demo-v2' else 7
        assert len(weights)==expected
        assert sha(OUT/'staging'/tree/('ckpts' if tree=='dark-demo-v2' else 'weights')/NAME)==SHA
        # Previously shipped weights must remain byte-identical for rollback.
        for name in weights:
            if name.endswith(NAME):continue
            old=ROOT/('platform/dark-demo-v2' if tree=='dark-demo-v2' else tree)/name
            assert sha(old)==weights[name]['sha256'],('old weight changed',name)
        reports.append(dict(archive=str(path.relative_to(ROOT)),size=path.stat().st_size,
            sha256=meta['sha256'],md5=hashlib.md5(path.read_bytes()).hexdigest(),
            files=len(files),weights=len(weights),runtime_identical_to_tested_extraction=True))

    # Preserve the previous default pointers as metadata; versioned old archives remain.
    prior={}
    for p in [ROOT/'platform/dark-demo-v2.tar.gz',ROOT/'NVIDIA-Box.zip']:
        prior[str(p.relative_to(ROOT))]=dict(size=p.stat().st_size,sha256=sha(p))
    before=OUT/'before/default-archives.json'
    if not before.exists():before.write_text(json.dumps(prior,indent=2)+'\n')

    updates={
        'dark-demo-v2':['README.md','CLAUDE.md','darkpipe/constants.py','ckpts/'+NAME],
        'NVIDIA-Box':['README.md','VERIFICATION.md','behavior_infer.py','check_env.py','manifest.json','weights/'+NAME],
    }
    for tree,names in updates.items():
        target=ROOT/('platform/dark-demo-v2' if tree=='dark-demo-v2' else tree)
        for name in names:
            dest=target/name
            backup=OUT/'before'/('platform/dark-demo-v2' if tree=='dark-demo-v2' else tree)/name
            if dest.exists() and not backup.exists():
                backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(dest,backup)
            atomic_copy(OUT/'staging'/tree/name,dest)

    aliases=[(deliveries[0][0],ROOT/'platform/dark-demo-v2.tar.gz'),
             (deliveries[1][0],ROOT/'platform/dark-demo-v2.zip'),
             (deliveries[2][0],ROOT/'NVIDIA-Box.zip')]
    for src,dest in aliases:
        atomic_copy(src,dest)
        meta=json.loads(Path(str(src)+'.manifest.json').read_text());meta['archive']=dest.name
        side=Path(str(dest)+'.manifest.json');tmp=side.with_suffix('.tmp')
        tmp.write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n');tmp.replace(side)
        assert sha(dest)==sha(src)
    for path,tree in deliveries:
        m=json.loads(Path(str(path)+'.manifest.json').read_text());files=m.get('files') or m['contents']['files']
        base=ROOT/('platform/dark-demo-v2' if tree=='dark-demo-v2' else tree)
        assert all(sha(base/n)==info['sha256'] for n,info in files.items()),tree
    release=dict(model=NAME,model_sha256=SHA,selected='epoch-04',defaults={'reco_min_conf':.75,'proc_max_side':720,'pose_overlay':True,'clip_pre_sec':2,'clip_post_sec':2},
        packages=reports,aliases={str(b.relative_to(ROOT)):str(a.relative_to(ROOT)) for a,b in aliases},
        image=json.loads((OUT/'image-verification.json').read_text()),
        validation={'operator_cpu':108,'box_cpu':31,'box_environment':8,'gpu_matched_samples':10,'offline':'real-offline/verification.json'},
        no_platform_upload=True)
    (OUT/'release.json').write_text(json.dumps(release,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(release,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
