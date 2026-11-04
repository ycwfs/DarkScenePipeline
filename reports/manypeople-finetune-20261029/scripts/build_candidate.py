"""Local isolated checkpoint/configuration bundle; never replaces production assets."""
import shutil
import subprocess
import torch
from common import *


def main():
    lock=read(OUT/'selection-lock.json');protocol=read(OUT/'protocol.json');config=lock['config']
    ckpts=OUT/'candidate/ckpts';ckpts.mkdir(parents=True,exist_ok=True)
    files={'rgb':'videomamba_t_behavior_32f_person_roi_20261023.pth','skeleton':'stgcn_person_20261024.pth'}
    for path in (OP/'ckpts').iterdir():
        if path.is_file() and not (ckpts/path.name).exists():(ckpts/path.name).symlink_to(path.resolve())
    selected={}
    for branch,file in files.items():
        which=config[branch];m=lock['models'][branch][which];source=Path(m['model']);assert sha(source)==m['sha256']
        filename=('videomamba_t_person_roi_crowded_20261030.pth'if branch=='rgb'else'stgcn_person_crowded_20261030.pth')
        if lock['epochs'][branch]==0:
            filename='videomamba_t_person_roi_retained_20261023.pth'if branch=='rgb'else'stgcn_person_retained_20261024.pth'
        exported=OUT/'models'/filename
        exported.parent.mkdir(exist_ok=True)
        if lock['epochs'][branch]==0:
            shutil.copy2(source,exported)
        else:
            # Our local training checkpoint includes TorchVersion metadata, which
            # the production weights_only loader rejects. Export tensor weights
            # with plain metadata; preserve the immutable training checkpoint.
            training=torch.load(source,map_location='cpu',weights_only=False)
            tensors=training.get('model',training)
            torch.save(dict(model=tensors,epoch=int(lock['epochs'][branch]),
                source_sha256=m['sha256']),exported)
            loaded=torch.load(exported,map_location='cpu',weights_only=True)['model']
            assert loaded.keys()==tensors.keys()
            assert all(torch.equal(loaded[k],v) for k,v in tensors.items())
        dest=ckpts/file
        if dest.is_symlink():dest.unlink()
        if dest.exists():dest.unlink()
        dest.symlink_to(exported)
        selected[branch]=dict(path=str(dest),sha256=sha(dest),source=str(source),selection=which,
            export_path=str(exported),epoch=lock['epochs'][branch],source_sha256=m['sha256'],
            exported_tensors_identical=True)
    candidate=dict(models=selected,parameters=dict(skel_fusion_alpha=config['alpha'],lum_gate=config['gate'],
        reco_min_conf=config['action'],object_min_conf=config['object'],wave_pose_min_frames=3,running_pose_min_speed=.35,
        person_object_prior=True,proc_max_side=720,clip_pre_sec=2.,clip_post_sec=2.),
        checkpoint_dir=str(ckpts),operator_code=str(OP),selection_lock_sha256=sha(OUT/'selection-lock.json'),
        deployment_status='Local experimental candidate; other checkpoint symlinks are local references, not an upload package.',production_modified=False)
    save(OUT/'candidate-config.json',candidate)
    rows=read(OUT/'dataset.json')['records'];annotations=read(OUT/'annotations.json')['annotations']
    jobs=[];sources={s['number']:s for s in read(REVIEW/'sources.json')}
    # Label-defined display cases, selected without consulting before/after model outcomes.
    for label in ['other','drink','chase','wave']:
        candidates=[r for r in rows if r['scope']=='new' and r['split']=='test' and r['label_kind']=='human' and r['label']==label]
        r=min(candidates,key=lambda x:(x['number'],x['start']))
        source=sources[r['number']];start=max(0,r['start']-3);duration=min(10.,source['duration']-start)
        file=OUT/'demo-inputs'/f'new-{label}.mkv';file.parent.mkdir(exist_ok=True)
        if not file.exists():
            subprocess.run(['ffmpeg','-v','error','-nostdin','-ss',str(start),'-i',r['source'],'-t',str(duration),'-an',
                '-c:v','ffv1','-level','3','-threads','2',str(file)],check=True)
        jobs.append(dict(id='new-'+label,source=str(file),original=r['source'],start=start,duration=duration,
            reviewed_target=r['id'],selection='First human-confirmed test window for this class; no prediction outcome filtering',processed=False))
    oldvideos={v['id']:v for v in read(CURRENT/'protocol.json')['videos']}
    used=set()
    for key,lo,hi in [('dark',0,35),('dim',35,75),('bright',75,256)]:
        candidates=[]
        for split in ('test','val','train'):
            candidates=[r for r in rows if r['scope']=='old' and r['split']==split and r.get('current_index')is not None
                and r['raw_luma']is not None and lo<=r['raw_luma']<hi and r['label']!='other' and r['source_id']not in used]
            if candidates:break
        if not candidates:continue
        r=min(candidates,key=lambda r:(r['source_id'],r['end']));used.add(r['source_id']);v=oldvideos[r['source_id']]
        jobs.append(dict(id='old-'+key,source=r['source'],original=r['source'],processed=r.get('processed',False),
            reviewed_target=r['id'],raw_luma=r['raw_luma'],label=r['label'],split=r['split'],
            selection='First positive source in raw-luma stratum; test then val then train fallback, no prediction-outcome filtering; visual function demo only'))
    save(OUT/'demo-jobs.json',jobs)
    print('CANDIDATE',candidate,'DEMOS',len(jobs),flush=True)


if __name__=='__main__':main()
