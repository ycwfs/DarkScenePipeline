"""Joint validation checkpoint/parameter selection; no test labels are scored here."""
import csv
import itertools
import sys
import time
from common import *
sys.path.insert(0,str(OP))
import numpy as np
import torch
from darkpipe.stages.skeleton_aux import STGCN
from darkpipe.person.object_prior import filter_object_evidence
from evaluation_metrics import metrics


def main():
    assert not (OUT/'selection-lock.json').exists()
    protocol=read(OUT/'protocol.json');amend=read(OUT/'protocol-amendment-before-training.json')
    joint=read(OUT/'joint-selection-protocol.json');rows=read(OUT/'dataset.json')['records'];index={r['id']:i for i,r in enumerate(rows)}
    ids=np.load(OUT/'rgb/val-00.npz')['indices'];vrows=[rows[i]for i in ids]
    assert all(r['split']=='val' and r['label_kind']!='pseudo'for r in vrows)
    rgb=np.stack([np.load(OUT/'rgb'/f'val-{epoch:02d}.npz')['probs']for epoch in range(7)])
    for epoch in range(7):assert np.array_equal(np.load(OUT/'rgb'/f'val-{epoch:02d}.npz')['indices'],ids)
    cache=np.load(OUT/'skeleton-inputs.npz');features=cache['features'][:,ids];valid=cache['valid'][:,ids]
    wave=cache['wave_counts'][:,ids];speed=cache['speed'][:,ids]
    net=STGCN(num_classes=10).cuda().eval();torch.set_num_threads(2)
    skeleton=[];models={'rgb':{},'skeleton':{}}
    for branch in models:
        epochs=range(7)if branch=='rgb'else range(21)
        for epoch in epochs:
            path=Path(protocol['models'][branch]['path'])if epoch==0 else OUT/branch/f'epoch-{epoch:02d}.pth'
            models[branch][str(epoch)]=dict(model=str(path),sha256=sha(path),epoch=epoch,trained_this_round=epoch>0)
    with torch.inference_mode():
        for epoch in range(21):
            state=torch.load(models['skeleton'][str(epoch)]['model'],map_location='cpu',weights_only=False)
            net.load_state_dict(state.get('model',state));gates=[]
            for x in features:
                gates.append(np.concatenate([net(torch.from_numpy(x[i:i+128]).cuda()).softmax(1).cpu().numpy()for i in range(0,len(x),128)]))
            skeleton.append(gates)
    skeleton=np.array(skeleton,np.float32)
    np.savez_compressed(OUT/'joint-validation-inputs.npz',indices=ids,rgb=rgb,skeleton=skeleton)
    luma=np.array([np.nan if r['raw_luma']is None else r['raw_luma']for r in vrows])
    support=np.full((len(GATES),len(ids),2),np.inf,np.float32);local={r['id']:j for j,r in enumerate(vrows)};covered=set()
    for path in (OUT/'object-evidence').glob('*.json'):
        for e in read(path)['records']:
            if e['id']not in local:continue
            j=local[e['id']];covered.add(j)
            for gi,g in enumerate(GATES):
                for ci,label in enumerate(('drink','pick_up')):
                    result=filter_object_evidence(label,e['actions'][label],vrows[j]['raw_luma'],.001,g)
                    if result['applied']:support[gi,j,ci]=max((d['confidence']for d in result['evidence']),default=0.)
    assert len(covered)==len(ids)
    truth=np.array([LABELS.index(r['label'])for r in vrows]);masks={d:np.array([r['scope']==d for r in vrows])for d in ('old','new')}
    def fast(pred):
        output=[]
        for domain in ('old','new'):
            mask=masks[domain];cm=np.bincount(truth[mask]*10+pred[mask],minlength=100).reshape(10,10)
            sup=cm.sum(1);tp=np.diag(cm);recall=tp/np.maximum(sup,1);f1=2*tp/np.maximum(sup+cm.sum(0),1)
            output.append(dict(fp=int(sup[9]-tp[9]),fpr=float((sup[9]-tp[9])/max(sup[9],1)),
                recall=recall,positive_macro=float(recall[:9][sup[:9]>0].mean()),positive_tp=int(sum(tp[:9])),
                macro_f1=float(f1[sup>0].mean()),supported=sup))
        return output
    def fuse(ri,si,gi,alpha):
        sk=skeleton[si,gi].copy();sk[~valid[gi]]=np.eye(10,dtype=np.float32)[9]
        fused=alpha*rgb[ri]+(1-alpha)*sk;fused[luma<GATES[gi]]=rgb[ri,luma<GATES[gi]]
        return fused
    def decide(fused,gi,tau,obj):
        p=fused.argmax(1);p[(p!=9)&(fused.max(1)<tau)]=9
        p[(p==6)&(support[gi,:,0]<obj)]=9;p[(p==7)&(support[gi,:,1]<obj)]=9
        p[(p==0)&(wave[gi]<3)]=9;p[(p==2)&np.isfinite(speed[gi])&(speed[gi]<.35)]=9
        return p
    original=dict(rgb='baseline',skeleton='baseline',alpha=.972,gate=35.,action=.925,object=.581)
    base_pred=decide(fuse(0,0,GATES.index(35.),.972),GATES.index(35.),.925,.581);baseline=fast(base_pred)
    old_supported=(baseline[0]['supported']>0);old_supported[9]=False
    params=dict(protocol['tuning']);params.update(amend.get('grid_extension',{}));alphas=[a for a in params['alpha']if a<.999]
    count=accepted_count=0;best=None;winner=None;best_by_pair=[];started=time.monotonic()
    fields=['rgb_epoch','skeleton_epoch','alpha','gate','action','object','accepted','old_fp','new_fp','macro_f1','positive_tp']
    with (OUT/'joint-parameter-grid.csv').open('w')as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for ri,si in itertools.product(range(7),range(21)):
            pair_best=None;pair_winner=None
            for gi,alpha in itertools.product([GATES.index(g) for g in params['gate']],alphas):
                fused=fuse(ri,si,gi,alpha)
                for tau,obj in itertools.product(params['action'],params['object']):
                    p=decide(fused,gi,tau,obj);sc=fast(p)
                    accepted=bool(np.all(sc[0]['recall'][old_supported]>=baseline[0]['recall'][old_supported]-.05-1e-9)
                        and sc[1]['positive_macro']>=baseline[1]['positive_macro']-.10-1e-9)
                    fpr=(sc[0]['fpr']+sc[1]['fpr'])/2;f1=(sc[0]['macro_f1']+sc[1]['macro_f1'])/2;tp=sc[0]['positive_tp']+sc[1]['positive_tp']
                    distance=abs(alpha-.972)+abs(GATES[gi]-35)/100+abs(tau-.925)+abs(obj-.581)
                    key=(-fpr,f1,tp,-distance,ri>0,si>0,-ri,-si)
                    config=dict(rgb_epoch=ri,skeleton_epoch=si,alpha=alpha,gate=GATES[gi],action=tau,object=obj)
                    writer.writerow(dict(**config,accepted=accepted,old_fp=sc[0]['fp'],new_fp=sc[1]['fp'],macro_f1=f1,positive_tp=tp))
                    count+=1;accepted_count+=accepted
                    if accepted and (pair_best is None or key>pair_best):pair_best=key;pair_winner=dict(config=config,key=key)
                    if accepted and (best is None or key>best):best=key;winner=dict(config=config,prediction=p.copy())
            best_by_pair.append(pair_winner or dict(config=dict(rgb_epoch=ri,skeleton_epoch=si),feasible=False))
            if si==20:
                save(OUT/'joint-selection-progress.json',dict(rgb_epochs_finished=ri+1,settings=count,accepted=accepted_count,seconds=time.monotonic()-started))
                print('JOINT',ri+1,'/7',count,'settings',round(time.monotonic()-started,1),'seconds',flush=True)
    assert winner is not None
    c=winner['config'];selected_models={b:models[b][str(c[b+'_epoch'])]for b in ('rgb','skeleton')}
    config=dict(rgb='selected',skeleton='selected',alpha=c['alpha'],gate=c['gate'],action=c['action'],object=c['object'])
    validation={d:metrics([r for r in vrows if r['scope']==d],winner['prediction'][masks[d]])for d in ('old','new')}
    baseval={d:metrics([r for r in vrows if r['scope']==d],base_pred[masks[d]])for d in ('old','new')}
    # This file is the only authorization to open test metrics in evaluate_selected.py.
    save(OUT/'selection-lock.json',dict(config=config,epochs=dict(rgb=c['rgb_epoch'],skeleton=c['skeleton_epoch']),
        validation=validation,baseline_validation=baseval,baseline_config=original,criterion=protocol['tuning']['objective'],
        grid_count=count,accepted_count=accepted_count,selected_models=selected_models,
        models={b:dict(baseline=models[b]['0'],selected=selected_models[b],candidate=read(OUT/'predictions'/f'{b}-candidate.json')if (OUT/'predictions'/f'{b}-candidate.json').exists()else None)for b in ('rgb','skeleton')},
        all_checkpoint_models=models,protocol_sha256=sha(OUT/'protocol.json'),joint_protocol_sha256=sha(OUT/'joint-selection-protocol.json'),
        dataset_sha256=sha(OUT/'dataset.json'),test_used_for_selection=False))
    save(OUT/'joint-best-by-model.json',best_by_pair)
    print('LOCKED',json.dumps(dict(config=config,epochs=c,validation=validation)),flush=True)


if __name__=='__main__':main()
