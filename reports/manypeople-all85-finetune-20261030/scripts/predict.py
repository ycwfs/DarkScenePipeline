"""Paired old/candidate probabilities on identical inputs, without fitting or selecting."""
import argparse
import sys
from common import *
sys.path.insert(0,str(OP))
import numpy as np
import torch
from torch.utils.data import DataLoader
from train_models import Windows,infer
from darkpipe.stages.recognize import BehaviorRecognizer
from darkpipe.stages.skeleton_aux import STGCN


def main():
    p=argparse.ArgumentParser();p.add_argument('branch',choices=['rgb','skeleton']);p.add_argument('--which',choices=['baseline','candidate','selected','last','trained'],required=True);a=p.parse_args()
    dest=OUT/'predictions'/f'{a.branch}-{a.which}.npz';dest.parent.mkdir(exist_ok=True)
    assert not dest.exists()
    protocol=read(OUT/'protocol.json');rows=read(OUT/'dataset.json')['records']
    checkpoint=(Path(read(OUT/'selection-lock.json')['best_trained_models'][a.branch]['model'])if a.which=='trained'else
        Path(protocol['models'][a.branch]['path'])if a.which=='baseline'else
        OUT/a.branch/f"epoch-{protocol['training']['rgb_epochs' if a.branch=='rgb' else 'skeleton_epochs']:02d}.pth"if a.which=='last'else
        Path(read(OUT/'selection-lock.json')['selected_models'][a.branch]['model'])if a.which=='selected'else
        Path(read(OUT/a.branch/'selection.json')['path']))
    torch.set_num_threads(3)
    evaluated=scored_indices(rows)
    evaluated_mask=np.zeros(len(rows),dtype=bool);evaluated_mask[evaluated]=True
    if a.branch=='rgb':
        rec=BehaviorRecognizer(str(OP/'ckpts'),skel_fusion_alpha=1.);rec.ckpt=str(checkpoint);rec.load('cuda:0')
        loader=DataLoader(Windows(rows,evaluated,'rgb',None),batch_size=8,num_workers=3,pin_memory=True,
            multiprocessing_context='spawn')
        probs=np.full((len(rows),10),np.nan,dtype=np.float32)
        probs[evaluated]=infer(rec.net,loader,'rgb');rec.close()
    else:
        net=STGCN(num_classes=10).cuda();state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        net.load_state_dict(state.get('model',state));features=np.load(OUT/'skeleton-inputs.npz')['features'];parts=[]
        for x in features:
            loader=DataLoader(Windows(rows,list(range(len(rows))),'skeleton',x),batch_size=128,num_workers=0)
            parts.append(infer(net,loader,'skeleton'))
        probs=np.stack(parts)
    np.savez_compressed(dest,probs=probs,ids=np.array([r['id']for r in rows]),evaluated_mask=evaluated_mask)
    save(dest.with_suffix('.json'),dict(model=str(checkpoint),sha256=sha(checkpoint),dataset_sha256=sha(OUT/'dataset.json'),
        samples=len(evaluated)if a.branch=='rgb'else len(rows),test_labels_used=False,selection_used=False,
        rgb_scope='Old canonical validation/test plus all human new; all 85 new positives are training-fit diagnostics',evaluated_indices=evaluated))
    print('PREDICTED',a.branch,a.which,probs.shape,flush=True)


if __name__=='__main__':main()
