"""Export selected weights and verify production loading on real cached windows."""
import argparse
import shutil
import sys
from common import *
sys.path.insert(0,str(OP))
import numpy as np
import torch
from darkpipe.stages.recognize import BehaviorRecognizer
from darkpipe.stages.skeleton_aux import STGCN
from train_models import Windows


@torch.inference_mode()
def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--which',choices=['selected','trained','last'],default='trained');args=parser.parse_args()
    lock=read(OUT/'selection-lock.json');rows=read(OUT/'dataset.json')['records'];scored=set(scored_indices(rows))
    torch.set_num_threads(3)
    chosen=[]
    for domain in ['old','new']:
        for label in ['other','drink','wave','chase']:
            eligible=[i for i,r in enumerate(rows) if i in scored and r['scope']==domain
                      and r['label']==label and r['label_kind']!='pseudo']
            if eligible:chosen.append(eligible[0])
    results={};exported=OUT/'models'/args.which;exported.mkdir(parents=True,exist_ok=True)
    for branch in ['rgb','skeleton']:
        selected=(lock['selected_models'][branch] if args.which=='selected' else lock['best_trained_models'][branch] if args.which=='trained' else lock['all_checkpoint_models'][branch]['6' if branch=='rgb' else '20']);source=Path(selected['model'])
        assert sha(source)==selected['sha256']
        dest=exported/f"{branch}-epoch-{selected['epoch']:02d}.pth"
        trained=torch.load(source,map_location='cpu',weights_only=False)
        tensors=trained.get('model',trained)
        if selected['epoch']==0:shutil.copy2(source,dest)
        else:
            metadata=dict(model=tensors,epoch=int(selected['epoch']),source_sha256=sha(source),
                class_order=LABELS,training_protocol_sha256=sha(OUT/'protocol.json'))
            if branch=='rgb':
                metadata.update(preprocess='fullfov224x400',input_scope='tracked_person_roi',
                                person_preprocessing=dict(margin=.15,pad=114,span_sec=1.,frames=32,proc_max_side=720))
            torch.save(metadata,dest)
        loaded=torch.load(dest,map_location='cpu',weights_only=True)
        loaded=loaded.get('model',loaded)
        assert loaded.keys()==tensors.keys() and all(torch.equal(loaded[k],v) for k,v in tensors.items())
        reference=np.load(OUT/'predictions'/f'{branch}-{args.which}.npz')['probs']
        if branch=='rgb':
            recognizer=BehaviorRecognizer(str(OP/'ckpts'),skel_fusion_alpha=1.)
            recognizer.ckpt=str(dest);recognizer.load('cuda:0')
            assert tuple(recognizer.cfg['fov'])==(224,400)
            ds=Windows(rows,chosen,'rgb',None)
            x=torch.stack([ds[j][0] for j in range(len(chosen))]).cuda()
            with torch.autocast('cuda',dtype=torch.float16):p=recognizer.net(x).softmax(1).float().cpu().numpy()
            expected=reference[chosen];recognizer.close()
        else:
            net=STGCN(num_classes=10).cuda().eval();net.load_state_dict(loaded)
            features=np.load(OUT/'skeleton-inputs.npz')['features'][GATES.index(35.),chosen]
            p=net(torch.from_numpy(features).cuda()).softmax(1).cpu().numpy()
            expected=reference[GATES.index(35.),chosen]
        error=float(np.max(np.abs(p-expected)))
        assert error<.003,(branch,error)
        assert np.array_equal(p.argmax(1),expected.argmax(1)),branch
        results[branch]=dict(path=str(dest),sha256=sha(dest),source=str(source),source_sha256=sha(source),
            epoch=selected['epoch'],tensors_identical=True,weights_only_load=True,
            actual_inference_windows=len(chosen),max_probability_abs_difference=error,argmax_equal=True)
    protocol=read(OUT/'protocol.json')
    assert sha(Path(protocol['previous_experiment'])/'dataset.json')==protocol['previous_dataset_sha256']
    for model in protocol['models'].values():assert sha(model['path'])==model['sha256']
    save(OUT/f'export-{args.which}-verification.json',dict(models=results,production_models_unchanged=True,
        previous_dataset_unchanged=True,upload_package=False,chosen_ids=[rows[i]['id'] for i in chosen]))
    print(json.dumps(results))


if __name__=='__main__':main()
