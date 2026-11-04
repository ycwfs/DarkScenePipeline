"""Render independent ROI demos using a selected checkpoint; production defaults stay fixed."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import cv2
import torch

OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT/'reports/person-action-20261023/scripts'))
from demo import PersonModels,run_sample,RetinexformerStage,OP
from darkpipe.stages.recognize import BehaviorRecognizer


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--output',type=Path,default=OUT/'demo')
    parser.add_argument('--samples',type=Path,default=OUT/'demo-samples.json')
    args=parser.parse_args()
    checkpoint=args.checkpoint or Path(json.loads((OUT/'selection.json').read_text())['path'])
    torch.set_num_threads(4);cv2.setNumThreads(2)
    for folder in ['videos','stills','events','crops']:(args.output/folder).mkdir(parents=True,exist_ok=True)
    if (args.output/'sample-results.json').exists():raise RuntimeError('Refusing to overwrite existing demo')
    models=PersonModels('cuda:0')
    models.rgb.close()
    models.rgb=BehaviorRecognizer(str(OP/'ckpts'),skel_fusion_alpha=1.)
    models.rgb.ckpt=str(checkpoint);models.rgb.load('cuda:0');models.rgb.warmup()
    enhancer=RetinexformerStage(str(OP/'ckpts'),chunk=1);enhancer.load('cuda:0')
    results=[]
    try:
        for item in json.loads(args.samples.read_text()):
            results.append(run_sample(item,models,enhancer,args.output))
            (args.output/'sample-results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    finally:models.close();enhancer.close()
    provenance=dict(checkpoint=str(checkpoint),sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        alpha=.7,threshold=.75,lum_gate=75.,production_modified=False,
        inputs='same-track 32 RGB crops and corresponding pose samples',
        caveat='Visual demonstrations include training examples; numerical comparison uses the separately frozen test set.')
    (args.output/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
