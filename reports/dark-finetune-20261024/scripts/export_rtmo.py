"""Export RTMO-s to the existing rtmlib dets/keypoints interface (batch 1)."""
import rtmo_support
import argparse
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from torchvision.ops import nms

from prepare import OUT, ROOT, save
from train_pose import letterbox


class ExportRTMO(torch.nn.Module):
    def __init__(self,model):
        super().__init__();self.model=model
        # Fuse the official RepVGG neck blocks; retain mathematically identical DCC.
        for m in list(model.modules()):
            if m.__class__.__name__ in ['HybridEncoder','RepVGGBlock']:
                m.switch_to_deploy(dict(input_size=(640,640)))
        self.register_buffer('priors',torch.cat(model.head.prior_generator.grid_priors(
            [(40,40),(20,20)],dtype=torch.float32,device='cpu',with_stride=True)))

    def forward(self,x):
        head=self.model.head
        cls,bbox,_,vis,vec=head(self.model.extract_feat(x))
        scores=head._flatten_predictions(cls).sigmoid()[0,:,0]
        boxes=head.decode_bbox(head._flatten_predictions(bbox),self.priors[:,:2],self.priors[:,-1])[0]
        visibility=head._flatten_predictions(vis).sigmoid()[0]
        features=head._flatten_predictions(vec)[0]
        # Decode DCC only after the same class-agnostic NMS. Keep a single
        # below-threshold fallback to make empty-image ONNX tensors well-defined;
        # rtmlib's .45 threshold removes it.
        candidates=torch.cat([torch.where(scores>.1)[0],scores.argmax().reshape(1)])
        keep=nms(boxes[candidates],scores[candidates],.45)[:100]
        indices=candidates[keep];top=scores[indices]
        boxes=boxes[indices];features=features[indices];visibility=visibility[indices]
        cs=torch.cat([(boxes[:,:2]+boxes[:,2:])/2,(boxes[:,2:]-boxes[:,:2])*head.bbox_padding],dim=1)
        xy=head.dcc.forward_test(features,cs,self.priors[indices,:2])
        return torch.cat([boxes,top[:,None]],dim=1)[None],torch.cat([xy,visibility[:,:,None]],dim=2)[None]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',default='rtmo-s_body7_original.pth')
    parser.add_argument('--output',default='rtmo-s_original_export.onnx');args=parser.parse_args()
    torch.set_num_threads(2);cv2.setNumThreads(1)
    model=rtmo_support.load_model(OUT/'models'/args.checkpoint)
    wrapper=ExportRTMO(model).eval();output=OUT/'models'/args.output
    image=cv2.imread(str(next((OUT/'dataset/images/val').glob('*.png'))))
    x=letterbox(image)[0][None].float()
    with torch.no_grad():expected=wrapper(x)
    torch.onnx.export(wrapper,x,str(output),opset_version=17,dynamo=False,
        input_names=['input'],output_names=['dets','keypoints'],do_constant_folding=True)
    onnx.checker.check_model(onnx.load(output))
    opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
    session=ort.InferenceSession(str(output),sess_options=opts,providers=['CPUExecutionProvider'])
    actual=session.run(None,{'input':x.numpy()})
    # Top-k order can differ only at almost equal scores; compare confident entries.
    mask=expected[0][0,:,4].numpy()>=.45
    errors=[float(np.max(np.abs(a[0,mask]-b.numpy()[0,mask]))) if mask.any() else 0.
            for a,b in zip(actual,expected)]
    assert max(errors)<.1,errors
    save(output.with_suffix('.verification.json'),dict(checkpoint=args.checkpoint,
         confident_count=int(mask.sum()),max_absolute_errors=errors,onnx_check=True,
         interface='B1 dets[1,N,5], keypoints[1,N,17,3]; NMS before DCC, caller threshold .45'))
    print(output,errors,flush=True)


if __name__=='__main__':main()
