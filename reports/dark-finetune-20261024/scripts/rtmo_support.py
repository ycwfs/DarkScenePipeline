"""Isolated official MMPose RTMO runtime; unused MMCV compiled ops fail loudly.

RTMO uses native torch NMS and standard convolutions. Registry imports other
architectures which import MMCV extensions; none may silently run a fake op.
"""
from pathlib import Path
import sys
from types import SimpleNamespace

OUT = Path(__file__).resolve().parents[1]
VENDOR = OUT/'vendor'
sys.path.insert(0,str(VENDOR))
import torch


def setup() -> None:
    from mmcv.utils import ext_loader
    def unavailable(name):
        def op(*args, **kwargs):
            raise RuntimeError(f'Compiled MMCV op {name} is unavailable; no fallback executed')
        return op
    ext_loader.load_ext = lambda module, names: SimpleNamespace(**{n:unavailable(n) for n in names})
    from mmpose.utils import register_all_modules
    register_all_modules()


def config():
    from mmengine.config import Config
    path=VENDOR/'mmpose/.mim/configs/body_2d_keypoint/rtmo/body7/rtmo-s_8xb32-600e_body7-640x640.py'
    cfg=Config.fromfile(str(path))
    cfg.model.backbone.init_cfg=None
    cfg.model.init_cfg=None
    cfg.model.data_preprocessor.batch_augments=None
    meta=str(VENDOR/'mmpose/.mim/configs/_base_/datasets/coco.py')
    cfg.model.head.assigner.oks_calculator.metainfo=meta
    cfg.model.head.loss_oks.metainfo=meta
    # Match the final training stage of the official 600-epoch checkpoint.
    cfg.model.head.proxy_target_cc=True
    cfg.model.head.loss_mle.loss_weight=5.
    cfg.model.head.loss_oks.loss_weight=10.
    # Unknown joints are masked coordinate targets, not visibility negatives.
    cfg.model.head.loss_vis.loss_weight=0.
    cfg.model.test_cfg.update(score_thr=.45,nms_thr=.45)
    return cfg


def load_model(path: Path, device='cpu'):
    setup()
    from mmpose.registry import MODELS
    model=MODELS.build(config().model)
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    return model.to(device).eval()


if __name__=='__main__':
    torch.set_num_threads(2)
    model=load_model(OUT/'models/rtmo-s_body7_original.pth')
    print('RTMO strict load',sum(p.numel() for p in model.parameters()))
    with torch.no_grad():
        result=model(torch.zeros(1,3,640,640),None,mode='tensor')
    print('native forward',type(result))
