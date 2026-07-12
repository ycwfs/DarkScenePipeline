"""Low-light enhancement stage: Retinexformer (NTIRE weight), fp16, chunked."""
import os

import torch

from ..constants import CKPT_FILES
from ..utils import bgr_batch_to_tensor, reflect_pad_to, tensor_to_bgr_list
from .base import FrameStage


class RetinexformerStage(FrameStage):
    name = "enhance:retinexformer"

    def __init__(self, ckpt_dir: str, chunk: int = 32):
        self.ckpt = os.path.join(ckpt_dir, CKPT_FILES["retinexformer"])
        self.chunk = chunk
        self.net = None
        self.device = "cuda"

    def load(self, device: str) -> None:
        from ..vendor.retinexformer_arch import RetinexFormer
        net = RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1,
                            num_blocks=[1, 2, 2])
        net.load_state_dict(torch.load(self.ckpt, map_location="cpu")["params"], strict=True)
        self.device = device
        self.net = net.to(device).eval().half()

    @torch.inference_mode()
    def __call__(self, frames: list) -> list:
        outs = []
        for i in range(0, len(frames), self.chunk):
            x = bgr_batch_to_tensor(frames[i:i + self.chunk], self.device, torch.float16)
            x, (h, w) = reflect_pad_to(x, 4)
            y = self.net(x)[:, :, :h, :w]
            outs.extend(tensor_to_bgr_list(y))
        return outs

    def close(self) -> None:
        self.net = None
        torch.cuda.empty_cache()
