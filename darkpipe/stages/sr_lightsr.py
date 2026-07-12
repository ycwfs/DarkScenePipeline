"""Super-resolution stage: MambaIRv2 lightSR x2.

Invariants (validated):
- net stays fp32; speed comes from torch.autocast fp16 (net.half() breaks at ASSM's
  out_norm: the selective scan hard-casts its output to fp32 and LayerNorm rejects the mix).
- torch.manual_seed(0) before EVERY forward: ASSM uses gumbel_softmax(hard=True) at eval
  time; unseeded outputs are non-reproducible.
- at load(): autocast output is PSNR-compared to fp32 on a probe frame; < 45 dB falls back
  to pure fp32 for the session.
"""
import os

import torch

from ..constants import CKPT_FILES
from ..utils import bgr_batch_to_tensor, psnr_uint8, reflect_pad_to, tensor_to_bgr_list
from .base import FrameStage

LIGHTSR_KWARGS = dict(upscale=2, in_chans=3, img_size=64, img_range=1.0, embed_dim=48,
                      d_state=8, depths=[5, 5, 5, 5], num_heads=[4, 4, 4, 4], window_size=16,
                      inner_rank=32, num_tokens=64, convffn_kernel_size=5, mlp_ratio=1.0,
                      upsampler="pixelshuffledirect", resi_connection="1conv")


class LightSRStage(FrameStage):
    name = "sr:lightsr_x2"

    def __init__(self, ckpt_dir: str, chunk: int = 8, force_fp32: bool = False):
        self.ckpt = os.path.join(ckpt_dir, CKPT_FILES["lightsr_x2"])
        self.chunk = chunk
        self.autocast = not force_fp32
        self.net = None
        self.device = "cuda"

    def load(self, device: str) -> None:
        from ..vendor.mambairv2light_arch import MambaIRv2Light
        net = MambaIRv2Light(**LIGHTSR_KWARGS)
        sd = torch.load(self.ckpt, map_location="cpu", weights_only=True)
        net.load_state_dict(sd["params"], strict=True)
        self.device = device
        self.net = net.to(device).eval()
        if self.autocast:
            probe = [torch.randint(0, 255, (96, 128, 3), dtype=torch.uint8).numpy()]
            a = self._forward(probe, autocast=True)[0]
            b = self._forward(probe, autocast=False)[0]
            p = psnr_uint8(a, b)
            if p < 45:
                print(f"[lightsr] autocast PSNR {p:.1f} dB < 45 -> falling back to fp32")
                self.autocast = False

    @torch.inference_mode()
    def _forward(self, frames, autocast):
        outs = []
        for i in range(0, len(frames), self.chunk):
            x = bgr_batch_to_tensor(frames[i:i + self.chunk], self.device, torch.float32)
            x, (h, w) = reflect_pad_to(x, 16)
            torch.manual_seed(0)
            with torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
                y = self.net(x)
            outs.extend(tensor_to_bgr_list(y.float()[:, :, : h * 2, : w * 2]))
        return outs

    def __call__(self, frames: list) -> list:
        return self._forward(frames, self.autocast)

    def close(self) -> None:
        self.net = None
        torch.cuda.empty_cache()
