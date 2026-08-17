"""Super-resolution stage: CATANet x2 (CVPR 2025).

An alternative to `lightsr_x2`, and the faster of the two at batch 1 — which is the batch
serve mode uses. Measured on an RTX 3090 at 640x480 input: 828 ms/frame vs lightSR's 1018,
which is what brings serve latency from 1082-1123 ms (over the 1 s spec) down to 825-903 ms.
Batched the advantage inverts (788 ms/frame vs lightSR's 738 at chunk 4), so this is the
real-time option, not the offline-throughput one. Neither reaches the 15 fps target.

Notes specific to this arch:
- fp16 autocast buys ~2%, not the ~2x it buys a matmul-bound net: the cost here is token
  clustering — argsort/gather/scatter/normalize — which is memory-bound and which autocast
  does not touch. It is still requested for uniformity with lightSR and guarded by the same
  PSNR self-check, which this arch fails (40-42 dB) and therefore runs fp32 in practice.
- Memory, not compute, is the binding constraint: 10.3 GiB for a single 640x480 frame
  (~3x lightSR), so chunk stays at 1 by default; chunk 4 at that size OOMs on a 24 GB card
  and is only recoverable through the shared halving helper.
- No seeding needed, unlike lightSR: there is no stochastic op at eval. The `initted`
  buffers ship True, so the token centers come from training rather than being re-estimated
  per frame.
"""
import os

import torch

from ..constants import sr_ckpt_file
from ..utils import free_device_cache, bgr_batch_to_tensor, chunked, psnr_uint8, tensor_to_bgr_list
from .base import FrameStage


class CATANetStage(FrameStage):
    name = "sr:catanet_x2"
    post_recognition = True

    def __init__(self, ckpt_dir: str, chunk: int = 1, force_fp32: bool = False,
                 scale: int = 2):
        # x4 takes a different upsampler path in the arch (two PixelShuffle stages instead
        # of one); x2/x3 share theirs. Weights are per-scale, from the CATANet release.
        self.scale = scale
        self.name = f"sr:catanet_x{scale}"
        self.ckpt = os.path.join(ckpt_dir, sr_ckpt_file("catanet", scale))
        self.chunk = chunk
        self.autocast = not force_fp32
        self.net = None
        self.device = "cuda"

    def load(self, device: str) -> None:
        from ..vendor.catanet_arch import CATANet
        net = CATANet(upscale=self.scale)
        sd = torch.load(self.ckpt, map_location="cpu", weights_only=True)
        net.load_state_dict(sd["params"], strict=True)
        self.device = device
        self.net = net.to(device).eval()
        if self.autocast:
            probe = [torch.randint(0, 255, (96, 128, 3), dtype=torch.uint8).numpy()]
            p = psnr_uint8(self._batch(probe, True)[0], self._batch(probe, False)[0])
            if p < 45:
                print(f"[catanet] autocast PSNR {p:.1f} dB < 45 -> falling back to fp32")
                self.autocast = False

    @torch.inference_mode()
    def _batch(self, frames, autocast):
        # No reflect-pad wrapper: CATANet's patch_divide/IASA pad internally to the per-block
        # patch and group sizes, so odd frame sizes come back at exactly 2x (asserted in tests).
        x = bgr_batch_to_tensor(frames, self.device, torch.float32)
        with torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
            y = self.net(x)
        return tensor_to_bgr_list(y.float())

    def __call__(self, frames: list) -> list:
        return chunked(self, frames, lambda b: self._batch(b, self.autocast))

    def close(self) -> None:
        self.net = None
        free_device_cache(self.device)
