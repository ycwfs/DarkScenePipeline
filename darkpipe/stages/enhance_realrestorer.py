"""RealRestorer enhancement stage (diffusion, whole-video, OFFLINE ONLY, ~45 s/frame).

Heavy imports (transformers, the 39 GiB bundle) happen only in load() — selecting
retinexformer never touches this stack.
"""
import os

import cv2
import numpy as np

from ..constants import CKPT_FILES, REALRESTORER_PROMPT
from .base import FrameStage


class RealRestorerStage(FrameStage):
    name = "enhance:realrestorer"
    whole_video = True

    def __init__(self, ckpt_dir: str, bundle: str | None = None, steps: int = 28,
                 cfg_scale: float = 3.0, chunk: int = 8, prompt: str | None = None,
                 size_level: int = 512):
        self.bundle = bundle or os.path.join(ckpt_dir, CKPT_FILES["realrestorer"])
        self.steps, self.cfg_scale, self.chunk = steps, cfg_scale, chunk
        self.prompt = prompt or REALRESTORER_PROMPT
        self.size_level = size_level
        self.components = None
        self.device = "cuda"

    def load(self, device: str) -> None:
        import torch  # noqa: F401  (heavy stack loads here only)
        from ..vendor.realrestorer.loader import build_components
        self.device = device
        print(f"[realrestorer] loading bundle {self.bundle} (~39 GiB, may take minutes)")
        self.components = build_components(self.bundle)

    def __call__(self, frames: list) -> list:
        from PIL import Image
        from ..vendor.realrestorer.runner import restore_frames
        pils = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
        outs = restore_frames(self.components, pils, self.prompt, steps=self.steps,
                              guidance=self.cfg_scale, chunk=self.chunk,
                              size_level=self.size_level, device=self.device)
        return [cv2.cvtColor(np.array(o), cv2.COLOR_RGB2BGR) for o in outs]

    def close(self) -> None:
        self.components = None
        import torch
        torch.cuda.empty_cache()
