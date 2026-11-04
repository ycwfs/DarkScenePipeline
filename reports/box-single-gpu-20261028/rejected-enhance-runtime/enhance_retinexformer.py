"""Low-light enhancement stage: Retinexformer (NTIRE weight), fp16, chunked."""
import os

import torch

from ..constants import CKPT_FILES
from ..utils import free_device_cache, bgr_batch_to_tensor, chunked, reflect_pad_to, tensor_to_bgr_list
from .base import FrameStage
from ..box_accel import GraphForward, graphs_enabled
from ..box_onnx import TensorRTSession, backend


class RetinexformerStage(FrameStage):
    name = "enhance:retinexformer"

    def __init__(self, ckpt_dir: str, chunk: int = 32):
        self.ckpt = os.path.join(ckpt_dir, CKPT_FILES["retinexformer"])
        self.chunk = chunk
        self.net = None
        self.graph = None
        self.trt_session = None
        self.device = "cuda"

    def load(self, device: str) -> None:
        from ..vendor.retinexformer_arch import RetinexFormer
        net = RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1,
                            num_blocks=[1, 2, 2])
        net.load_state_dict(torch.load(self.ckpt, map_location="cpu")["params"], strict=True)
        self.device = device
        self.net = net.to(device).eval().half()
        if backend() == 'tensorrt':
            onnx = os.path.join(os.path.dirname(self.ckpt), 'retinexformer_ntire_720_box.onnx')
            if not os.path.isfile(onnx):
                raise FileNotFoundError(f'Box TensorRT enhancement model missing: {onnx}')
            self.trt_session = TensorRTSession(onnx, device)
        if graphs_enabled():
            self.graph = GraphForward(self.net)

    @torch.inference_mode()
    def _batch(self, frames: list) -> list:
        x = bgr_batch_to_tensor(frames, self.device, torch.float16)
        x, (h, w) = reflect_pad_to(x, 4)
        # The shipped engine targets the default landscape 720 input. Other aspect ratios
        # retain the trained PyTorch model instead of stretching the image or disabling it.
        if self.trt_session is not None and tuple(x.shape) == (1, 3, 404, 720):
            y = self.trt_session.run_tensor(x)[:, :, :h, :w]
        else:
            y = (self.graph or self.net)(x)[:, :, :h, :w]
        return tensor_to_bgr_list(y)

    def __call__(self, frames: list) -> list:
        return chunked(self, frames, self._batch)

    def close(self) -> None:
        if self.graph is not None:
            self.graph.close()
        self.net = None
        self.trt_session = None
        free_device_cache(self.device)
