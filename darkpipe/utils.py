"""Small shared helpers: color/tensor conversion, padding, PSNR."""
import cv2
import numpy as np
import torch
import torch.nn.functional as F


def bgr_batch_to_tensor(frames_bgr, device, dtype):
    """list of BGR uint8 HxWx3 -> [B,3,H,W] in [0,1]."""
    rgb = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr])
    return torch.from_numpy(rgb).to(device).permute(0, 3, 1, 2).to(dtype).div_(255.0)


def tensor_to_bgr_list(y):
    """[B,3,H,W] in [0,1] -> list of BGR uint8."""
    out = y.clamp(0, 1).mul(255).round_().permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()
    return [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in out]


def reflect_pad_to(x, multiple):
    """Pad [B,C,H,W] on bottom/right so H,W are multiples of `multiple`. Returns x, (h, w)."""
    h, w = x.shape[2:]
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), "reflect")
    return x, (h, w)


def psnr_uint8(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse < 1e-9 else 10 * np.log10(255.0 ** 2 / mse)
