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


def free_device_cache(device):
    """Release cached blocks on `device` -- the stage's card, not whichever one is current.

    `torch.cuda.empty_cache()` frees the *current* device, which is cuda:0 unless something
    set it, so a stage on cuda:3 was freeing a card it never allocated on. That was invisible
    while every stage in a process shared one card; serve mode now runs enhance instances on
    several at once, where it would leak one card's weights per extra device.
    """
    if device is None or not str(device).startswith("cuda") or not torch.cuda.is_available():
        return
    with torch.cuda.device(device):
        torch.cuda.empty_cache()


def bind_thread_device(device):
    """Make `device` this thread's current CUDA device. Call it first in any worker thread.

    PyTorch's current device is thread-local, and every thread starts on the process default
    (cuda:0). Placement is already explicit everywhere here, and ATen's per-op device guards
    make that correct on its own -- so this is not a correctness fix for our own code. What it
    covers is everything that reads the CURRENT device rather than a tensor's: third-party CUDA
    extensions (mamba_ssm / causal_conv1d, the VideoMamba selective-scan kernels) commonly take
    their stream from at::cuda::getCurrentCUDAStream() with no device argument, and cuBLAS/cuDNN
    workspaces and autotune caches are per current device.

    Added while chasing a 242 -> 312 ms inference regression that appeared when recognition
    moved from cuda:0 (shared with enhance) to a card of its own -- i.e. it got slower after
    being given more resources, which points at stream/allocator state rather than contention.
    Whether this is that cause is still unmeasured; it is worth doing either way, because a
    worker submitting through another card's stream state is not a thing to leave in place.
    """
    if device is None or not str(device).startswith("cuda") or not torch.cuda.is_available():
        return
    try:
        torch.cuda.set_device(device)
    except Exception as e:                           # noqa: BLE001 - never fatal
        print(f"[warn] set_device({device}) 失败，线程仍用默认卡：{e}")


def reflect_pad_to(x, multiple):
    """Pad [B,C,H,W] on bottom/right so H,W are multiples of `multiple`. Returns x, (h, w)."""
    h, w = x.shape[2:]
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), "reflect")
    return x, (h, w)


def chunked(stage, frames, fn):
    """Apply `fn` over `frames` in batches of `stage.chunk`, halving the chunk on OOM.

    The safe batch size is a function of resolution, not just of the card, and the enhancer,
    SR and recognizer share one GPU — so no fixed default is right everywhere. Measured on a
    24 GB 3090: the shipped enhance chunk of 32 needs a 2.2 GiB allocation at 1280x720 and
    dies, while the same chunk is comfortable at 640x480. Rather than make the caller pick a
    number per resolution, each stage finds its own ceiling on the first failure and keeps the
    reduced value for the session (falling back to 1 frame at a time in the worst case, which
    is what serve mode already does successfully at 720p).
    """
    outs = []
    i = 0
    while i < len(frames):
        try:
            outs.extend(fn(frames[i:i + stage.chunk]))
        except torch.cuda.OutOfMemoryError:
            if stage.chunk == 1:
                raise
            stage.chunk = max(1, stage.chunk // 2)
            free_device_cache(getattr(stage, "device", None))
            print(f"[{stage.name}] OOM -> chunk {stage.chunk}", flush=True)
            continue
        i += stage.chunk
    return outs


def psnr_uint8(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse < 1e-9 else 10 * np.log10(255.0 ** 2 / mse)
