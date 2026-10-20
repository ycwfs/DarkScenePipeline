"""低照度增强模型 · 独立推理脚本（Retinexformer，权重 weights/NTIRE.pth）。

加载、前处理、推理、后处理与 DarkScenePipeline 平台链路（enhance=retinexformer）完全一致：

    BGR uint8 帧
      -> RGB、/255 归一化                  （BGR/255，RGB 通道序）
      -> 底/右边 reflect pad 到 4 的倍数    （网络要求 H、W 为 4 的倍数）
      -> fp16 前向
      -> 裁回原尺寸、clamp 到 [0,1]、x255 四舍五入回 uint8
      -> RGB 转回 BGR
    = 增强后的 BGR uint8 帧

权重说明见 README.md。用法：

    python lowlight_infer.py --input sample_data/dark_frame.png
    python lowlight_infer.py --input sample_data/ --output out/
    python lowlight_infer.py --input demo.mp4 --output out/demo_enhanced.mp4
"""
import argparse
import os
import time

import cv2
import numpy as np
import torch

from video_io import (downscale_long_side, encode_video, is_image, list_images,
                      read_image, read_video)

DEFAULT_CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "NTIRE.pth")


class LowLightEnhancer:
    """Retinexformer 增强模型封装：构造一次，enhance() 可反复调用。

    参数
      ckpt    权重路径（NTIRE.pth，键名为 params）
      device  "cuda" / "cpu"，缺省自动；模型只能在 GPU 上跑 fp16，CPU 自动回退 fp32
      fp16    与平台一致的半精度推理；传 False 用 fp32
      chunk   一次送 GPU 的帧数。显存占用 ∝ 分辨率 × chunk：None（默认）按首帧分辨率
              自动取值（约等于 720p 下取 4，与平台默认同量级）；显存不足时手动调小。
    """

    def __init__(self, ckpt: str = DEFAULT_CKPT, device: str | None = None,
                 fp16: bool = True, chunk: int | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if fp16 and not str(device).startswith("cuda"):
            fp16 = False
        self.ckpt, self.device, self.fp16, self.chunk = ckpt, device, fp16, chunk
        self._chunk = chunk
        from models.retinexformer_arch import RetinexFormer
        net = RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1,
                            num_blocks=[1, 2, 2])
        sd = torch.load(ckpt, map_location="cpu")
        net.load_state_dict(sd["params"] if "params" in sd else sd, strict=True)
        self.net = net.to(device).eval()
        if fp16:
            self.net.half()
        self.dtype = torch.float16 if fp16 else torch.float32

    @torch.inference_mode()
    def enhance(self, frames_bgr: list) -> list:
        """list[BGR uint8 HxWx3] -> list[BGR uint8]（增强后），内部按 chunk 分批。"""
        if self._chunk is None:  # 按首帧分辨率定 chunk：总像素量约等于 720p x 4 帧
            h, w = frames_bgr[0].shape[:2]
            self._chunk = max(1, round(4 * (1280 * 720) / (h * w)))
        out = []
        for i in range(0, len(frames_bgr), self._chunk):
            out.extend(self._batch(frames_bgr[i:i + self._chunk]))
        return out

    def _batch(self, frames: list) -> list:
        rgb = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames])
        x = torch.from_numpy(rgb).to(self.device).permute(0, 3, 1, 2).to(self.dtype).div_(255.0)
        # 右/下 reflect pad 到 4 的倍数，推理完裁回原尺寸
        h, w = x.shape[2:]
        ph, pw = (4 - h % 4) % 4, (4 - w % 4) % 4
        if ph or pw:
            x = torch.nn.functional.pad(x, (0, pw, 0, ph), "reflect")
        y = self.net(x)[:, :, :h, :w]
        arr = y.clamp(0, 1).mul(255).round_().permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()
        return [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in arr]


def decide_output(src, a):
    """输出路径：单输入时，图片扩展名或视频的 .mp4 输出视为目标文件，否则视为目录；
    不给 --output 时写到输入旁 <名字>_enhanced.<ext>。"""
    stem, ext = os.path.splitext(os.path.basename(src))
    srcs = a._srcs  # single-input file path only when exactly one input
    video_file = not is_image(src) and a.output and a.output.lower().endswith(".mp4")
    if a.output and len(srcs) == 1 and (is_image(a.output) or video_file):
        parent = os.path.dirname(a.output)
        if parent:
            os.makedirs(parent, exist_ok=True)
        return a.output
    outdir = a.output or (os.path.dirname(src) or ".")
    os.makedirs(outdir, exist_ok=True)
    return os.path.join(outdir, f"{stem}_enhanced{ext}")


def run_one(model, src, dst, proc_max_side):
    if is_image(src):
        frames, fps = [read_image(src)], 0.0
    else:
        frames, fps = read_video(src)
        dst = os.path.splitext(dst)[0] + ".mp4"  # 视频输入一律出 mp4
    if proc_max_side:
        frames = [downscale_long_side(f, proc_max_side) for f in frames]

    t0 = time.time()
    out_frames = model.enhance(frames)
    dt = time.time() - t0

    if fps > 0:
        encode_video(out_frames, dst, fps)
    else:
        cv2.imwrite(dst, out_frames[0])
    mean_in = float(np.stack(frames).astype(np.float32).mean())
    mean_out = float(np.stack(out_frames).astype(np.float32).mean())
    print(f"[done] {os.path.basename(src)} -> {dst}  "
          f"({len(frames)} 帧, {dt:.2f}s, {len(frames) / dt:.1f} 帧/s)  "
          f"平均亮度 {mean_in:.1f} -> {mean_out:.1f}")


def main():
    ap = argparse.ArgumentParser(description="Retinexformer 低照度增强 · 独立推理")
    ap.add_argument("--input", required=True, help="图片 / 图片目录 / 视频路径")
    ap.add_argument("--output", default=None,
                    help="输出文件（单输入）或目录（多输入）；缺省写到输入旁 <名字>_enhanced.<ext>")
    ap.add_argument("--weights", default=DEFAULT_CKPT)
    ap.add_argument("--device", default=None, help="cuda / cpu，缺省自动")
    ap.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--proc-max-side", type=int, default=720,
                    help="增强前先把长边缩到该值（当前部署服务链路默认 720）；0 = 不缩放")
    ap.add_argument("--chunk", type=int, default=None,
                    help="一次送 GPU 的帧数；缺省按输入分辨率自动（约 720p x 4 帧的像素量），显存不足调小")
    a = ap.parse_args()

    model = LowLightEnhancer(a.weights, a.device, a.precision == "fp16", a.chunk)
    print(f"[load] {a.weights}  device={model.device} "
          f"precision={'fp16' if model.fp16 else 'fp32'}")

    if os.path.isdir(a.input):
        srcs = list_images(a.input)
    else:
        srcs = [a.input]
    a._srcs = srcs
    for src in srcs:
        run_one(model, src, decide_output(src, a), a.proc_max_side)


if __name__ == "__main__":
    main()
