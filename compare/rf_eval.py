"""Retinexformer inference on a low-light folder, using DarkScenePipeline's
vendored architecture (self-contained: torch + einops only).

Config matches DarkScenePipeline's enhance stage / the Retinexformer NTIRE weight:
  RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1,2,2])
Weights are stored under the 'params' key. Images are loaded exactly like the
metric script (Image.open().convert('RGB') + ToTensor), reflect-padded to a
multiple of 4 (denoiser downsamples x4), forwarded fp32, then cropped back.
"""
import os
import sys
import time
import argparse

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

RF_VENDOR = "/ssd1/wfs/project/lowlight/DarkScenePipeline/darkpipe/vendor"
sys.path.insert(0, RF_VENDOR)
from retinexformer_arch import RetinexFormer  # noqa: E402

_IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--low_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--weight", required=True)
    ap.add_argument("--factor", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.set_grad_enabled(False)

    net = RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1,
                        num_blocks=[1, 2, 2]).cuda().eval()
    sd = torch.load(args.weight, map_location="cpu")
    sd = sd["params"] if "params" in sd else sd
    net.load_state_dict(sd, strict=True)

    to_t = transforms.ToTensor()
    files = sorted(f for f in os.listdir(args.low_dir) if f.lower().endswith(_IMG_EXT))

    t0 = time.time()
    for f in files:
        img = Image.open(os.path.join(args.low_dir, f)).convert("RGB")
        x = to_t(img).unsqueeze(0).cuda()               # (1,3,H,W) in [0,1]
        _, _, h, w = x.shape
        fac = args.factor
        H = (h + fac - 1) // fac * fac
        W = (w + fac - 1) // fac * fac
        x = F.pad(x, (0, W - w, 0, H - h), mode="reflect")
        y = net(x)[:, :, :h, :w]
        y = torch.clamp(y, 0, 1).cpu()
        transforms.ToPILImage()(y.squeeze(0)).save(os.path.join(args.out_dir, f))
    torch.cuda.synchronize()
    print(f"[retinexformer] weight={os.path.basename(args.weight)} -> {len(files)} imgs "
          f"in {time.time()-t0:.1f}s -> {args.out_dir}")


if __name__ == "__main__":
    main()
