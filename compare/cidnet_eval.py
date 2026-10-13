"""HVI-CIDNet inference on a low-light folder, faithful to HVI-CIDNet/eval.py.

Replicates eval.py's per-dataset behaviour:
  real: model.trans.gated2 = True; model.trans.alpha = alpha
  syn : no gating flags set (matches eval.py, where v2 flag is only lol_v2_real)
  output = model(input ** gamma), then clamp[0,1].
Loads images through the repo's own get_eval_set (ToTensor, full size).
"""
import os
import sys
import time
import argparse

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

HVI_DIR = "/ssd1/wfs/project/lowlight/HVI-CIDNet"
sys.path.insert(0, HVI_DIR)
from data.data import get_eval_set          # noqa: E402
from net.CIDNet import CIDNet               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--low_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--weight", required=True)
    ap.add_argument("--mode", choices=["real", "syn"], required=True)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.set_grad_enabled(False)

    model = CIDNet().cuda().eval()
    model.load_state_dict(torch.load(args.weight, map_location="cpu"))
    if args.mode == "real":
        model.trans.gated2 = True
        model.trans.alpha = args.alpha

    ds = get_eval_set(args.low_dir)
    dl = DataLoader(dataset=ds, num_workers=2, batch_size=1, shuffle=False)

    t0 = time.time()
    n = 0
    for batch in dl:
        inp, name = batch[0], batch[1]
        inp = inp.cuda()
        out = model(inp ** args.gamma)
        out = torch.clamp(out, 0, 1).cpu()
        transforms.ToPILImage()(out.squeeze(0)).save(os.path.join(args.out_dir, name[0]))
        n += 1
    torch.cuda.synchronize()
    print(f"[cidnet] {args.mode} weight={os.path.basename(args.weight)} alpha={args.alpha} "
          f"gamma={args.gamma} -> {n} imgs in {time.time()-t0:.1f}s -> {args.out_dir}")


if __name__ == "__main__":
    main()
