"""Generate parity-reference fixtures from the OLD validated envs. Run twice:

  1) Retinexformer conda env:  --env retinexformer
       saves dark_frame.npy (input), enh_frame.npy + ref_retinexformer.npy (fp16 output)
  2) videomamba conda env (PYTHONPATH=<MambaIR repo>):  --env videomamba
       saves ref_lightsr.npy (fp32, seed0), clip16_tensor.npy + ref_r3d_logits.npy,
             clip32_tensor.npy + ref_videomamba_logits.npy   (fp32 forwards, no autocast)

The new env's scripts/check_parity.py compares against these.
"""
import argparse, os, sys
import numpy as np

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "assets")
RETX = "/data1/data1/wfs/project/low-light/Retinexformer"
ARID = f"{RETX}/data/Action_Recognition_in_the_Dark"


def env_retinexformer():
    import cv2, torch
    import torch.nn.functional as F
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "rfarch", f"{RETX}/basicsr/models/archs/RetinexFormer_arch.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    net = m.RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1, 2, 2])
    net.load_state_dict(torch.load(f"{RETX}/pretrain_model/NTIRE.pth", map_location="cpu")["params"])
    net = net.cuda().eval().half()

    cap = cv2.VideoCapture(f"{ARID}/clips_v1.5/Sit/Sit_12_20.mp4")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
    ok, dark = cap.read(); cap.release(); assert ok
    np.save(f"{ASSETS}/dark_frame.npy", dark)

    with torch.inference_mode():
        rgb = cv2.cvtColor(dark, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).cuda().permute(2, 0, 1).unsqueeze(0).half().div_(255.0)
        h, w = x.shape[2:]
        H, W = (h + 3) // 4 * 4, (w + 3) // 4 * 4
        x = F.pad(x, (0, W - w, 0, H - h), "reflect")
        y = net(x)[:, :, :h, :w].clamp_(0, 1).mul_(255).round_()
        out = cv2.cvtColor(y[0].permute(1, 2, 0).to(torch.uint8).cpu().numpy(), cv2.COLOR_RGB2BGR)
    np.save(f"{ASSETS}/ref_retinexformer.npy", out)
    np.save(f"{ASSETS}/enh_frame.npy", out)
    print("retinexformer refs saved:", dark.shape, "->", out.shape)


def env_videomamba():
    import cv2, torch, yaml
    import torch.nn.functional as F
    sys.path.insert(0, "/data1/data1/wfs/project/low-light/MambaIR")
    from basicsr.archs import build_network

    enh = np.load(f"{ASSETS}/enh_frame.npy")

    # ---- lightSR fp32 seed0 ----
    opt = yaml.safe_load(open("/data1/data1/wfs/project/low-light/MambaIR/options/test/mambairv2/test_MambaIRv2_lightSR_x2.yml"))
    sr = build_network(dict(opt["network_g"]))
    sd = torch.load("/data1/data1/wfs/project/low-light/MambaIR/ckpts/v1.0/mambairv2_lightSR_x2.pth",
                    map_location="cpu", weights_only=True)
    sr.load_state_dict(sd["params"], strict=True); sr = sr.cuda().eval()
    with torch.inference_mode():
        rgb = cv2.cvtColor(enh, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).cuda().permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
        h, w = x.shape[2:]
        ph, pw = (16 - h % 16) % 16, (16 - w % 16) % 16
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), "reflect")
        torch.manual_seed(0)
        y = sr(x)[:, :, : h * 2, : w * 2].clamp_(0, 1).mul_(255).round_()
        out = cv2.cvtColor(y[0].permute(1, 2, 0).to(torch.uint8).cpu().numpy(), cv2.COLOR_RGB2BGR)
    np.save(f"{ASSETS}/ref_lightsr.npy", out)
    print("lightSR ref saved:", out.shape)
    del sr; torch.cuda.empty_cache()

    # ---- clip tensors via the deployed preprocessing ----
    import importlib.util
    spec = importlib.util.spec_from_file_location("pf", f"{RETX}/Enhancement/pipeline_full.py")
    pf = importlib.util.module_from_spec(spec); spec.loader.exec_module(pf)
    frames = pf.read_frames(f"{ARID}/clips_enhanced/Sit/Sit_12_20.mp4")
    t16 = pf.clip_tensor(frames, pf.RECO_CFG["r3d"]).numpy()
    t32 = pf.clip_tensor(frames, pf.RECO_CFG["videomamba"]).numpy()
    np.save(f"{ASSETS}/clip16_tensor.npy", t16)
    np.save(f"{ASSETS}/clip32_tensor.npy", t32)

    # ---- r3d logits fp32 ----
    r3d = pf.load_recognizer("r3d").float()
    with torch.inference_mode():
        lo = r3d(torch.from_numpy(t16).cuda().float()).cpu().numpy()
    np.save(f"{ASSETS}/ref_r3d_logits.npy", lo)
    print("r3d logits:", lo.round(3))
    del r3d; torch.cuda.empty_cache()

    # ---- videomamba logits fp32 ----
    vm = pf.load_recognizer("videomamba").float()
    with torch.inference_mode():
        lo = vm(torch.from_numpy(t32).cuda().float()).cpu().numpy()
    np.save(f"{ASSETS}/ref_videomamba_logits.npy", lo)
    print("videomamba logits:", lo.round(3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=["retinexformer", "videomamba"], required=True)
    a = ap.parse_args()
    os.makedirs(ASSETS, exist_ok=True)
    env_retinexformer() if a.env == "retinexformer" else env_videomamba()
