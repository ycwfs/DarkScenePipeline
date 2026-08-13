"""Denoising options for enhanced low-light frames: measured, and rendered for eyeballing.

Two questions this answers, both on real deployment footage rather than on ARID:

  * where the denoiser belongs. Denoising the dark input BEFORE enhancement scored worse
    than denoising after, on every scene tried (59-65% noise removed against 73-75%, at
    similar or worse edge retention). The reason is legible in the numbers: a dark frame has
    too little signal for NLM's patch-similarity test to be reliable, and the enhancer is
    already doing 64-76% of the denoising by itself, so a pre-pass mostly duplicates it while
    whatever survives still gets amplified.
  * what it costs. NLM is the only method that removes a useful amount, and its cost is the
    search-window size, not the filter strength: window 7 buys 51% for 118 ms, window 15
    buys 75% for 330 ms.

Note the measurement trap this ran into first: the delivered outputs are downscaled,
enhanced, then upscaled x2, and bicubic upscaling correlates neighbouring pixels, so a
per-pixel noise metric reads *lower* on the noisy output than on the input (two of the three
production clips measured 0.00). Everything here is therefore measured at the processing
resolution, before SR.

    python denoise_probe.py --video <input.mp4> --measure     # numbers for one clip
    python denoise_probe.py --video <input.mp4> --sheet out.png
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

METHODS = [
    ("bilateral d5", lambda x: cv2.bilateralFilter(x, 5, 40, 40)),
    ("bilateral d9", lambda x: cv2.bilateralFilter(x, 9, 60, 60)),
    ("NLM h3 win7", lambda x: cv2.fastNlMeansDenoisingColored(x, None, 3, 3, 7, 7)),
    ("NLM h3 win11", lambda x: cv2.fastNlMeansDenoisingColored(x, None, 3, 3, 7, 11)),
    ("NLM h3 win15", lambda x: cv2.fastNlMeansDenoisingColored(x, None, 3, 3, 7, 15)),
]


def noise(bgr):
    """Flat-block noise estimate: 10th-percentile local std over 8x8 blocks.

    The low percentile is what keeps texture out of it -- the flattest blocks are the ones
    where whatever is left is noise.
    """
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = g.shape
    h -= h % 8
    w -= w % 8
    b = g[:h, :w].reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 64)
    return float(np.percentile(b.std(axis=1), 10))


def edge_energy(bgr, mask):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0),
                          cv2.Sobel(g, cv2.CV_32F, 0, 1))[mask].mean())


def enhanced_frames(video, idxs, proc_max_side, device, ckpt_dir):
    from darkpipe.config import PipelineConfig, validate
    from darkpipe.stages import build_stages
    from darkpipe.stages.downscale import DownscaleStage
    cap = cv2.VideoCapture(video)
    raw = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if ok:
            raw.append(f)
    cap.release()
    if not raw:
        sys.exit(f"error: no frames read from {video}")
    small = DownscaleStage(proc_max_side)(raw) if proc_max_side else raw
    cfg = validate(PipelineConfig(input=video, enhance="retinexformer", sr="off",
                                  recognize="off", ckpt_dir=ckpt_dir, device=device,
                                  output="/tmp/unused.mp4"))
    stages, _ = build_stages(cfg)
    for s in stages:
        s.load(device)
    return raw, small, stages[0](list(small))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames", default="10,30,50,70")
    ap.add_argument("--proc-max-side", type=int, default=1280)
    ap.add_argument("--ckpt-dir", default="/ssd1/wfs/project/lowlight/DarkScenePipeline/ckpts")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--sheet", default="")
    ap.add_argument("--crop", default="", help="x,y,w,h at processing resolution")
    a = ap.parse_args()

    idxs = [int(x) for x in a.frames.split(",")]
    raw, small, enh = enhanced_frames(a.video, idxs, a.proc_max_side, a.device, a.ckpt_dir)

    if a.measure:
        g0 = cv2.cvtColor(enh[0], cv2.COLOR_BGR2GRAY).astype(np.float32)
        gm = np.hypot(cv2.Sobel(g0, cv2.CV_32F, 1, 0), cv2.Sobel(g0, cv2.CV_32F, 0, 1))
        E = gm > np.percentile(gm, 80)
        bn, be = np.mean([noise(x) for x in enh]), edge_energy(enh[0], E)
        print(f"  source {raw[0].shape[1]}x{raw[0].shape[0]} luma {np.mean([x.mean() for x in raw]):.1f} "
              f"noise {np.mean([noise(x) for x in small]):.2f}")
        print(f"  {'method':<16}{'noise':>7}{'reduced':>9}{'edges':>7}{'ms/frame':>10}")
        print(f"  {'enhance only':<16}{bn:7.2f}{0:8.0f}%{100:6.0f}%{0.0:9.0f}")
        for name, fn in METHODS:
            fn(enh[0])
            t = time.time()
            out = [fn(x) for x in enh]
            ms = (time.time() - t) / len(enh) * 1000
            n = np.mean([noise(x) for x in out])
            print(f"  {name:<16}{n:7.2f}{(1 - n / bn) * 100:8.0f}%"
                  f"{edge_energy(out[0], E) / be * 100:6.0f}%{ms:9.0f}")
        # placement: before vs after. The pre-pass has to re-run enhancement, so this is the
        # only fair way to compare the two orders.
        from darkpipe.config import PipelineConfig, validate
        from darkpipe.stages import build_stages
        cfg = validate(PipelineConfig(input=a.video, enhance="retinexformer", sr="off",
                                      recognize="off", ckpt_dir=a.ckpt_dir, device=a.device,
                                      output="/tmp/unused.mp4"))
        st, _ = build_stages(cfg)
        for s in st:
            s.load(a.device)
        nlm = dict(METHODS)["NLM h3 win15"]
        pre = st[0]([nlm(x) for x in small])
        both = [nlm(x) for x in pre]
        for label, seq in (("denoise->enhance", pre), ("denoise->enh->denoise", both)):
            n = np.mean([noise(x) for x in seq])
            print(f"  {label:<16}{n:7.2f}{(1 - n / bn) * 100:8.0f}%"
                  f"{edge_energy(seq[0], E) / be * 100:6.0f}%{'':>9}")

    if a.sheet:
        base = enh[0]
        if a.crop:
            X, Y, W, H = (int(v) for v in a.crop.split(","))
        else:
            X, Y, W, H = 820, 150, 300, 190
        crop = base[Y:Y + H, X:X + W]
        show = [("(1) enhanced", crop)] + [(f"({i + 2}) {n}", fn(crop))
                                           for i, (n, fn) in enumerate(METHODS) if "NLM" in n
                                           or "d5" in n]

        def panel(name, img, zoom, width):
            big = cv2.resize(img, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
            big = cv2.resize(big, (width, int(big.shape[0] * width / big.shape[1])),
                             interpolation=cv2.INTER_NEAREST)
            bar = np.full((26, width, 3), 20, np.uint8)
            cv2.putText(bar, name, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (245, 245, 245),
                        1, cv2.LINE_AA)
            return np.vstack([bar, big])

        Wp = 560
        r1 = np.hstack([panel(n, i, 2, Wp) for n, i in show])
        r2 = np.hstack([panel(n, i[70:130, 60:170], 5, Wp) for n, i in show])
        sheet = np.vstack([r1, np.full((12, r1.shape[1], 3), 40, np.uint8), r2])
        cv2.imwrite(a.sheet, sheet)
        print(f"  [sheet] {a.sheet}  {sheet.shape[1]}x{sheet.shape[0]}")


if __name__ == "__main__":
    main()
