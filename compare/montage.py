"""Build side-by-side comparison montages with arbitrary labelled columns.

Usage: repeat --col LABEL DIR for each column (order = display order); the tool
auto-selects N evenly-spaced scenes (by filename) shared across all columns.
"""
import os
import argparse

from PIL import Image, ImageDraw, ImageFont

PANEL_W = 300
PAD = 6
HEADER_H = 22


def load_resized(path, w):
    im = Image.open(path).convert("RGB")
    ow, oh = im.size
    return im.resize((w, round(oh * w / ow)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--col", action="append", nargs=2, metavar=("LABEL", "DIR"), required=True)
    ap.add_argument("--pick_from", required=True, help="dir whose filenames drive scene selection")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    labels = [c[0] for c in args.col]
    dirs = [c[1] for c in args.col]
    ncol = len(dirs)

    files = sorted(f for f in os.listdir(args.pick_from)
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))
    idx = [round(i * (len(files) - 1) / max(args.n - 1, 1)) for i in range(args.n)]
    picks = [files[i] for i in idx]

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    rows = []
    for name in picks:
        panels = [load_resized(os.path.join(d, name), PANEL_W) for d in dirs]
        rh = max(p.height for p in panels)
        row = Image.new("RGB", (PANEL_W * ncol + PAD * (ncol + 1), rh + PAD * 2), (20, 20, 20))
        for c, p in enumerate(panels):
            row.paste(p, (PAD + c * (PANEL_W + PAD), PAD))
        rows.append(row)

    grid_w = rows[0].width
    total_h = HEADER_H * 2 + sum(r.height for r in rows)
    canvas = Image.new("RGB", (grid_w, total_h), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    if args.title:
        draw.text((4, 3), args.title, fill=(120, 220, 120), font=font)
    for c, title in enumerate(labels):
        draw.text((PAD + c * (PANEL_W + PAD) + 4, HEADER_H + 3), title, fill=(255, 255, 255), font=font)

    y = HEADER_H * 2
    for r in rows:
        canvas.paste(r, (0, y))
        y += r.height

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    canvas.save(args.out)
    print(f"saved {args.out} ({args.n} scenes: {', '.join(picks)})")


if __name__ == "__main__":
    main()
