"""Label-bar rendering: the recognition result is displayed BELOW the video frame
(spec: '... including the text of the action recognition results displayed below the
video'), never overlaid on the image content.

The behaviour name is drawn in Chinese, which cv2.putText cannot do at all -- its Hershey
fonts are vector strokes with no CJK glyphs, and it silently renders '????' rather than
failing. So the text goes through PIL with a bundled font. The font is a *subset* of Noto
Sans CJK containing only the ~120 characters the bar can ever draw (the ten behaviour names,
"识别中...", and ASCII): 15 KB instead of a 19 MB .ttc, small enough to ship in the zip and
therefore no image rebuild. Noto is OFL-1.1; the licence ships beside it.

Falling back to English rather than to mojibake matters here: if the font were ever missing,
'????' on a monitoring wall looks like a decoding bug in the video, not a missing asset.
"""
import os

import cv2
import numpy as np

from .constants import DISPLAY_TO_ZH, ZH_RECOGNIZING

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets",
                         "NotoSansCJK-subset.ttf")
_font_cache = {}
_warned = []


def _cjk_font(size):
    """PIL font at `size`, or None if the bundled file is unusable. Cached per size."""
    size = max(8, int(size))
    if size in _font_cache:
        return _font_cache[size]
    font = None
    try:
        from PIL import ImageFont
        font = ImageFont.truetype(FONT_PATH, size)
    except Exception as e:                                    # noqa: BLE001
        if not _warned:
            _warned.append(1)
            print(f"[render] 中文字体不可用（{e}），标签条回退为英文: {FONT_PATH}")
    _font_cache[size] = font
    return font


def draw_text(img, text, xy, size, color, font=None):
    """Draw `text` at `xy` (left, baseline-ish). Returns the width drawn.

    Falls back to cv2's Hershey font when the CJK font is missing, which also means an
    ASCII-only caller never depends on PIL.
    """
    font = font or _cjk_font(size)
    if font is None:
        scale = size / 30.0
        cv2.putText(img, text, xy, FONT, scale, color, 2, cv2.LINE_AA)
        return cv2.getTextSize(text, FONT, scale, 2)[0][0]
    from PIL import Image, ImageDraw
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    x, y = xy
    draw.text((x, y - size), text, font=font, fill=tuple(int(c) for c in color[::-1]))
    img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return int(draw.textlength(text, font=font))


def append_label_bar(frame, event, extra: str = ""):
    """Returns frame with a text bar stacked underneath. event may be None (warmup)."""
    h, w = frame.shape[:2]
    # Even height, always. H.264 with yuv420p rejects odd dimensions outright, and the bar is
    # what makes them odd: 0.08 * 2160 rounds to 173, so a 1080p source upscaled x2 came out
    # 3840x2333 and libx264 refused to open ("height not divisible by 2"). It went unnoticed
    # for a while because MJPEG does not care, and cv2's mp4 writer silently rounds down --
    # only the H.264 outputs (FLV/HLS/RTSP push) actually fail.
    bar_h = max(48, round(0.08 * h))
    bar_h += bar_h & 1
    bar = np.full((bar_h, w, 3), 32, np.uint8)
    size = round(bar_h * 0.55)
    if event is None:
        txt, color = ZH_RECOGNIZING, (170, 170, 170)
    else:
        # Events carry the English display name (it is the identity used by clip paths and
        # downstream parsers); only what is painted becomes Chinese.
        name = DISPLAY_TO_ZH.get(event.label, event.label)
        txt = f"{name}  {event.confidence * 100:.0f}%"
        color = (0, 200, 0) if event.confidence >= 0.5 else (200, 200, 200)
    y = (bar_h + size) // 2
    draw_text(bar, txt, (max(8, int(0.02 * w)), y), size, color)
    right = (event.model if event else "") + ((" | " + extra) if extra else "")
    if right:
        small = round(size * 0.6)
        font = _cjk_font(small)
        if font is not None:
            from PIL import ImageDraw, Image
            rw = int(ImageDraw.Draw(Image.new("RGB", (1, 1))).textlength(right, font=font))
        else:
            rw = cv2.getTextSize(right, FONT, small / 30.0, 1)[0][0]
        draw_text(bar, right, (max(8, w - rw - 8), y), small, (150, 150, 150))
    return np.vstack([frame, bar])
