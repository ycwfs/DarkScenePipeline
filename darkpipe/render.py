"""Label-bar rendering: the recognition result is displayed BELOW the video frame
(spec: '... including the text of the action recognition results displayed below the
video'), never overlaid on the image content."""
import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX


def append_label_bar(frame, event, extra: str = ""):
    """Returns frame with a text bar stacked underneath. event may be None (warmup)."""
    h, w = frame.shape[:2]
    bar_h = max(48, round(0.08 * h))
    bar = np.full((bar_h, w, 3), 32, np.uint8)
    scale = bar_h / 48 * 0.9
    if event is None:
        txt, color = "recognizing...", (170, 170, 170)
    else:
        txt = f"{event.label}  {event.confidence * 100:.0f}%"
        color = (0, 200, 0) if event.confidence >= 0.5 else (200, 200, 200)
    (tw, th), base = cv2.getTextSize(txt, FONT, scale, 2)
    y = (bar_h + th) // 2
    cv2.putText(bar, txt, (max(8, int(0.02 * w)), y), FONT, scale, color, 2, cv2.LINE_AA)
    right = (event.model if event else "") + ((" | " + extra) if extra else "")
    if right:
        (rw, _), _ = cv2.getTextSize(right, FONT, scale * 0.6, 1)
        cv2.putText(bar, right, (w - rw - 8, y), FONT, scale * 0.6, (150, 150, 150), 1,
                    cv2.LINE_AA)
    return np.vstack([frame, bar])
