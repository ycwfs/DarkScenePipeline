"""Label-bar geometry: bar is appended BELOW the frame, correct height, text present."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from darkpipe.render import append_label_bar
from darkpipe.stages.base import RecognitionEvent


def test_bar_below_frame():
    frame = np.zeros((480, 640, 3), np.uint8)
    ev = RecognitionEvent(frame_index=0, timestamp=0.0, label="Walk", confidence=0.91,
                          topk=[("Walk", 0.91)], model="videomamba", window=32)
    out = append_label_bar(frame, ev)
    bar_h = max(48, round(0.08 * 480))
    assert out.shape == (480 + bar_h, 640, 3)
    assert (out[:480] == 0).all()          # frame content untouched (no overlay)
    assert out[480:].max() > 0             # bar contains rendered text


def test_bar_warmup():
    frame = np.zeros((240, 320, 3), np.uint8)
    out = append_label_bar(frame, None)
    assert out.shape[0] > 240 and out[240:].max() > 0
