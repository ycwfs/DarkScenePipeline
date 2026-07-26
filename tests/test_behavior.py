"""Behavior label set, per-recognizer vocabularies, and the X-CLIP fast-path decomposition."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CKPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ckpts")
XCLIP = os.path.join(CKPTS, "xclip-base-patch16-zero-shot")


def test_behavior_label_set_covers_the_requirement():
    from darkpipe.constants import BEHAVIOR_DISPLAY, BEHAVIOR_PROMPTS, BEHAVIORS
    required = ["wave", "throw", "chase", "fall", "fight", "talk", "drink", "pick_up",
                "shake_hands"]
    assert BEHAVIORS == required + ["other"]      # 9 behaviors + explicit negative class
    assert set(BEHAVIOR_DISPLAY) == set(BEHAVIORS)
    assert set(BEHAVIOR_PROMPTS) == set(BEHAVIORS)
    assert all(len(v) >= 2 for v in BEHAVIOR_PROMPTS.values())


def test_labels_are_per_recognizer_not_global():
    """ARID recognizers must keep speaking ARID-11 after the behavior classes were added."""
    from darkpipe.constants import BEHAVIORS, CLASSES
    from darkpipe.stages.recognize import (BehaviorRecognizer, R3DRecognizer,
                                           VideoMambaRecognizer)
    from darkpipe.stages.recognize_xclip import XCLIPRecognizer
    assert R3DRecognizer.labels == CLASSES
    assert VideoMambaRecognizer.labels == CLASSES
    assert BehaviorRecognizer.labels == BEHAVIORS
    assert XCLIPRecognizer("x").labels == BEHAVIORS
    assert XCLIPRecognizer("x", labels=["a", "b"]).labels == ["a", "b"]


def test_open_vocabulary_prompts():
    from darkpipe.constants import BEHAVIOR_PROMPTS
    from darkpipe.stages.recognize_xclip import prompts_for
    assert prompts_for("chase") == BEHAVIOR_PROMPTS["chase"]      # curated ensemble
    assert "a video of loitering" in prompts_for("loitering")     # generic fallback


def test_span_window_resamples_to_T():
    """With span set, a buffer shorter or longer than T still yields exactly T frames."""
    import numpy as np

    from darkpipe.stages.recognize import VideoMambaRecognizer
    r = VideoMambaRecognizer.__new__(VideoMambaRecognizer)
    r.window, r.span = 32, 1.0
    for n, fps in ((7, 7.0), (32, 32.0), (60, 60.0)):
        r.buf = [(np.full((1, 1, 1), i, np.float32), i / fps) for i in range(n)]
        f = r._window_frames()
        assert len(f) == 32
        assert f[0][0, 0, 0] == 0 and f[-1][0, 0, 0] == n - 1   # spans the whole buffer


def test_span_fires_on_wall_clock_timestamps():
    """Serve mode's regression: irregular timestamps must still produce events.

    `_trim` bounds the buffer to <= span, so a `covered >= span` readiness test can only pass
    on exact float equality -- which offline's i/fps timestamps hit and serve's wall clock does
    not. That combination left serve mode showing "recognizing..." indefinitely.
    """
    import numpy as np

    from darkpipe.stages.recognize import VideoMambaRecognizer
    r = VideoMambaRecognizer.__new__(VideoMambaRecognizer)
    r.window, r.span, r.stride = 32, 1.0, 16
    r.buf, r._filled_pushes = __import__("collections").deque(), 0

    t, fired = 0.0, 0
    for i in range(200):                       # ~27 fps, jittered: never 32 frames inside 1 s
        t += 1 / 27.0 + (0.002 if i % 3 else -0.001)
        r.buf.append((np.zeros((1, 1, 1), np.float32), t))
        r._trim()
        assert r.buf[-1][1] - r.buf[0][1] <= r.span      # the <= span guarantee still holds
        if r._ready():
            if r._filled_pushes % r.stride == 0:
                fired += 1
            r._filled_pushes += 1
    assert fired >= 9, f"{fired} events in 200 frames at 27 fps -- serve would look frozen"
    assert len(r.buf) < r.window                # and it never reached the frame-count fallback


def test_reset_forgets_the_stream_but_not_the_weights():
    """Rendering several clips through one loaded recognizer (compare/behavior/class_video.py)
    must not let the tail of clip N fill the first window of clip N+1."""
    import numpy as np

    from darkpipe.stages.recognize import BehaviorRecognizer
    r = BehaviorRecognizer.__new__(BehaviorRecognizer)
    r.buf = [(np.zeros((1, 1, 1), np.float32), i / 30.0) for i in range(32)]
    r.net, r._filled_pushes = "loaded-weights", 17
    r.reset()
    assert len(r.buf) == 0 and r._filled_pushes == 0
    assert r.net == "loaded-weights"           # reset() is not close()


def test_open_set_reject_rule():
    """Below tau -> `other` wins and carries 1-max(named); above tau -> distribution intact."""
    import numpy as np

    from darkpipe.constants import BEHAVIORS
    from darkpipe.stages.recognize_xclip import XCLIPRecognizer
    r = XCLIPRecognizer("x", reject_tau=0.4)
    o = BEHAVIORS.index("other")

    p = np.full(10, 0.02)
    p[BEHAVIORS.index("wave")] = 0.35                    # confident-ish, but under tau
    out = r.reject(p)
    assert BEHAVIORS[int(out.argmax())] == "other"
    assert abs(out[o] - 0.65) < 1e-6

    p = np.full(10, 0.02)
    p[BEHAVIORS.index("wave")] = 0.82                    # clears tau -> untouched
    assert np.array_equal(r.reject(p), p)

    r.reject_tau = 0.0                                   # disabled -> never fires
    p = np.full(10, 0.02)
    p[BEHAVIORS.index("wave")] = 0.1
    assert np.array_equal(r.reject(p), p)

    r2 = XCLIPRecognizer("x", labels=["loitering", "running"], reject_tau=0.4)
    p = np.array([0.1, 0.9])                             # no `other` label -> rule is a no-op
    assert np.array_equal(r2.reject(p), p)


def test_cli_accepts_the_new_recognizers():
    from darkpipe.cli import build_parser
    p = build_parser()
    for r in ("off", "r3d", "videomamba", "behavior", "xclip"):
        assert p.parse_args(["--input", "x.mp4", "--recognize", r]).recognize == r


def test_config_label_list():
    from darkpipe.config import PipelineConfig
    assert PipelineConfig().label_list() is None
    assert PipelineConfig(labels="climbing a fence, loitering").label_list() == \
        ["climbing a fence", "loitering"]


@pytest.mark.skipif(not os.path.exists(os.path.join(XCLIP, "pytorch_model.bin")),
                    reason="X-CLIP weights not downloaded")
def test_xclip_fast_path_matches_stock_forward():
    """XCLIPRecognizer._infer caches the text tower and inlines XCLIPModel.forward.

    This asserts the decomposition is exact: same probabilities as calling the stock model
    with input_ids + pixel_values, on the same 32 frames and the same prompt list.
    """
    import numpy as np
    import torch
    from transformers import AutoTokenizer

    from darkpipe.stages.recognize_xclip import XCLIPRecognizer, prompts_for

    labels = ["wave", "fight", "chase"]
    r = XCLIPRecognizer(CKPTS, labels=labels)
    r.load("cuda")

    rng = np.random.default_rng(0)
    frames = rng.normal(0, 1, (32, 224, 224, 3)).astype(np.float32)
    fast = r._infer(frames)

    prompts = [p for lab in labels for p in prompts_for(lab)]
    tok = AutoTokenizer.from_pretrained(XCLIP)
    batch = tok(prompts, padding=True, return_tensors="pt").to("cuda")
    px = torch.from_numpy(frames).permute(0, 3, 1, 2).unsqueeze(0).to("cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        scale = r.net.logit_scale.exp().float()
        sim = r.net(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                    pixel_values=px).logits_per_video[0].float() / scale   # back to cosine
        ref = sim.view(len(labels), -1).mean(1)                            # per-label ensemble
        ref = (ref * scale).softmax(0).cpu().numpy()

    assert np.abs(fast - ref).max() < 2e-3, f"{fast} vs {ref}"
    assert len(r.labels) == 3
    r.close()
