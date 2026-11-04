"""Regression tests for the causal low-confidence candidate recovery rules."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "dark-demo-v2"))

from darkpipe.person.recheck import CandidateRecheck  # noqa: E402


def test_continuation_requires_anchor_and_gap():
    key = ("session", 1, 1)
    state = CandidateRecheck(True, .85, .925)
    assert state.update(key=key, action="wave", confidence=.90,
                       t_start=1.0, t_end=1.5) is None
    state.observe_high(key, "wave", 2.0)
    assert state.update(key=key, action="wave", confidence=.90,
                       t_start=2.2, t_end=2.7).mechanism == "continuation"
    # The anchor window is now too old, so this cannot be recovered.
    assert state.update(key=key, action="wave", confidence=.90,
                       t_start=3.0, t_end=3.5) is None


def test_running_requires_two_windows_and_keeps_chain():
    key = ("session", 2, 1)
    state = CandidateRecheck(True, .85, .925)
    assert state.update(key=key, action="chase", confidence=.90,
                       t_start=1.0, t_end=1.5, run_speed=.40) is None
    assert state.update(key=key, action="chase", confidence=.90,
                       t_start=1.5, t_end=2.0, run_speed=.40).mechanism == "running_dual"
    assert state.update(key=key, action="chase", confidence=.90,
                       t_start=2.0, t_end=2.5, run_speed=.40).mechanism == "running_dual"
    # A slow candidate breaks the running chain.
    assert state.update(key=key, action="chase", confidence=.90,
                       t_start=2.5, t_end=3.0, run_speed=.20) is None


def test_band_and_action_switch_reset_state():
    key = ("session", 3, 1)
    state = CandidateRecheck(True, .85, .925)
    state.observe_high(key, "wave", 1.0)
    assert state.update(key=key, action="drink", confidence=.90,
                       t_start=1.1, t_end=1.6) is None
    assert state.update(key=key, action="wave", confidence=.80,
                       t_start=1.1, t_end=1.6) is None
    assert state.update(key=key, action="wave", confidence=.90,
                       t_start=1.1, t_end=1.6) is None


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("recheck regression tests passed")
