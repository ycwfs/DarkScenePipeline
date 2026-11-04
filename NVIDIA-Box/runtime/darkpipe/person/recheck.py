"""Causal recovery for action windows just below the publication threshold.

The normal recognizer publishes a named action only when its fused probability reaches
``action_threshold``.  This module is deliberately a small, model independent state
machine: a low confidence window can be recovered only after it has passed the existing
object/wave/running priors and one of the two frozen temporal rules is satisfied.

Only the switch and lower confidence bound are configurable.  The temporal constants are
part of the validated release protocol and are kept here so platform and Box cannot drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


ANCHOR_MAX_GAP_S = 1.2
ADJACENT_MAX_GAP_S = 0.65
DUAL_WINDOW_MIN_SPAN_S = 0.5
RUN_MIN_SPEED = 0.35
LOWER_FLOOR = 0.50


@dataclass(frozen=True)
class RecheckDecision:
    action: str
    confidence: float
    mechanism: str  # continuation | running_dual


class CandidateRecheck:
    """Per action-worker causal state.

    A task key contains the session, track id and history generation.  Keeping state in
    the action worker is safe because the scheduler assigns one key to one worker for its
    whole generation; it also avoids a lock on the real-time output thread.
    """

    def __init__(self, enabled: bool = True, lower: float = 0.85,
                 upper: float = 0.925):
        self.enabled = bool(enabled)
        self.lower = float(lower)
        self.upper = float(upper)
        self._anchors: Dict[Tuple, Tuple[str, float]] = {}
        self._chains: Dict[Tuple, dict] = {}
        self._running: Dict[Tuple, dict] = {}

    def in_band(self, confidence: float) -> bool:
        return (self.enabled and self.lower <= float(confidence) < self.upper)

    def reset(self, key: tuple) -> None:
        self._chains.pop(key, None)
        self._running.pop(key, None)

    def observe_high(self, key: tuple, action: str, t_end: float) -> None:
        """Record a normally confirmed action as an anchor.

        Low confidence recoveries never call this method, so they cannot extend the
        high-confidence anchor indefinitely.
        """
        self._anchors[key] = (str(action), float(t_end))
        self.reset(key)

    def update(self, *, key: tuple, action: str, confidence: float,
               t_start: float, t_end: float,
               run_speed: Optional[float] = None) -> Optional[RecheckDecision]:
        """Consume a prior-approved candidate and return a recovery decision if any."""
        if not self.in_band(confidence):
            self.reset(key)
            self._anchors.pop(key, None)
            return None
        action = str(action)
        confidence = float(confidence)
        t_start, t_end = float(t_start), float(t_end)
        decision = self._continuation(key, action, confidence, t_start, t_end)

        if action == "chase" and run_speed is not None and run_speed >= RUN_MIN_SPEED:
            running = self._running.get(key)
            if running is None or t_start - running["last_start"] > ADJACENT_MAX_GAP_S:
                self._running[key] = dict(first_start=t_start, last_start=t_start,
                                          last_end=t_end, confirmed=False)
            else:
                # A new candidate is part of the same causal chain.  Keep the first
                # window so the required tail-to-tail span cannot be faked by jitter.
                if not running["confirmed"]:
                    if (t_end - running["first_start"] >= DUAL_WINDOW_MIN_SPAN_S
                            and t_start - running["last_start"] <= ADJACENT_MAX_GAP_S):
                        running["confirmed"] = True
                        if decision is None:
                            decision = RecheckDecision(action, confidence, "running_dual")
                elif decision is None:
                    decision = RecheckDecision(action, confidence, "running_dual")
                running["last_start"], running["last_end"] = t_start, t_end
        else:
            self._running.pop(key, None)
        return decision

    def _continuation(self, key: tuple, action: str, confidence: float,
                      t_start: float, t_end: float) -> Optional[RecheckDecision]:
        chain = self._chains.get(key)
        if chain is not None:
            if (chain["action"] == action
                    and t_start - chain["last_end"] <= ADJACENT_MAX_GAP_S
                    and t_end - chain["anchor_end"] <= ANCHOR_MAX_GAP_S):
                chain["last_end"] = t_end
                return RecheckDecision(action, confidence, "continuation")
            self._chains.pop(key, None)

        anchor = self._anchors.get(key)
        if anchor is None:
            return None
        if anchor[0] != action:
            # A different named candidate between the anchor and this window breaks
            # "same action, adjacent window" continuity.
            self._anchors.pop(key, None)
            return None
        anchor_end = anchor[1]
        if (t_start - anchor_end <= ADJACENT_MAX_GAP_S
                and t_end - anchor_end <= ANCHOR_MAX_GAP_S):
            self._chains[key] = dict(action=action, anchor_end=anchor_end, last_end=t_end)
            return RecheckDecision(action, confidence, "continuation")
        return None
