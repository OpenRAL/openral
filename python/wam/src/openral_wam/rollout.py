"""Rollout — the typed output of a :class:`~openral_wam.WorldModel`.

A rollout is a *predicted* trajectory: given a current
:class:`~openral_core.WorldState` and a candidate action chunk, the
WAM emits the predicted states (and optionally predicted frames /
rewards) for ``horizon`` steps. The planning layer uses the rollout to
gate the action chunk (CLAUDE.md §6.3 pattern 1), anticipate failures
(pattern 2), or propose alternative subgoals (pattern 3).

The schema deliberately holds **predictions only** — never executed
state, never sensor frames captured from the real world. Real frames
live in the trace; predicted frames live here for failure
anticipation's discrepancy detector.
"""

from __future__ import annotations

from openral_core import WorldState
from pydantic import BaseModel, ConfigDict, Field

__all__ = ["Rollout"]


class Rollout(BaseModel):
    """Predicted trajectory from one :class:`~openral_wam.WorldModel` call.

    Attributes:
        predicted_states: Predicted :class:`WorldState` for each of the
            ``horizon`` steps. Length must equal ``horizon``.
        predicted_rewards: Optional predicted reward per step. Length
            must equal ``horizon`` when populated; ``None`` when the
            WAM has no reward head.
        horizon: Number of steps predicted. Mirrors the action chunk's
            horizon for gating-pattern uses (CLAUDE.md §6.3 pattern 1).
        latency_ms: Wall-clock latency of the WAM call, recorded for
            the deadline policy and failure-anticipation budgets.
        confidence: Aggregate confidence over the rollout in
            ``[0.0, 1.0]``. Used by the gating pattern's commit
            threshold.

    Example:
        >>> from openral_core import JointState, WorldState
        >>> js = JointState(name=["j1"], position=[0.0], stamp_ns=0)
        >>> ws = WorldState(stamp_ns=0, joint_state=js)
        >>> r = Rollout(
        ...     predicted_states=[ws],
        ...     horizon=1,
        ...     latency_ms=12.0,
        ...     confidence=0.8,
        ... )
        >>> r.horizon
        1
    """

    model_config = ConfigDict(extra="forbid")

    predicted_states: list[WorldState] = Field(
        ...,
        description="Predicted WorldState per step; len(predicted_states) == horizon.",
        min_length=1,
    )
    predicted_rewards: list[float] | None = Field(
        default=None,
        description="Optional predicted reward per step; None when the WAM has no reward head.",
    )
    horizon: int = Field(
        ...,
        description="Number of predicted steps; matches the gated action chunk's horizon.",
        gt=0,
    )
    latency_ms: float = Field(
        ...,
        description="Wall-clock latency of the WAM call in milliseconds.",
        ge=0.0,
    )
    confidence: float = Field(
        ...,
        description="Aggregate WAM confidence in [0.0, 1.0].",
        ge=0.0,
        le=1.0,
    )
