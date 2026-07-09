"""NullWorldModel — identity stub satisfying the WorldModel Protocol.

Returns ``horizon`` copies of the input :class:`~openral_core.WorldState`,
zero predicted rewards, latency 0.0 ms, and confidence 1.0. Useful for
runner / planner plumbing tests where the WAM seam must be present but
no real prediction is needed.

This stub is **not** a fallback to be silently used in production —
CLAUDE.md §1.4 ("explicit beats implicit") forbids hidden fallbacks.
The mental-simulation gating pattern (CLAUDE.md §6.3 pattern 1) wired
against a :class:`NullWorldModel` is a no-op gate; replace with a
concrete adapter before relying on the gating signal.
"""

from __future__ import annotations

import structlog
from openral_core import Action, WorldState

from openral_wam.rollout import Rollout

__all__ = ["NullWorldModel"]

log = structlog.get_logger(__name__)


class NullWorldModel:
    """A WAM that predicts ``world_state`` unchanged for ``horizon`` steps.

    Useful for plumbing tests where a real generative WAM would
    require a Thor-class GPU or a cloud dispatch. The emitted
    :class:`~openral_wam.Rollout` carries the input state copied
    ``horizon`` times, no rewards, 0 ms latency, and confidence 1.0.

    Args:
        max_horizon: Largest horizon this stub will accept. Defaults
            to 16, which matches the SmolVLA chunk size.

    Example:
        >>> from openral_core import Action, ControlMode, JointState, WorldState
        >>> wam = NullWorldModel(max_horizon=8)
        >>> js = JointState(name=["j1"], position=[0.0], stamp_ns=0)
        >>> ws = WorldState(stamp_ns=0, joint_state=js)
        >>> chunk = Action(control_mode=ControlMode.JOINT_POSITION, horizon=4)
        >>> r = wam.rollout(ws, chunk, horizon=4)
        >>> r.horizon
        4
        >>> r.confidence
        1.0
    """

    max_horizon: int

    def __init__(self, max_horizon: int = 16) -> None:
        """Initialise with an advertised maximum horizon."""
        if max_horizon <= 0:
            raise ValueError(f"max_horizon must be > 0, got {max_horizon!r}")
        self.max_horizon = max_horizon

    def rollout(
        self,
        world_state: WorldState,
        action_chunk: Action,
        horizon: int,
    ) -> Rollout:
        """Return a :class:`Rollout` with ``world_state`` copied ``horizon`` times."""
        del action_chunk
        if horizon <= 0 or horizon > self.max_horizon:
            raise ValueError(
                f"horizon must satisfy 0 < horizon <= max_horizon={self.max_horizon}, "
                f"got {horizon!r}"
            )
        log.debug("wam.null_rollout", horizon=horizon)
        return Rollout(
            predicted_states=[world_state] * horizon,
            predicted_rewards=None,
            horizon=horizon,
            latency_ms=0.0,
            confidence=1.0,
        )
