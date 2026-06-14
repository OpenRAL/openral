"""WorldModel Protocol — the seam every WAM adapter implements.

CLAUDE.md §6.3 calls out three integration patterns for WAMs:

1. **Mental simulation (gating)** — sample N short rollouts before
   committing an action chunk.
2. **Failure anticipation** — continuous predicted-vs-observed
   discrepancy detector.
3. **Replanning loop** — propose alternative subgoals as visual
   prompts.

All three consume the same surface — a function from
``(WorldState, ActionChunk, horizon) -> Rollout`` — so the Protocol
below is sufficient for v0.2's "scaffold-only" deliverable. Concrete
adapters (Cosmos Predict, UnifoLM-WMA-0, IRASim) layer on top in v0.3+
and add their own configuration without breaking this seam.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from openral_core import Action, WorldState

from openral_wam.rollout import Rollout

__all__ = ["WorldModel"]


@runtime_checkable
class WorldModel(Protocol):
    """Structural protocol for a World Action Model.

    Implementations predict a future trajectory given a starting
    :class:`~openral_core.WorldState` and a candidate
    :class:`~openral_core.Action` chunk. The planning layer feeds the
    returned :class:`Rollout` into one of the three integration patterns
    documented in CLAUDE.md §6.3.

    Attributes:
        max_horizon: Maximum horizon the model can predict in one call.
            Reflected on the trace span for budget enforcement; the
            planning layer never asks for more than this.
    """

    max_horizon: int

    def rollout(
        self,
        world_state: WorldState,
        action_chunk: Action,
        horizon: int,
    ) -> Rollout:
        """Predict ``horizon`` steps of future state given ``action_chunk``.

        Args:
            world_state: The starting :class:`~openral_core.WorldState`
                snapshot — typically the same one fed to the S1 skill.
            action_chunk: The :class:`~openral_core.Action` chunk the
                planning layer wants to gate or anticipate.
            horizon: Number of steps to predict. Must satisfy
                ``0 < horizon <= max_horizon``.

        Returns:
            A :class:`Rollout` of length ``horizon`` with predicted
            states and (optionally) rewards.

        Raises:
            ROSConfigError: When ``horizon`` exceeds :attr:`max_horizon`.
            ROSInferenceTimeout: When the model exceeds its declared
                latency budget. Per CLAUDE.md §10 every WAM error is
                typed.
        """
        ...
