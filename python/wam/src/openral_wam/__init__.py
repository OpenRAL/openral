"""OpenRAL World Action Model (WAM) layer.

WAMs are generative simulators for **mental simulation** (gating action
chunks before commit), **failure anticipation** (predicted-vs-observed
discrepancy detection), and **replanning loops** (proposing subgoals as
visual prompts) — CLAUDE.md §6.3. They are an *optional* planning-layer
component: v0.2 ships a ``NullWorldModel`` default and the Protocol;
concrete adapters (Cosmos Predict, UnifoLM-WMA-0, IRASim) land in v0.3+.

Ships the **Protocol surface only**:

- :class:`WorldModel`: structural Protocol every WAM adapter satisfies
  (``rollout(world_state, action_chunk, horizon) -> Rollout``).
- :class:`Rollout`: Pydantic model holding predicted states, rewards,
  and the latency budget.
- :class:`NullWorldModel`: identity stub returning the input state
  unchanged for ``horizon`` steps; for plumbing tests.
"""

from __future__ import annotations

from importlib.metadata import version as _pkg_version

from openral_wam.null_wam import NullWorldModel
from openral_wam.protocol import WorldModel
from openral_wam.rollout import Rollout

__all__ = [
    "NullWorldModel",
    "Rollout",
    "WorldModel",
]
__version__ = _pkg_version("openral-wam")
