"""OpenRAL World Action Model (WAM) layer.

WAMs are generative simulators used for **mental simulation** (gating
action chunks before commit), **failure anticipation** (continuous
predicted-vs-observed discrepancy detection), and **replanning loops**
(proposing alternative subgoals as visual prompts). Per CLAUDE.md §6.3
they are an *optional* planning-layer component — v0.2 ships a
``NullWorldModel`` default and the Protocol; concrete adapters
(Cosmos Predict, UnifoLM-WMA-0, IRASim) land in v0.3+.

This module ships the **Protocol surface only**:

- :class:`WorldModel`: structural Protocol every WAM adapter satisfies
  (``rollout(world_state, action_chunk, horizon) -> Rollout``).
- :class:`Rollout`: Pydantic model holding predicted frames, states,
  rewards, and the latency budget.
- :class:`NullWorldModel`: identity stub returning the input state
  unchanged for ``horizon`` steps; for plumbing tests.

See CLAUDE.md §6.3 for the three integration patterns (gating /
anticipation / replanning) and ``docs/architecture/repo-state-map.html``
Layer 5 for what's planned next.
"""

from __future__ import annotations

from openral_wam.null_wam import NullWorldModel
from openral_wam.protocol import WorldModel
from openral_wam.rollout import Rollout

__all__ = [
    "NullWorldModel",
    "Rollout",
    "WorldModel",
]
__version__ = "0.1.0"
