"""ADR-0018 F10 — prompt_router_node and adapter registry.

The router is a single lifecycle node that fans in operator prompts
from any external source (CLI, dashboard WebSocket, voice, Slack)
into a normalised :class:`openral_msgs/PromptStamped` stream on
``/openral/prompt`` (consumed by the F4 reasoner).

v1 adapters (per ADR-0018 §3 / capability review §3.F10):

- CLI: ``openral prompt "do X"`` publishes a one-shot PromptStamped and
  exits. Lives in :mod:`openral_cli.prompt` — does not load through
  this package's adapter registry because it is a separate process.
- WebSocket / voice / Slack: out-of-scope for v1; the registry shape
  here reserves the entry-point slot.
"""

from __future__ import annotations

from openral_prompt_router.prompt_router_node import PromptRouterNode

__all__ = ["PromptRouterNode"]
