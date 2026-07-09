# openral-reasoner

OpenRAL S2 reasoner — the event-driven slow planning loop (CLAUDE.md §3
Layer 4). `ReasonerCore` consumes a `WorldState` snapshot plus rolling
event buffers (failures, perception events, operator prompts) and emits
**exactly one** typed `ReasonerToolCall` per tick via the LLM's
structured tool-use API — no free-form JSON. The ROS-side
`reasoner_node` (in [`packages/openral_reasoner_ros/`](../../packages/openral_reasoner_ros/))
wraps this core with rclpy subscriptions and dispatch plumbing.

## Layer

CLAUDE.md §3 Layer 4 — S2 slow reasoning: event-driven with a ~0.2 Hz
heartbeat, sitting between the `WorldStateAggregator` and the S1 skill
executor. Dispatch is direct typed tool calls.

## Design

- ROS 2 reasoner supervisor: direct typed tool-call dispatch.
- Symbolic S2 reasoner: authored playbooks, self-maintained memory,
  success-gated task queue.
- VLM-adjudicated completion, grounding-before-decompose, detection
  identity.

## Public surface

```python
from openral_reasoner import (
    ReasonerCore, ReasonerTickResult,          # the S2 tick loop
    ToolPalette, build_tool_palette,           # registry -> LLM tool palette
    AnthropicToolUseClient,                    # provider clients
    OpenAICompatibleToolUseClient,
    build_tool_use_client_from_env,            # OPENRAL_REASONER_LLM_* selection
    ContextRenderer,                           # WorldState/event -> prompt context
    MemoryStore, MissionState,                 # self-maintained memory + mission ladder
    CriticWatchdog, SpatialMemoryQuerier,      # critic gating + spatial recall
    plan_active_search,                        # active-search frontier planning
)
```

- `ReasonerCore` — the tick loop: render context → call the LLM with the
  tool palette → validate into a `ReasonerToolCall` → bounded replanning
  ladder on failure (retry → param-tweak → substitute-skill → goal-replan
  → human-handoff).
- `AnthropicToolUseClient` / `OpenAICompatibleToolUseClient` — concrete
  `ToolUseClient` implementations; selected at activate-time via
  `OPENRAL_REASONER_LLM_*` env vars (`PROVIDER` ∈ {`anthropic`,
  `openai-compatible`, `openrouter`}). No hidden default.
- `ToolPalette` / `build_tool_palette` — generated from the local skill
  registry, rebuilt on `/openral/skill_registry_changed`.
- `MemoryStore`, `MissionState`, `evaluate_task_verdict` — self-maintained
  memory and success-gated task queue.

See [`docs/methods/06-reasoning-wam-safety-observability.md`](../../docs/methods/06-reasoning-wam-safety-observability.md)
for the full symbol inventory and
[`docs/reference/reasoner-design.md`](../../docs/reference/reasoner-design.md)
for the design walkthrough.
