# openral-reasoner

OpenRAL S2 reasoner (CLAUDE.md §3 Layer 4): the event-driven slow planning
loop, with a ~0.2 Hz heartbeat, between the `WorldStateAggregator` and the S1
skill executor. `ReasonerCore` consumes a `WorldState` snapshot plus rolling
event buffers (failures, perception events, operator prompts) and emits
**exactly one** typed `ReasonerToolCall` per tick via the LLM's structured
tool-use API (no free-form JSON); dispatch is direct typed tool calls. The
ROS-side `reasoner_node` (in
[`packages/openral_reasoner_ros/`](https://github.com/OpenRAL/openral/tree/master/packages/openral_reasoner_ros)) wraps
this core with rclpy subscriptions and dispatch plumbing.

## Public surface

```python
from openral_reasoner import (
    ReasonerCore, ReasonerTickResult,          # the S2 tick loop
    ToolPalette, build_tool_palette,           # registry -> LLM tool palette
    AnthropicToolUseClient,                    # provider clients
    OpenAICompatibleToolUseClient,
    build_tool_use_client_from_env,            # model-first env selection
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
- `AnthropicToolUseClient` / `OpenAICompatibleToolUseClient` (plus
  `Cosmos3ToolUseClient` for the local `cosmos3-edge` sidecar) — concrete
  `ToolUseClient` implementations; selected at activate-time from the curated
  `openral_core.REASONER_MODELS` registry via `OPENRAL_REASONER_MODEL`.
  `OPENRAL_REASONER_ENDPOINT` overrides where the model runs. No hidden default.
- `ToolPalette` / `build_tool_palette` — generated from the local skill
  registry, rebuilt on `/openral/skill_registry_changed`.
- `MemoryStore`, `MissionState`, `evaluate_task_verdict` — self-maintained
  memory and success-gated task queue.

See the [symbol inventory](https://github.com/OpenRAL/openral/blob/master/docs/methods/06-reasoning-wam-safety-observability.md)
and the [reasoner design](https://docs.openral.com/reference/reasoner-design/).
