# `openral_reasoner_ros`

ROS 2 lifecycle wrapper for the OpenRAL S2 reasoner (ADR-0018 F4).

## What it does

A thin rclpy lifecycle node around
[`openral_reasoner.ReasonerCore`](../../python/reasoner/src/openral_reasoner/core.py).
Subscribes to:

- `/openral/world_state_slow` (`openral_msgs/WorldStateStamped`, 5 Hz)
- `/openral/failure/{hal,sensor,rskill,safety,wam,critic}` (`openral_msgs/FailureTrigger`)
- `/openral/perception/{motion,objects,ocr,scene_change}` (`openral_msgs/PromptStamped`)
- `/openral/prompt` (`openral_msgs/PromptStamped`)

Per the ADR-0018 amendment of 2026-05-25 the reasoner is
**event-driven** with a slow heartbeat. The periodic timer ticks at
`tick_hz` (default 0.2 Hz = one every 5 s; was 5 Hz pre-amendment).
Event preemption is the primary trigger:

- `/openral/failure/safety` (Tier A) preempts on `severity ≥ SEVERITY_WARN`.
- `/openral/failure/{hal,sensor,rskill,wam,critic}` (Tier B/C) preempts
  on `severity ≥ SEVERITY_FAIL`.
- `/openral/prompt` (Tier D) always preempts.

All preemptions are subject to the 100 ms min-interval per ADR-0018 §4.
Heartbeat ticks that see no new event since the last successful tick
are short-circuited inside `ReasonerCore` with
`suppressed_reason="heartbeat_idle"`.

Each tick the LLM picks one of four typed tool calls
([`openral_core.ReasonerToolCall`](../../python/core/src/openral_core/schemas.py)):

| Tool | Dispatch target | What's wired today |
|---|---|---|
| `ExecuteRskillTool` | action goal on `/openral/execute_rskill` (F1) | ✅ `rclpy_action.ActionClient` — sends a goal with `deadline_s`, streams feedback to the warning log, emits a `FailureTrigger` on `/openral/failure/rskill` with `KIND_CONTROLLER` (rejection / abort / server-unavailable) or `KIND_TIMEOUT` (deadline_s expired) |
| `LifecycleTransitionTool` | service call on `<node>/change_state` | ✅ generic `lifecycle_msgs/srv/ChangeState` client — `configure` / `activate` / `deactivate` / `cleanup` only (`shutdown` reserved for the safety supervisor, CLAUDE.md §6 Layer 6) |
| `EmitPromptTool` | publish on the target `PromptStamped` topic | ✅ one-line publish; stamps the active OTel `traceparent` into `metadata_json` |
| `ReloadGstPipelineTool` | service call on `/openral/sensors/<id>/reload_pipeline` | ⚠️ log-and-acknowledge stub — F6 sensor-package service IDL is not yet on disk (tracked in [GH-126](https://github.com/OpenRAL/openral/issues/126)) |

The reasoner **never** publishes `openral_msgs/ActionChunk` — actuation
authority lives behind the F1 action server + the F5 safety boundary
(ADR-0018 §4 "Holds no authority over actuation").

## LLM provider

The reasoner is wire-protocol agnostic — every provider satisfies the
`ToolUseClient` Protocol. Two concrete clients ship in
[`openral_reasoner.tool_use`](../../python/reasoner/src/openral_reasoner/tool_use.py):

- `AnthropicToolUseClient` — Anthropic SDK; `OPENRAL_REASONER_LLM_PROVIDER=anthropic`.
- `OpenAICompatibleToolUseClient` — OpenAI SDK pointed at any
  OpenAI-protocol endpoint (cloud OpenAI, local vLLM, Ollama-OpenAI);
  `OPENRAL_REASONER_LLM_PROVIDER=openai-compatible`.
- `OPENRAL_REASONER_LLM_PROVIDER=openrouter` — convenience preset on
  top of `OpenAICompatibleToolUseClient` that pre-fills the OpenRouter
  base URL (`https://openrouter.ai/api/v1`) so users don't have to
  memorise it. Auth is always required.

### Named provider presets

`OPENRAL_REASONER_LLM_PROVIDER` accepts these named values. Each is just
a `ToolUseClient` selection plus, for the cloud presets, a pre-filled
`OPENRAL_REASONER_LLM_BASE_URL` so you don't hand-configure it. An
explicit `OPENRAL_REASONER_LLM_BASE_URL` always overrides the preset
(proxy / staging gateway). `gemini` / `xai` / `deepseek` are thin
shortcuts over the same `OpenAICompatibleToolUseClient` as `openrouter`,
pointed at each vendor's own OpenAI-compatible endpoint.

| `PROVIDER` | Client | Default base URL | API key |
|---|---|---|---|
| `anthropic` | `AnthropicToolUseClient` | `https://api.anthropic.com` | required |
| `openai-compatible` | `OpenAICompatibleToolUseClient` | `https://api.openai.com/v1` (set yours) | optional¹ |
| `ollama` | `OpenAICompatibleToolUseClient` | `http://localhost:11434/v1` | none |
| `openrouter` | `OpenAICompatibleToolUseClient` | `https://openrouter.ai/api/v1` | required |
| `gemini` | `OpenAICompatibleToolUseClient` | `https://generativelanguage.googleapis.com/v1beta/openai/` | required |
| `xai` | `OpenAICompatibleToolUseClient` | `https://api.x.ai/v1` | required |
| `deepseek` | `OpenAICompatibleToolUseClient` | `https://api.deepseek.com` | required |

¹ `openai-compatible` ignores the key for local endpoints (vLLM /
llama-server) that don't enforce auth; set it when targeting cloud OpenAI.

```bash
# Gemini (Google AI Studio key)
export OPENRAL_REASONER_LLM_PROVIDER=gemini
export OPENRAL_REASONER_LLM_MODEL=gemini-2.5-flash
export OPENRAL_REASONER_LLM_API_KEY=...

# xAI (Grok)
export OPENRAL_REASONER_LLM_PROVIDER=xai
export OPENRAL_REASONER_LLM_MODEL=grok-4
export OPENRAL_REASONER_LLM_API_KEY=xai-...

# DeepSeek (direct)
export OPENRAL_REASONER_LLM_PROVIDER=deepseek
export OPENRAL_REASONER_LLM_MODEL=deepseek-chat
export OPENRAL_REASONER_LLM_API_KEY=sk-...

uv add openai --package openral-reasoner      # one-time, all three
```

No cloud lock-in: the open-core path requires the deployment to pick
the endpoint explicitly via env (`OPENRAL_REASONER_LLM_PROVIDER`,
`OPENRAL_REASONER_LLM_MODEL`, `OPENRAL_REASONER_LLM_API_KEY`,
`OPENRAL_REASONER_LLM_BASE_URL`). Tests use a deterministic
`FakeToolUseClient` under
[`tests/integration/fakes/`](../../tests/integration/fakes/) — the only
test double permitted at this process boundary per CLAUDE.md §1.11.

## System prompt

The base system prompt (`openral_reasoner.DEFAULT_SYSTEM_PROMPT`) is a
robot-agnostic operating brief: one-tool-per-tick semantics, faithful
adherence to the operator goal, robot/scene-matched skill selection,
locate-before-manipulate (`recall_object`), navigate-to-approach
(`resolve_place` / Nav2 navigation skills), per-tick progress
evaluation, and observe-but-never-bypass safety/e-stop handling
("Python proposes, C++ disposes").

At `on_configure` the node calls
[`resolve_reasoner_system_prompt`](../../python/reasoner/src/openral_reasoner/tool_use.py),
which composes the prompt in two parts:

1. **Base brief** — `DEFAULT_SYSTEM_PROMPT`, unless the deployment sets
   `OPENRAL_REASONER_SYSTEM_PROMPT` to a non-empty value, which replaces
   it. (A whitespace-only value is treated as unset.)
2. **`## THIS ROBOT` block** — appended by `render_robot_context_prompt`
   from the active robot's `RobotCapabilities` (loaded from the
   `robot_yaml` ROS parameter, or supplied via the `robot_capabilities`
   constructor arg). It lists the robot's embodiment tags, whether it
   can locomote (which gates the navigate-to-approach rule — a
   fixed-base arm is told it cannot drive to a target and should hand
   off instead), its manipulation / sensing hardware, payload, and
   control modes.

The robot block is appended to whichever base is in effect, so a custom
brief still carries the factual body description it cannot hardcode.
With no robot wired the prompt stays at the (possibly overridden) base
brief alone.

## Baseline LLM (recommended configurations)

The reasoner is event-driven with a 0.2 Hz heartbeat (one tick every
5 s) per the ADR-0018 amendment of 2026-05-25; it sees no pixels and
picks exactly one of four typed tool calls per tick from a small
palette. This is a constrained tool-use task — a small instruction-
tuned model with reliable function-calling is plenty. Three baselines:

### Paid baseline — Anthropic Haiku 4.5

Cheap (~$1/$5 per Mtok), ~0.74 s TTFT, native tool use, matches Sonnet-4
on agentic benchmarks. Recommended default when an API key is acceptable.

```bash
export OPENRAL_REASONER_LLM_PROVIDER=anthropic
export OPENRAL_REASONER_LLM_MODEL=claude-haiku-4-5
export OPENRAL_REASONER_LLM_API_KEY=sk-ant-...
uv add anthropic --package openral-reasoner   # one-time
```

### Free baseline — OpenRouter

OpenRouter exposes `:free` variants of DeepSeek v3, Llama-4-Maverick,
and Qwen3-235B that all pass tool-calling tests as of early 2026. Pick
one directly, or use the `openrouter/free` auto-router (non-deterministic
but always cheapest).

```bash
export OPENRAL_REASONER_LLM_PROVIDER=openrouter
export OPENRAL_REASONER_LLM_MODEL=deepseek/deepseek-chat-v3:free
export OPENRAL_REASONER_LLM_API_KEY=sk-or-...
uv add openai --package openral-reasoner      # one-time
```

### Local baseline — Ollama + Qwen3 8B

Single-binary install, strong tool-calling, runs on a laptop GPU or CPU.
One command sets it up; `openral doctor` then surfaces a green "Reasoner LLM"
+ "Ollama" row:

```bash
just bootstrap-ollama          # installs ollama, starts the daemon, pulls qwen3:8b

export OPENRAL_REASONER_LLM_PROVIDER=openai-compatible
export OPENRAL_REASONER_LLM_MODEL=qwen3:8b
export OPENRAL_REASONER_LLM_BASE_URL=http://localhost:11434/v1
uv add openai --package openral-reasoner      # one-time
```

Run `openral doctor` after exporting the envs — the reasoner row reports
exactly which variable is missing if any, and TCP-probes the Ollama
endpoint when `BASE_URL` is loopback.

## Synopsis

```bash
just ros2-build      # builds openral_msgs + openral_reasoner_ros
source install/setup.bash

# One of the three baseline configs above, e.g.:
export OPENRAL_REASONER_LLM_PROVIDER=anthropic
export OPENRAL_REASONER_LLM_MODEL=claude-haiku-4-5
export OPENRAL_REASONER_LLM_API_KEY=sk-ant-...

ros2 run openral_reasoner_ros reasoner_node
ros2 lifecycle set /openral_reasoner configure
ros2 lifecycle set /openral_reasoner activate
```

## Observability — what to expect on the dashboard

Each `ReasonerCore.tick` opens an OTel span named `reasoner.tick`
(see `openral_observability.reasoner_span`) with these attributes:

| Attribute | When set | Meaning |
|---|---|---|
| `reasoner.tick.idx` | Always | Monotonic per-`ReasonerCore` tick counter. |
| `reasoner.model` | When the client has a `model_id` | LLM model identifier (e.g. `claude-opus-4-7`). |
| `reasoner.force` | Always | `True` when the tick was preempted by `FailureTrigger.severity ≥ FAIL` or a new operator prompt. |
| `reasoner.tool` | Successful + retry-cap suppressed ticks | Which of the four `ReasonerToolCall` variants the LLM picked. |
| `reasoner.rskill_id` | When tool=`execute_skill` | Skill id the LLM chose. |
| `reasoner.suppressed_reason` | Suppressed ticks | One of `palette_empty` / `retry_cap` / `heartbeat_idle`. The `min_interval` and `heartbeat_idle` short-circuits fire BEFORE the span opens (so dashboards don't show noise). |
| `reasoner.tier` | Always | Trigger tier that drove this call: `A` (safety), `B` (replan: hal/sensor/rskill/wam), `C` (critic), `D` (operator/perception), or `heartbeat`. |
| `reasoner.error_kind` | Provider failure | `ROSPlanningError` subclass name; an `exception` event is added to the span. |

The active W3C `traceparent` captured inside this span is threaded
through onto the outbound `EmitPromptTool` `PromptStamped.metadata_json`
(per ADR-0018 §6) so the F7 bag↔OTel correlator can join the
published prompt back to the producing tick.

Spans are emitted via `opentelemetry-sdk` — no provider installed
when `configure_observability` was not called, which makes the helper
a no-op (cost <1 µs). The
`/just docker-smoke-x86-reasoner` smoke explicitly installs a real
provider so the round-trip can be observed end-to-end inside the
deploy image.

## CLAUDE.md amendment (ADR-0018 §9)

The §6.2 dual-system pattern and §7.6 working-with-the-planner wording
were amended in the same PR that introduced this package. Before F4,
the Reasoner was specified to emit BehaviorTree.CPP v4 XML; F4 pivots
to direct typed tool-call dispatch. A future `bt_executor_node`
consuming `BehaviorTreeXml` plans alongside direct tool calls is left
as an explicit follow-up — F4's contract does not preclude it.

## See also

- [ADR-0018](../../docs/adr/0018-ros2-reasoner-supervisor.md) — graph contract (incl. the F4 dispatch decisions).
- [`openral_reasoner.core`](../../python/reasoner/src/openral_reasoner/core.py) — transport-agnostic orchestrator.
- [`packages/openral_prompt_router`](../openral_prompt_router/) — F10 prompt fan-in.
