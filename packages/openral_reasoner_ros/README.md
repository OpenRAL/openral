# `openral_reasoner_ros`

ROS 2 lifecycle wrapper for the OpenRAL S2 reasoner.

## What it does

A thin rclpy lifecycle node around
[`openral_reasoner.ReasonerCore`](../../python/reasoner/src/openral_reasoner/core.py).
Subscribes to:

- `/openral/world_state_slow` (`openral_msgs/WorldStateStamped`, 5 Hz)
- `/openral/failure/{hal,sensor,rskill,safety,wam,critic}` (`openral_msgs/FailureTrigger`)
- `/openral/perception/{motion,objects,ocr,scene_change}` (`openral_msgs/PromptStamped`)
- `/openral/prompt` (`openral_msgs/PromptStamped`)

Since the 2026-05-25 amendment the reasoner is
**event-driven** with a slow heartbeat. The periodic timer ticks at
`tick_hz` (default 0.2 Hz = one every 5 s; was 5 Hz pre-amendment).
Event preemption is the primary trigger:

- `/openral/failure/safety` (Tier A) preempts on `severity ≥ SEVERITY_WARN`.
- `/openral/failure/{hal,sensor,rskill,wam,critic}` (Tier B/C) preempts
  on `severity ≥ SEVERITY_FAIL`.
- `/openral/prompt` (Tier D) always preempts.

All preemptions are subject to the 100 ms min-interval.
Heartbeat ticks that see no new event since the last successful tick
are short-circuited inside `ReasonerCore` with
`suppressed_reason="heartbeat_idle"` — **except while an
`execute_rskill` goal is in flight**, when the heartbeat stays live so
the reasoner can poll `query_task_progress` mid-execution. The retry
cap is keyed on the **identical tool call** (same tool + same
arguments): a healthy navigate → pick → place sequence never trips it,
a verbatim repeat does, and after a capped streak further
unchanged-context ticks are held *before* the LLM call
(`suppressed_reason="retry_cap_hold"`) until new context arrives.

Each tick the LLM picks one typed tool call
([`openral_core.ReasonerToolCall`](../../python/core/src/openral_core/schemas.py)).
The effect/actuation variants:

| Tool | Dispatch target | What's wired today |
|---|---|---|
| `ExecuteRskillTool` | action goal on `/openral/execute_rskill` (F1) | ✅ `rclpy.action.ActionClient` — sends a goal with `deadline_s`, streams feedback to the warning log, emits a `FailureTrigger` on `/openral/failure/rskill` with `KIND_CONTROLLER` (rejection / abort / server-unavailable) or `KIND_TIMEOUT` (deadline_s expired). One goal at a time: while a goal is in flight a second dispatch is refused with feedback and the context carries an `in_flight:` line |
| `LifecycleTransitionTool` | service call on `<node>/change_state` | ✅ generic `lifecycle_msgs/srv/ChangeState` client — `configure` / `activate` / `deactivate` / `cleanup` only (`shutdown` reserved for the safety supervisor, CLAUDE.md §6 Layer 6) |
| `EmitPromptTool` | publish on `target_topic` | ✅ publishes on the call's `target_topic` (per-topic publisher cache; `/openral/prompt` reuses the cascade publisher); stamps the active OTel `traceparent` into `metadata_json` |
| `WaitTool` | none (deliberate no-op) | ✅ the forced tool choice needs an explicit "observe and wait" option — logged with its rationale, no ROS traffic |
| `ReloadGstPipelineTool` | service call on `/openral/sensors/<id>/reload_pipeline` | ⚠️ log-and-acknowledge stub — F6 sensor-package service IDL is not yet on disk (tracked in [GH-126](https://github.com/OpenRAL/openral/issues/126)) |

**Mission handling on `/openral/prompt`:** a genuine operator prompt
(re)builds the mission queue only when no mission is in progress (none,
finished, or not yet started — a pre-work resend still replaces). While
a mission is mid-flight, a prompt is treated as *guidance* (it reaches
the LLM via the PROMPTS context section) unless its `metadata_json`
carries `{"new_goal": true}` (`openral prompt --new-goal "..."` stamps
it) — so an operator answer to a reasoner question can never silently
discard the task queue. The reasoner's own
cascade re-prompts (`spatial_memory` / `detector` / `scene_vlm` /
`reward_monitor` / `memory` / `mission` frame_ids) never rebuild the
mission and never reset the search budgets or the retry-cap streak.

**Crash-safe ladder resume:** set the `ladder_state_path` ROS parameter
to a writable JSON path and the mission ledger + every replanning-ladder
bound (attempts, subdivision offers, decompose nudges, the per-task
locate budget) is snapshotted after each mutation and restored at
`on_configure` — a restarted reasoner resumes the mission where it
stopped instead of resetting every cap. Empty (default) disables
persistence.

**Dispatch-phase watchdog:** `_rskill_inflight` (the one-goal-at-a-time
busy latch) is bounded by the `dispatch_watchdog_s` ROS parameter
(default 30 s; `<= 0` disables): if neither the VRAM-peer eviction nor
the goal response resolves within the ceiling — a runner or peer that
died *after* the readiness probe; rclpy futures never time out on their
own — the watchdog releases the latch, reactivates the peers, and emits
a `KIND_CONTROLLER` FailureTrigger (`state="dispatch_timeout"`) so the
ladder handles it instead of every future dispatch being refused as
busy forever. The expired dispatch generation invalidates its remaining
callbacks; a late accepted goal is canceled and cannot overwrite a newer
dispatch.

The reasoner **never** publishes `openral_msgs/ActionChunk` — actuation
authority lives behind the F1 action server + the F5 safety boundary
("Holds no authority over actuation").

## Reasoner model registry

Reasoner selection is **model-first** (ADR-0088). The primary knob is
`OPENRAL_REASONER_MODEL`, a key in the curated
[`openral_core.REASONER_MODELS`](../../python/core/src/openral_core/schemas.py)
registry. Registry membership means the model has cleared OpenRAL's robotics
tool-calling contract; there is deliberately no separate compatibility flag.
The registry resolves the wire dialect, served model id, default endpoint,
auth requirement, hosting mode, and local-compute floor.

| `OPENRAL_REASONER_MODEL` | Served model | Client | Hosting / endpoint | Auth |
|---|---|---|---|---|
| `claude-opus-4-8` | `claude-opus-4-8` | `AnthropicToolUseClient` | Anthropic cloud | required |
| `gpt-5.5` | `openai/gpt-5.5` | `OpenAICompatibleToolUseClient` | OpenRouter cloud | required |
| `gpt-5.6` | `openai/gpt-5.6` | `OpenAICompatibleToolUseClient` | OpenRouter cloud | required |
| `cosmos3-edge` | `nvidia/Cosmos3-Edge` | `Cosmos3ToolUseClient` | managed local `http://127.0.0.1:8901/v1` | none |

The env contract is:

- `OPENRAL_REASONER_MODEL` — required registry key, or a raw model id for the
  explicit uncurated escape hatch.
- `OPENRAL_REASONER_ENDPOINT` — optional URL override. Location lives here,
  not in a provider name. For an OpenAI-dialect cloud model, an override means
  the operator owns auth policy for that endpoint; Anthropic-compatible
  endpoints still require a key.
- `OPENRAL_REASONER_API_KEY` — required only when the resolved endpoint needs it.
- `OPENRAL_REASONER_MAX_TOKENS` / `OPENRAL_REASONER_TIMEOUT_S` — optional
  per-call overrides. The `ollama` / `vllm` named endpoints cap a call at
  **60 s**; a large local model on a long-running mission reaches that ceiling
  (`tick error: … Request timed out`), and `OPENRAL_REASONER_TIMEOUT_S` is the
  way past it. Before raising it, read `llm_s` / `prompt_tokens` on the tick
  (below) — a ceiling hit driven by a growing prompt wants context management,
  not a longer timeout.
- `OPENRAL_REASONER_DIALECT=anthropic|openai` — required only for an uncurated
  raw model id.

```bash
# Curated cloud model
export OPENRAL_REASONER_MODEL=gpt-5.5
export OPENRAL_REASONER_API_KEY=sk-or-...

# Curated managed-local model
export OPENRAL_REASONER_MODEL=cosmos3-edge

# Uncurated escape hatch: explicit and warned by the factory + doctor
export OPENRAL_REASONER_MODEL=qwen3:8b
export OPENRAL_REASONER_ENDPOINT=http://localhost:11434/v1
export OPENRAL_REASONER_DIALECT=openai
```

`OPENRAL_REASONER_MAX_TOKENS` defaults to `16384` for the curated GPT-5.x
entries so OpenRouter does not reserve their full output window and reject a
low-balance request with HTTP 402. Anthropic keeps its client default. Managed
Cosmos gets a 120 s first-call timeout for kernel compilation.

For `cosmos3-edge`, a down managed endpoint auto-starts
`tools/cosmos3_reasoner_sidecar.py`; disable with
`OPENRAL_COSMOS3_AUTOSTART=0`. `OPENRAL_COSMOS3_BOOT_TIMEOUT_S` bounds first
boot. An explicit `OPENRAL_REASONER_ENDPOINT` points the same curated model at
a self-managed vLLM/NIM endpoint.

### Legacy migration

The old `OPENRAL_REASONER_LLM_{PROVIDER,MODEL,API_KEY,BASE_URL,...}` contract
was **removed in 0.3.0** (deprecated in 0.2.0 per ADR-0088). It is no longer
read at all: an environment that sets only the old vars now fails with
"`OPENRAL_REASONER_MODEL` is unset" rather than silently selecting a model.

Each var maps across directly:

| Legacy | Model-first |
|---|---|
| `OPENRAL_REASONER_LLM_PROVIDER` + `OPENRAL_REASONER_LLM_MODEL` | `OPENRAL_REASONER_MODEL` (+ `OPENRAL_REASONER_ENDPOINT`) |
| `OPENRAL_REASONER_LLM_BASE_URL` | `OPENRAL_REASONER_ENDPOINT` |
| `OPENRAL_REASONER_LLM_API_KEY` | `OPENRAL_REASONER_API_KEY` |
| `OPENRAL_REASONER_LLM_MAX_TOKENS` | `OPENRAL_REASONER_MAX_TOKENS` |
| `OPENRAL_REASONER_LLM_TIMEOUT_S` | `OPENRAL_REASONER_TIMEOUT_S` |

`PROVIDER` split into two axes — *which model* and *where it runs* — so each
old provider value maps onto a model plus an endpoint:

| Legacy `PROVIDER=` | Model-first replacement |
|---|---|
| `anthropic` | curated `claude-opus-4-8`, or `ENDPOINT=anthropic` + a raw model id |
| `openrouter` / `gemini` / `xai` / `deepseek` / `huggingface` | `ENDPOINT=<name>` + `API_KEY` |
| `ollama` / `vllm` | `ENDPOINT=<name>` (no key) |
| `cosmos` | curated `cosmos3-edge` (managed autostart included) |
| `openai-compatible` | `ENDPOINT=<url>` + `DIALECT=openai` |

A named endpoint carries its own base URL, dialect, auth posture, cold-start
timeout and `tool_choice` quirk, so `DIALECT` is needed only for a bare URL.

Tests use a deterministic `FakeToolUseClient` under
[`tests/integration/fakes/`](../../tests/integration/fakes/) — the only test
double permitted at this process boundary per CLAUDE.md §1.11.

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

## Curated reasoner models

The library factory has no default and refuses to guess. `openral deploy sim`
does default to `OPENRAL_REASONER_MODEL=gpt-5.5`, with
`OPENRAL_REASONER_MAX_TOKENS=16384`; it was the most reliable model in live
collective-goal decomposition tests. The default needs
`OPENRAL_REASONER_API_KEY` and fails loudly without it.

### Cloud — GPT-5.5 / GPT-5.6 via OpenRouter

```bash
export OPENRAL_REASONER_MODEL=gpt-5.5  # or gpt-5.6
export OPENRAL_REASONER_API_KEY=sk-or-...
uv add openai --package openral-reasoner
```

### Cloud — Claude Opus 4.8

```bash
export OPENRAL_REASONER_MODEL=claude-opus-4-8
export OPENRAL_REASONER_API_KEY=sk-ant-...
uv add anthropic --package openral-reasoner
```

### Managed local — NVIDIA Cosmos 3 Edge

[Cosmos 3 Edge](../../docs/reference/cosmos3-edge-reasoner.md) (released
2026-07-20) is the 4B on-device tier of NVIDIA's Cosmos 3 omnimodal
world-model family, built for exactly this job: physical reasoning, task
planning, and embodied decision making on Jetson Thor / RTX-class GPUs. Unlike
the general-purpose LLM baselines above it is a *physical-AI-native* planner —
trained on robotics/AV/warehouse data, with spatial grounding and
physical-plausibility judgment — and its `describe_image` completion gate runs
on the same local model (no separate cloud VLM). Weights are
**OpenMDW-1.1** (commercial use OK). One env var is enough; the managed vLLM
server auto-starts on the first tick (first boot downloads ~8 GB):

```bash
export OPENRAL_REASONER_MODEL=cosmos3-edge
uv add openai --package openral-reasoner      # one-time (client SDK only)
```

Requires an NVIDIA GPU (Ampere+; BF16 is the only officially-tested
precision; **≥12 GB recommended**). An 8 GB 4070 was enough for validation,
but the knobs differ by serving stack: the pinned stable stack needed
`OPENRAL_COSMOS3_GPU_MEM_UTIL=0.95` merely to boot (inference remains blocked);
the working vLLM-main native implementation needed expandable CUDA segments
plus `--kv-cache-dtype fp8`. Pre-warm the pinned sidecar with
`python tools/cosmos3_reasoner_sidecar.py`, or point
`OPENRAL_REASONER_ENDPOINT` at a compatible self-managed server.

> ⚠️ **Live status (2026-07-21): works end-to-end on vLLM `main`; blocked on
> the pinned stable release.** On vLLM nightly (native Edge model from
> [vllm#48291](https://github.com/vllm-project/vllm/pull/48291) + the one-line
> weight-filter from open [vllm#49190](https://github.com/vllm-project/vllm/pull/49190))
> a real reasoner tick returned a **validated typed tool call in ~1.5 s** and
> `describe_image` answered correctly, live on an 8 GB 4070 (`--kv-cache-dtype
> fp8` needed there). The sidecar's hash-locked stable vLLM (0.24.0) predates
> the native model and still crashes on inference via the Transformers-fallback
> `get_rope_index` bug — the lock is bumped the moment a vLLM release contains
> #48291 + #49190. Full findings in the
> [assessment page](../../docs/reference/cosmos3-edge-reasoner.md). **Use a
> curated cloud model above as the working reasoner today.**

### Uncurated local endpoint — explicit escape hatch

Models outside `REASONER_MODELS` are not claimed compatible. They still work
when the model id, endpoint, and dialect are explicit; the factory and doctor
warn that robotics tool-calling reliability is unverified.

```bash
just bootstrap-ollama
export OPENRAL_REASONER_MODEL=qwen3:8b
export OPENRAL_REASONER_ENDPOINT=http://localhost:11434/v1
export OPENRAL_REASONER_DIALECT=openai
```

## Synopsis

```bash
just ros2-build      # builds openral_msgs + openral_reasoner_ros
source install/setup.bash

# One curated model, e.g.:
export OPENRAL_REASONER_MODEL=claude-opus-4-8
export OPENRAL_REASONER_API_KEY=sk-ant-...

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
| `reasoner.mission_json` | When a mission is active | `MissionState.to_summary()` JSON — the ordered task queue (id/text/status/attempts/verdict) the live dashboard renders as the Mission card checklist. Absent on bare-goal deploys. |
| `reasoner.error_kind` | Provider failure | `ROSPlanningError` subclass name; an `exception` event is added to the span. |
| `reasoner.llm_s` | Every tick that reached the LLM | Wall-clock of the provider round-trip **alone**. `ReasonerTickResult.elapsed_s` is end-to-end tick time (context render + call + bookkeeping), so `elapsed_s - llm_s` is the reasoner-side overhead — without the split a tick that grew from 6 s to 99 s cannot be attributed. |
| `reasoner.prompt_tokens` | When the client reports usage | Provider-reported prompt tokens (Anthropic cache reads included). Tick latency that grows in step with this is a context-size problem — window or summarize the history rather than raising `OPENRAL_REASONER_TIMEOUT_S`. |

Both also appear on the `reasoner.tick.selected` structured log, so the split
is readable from the log stream without opening Jaeger.

The active W3C `traceparent` captured inside this span is threaded
through onto the outbound `EmitPromptTool` `PromptStamped.metadata_json`
so the F7 bag↔OTel correlator can join the
published prompt back to the producing tick.

Spans are emitted via `opentelemetry-sdk` — no provider installed
when `configure_observability` was not called, which makes the helper
a no-op (cost <1 µs). The
`/just docker-smoke-x86-reasoner` smoke explicitly installs a real
provider so the round-trip can be observed end-to-end inside the
deploy image.

## CLAUDE.md amendment

The §3 dual-system pattern wording was amended in the same PR that
introduced this package to specify **direct typed `ReasonerToolCall`
dispatch** as the reasoner's output contract — the LLM picks exactly one
typed tool call per tick and the node routes it onto the ROS graph.

## See also

- [`openral_reasoner.core`](../../python/reasoner/src/openral_reasoner/core.py) — transport-agnostic orchestrator.
- [`packages/openral_prompt_router`](../openral_prompt_router/) — F10 prompt fan-in.
