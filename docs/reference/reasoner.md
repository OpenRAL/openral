# Reasoner (S2) Reference

The **reasoner** is OpenRAL's slow, deliberative control layer — the **S2** half
of the dual-system architecture. Where an `rSkill` (S1) is a fast visuomotor
policy running at 30–200 Hz, the reasoner is an **event-driven LLM supervisor**
that closes the loop `context → LLM → one typed tool call` at a slow cadence.
It decides *what to do next*; it never drives motors itself.

- **Core (transport-agnostic):** [`openral_reasoner.ReasonerCore`](https://github.com/OpenRAL/openral/blob/master/python/reasoner/src/openral_reasoner/core.py)
- **ROS 2 lifecycle node:** [`openral_reasoner_ros.reasoner_node`](https://github.com/OpenRAL/openral/blob/master/packages/openral_reasoner_ros/) — full contract in its [README](https://github.com/OpenRAL/openral/blob/master/packages/openral_reasoner_ros/README.md)
- **Design:** [ADR-0018](../adr/0018-ros2-reasoner-supervisor.md) (graph + F4 tool-dispatch), [ADR-0039](../adr/0039-llm-task-planning-active-search.md) (active search), ADR-0043/0047/0056/0057 (query tools)

> **Authority boundary.** The reasoner **never** publishes `openral_msgs/ActionChunk`.
> Actuation lives behind the S1 skill runner (`/openral/execute_rskill` action
> server) and the F5 safety boundary. The reasoner *proposes*; the C++ safety
> kernel *disposes* (see [ADR-0020](../adr/0020-cpp-safety-kernel.md) + the [hazard log](hazard-log.md)).

---

## Cadence & event model

Per the [ADR-0018](../adr/0018-ros2-reasoner-supervisor.md) amendment of
2026-05-25, the reasoner is **event-driven with a slow heartbeat**:

- **Heartbeat** — a periodic timer ticks at `tick_hz` (default **0.2 Hz**, one
  tick every 5 s). A heartbeat tick that sees no new event since the last
  successful tick is short-circuited inside `ReasonerCore` with
  `suppressed_reason="heartbeat_idle"` (no LLM call, no span).
- **Event preemption** is the primary trigger, subject to a hard **100 ms
  min-interval** between ticks:

| Tier | Source | Preempts on |
|---|---|---|
| **A — safety** | `/openral/failure/safety` | `severity ≥ SEVERITY_WARN` |
| **B — execution** | `/openral/failure/{hal,sensor,rskill,wam}` | `severity ≥ SEVERITY_FAIL` |
| **C — critic** | `/openral/failure/critic` | `severity ≥ SEVERITY_FAIL` |
| **D — operator** | `/openral/prompt` | always |

`/openral/perception/{motion,objects,ocr,scene_change}` events are informational
context (not preemptive on their own).

---

## The tool-call contract

Each tick the LLM emits **exactly one** variant of the
[`ReasonerToolCall`](https://github.com/OpenRAL/openral/blob/master/python/core/src/openral_core/schemas.py)
discriminated union (discriminator field: `tool`). Output is **structured** —
the provider's tool-use API returns a Pydantic-validated object, never free-form
JSON. Extending the palette requires a new variant in `openral_core` **and** the
matching dispatch in `reasoner_node` (CLAUDE.md §3).

### Effect tools (ADR-0018 §4)

| Tool (`tool=`) | Dispatch | Notes |
|---|---|---|
| `ExecuteRskillTool` (`execute_rskill`) | action goal on `/openral/execute_rskill` | `rskill_id`, `prompt`, `goal_params_json` (ADR-0026), `deadline_s`. Emits a `FailureTrigger` on rejection/abort/timeout. |
| `LifecycleTransitionTool` (`lifecycle_transition`) | `<node>/change_state` service | `configure` / `activate` / `deactivate` / `cleanup` only — `shutdown` is reserved for the safety supervisor (CLAUDE.md §6). |
| `EmitPromptTool` (`emit_prompt`) | publish on a `PromptStamped` topic | Stamps the active OTel `traceparent` into `metadata_json` (ADR-0018 §6). Used to stage multi-step plans / cascade prompts. |
| `ReloadGstPipelineTool` (`reload_gst_pipeline`) | `/openral/sensors/<id>/reload_pipeline` service | ⚠️ **log-and-acknowledge stub** today — the F6 sensor-service IDL is not yet on disk ([GH-126](https://github.com/OpenRAL/openral/issues/126)). |

### Read-only query tools

These hold **no actuation authority** — they read state and feed the result back
to the LLM as a re-prompt. Each is gated by a `ToolPalette` flag and only offered
when the corresponding service is present.

| Tool (`tool=`) | Reads | Gate | ADR |
|---|---|---|---|
| `RecallObjectTool` (`recall_object`) | spatial-memory scene graph — *"where did I last see X?"* | `spatial_memory_available` | 0038/0039 |
| `ResolvePlaceTool` (`resolve_place`) | spatial memory → navigation goal pose + path | `spatial_memory_available` | 0039 |
| `LocateInViewTool` (`locate_in_view`) | on-demand open-vocab detector — *"where is X right now?"* | `detector_available` | 0043/0056 |
| `QuerySceneTool` (`query_scene`) | scene VLM (Qwen3.5-4B) — free-text *"did the grasp succeed?"* | `scene_query_available` | 0047 |
| `QueryTaskProgressTool` (`query_task_progress`) | reward monitor (Robometer-4B) — windowed `progress_now` / `success_now` / trends / `stalled` | `task_progress_available` | 0057 |

`locate_in_view` carries an optional `detector` selector — `omdet-turbo-locator`
(fast, in-process) for simple "find X", `locateanything-3b` for complex referring
expressions. `recall_object` *remembers*; `locate_in_view` *looks now*.

---

## Tool palette & gating

The palette ([`openral_reasoner.palette.ToolPalette`](https://github.com/OpenRAL/openral/blob/master/python/reasoner/src/openral_reasoner/palette.py))
is built at `on_configure` and **rebuilt on `/openral/skill_registry_changed`**
(fired by `openral rskill install|remove`). `build_tool_palette()` filters
installed rSkills by:

- `RobotCapabilities` flags (`capabilities_required` ⊆ robot capabilities),
- embodiment-tag intersection (`embodiment_tags`),
- `role == "s1"` (S0/S2 excluded from the actuation palette),
- license posture (commercial-deployment gate).

Per ADR-0022, the LLM sees **one tool per skill** (`execute_rskill__<slug>`) with
a real description + action/object/scene discriminators, not one opaque tool with
an enum. Continuous detectors are surfaced as `continuous_detectors` so the LLM is
told *what is already tracked for free* and only reaches for `locate_in_view` on
something outside that coverage.

**The palette is closed:** the LLM cannot dispatch a skill that isn't installed,
capability-matched, and licensed.

---

## LLM provider selection

The reasoner is wire-protocol agnostic — every provider satisfies the
`ToolUseClient` Protocol, selected at `on_configure` from environment variables.
There is **no default** (no cloud lock-in); the deployment picks the endpoint.

| `OPENRAL_REASONER_LLM_PROVIDER` | Client | Default base URL | API key |
|---|---|---|---|
| `anthropic` | `AnthropicToolUseClient` | `https://api.anthropic.com` | required |
| `openai-compatible` | `OpenAICompatibleToolUseClient` | `https://api.openai.com/v1` (set yours) | optional |
| `ollama` | OpenAI-compatible preset | `http://localhost:11434/v1` | none |
| `vllm` | OpenAI-compatible preset | `http://localhost:8000/v1` | none |
| `openrouter` | OpenAI-compatible preset | `https://openrouter.ai/api/v1` | required |
| `gemini` | OpenAI-compatible preset | `https://generativelanguage.googleapis.com/v1beta/openai/` | required |
| `xai` | OpenAI-compatible preset | `https://api.x.ai/v1` | required |
| `deepseek` | OpenAI-compatible preset | `https://api.deepseek.com` | required |

Other env: `OPENRAL_REASONER_LLM_MODEL` (required), `OPENRAL_REASONER_LLM_API_KEY`
(conditional), `OPENRAL_REASONER_LLM_BASE_URL` (overrides any preset).

```bash
# Paid baseline — cheap, fast, native tool use
export OPENRAL_REASONER_LLM_PROVIDER=anthropic
export OPENRAL_REASONER_LLM_MODEL=claude-haiku-4-5
export OPENRAL_REASONER_LLM_API_KEY=sk-ant-...

# Local baseline — runs on a laptop GPU/CPU
just bootstrap-ollama                            # installs ollama, pulls qwen3:8b
export OPENRAL_REASONER_LLM_PROVIDER=ollama
export OPENRAL_REASONER_LLM_MODEL=qwen3:8b
```

`openral doctor` reports a green "Reasoner LLM" row once the envs are set (and
TCP-probes a loopback Ollama/vLLM endpoint). Tests use a deterministic
`FakeToolUseClient` — the only test double permitted at this boundary (CLAUDE.md §1.11).

---

## System prompt & context

At `on_configure`, `resolve_reasoner_system_prompt(...)` composes the prompt in
two parts:

1. **Base brief** — `DEFAULT_SYSTEM_PROMPT` (robot-agnostic operating brief:
   one-tool-per-tick, faithful goal adherence, locate-before-manipulate,
   navigate-to-approach, observe-but-never-bypass safety), overridable via
   `OPENRAL_REASONER_SYSTEM_PROMPT`.
2. **`## THIS ROBOT` block** — rendered from the active `RobotCapabilities`
   (embodiment tags, whether it can locomote — which gates the navigate-to-approach
   rule — manipulation/sensing hardware, payload, control modes).

Each tick the `ContextRenderer` assembles the per-tick situation report from the
subscribed topics: **world state** (`/openral/world_state_slow`, 5 Hz),
**failures** (the `/openral/failure/*` bus), **perception** events, and pending
**operator prompts**. The reasoner does **not** read pixels directly — vision
reaches it through the perception tools (`query_scene`, `query_task_progress`,
`locate_in_view`) and the GStreamer perception bus (ADR-0037).

---

## Bounded replanning

`ReasonerCore` enforces a **per-kind retry cap** (`retry_cap_per_kind`, default 3):
consecutive selections of the same tool kind beyond the cap are suppressed for one
tick with `suppressed_reason="retry_cap"`. The streak resets when context shifts
materially (new operator prompt, palette refresh). This is the concrete gate that
ships; the broader ladder (retry → param-tweak → substitute-skill → goal-replan →
human-handoff, CLAUDE.md §7.6) is partially realized — the substitute/replan rungs
are still being built out.

---

## Observability

Every `ReasonerCore.tick` opens an OTel span `reasoner.tick` (via
`openral_observability.reasoner_span`) with attributes including `reasoner.tick.idx`,
`reasoner.model`, `reasoner.force`, `reasoner.tool`, `reasoner.rskill_id`,
`reasoner.tier` (`A`/`B`/`C`/`D`/`heartbeat`), and `reasoner.suppressed_reason`
(`palette_empty` / `retry_cap` / `heartbeat_idle`). The captured `traceparent` is
threaded onto outbound `emit_prompt` payloads so the bag↔OTel correlator can join a
published prompt back to the producing tick. Watch it live on `openral dashboard`.

---

## Running it

```bash
just ros2-build
source install/setup.bash

export OPENRAL_REASONER_LLM_PROVIDER=anthropic
export OPENRAL_REASONER_LLM_MODEL=claude-haiku-4-5
export OPENRAL_REASONER_LLM_API_KEY=sk-ant-...

ros2 run openral_reasoner_ros reasoner_node
ros2 lifecycle set /openral_reasoner configure
ros2 lifecycle set /openral_reasoner activate
```

In practice the reasoner comes up as part of the deploy graph
(`openral deploy sim` / `openral deploy run`), which wires it alongside the HAL,
safety, world-state, and perception nodes.

---

## In development

The following reasoner work is on feature branches and **not yet merged to
`master`** — it will land incrementally:

- **Context grounding & reflection** — a richer per-tick situation report with a
  closed execution-feedback loop (ADR-0071 Phase 2).
- **`MEMORY.md` semantic memory** — a self-maintained narrative memory the reasoner
  reads each tick and edits through typed write tools, with consolidation under a
  cap (ADR-0071 Phases 4a–4c, 5).
- **`kind: playbook` rSkills** — human-authored Markdown decision procedures (SOPs)
  the reasoner reads as *content* (never the actuation palette): `decompose-mission`,
  `verify-outcome`, `clarify-ambiguity`, `preflight-reach`, `stage-for-manipulation`
  (ADR-0072).
- **Mission task-queue** — a typed `MissionState` queue
  (`pending → active → verifying → done|abandoned`) with reward-gated advance
  (ADR-0073).

These are **not** described as shipped until they reach `master` (CLAUDE.md §1.2).

---

## See also

- [`openral_reasoner_ros` README](https://github.com/OpenRAL/openral/blob/master/packages/openral_reasoner_ros/README.md) — full ROS wrapper contract, provider presets, baseline LLM configs.
- [ADR-0018](../adr/0018-ros2-reasoner-supervisor.md) — reasoner/supervisor graph + F4 dispatch.
- [ADR-0025](../adr/0025-reasoner-managed-background-services.md) — reasoner-managed SLAM/Nav2 background services.
- [ADR-0039](../adr/0039-llm-task-planning-active-search.md) — LLM task planning & active search.
- [rSkills reference](rskills.md) — the `kind: detector` / `vlm` / `reward` / `ros_action` skills the reasoner queries and dispatches.
</content>
</invoke>
