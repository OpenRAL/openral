# Reasoner (S2) Reference

The **reasoner** is OpenRAL's slow, deliberative control layer — the **S2** half
of the dual-system architecture. Where an `rSkill` (S1) is a fast visuomotor
policy running at 30–200 Hz, the reasoner is an **event-driven LLM supervisor**
that closes the loop `context → LLM → one typed tool call` at a slow cadence.
It decides *what to do next*; it never drives motors itself.

- **Core (transport-agnostic):** [`openral_reasoner.ReasonerCore`](https://github.com/OpenRAL/openral/blob/master/python/reasoner/src/openral_reasoner/core.py)
- **ROS 2 lifecycle node:** [`openral_reasoner_ros.reasoner_node`](https://github.com/OpenRAL/openral/blob/master/packages/openral_reasoner_ros/) — full contract in its [README](https://github.com/OpenRAL/openral/blob/master/packages/openral_reasoner_ros/README.md)
- **Design:** covers the supervisor graph + tool-dispatch, active search, and the read-only query tools
- **Design narrative (how it thinks):** [Reasoner Design & Decisions](reasoner-design.md) — the connective story across the decisions below, organized by logic problem (tick loop → grounding → decomposition → completion verdict → reward pairing → replanning → memory → LLM choice).

> **Authority boundary.** The reasoner **never** publishes `openral_msgs/ActionChunk`.
> Actuation lives behind the S1 skill runner (`/openral/execute_rskill` action
> server) and the F5 safety boundary. The reasoner *proposes*; the C++ safety
> kernel *disposes* (see the safety hazard log — private OpenRAL/management repo).

---

## Cadence & event model

As of the 2026-05-25 amendment,
the reasoner is **event-driven with a slow heartbeat**:

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

### Effect tools

| Tool (`tool=`) | Dispatch | Notes |
|---|---|---|
| `ExecuteRskillTool` (`execute_rskill`) | action goal on `/openral/execute_rskill` | `rskill_id`, `prompt`, `goal_params_json`, `deadline_s`. Emits a `FailureTrigger` on rejection/abort/timeout. |
| `LifecycleTransitionTool` (`lifecycle_transition`) | `<node>/change_state` service | `configure` / `activate` / `deactivate` / `cleanup` only — `shutdown` is reserved for the safety supervisor (CLAUDE.md §6). |
| `EmitPromptTool` (`emit_prompt`) | publish on a `PromptStamped` topic | Stamps the active OTel `traceparent` into `metadata_json`. Used to stage multi-step plans / cascade prompts. |
| `ReloadGstPipelineTool` (`reload_gst_pipeline`) | `/openral/sensors/<id>/reload_pipeline` service | ⚠️ **log-and-acknowledge stub** today — the F6 sensor-service IDL is not yet on disk ([GH-126](https://github.com/OpenRAL/openral/issues/126)). |

### Read-only query tools

These hold **no actuation authority** — they read state and feed the result back
to the LLM as a re-prompt. Each is gated by a `ToolPalette` flag and only offered
when the corresponding service is present.

| Tool (`tool=`) | Reads | Gate |
|---|---|---|
| `RecallObjectTool` (`recall_object`) | spatial-memory scene graph — *"where did I last see X?"* | `spatial_memory_available` |
| `ResolvePlaceTool` (`resolve_place`) | spatial memory → navigation goal pose + path | `spatial_memory_available` |
| `LocateInViewTool` (`locate_in_view`) | on-demand open-vocab detector — *"where is X right now?"* | `detector_available` |
| `QuerySceneTool` (`query_scene`) | scene VLM (Qwen3.5-4B) — free-text *"did the grasp succeed?"* | `scene_query_available` |
| `QueryTaskProgressTool` (`query_task_progress`) | reward monitor (Robometer-4B) — windowed `progress_now` / `success_now` / trends / `stalled` | `task_progress_available` |

`locate_in_view` carries an optional `detector` selector — `omdet-turbo-locator`
(fast, in-process) for simple "find X", `locateanything-3b` for complex referring
expressions. `recall_object` *remembers*; `locate_in_view` *looks now*.

### Memory & mission tools

These edit the reasoner's own state — its `MEMORY.md` file and its task ledger —
never the robot. They are advisory and hold no actuation authority.

| Tool (`tool=`) | Effect |
|---|---|
| `MemoryWriteTool` (`memory_write`) | the reasoner's first **write-capable** variant — `add` / `update` / `supersede` / `delete` an entry in the self-maintained `MEMORY.md` |
| `MemorySearchTool` (`memory_search`) | read-only query over the archival memory log |
| `DecomposeMissionTool` (`decompose_mission`) | write the deterministic `MissionState` task queue — populate/replace it, or flat-splice a blocked task into finer subtasks (`subdivide_active`) |

---

## Playbooks, memory & missions

Three S2 capabilities layer on top of the tool surface:

- **Playbooks (`kind: playbook`).** At palette-seed time the reasoner
  gathers installed, capability-matched playbook rSkills, reads their
  `PLAYBOOK.md` bodies, and appends a `## PLAYBOOKS` section to the system prompt
  — so the LLM follows the relevant authored decision procedure when its trigger
  matches the goal. Playbooks are `role: s2` content, never in the ExecuteRskill
  palette; every motion still crosses `execute_rskill` + the C++ safety kernel.
  Six ship in-tree: `decompose-mission`, `verify-outcome`, `clarify-ambiguity`,
  `preflight-reach`, `stage-for-manipulation`, `find-object`.
- **Self-maintained `MEMORY.md`.** A persistent semantic memory
  (`MemoryStore` / `MemoryEntry`) the reasoner reads each tick and edits through
  `memory_write`, with `consolidate()` (drop duplicates) and a
  `to_context_block(cap=N)` render that bounds the always-on `## MEMORY` block on
  a long-running robot. Loaded at deploy time via `openral deploy sim/run
  --memory-dir` (alongside `scene_graph.json` and the 2D nav map).
- **Sequential missions.** The operator goal seeds a single-task
  `MissionState`; the LLM decomposes it into the ordered queue via
  `decompose_mission` with at most one `active` (or `verifying`) `TaskState`.
  The queue advances only when the active task passes the reward/critic gate,
  rendered as a `## MISSION` ledger each tick.
  `DecomposeMissionTool` + `MissionState.subdivide_active` flat-splice a blocked
  task into finer subtasks on replan, bounded by `DEFAULT_MAX_SUBDIVIDE_DEPTH`
  before human-handoff.

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

The LLM sees **one tool per skill** (`execute_rskill__<slug>`) with
a real description + action/object/scene discriminators, not one opaque tool with
an enum. Continuous detectors are surfaced as `continuous_detectors` so the LLM is
told *what is already tracked for free* and only reaches for `locate_in_view` on
something outside that coverage.

**The palette is closed:** the LLM cannot dispatch a skill that isn't installed,
capability-matched, and licensed.

---

## Reasoner model selection

Selection is model-first (ADR-0088). `OPENRAL_REASONER_MODEL` names a curated
`openral_core.REASONER_MODELS` entry; registry membership means the model has
passed OpenRAL's robotics tool-calling contract. The entry resolves the client
dialect, served model id, endpoint, auth, hosting mode, and local-compute floor.
The library has no default; `openral deploy sim` defaults to `gpt-5.5`.

| Registry key | Served model | Hosting |
|---|---|---|
| `claude-opus-4-8` | `claude-opus-4-8` | Anthropic cloud |
| `gpt-5.5` | `openai/gpt-5.5` | OpenRouter cloud |
| `gpt-5.6` | `openai/gpt-5.6` | OpenRouter cloud |
| `cosmos3-edge` | `nvidia/Cosmos3-Edge` | managed local vLLM |

Other env: `OPENRAL_REASONER_ENDPOINT` (optional URL override),
`OPENRAL_REASONER_API_KEY` (conditional), and
`OPENRAL_REASONER_{MAX_TOKENS,TIMEOUT_S}`. A raw uncurated model id requires
both `OPENRAL_REASONER_ENDPOINT` and
`OPENRAL_REASONER_DIALECT=anthropic|openai`; doctor reports it as unverified.

```bash
# Curated cloud
export OPENRAL_REASONER_MODEL=gpt-5.5
export OPENRAL_REASONER_API_KEY=sk-or-...

# Curated managed local
export OPENRAL_REASONER_MODEL=cosmos3-edge

# Explicit uncurated local endpoint
export OPENRAL_REASONER_MODEL=qwen3:8b
export OPENRAL_REASONER_ENDPOINT=http://localhost:11434/v1
export OPENRAL_REASONER_DIALECT=openai
```

The old `OPENRAL_REASONER_LLM_*` provider-first contract remains a deprecated
one-release shim. `openral doctor` resolves the model registry directly,
checks auth, and probes loopback endpoints; a down managed-local endpoint is
informational only while autostart is enabled. Tests use the deterministic
`FakeToolUseClient` process-boundary double (CLAUDE.md §1.11).

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
`locate_in_view`) and the GStreamer perception bus.

---

## Bounded replanning

`ReasonerCore` enforces a **per-kind retry cap** (`retry_cap_per_kind`, default 3):
consecutive selections of the same tool kind beyond the cap are suppressed for one
tick with `suppressed_reason="retry_cap"`. The streak resets when context shifts
materially (new operator prompt, palette refresh). This is the concrete gate that
ships; the broader ladder (retry → param-tweak → substitute-skill → goal-replan →
human-handoff, CLAUDE.md §3) is partially realized — the substitute/replan rungs
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

export OPENRAL_REASONER_MODEL=claude-opus-4-8
export OPENRAL_REASONER_API_KEY=sk-ant-...

ros2 run openral_reasoner_ros reasoner_node
ros2 lifecycle set /openral_reasoner configure
ros2 lifecycle set /openral_reasoner activate
```

In practice the reasoner comes up as part of the deploy graph
(`openral deploy sim` / `openral deploy run`), which wires it alongside the HAL,
safety, world-state, and perception nodes.

---

## In development

The reasoner core, playbooks, the self-maintained `MEMORY.md`, and the sequential
mission task-queue have all landed on this integration branch.
Still in flight:

- **Dashboard mission card** — surfacing the `MissionState` ledger + the reward
  gate and attempts/cap ladder on the live `openral dashboard` (PR #122).

---

## See also

- [Reasoner Design & Decisions](reasoner-design.md) — the **why** behind every mechanism on this page, organized by logic problem.
- [`openral_reasoner_ros` README](https://github.com/OpenRAL/openral/blob/master/packages/openral_reasoner_ros/README.md) — full ROS wrapper contract, provider presets, baseline LLM configs.
- The reasoner/supervisor graph + tool-dispatch design.
- Reasoner-managed SLAM/Nav2 background services.
- LLM task planning & active search.
- Playbooks + self-maintained MEMORY.md.
- Success-gating + sequential mission task queue.
- [rSkills reference](rskills.md) — the `kind: detector` / `vlm` / `reward` / `ros_action` / `playbook` skills the reasoner reads and dispatches.
