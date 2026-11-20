# Reasoner (S2) Reference

The **reasoner** is OpenRAL's slow, deliberative control layer — the **S2** half
of the dual-system architecture. Where an `rSkill` (S1) is a fast visuomotor
policy running at 30–200 Hz, the reasoner is an **event-driven LLM supervisor**
that closes the loop `context → LLM → one typed tool call` at a slow cadence.
It decides *what to do next*; it never drives motors itself.

- **Core (transport-agnostic):** [`openral_reasoner.ReasonerCore`](https://github.com/OpenRAL/openral/blob/master/python/reasoner/src/openral_reasoner/core.py)
- **ROS 2 lifecycle node:** [`openral_reasoner_ros.reasoner_node`](https://github.com/OpenRAL/openral/blob/master/packages/openral_reasoner_ros/) — full contract in its [README](https://github.com/OpenRAL/openral/blob/master/packages/openral_reasoner_ros/README.md)
- **Design (the why):** [Reasoner Design & Decisions](reasoner-design.md), organized by logic problem (tick loop → grounding → decomposition → completion verdict → reward pairing → replanning → memory → LLM choice).

> **Authority boundary.** The reasoner **never** publishes `openral_msgs/ActionChunk`.
> Actuation lives behind the S1 skill runner (`/openral/execute_rskill` action
> server) and the F5 safety boundary. The reasoner *proposes*; the C++ safety
> kernel *disposes* (see the safety hazard log — private OpenRAL/management repo).

---

## Cadence & event model

The reasoner calls the LLM when something happens, never on a fixed timer
(issue #620):

- **Events** force a tick at once (table below), plus the reasoner's own: an
  `execute_rskill` result (`skill_result`), a tool result republished as a
  cascade prompt (`tool_result:<source>`), a task verdict or mission advance, and
  a new latch on `/openral/safety_status` (`safety_latched`, Tier A; skipped when a
  safety FailureTrigger forced a tick in the last 2 s, the same kernel stop). Requests
  that arrive while a tick is in flight coalesce into one replay; their names
  ride on it as `reasoner.tick.coalesced_triggers`. Forced ticks are not subject
  to the 100 ms `min_interval_s` (it only spaces timer beats).
- **The timer beat** (`tick_hz` ROS parameter, default **0.2 Hz**) only runs
  gates; a beat reaches the LLM only when one of these holds, recorded as its
  trigger:

  | Trigger | When |
  |---|---|
  | `context_changed` | nothing in flight, a mission or prompt is pending, and the context moved |
  | `backstop` | a skill is in flight and `clamp(max_execution_s / 4, 15 s, 30 s)` passed since the last LLM call (perception churn during a skill is bounded by the same period) |
  | `wait_timeout` | the model's `wait(timeout_s=…)` lapsed with a skill in flight or a task active |
  | `llm_error_retry` | the last LLM call failed in transport; retried after 5, 10, 20 s, then the reasoner hands off to the operator |

  Otherwise it is suppressed with no LLM call and no span: `heartbeat_idle`,
  `backstop_not_due`, `wait_hold`, `llm_gave_up`, `no_goal` (no mission, no prompt, nothing in
  flight), `safety_latched`, `mission_finished`, `llm_error_backoff`,
  `retry_cap_hold`, `min_interval`.
- **`wait(until=…, timeout_s=…)`** lets one decision cover a whole wait:
  `skill_result` ignores perception churn until the skill ends, `operator` waits
  for a reply, `next_event` ends on any new context, `timeout` only on the
  timeout. Any forced tick ends a wait early; `timeout_s` is bounded (≤ 300 s).

Every `reasoner.tick` span carries `reasoner.tick.trigger`,
`reasoner.tick.coalesced_triggers`, `reasoner.tick.skipped_since_last` and, for a
`wait`, `reasoner.wait.until` / `reasoner.wait.timeout_s`.

| Tier | Source | Preempts on |
|---|---|---|
| **A — safety** | `/openral/failure/safety` | `severity ≥ SEVERITY_WARN` |
| **B — execution** | `/openral/failure/{hal,sensor,rskill,wam}` | `severity ≥ SEVERITY_FAIL` |
| **B — object query recovery** (issue #418) | `/openral/failure/world_state` (`KIND_OBJECT_QUERY`) | matching failure cancels the active manipulation; reobserve and re-ground within the persisted task dispatch cap, then stay blocked until a new operator goal. No human handoff. |
| **C — critic** | `/openral/failure/critic` | `severity ≥ SEVERITY_FAIL` |
| **D — operator** | `/openral/prompt` | always |

`/openral/perception/{motion,objects,ocr,scene_change}` events are informational
context (not preemptive on their own); a record that repeats the previous one of
its kind is dropped and does not count as new context. The reasoner ignores the
FailureTriggers it publishes itself on `/openral/failure/rskill` (their outcome is
already in `## EXECUTION`).

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
| `LifecycleTransitionTool` (`lifecycle_transition`) | `<node>/change_state` service | `configure` / `activate` / `deactivate` / `cleanup` only — `shutdown` is reserved for the safety supervisor. |
| `EmitPromptTool` (`emit_prompt`) | publish on a `PromptStamped` topic | Stamps the active OTel `traceparent` into `metadata_json`. Used to stage multi-step plans / cascade prompts. |
| `ReloadGstPipelineTool` (`reload_gst_pipeline`) | `/openral/sensors/<id>/reload_pipeline` service | ⚠️ **log-and-acknowledge stub** today — the F6 sensor-service IDL is not yet on disk ([GH-126](https://github.com/OpenRAL/openral/issues/126)). |

### Read-only query tools

No actuation authority: they read state and feed the result back to the LLM as a
re-prompt. Each is gated by a `ToolPalette` flag and offered only when its service
is present.

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

These edit the reasoner's own state (its `MEMORY.md` and task ledger), never the
robot; no actuation authority.

| Tool (`tool=`) | Effect |
|---|---|
| `MemoryWriteTool` (`memory_write`) | the reasoner's first **write-capable** variant — `add` / `update` / `supersede` / `delete` an entry in the self-maintained `MEMORY.md` |
| `MemorySearchTool` (`memory_search`) | read-only query over the archival memory log |
| `DecomposeMissionTool` (`decompose_mission`) | write the deterministic `MissionState` task queue — populate/replace it, or flat-splice a blocked task into finer subtasks (`subdivide_active`) |

---

## Playbooks, memory & missions

- **Playbooks (`kind: playbook`).** At palette-seed time the reasoner gathers
  installed, capability-matched playbook rSkills, reads their `PLAYBOOK.md`
  bodies, and appends a `## PLAYBOOKS` section to the system prompt so the LLM
  follows the matching decision procedure. Playbooks are `role: s2` content,
  never in the ExecuteRskill palette; every motion still crosses
  `execute_rskill` + the C++ safety kernel. Six ship in-tree:
  `decompose-mission`, `verify-outcome`, `clarify-ambiguity`,
  `preflight-reach`, `stage-for-manipulation`, `find-object`.
- **Self-maintained `MEMORY.md`.** A persistent semantic memory
  (`MemoryStore` / `MemoryEntry`) the reasoner reads each tick and edits through
  `memory_write`, with `consolidate()` (drop duplicates) and a
  `to_context_block(cap=N)` render that bounds the always-on `## MEMORY` block on
  a long-running robot. Loaded at deploy time via `openral deploy sim/run
  --memory-dir` (alongside `scene_graph.json`, the 2D nav map and — when the file is
  present — the world-state object memory's `object_memory.json`, whose objects a
  restart restores as uncertain).
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
(`std_msgs/Empty`; the deploy launch fires it once a late action server such as
Nav2's `/navigate_to_pose` appears). `build_tool_palette()` filters
installed rSkills by:

- `RobotCapabilities` flags (`capabilities_required` ⊆ robot capabilities),
- embodiment-tag intersection (`embodiment_tags`),
- `kind in EXECUTE_RSKILL_KINDS` (policies and wrapped ROS servers — the same set the
  runner dispatches; detectors surface as coverage / `locate_in_view` options, and
  segmenters and the S2 kinds `vlm` / `reward` / `playbook` get no actuating tool),
- license posture (commercial-deployment gate).

The LLM sees one tool per skill (`execute_rskill__<slug>`) with a real description
+ action/object/scene discriminators, not one opaque tool with an enum. Continuous
detectors are surfaced as `continuous_detectors` so the LLM knows what is already
tracked and reaches for `locate_in_view` only outside that coverage.

The palette is closed: the LLM cannot dispatch a skill that isn't installed,
capability-matched, and licensed.

---

## Reasoner model selection

Selection is model-first (ADR-0088): `OPENRAL_REASONER_MODEL` names a curated
`openral_core.REASONER_MODELS` entry (registry membership means the model
cleared OpenRAL's robotics tool-calling contract), and the orthogonal
`OPENRAL_REASONER_ENDPOINT` overrides where it runs. The library has no
default.

```bash
export OPENRAL_REASONER_MODEL=gpt-5.5
export OPENRAL_REASONER_API_KEY=sk-or-...
```

Models that refuse forced tool use (the Claude 5 generation 400s on
`tool_choice` `any`) run `tool_choice: auto`: a prose reply is nudged once and
then fed back to the model as an invalid plan, and any other model whose 400
refuses forced tool use is downgraded to `auto` at runtime
(`reasoner.tool_choice_downgraded`).

Full model/endpoint matrix, named endpoints, the dialect rule, the legacy
migration table, and what `openral doctor`'s `Reasoner LLM` row checks:
[`openral_reasoner_ros` README §Reasoner model
registry](https://github.com/OpenRAL/openral/blob/master/packages/openral_reasoner_ros/README.md#reasoner-model-registry).

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

The param-tweak rung gets a crisp input for the most common manipulation failure, a
missed grasp (issue #353): the HAL publishes `KIND_GRASP_OUTCOME` on
`/openral/failure/hal` when a gripper close ends without an ATTACH, and
`ContextRenderer` renders one `[grasp_miss] hand=<jaw link> target=<id>` line per hand
in `## EXECUTION` with a `reflect_on_grasp_miss` hint naming the levers `execute_rskill`
already has (a re-grounded `grasp_target.object_id`, the other hand via
`grasp_target.contact_links`, a different skill). An `attach` for that hand clears it:
its own ATTACH, or another hand attaching the same target. The HAL counts the run per hand
and per target. The core still sequences nothing on it: per-target miss counting in
`ReasonerCore` and the
substitute / goal-replan escalation are deferred until trials show whether the LLM's
own re-dispatch recovers.

---

## Observability

Every `ReasonerCore.tick` opens an OTel span `reasoner.tick` (via
`openral_observability.reasoner_span`) with attributes including `reasoner.tick.idx`,
`reasoner.model`, `reasoner.force`, `reasoner.tool`, `reasoner.rskill_id`,
`reasoner.tier` (`A`/`B`/`C`/`D`/`heartbeat`), `reasoner.suppressed_reason` (on the span
only `palette_empty` and `retry_cap`; `min_interval` / `retry_cap_hold` / `mission_finished` /
`heartbeat_idle` suppress the tick before a span opens and appear only on the returned
`ReasonerTickResult`), `reasoner.error_kind`, `reasoner.llm_s`, `reasoner.inflight_skill` and
`reasoner.inflight_state` (the skill in flight and whether it is `dispatching` or `running`, if
any) and `reasoner.rationale` (the model's stated why, capped at 500 chars, stamped only when
the OTLP endpoint is on the deployer's own network or `OPENRAL_TELEMETRY_TEXT_CONSENT=1`,
CLAUDE.md §3). The dashboard store folds the in-flight skill and state, tier, LLM latency,
suppressed reason, error kind and rationale into `topics.reasoner.detail`, one line such as
`stocking running · heartbeat · 2.3s — The stocking skill is running…`, and appends the same
line to each `reasoner.tick` row of the event log (the `topics.reasoner` slot keeps only the
latest tick). The pinned robot console bundle does not render `detail` yet. The captured `traceparent` is
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

## See also

- [Reasoner Design & Decisions](reasoner-design.md) — the why behind every mechanism on this page.
- [`openral_reasoner_ros` README](https://github.com/OpenRAL/openral/blob/master/packages/openral_reasoner_ros/README.md) — full ROS wrapper contract, provider presets, baseline LLM configs.
- Private decision log (`OpenRAL/management`): ADR-0018 (reasoner + supervisor
  graph), ADR-0025 (reasoner-managed background services), ADR-0039 (LLM task
  planning & active search), ADR-0072 (playbooks + self-maintained memory),
  ADR-0073 (success gating + task queue), ADR-0088 (model registry).
- [rSkills reference](rskills.md) — the `kind: detector` / `vlm` / `reward` / `ros_action` / `playbook` skills the reasoner reads and dispatches.
