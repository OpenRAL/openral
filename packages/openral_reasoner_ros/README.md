# `openral_reasoner_ros`

ROS 2 lifecycle wrapper for the OpenRAL S2 reasoner.

## What it does

A thin rclpy lifecycle node around
[`openral_reasoner.ReasonerCore`](../../python/reasoner/src/openral_reasoner/core.py).
Subscribes to:

- `/openral/world_state_slow` (`openral_msgs/WorldStateStamped`, 5 Hz)
- `/openral/failure/{hal,sensor,rskill,safety,wam,critic,world_state}` (`openral_msgs/FailureTrigger`)
- `/openral/perception/{motion,objects,ocr,scene_change}` (`openral_msgs/PromptStamped`)
- `/openral/prompt` (`openral_msgs/PromptStamped`)

Since the 2026-05-25 amendment the reasoner is **event-driven** with a slow
heartbeat: the periodic timer ticks at `tick_hz` (default 0.2 Hz = one every
5 s). Event preemption is the primary trigger:

- `/openral/failure/safety` (Tier A) preempts on `severity ≥ SEVERITY_WARN`.
- `/openral/failure/{hal,sensor,rskill,wam,critic}` (Tier B/C) preempts
  on `severity ≥ SEVERITY_FAIL`.
- `/openral/failure/world_state` (issue #418 Addendum G): a matching
  `KIND_OBJECT_QUERY` (`ObjectQueryEvidence`, typed `ROSObjectNotInMemory`) records
  the action goal UUID, refusal reason, optional workspace, and bounded observation history. The
  reasoner cancels the matching manipulation before recovery, re-observes and
  revalidates within the persisted per-task dispatch budget, then leaves the task
  blocked until a new operator goal when the cap is spent. These failures do not
  produce a human handoff or alter e-stop behavior. When camera pose, robot mask or
  usable observation is unavailable, the recovery message says visibility is unknown
  and absence is unverified; a completed empty locate remains `object_not_found`.
- `/openral/prompt` (Tier D) always preempts.

A HAL grasp miss (`KIND_GRASP_OUTCOME`, issue #353) rides this ladder like any other
`hal` event: the bridge publishes it at `SEVERITY_FAIL` when a grasp declaration named
the hand (preempts) and `SEVERITY_WARN` for an undeclared close (buffered). The
`ContextRenderer` turns each miss into a per-hand `[grasp_miss]` line in `## EXECUTION`
with a `reflect_on_grasp_miss` hint (re-ground the target, try the other hand, or a
different skill), dropped by an `attach` for that hand (`SEVERITY_INFO`: its own ATTACH,
or another hand attaching the same target), an e-stop clear or a task change. It is a reflection only — no new rung: the node decides
nothing on it beyond the usual preemption.

All preemptions are subject to the 100 ms min-interval.
Heartbeat ticks that see no new event since the last successful tick
are short-circuited inside `ReasonerCore` with
`suppressed_reason="heartbeat_idle"` — **except while an
`execute_rskill` goal is in flight**, when the heartbeat stays live so
the reasoner can poll `query_task_progress` mid-execution. A *result-only*
skill (a wrapped ROS action with no `result_trajectory_field`, e.g. Nav2
`NavigateToPose`) has no progress to poll, so it does not keep the heartbeat
live unless a reward monitor (`query_task_progress`) is wired; its terminal result, prompts and safety events still wake the next tick.
The retry
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
| `LifecycleTransitionTool` | service call on `<node>/change_state` | ✅ generic `lifecycle_msgs/srv/ChangeState` client — `configure` / `activate` / `deactivate` / `cleanup` only (`shutdown` reserved for the safety supervisor, CLAUDE.md §3); `node` must be one of the palette's `node_ids` (the deployment's lifecycle peers) — the decoder rejects any other as `ROSReasonerInvalidPlan` and the dispatcher re-checks it, so the deadman or kernel can never be targeted |
| `EmitPromptTool` | publish on `target_topic` | ✅ publishes on the call's `target_topic`, which must be one of `EMIT_PROMPT_TOPICS` (`/openral/prompt`, `/openral/prompt_in/auto`) — validated in the schema and re-checked at dispatch, so the LLM can never open a publisher on another topic or speak as a human-priority router source (per-topic publisher cache; `/openral/prompt` reuses the cascade publisher); stamps the active OTel `traceparent` into `metadata_json` |
| `WaitTool` | none (deliberate no-op) | ✅ the forced tool choice needs an explicit "observe and wait" option — logged with its rationale, no ROS traffic |
| `ReloadGstPipelineTool` | service call on `/openral/sensors/<id>/reload_pipeline` | ⚠️ log-and-acknowledge stub — F6 sensor-package service IDL is not yet on disk (tracked in [GH-126](https://github.com/OpenRAL/openral/issues/126)) |

**Inactive reasoner:** prompt and failure subscriptions stay live while the node is
configured but not active; their payloads are kept in the context renderer for the
next activation, but no tick runs and nothing is dispatched until `on_activate`.

**Mission handling on `/openral/prompt`:** an operator prompt (re)builds the
mission queue only when no mission is in progress. Mid-flight, a prompt is
*guidance* (reaches the LLM via PROMPTS context) unless `metadata_json`
carries `{"new_goal": true}` (`openral prompt --new-goal "..."` stamps it).
The reasoner's own cascade re-prompts (`spatial_memory` / `detector` /
`scene_vlm` / `reward_monitor` / `memory` / `mission` frame_ids) never rebuild
the mission and never reset the search budgets or retry-cap streak.
One more case rebuilds mid-flight: the active task's last dispatch *succeeded*
and no reward monitor can verify it (`mission: task … succeeded with no
verifier` in the log) and no task is queued after it — the operator's next
prompt is then a new goal, not an aside after which the finished task resumes.
Queued tasks are never dropped this way, and once the reasoner prompts the
operator (`emit_prompt`, e.g. a clarification question) the next operator prompt
is a reply again.

**Task completion without a verifier.** A task normally completes only on a
reward / VLM verdict. A wrapped ROS action rSkill whose manifest sets
`ros_integration.completes_task_on_success: true` (the in-tree Nav2
`navigate_to_pose` skill; the setting is read when the goal is dispatched)
completes the active task when its goal succeeds —
Nav2 SUCCEEDED means "within the goal tolerance", so the reasoner must not
re-drive a 0.2 m residual. A goal the LLM sends with `final_step: false`
(the drive is one step of "drive to the kitchen and pick up the cup")
completes only its step: the task stays active for the rest. With no reward
monitor, the per-task attempt cap (`DEFAULT_MAX_ATTEMPTS`, 3) counts *failed*
dispatches (aborted, or cancelled on patience): a success nobody can judge is
not a failure, so a multi-skill task whose steps all work is not abandoned
after three of them. A task whose results no reward gate judges (no reward
monitor, or `final_step: false` steps) is also bounded by
`DEFAULT_MAX_TASK_DISPATCHES` (10) dispatches of any outcome, so one that
keeps succeeding without ever completing still ends. Without a reward monitor
or completion check a VLA run ends on its budget, which the runner reports as
a failure (`FAILURE_DEADLINE_MISSED`), so VLA runs still count toward the
failure cap; the successes the change stops counting are wrapped actions
(Nav2, MoveIt goals, look-at). Either bound abandons
through the usual abandon path (one subdivision offer, then hand-off). The
decisions log `mission: <complete|step|abandon> task <id> after '<rskill_id>' …`.

**What a success says.** The `## EXECUTION` line for a successful goal carries
the runner's `ExecuteRskill.Result.outcome_summary` before the trace id. For a
result-only wrapped action that is the terminal status, the result's scalar
fields and the goal that was sent, bounded to 400 characters:

```
[ok] skill=OpenRAL/rskill-nav2-mobile_base-navigate_to_pose: wrapped result-only action completed: status=STATUS_SUCCEEDED result{error_code=0} goal{pose.header.frame_id=map pose.pose.position.x=4 pose.pose.position.y=0 pose.pose.position.z=0 pose.pose.orientation.x=0 pose.pose.orientation.y=0 pose.pose.orientation.z=0 pose.pose.orientation.w=1} trace=1a2b3c4d
```

A failure carries the same `status=… result{…} goal{…}` line in its
`failure_reason`; the result's `error*` fields come first and its `header.*`
fields are dropped, so a MoveIt abort's error code survives the bound.

For direct `locate_in_view` tool calls, a missing service, failed request, or nonempty
`LocateInView.failure_reason` produces a bounded detector failure prompt that says
visibility is unknown and preserves the cause. Only a completed response with
`found=false` and an empty reason is rendered as “not visible.” These failures consume
the existing bounded locate-search budget; see the [`LocateInView` wire contract](../msgs/README.md#services--srv).

**Grasp / place targets — the reasoner names, perception grounds, the producer
measures** (real pick-and-place design §2.2-2.3). `ExecuteRskillTool` carries two
optional fields:

| Field | Type | Grounded at dispatch into |
|---|---|---|
| `grasp_target` | `GraspTargetRef{label, object_id?, contact_links=[]}` | a `GraspDeclaration` on the goal: `target_id="obj:<label or node_id>"`, `contact_links` (all `role: gripper` `child_link`s of ONE hand — a hand is the gripper joints hanging off one arm, so R1 Pro's two finger links are one hand; empty = the hand of a single-hand robot; with several hands an empty list is refused and the LLM must name the hand — defaulting would exempt both — and links of two hands, a non-gripper link, or only part of a hand's links — that would exempt one finger and the kernel would stop the grasp — are refused), `timeout_s` = patience + 10 s (capped), `search_box` = the 3D box of the recalled memory node (`object_id`) or of the ONE live lifted detection on `/openral/world_state_slow` carrying the label, padded by `grasp_target_voxel_m` (the deploy's octree cell, passed by `deploy_e2e`; no default — unset on a manifest with a gripper fails `on_configure` with `ROSConfigError`, never a per-call refusal the LLM would replan on) + `grasp_target_extrinsic_error_m` on every side (downward too, so the support layer the producer measures is inside it — a search hint with no support semantics) and gravity-aligned in the base frame. **Never a `region`.** |
| `place_target` | `PlaceTargetRef{label, object_id? \| place_node_id?}` — **optional hint, normally unset** | where to set the payload down is the policy's job, and the real place producer measures the surface directly under the carried payload without being told. When set, a region-less `PlaceDeclaration(target_id="surface:<label or node_id>")` whose `search_box` is the padded, gravity-aligned box of the memory node or the ONE live lifted detection carrying the label (same padding as a grasp target); the producer refuses a patch measured outside it. A hint that does not ground (unknown or ambiguous label, unknown memory node, no detector or memory) refuses the goal — no goal is sent, never a fall-back to "place anywhere"; the refusal tells the LLM to omit `place_target`. Nothing about the cell is surveyed or listed in the prompt. |

A target that grounds to nothing, to more than one instance with no
`object_id`, or to a box outside the robot base frame refuses the dispatch: no
goal is sent, the refusal is logged and recorded as a `failed` execution
telling the LLM to disambiguate (recall the object and pass its `node_id`, or
look for it). Naming a target arms
nothing: the runner strips any region, the HAL producers measure, and the
kernel trusts only the measured region — its trust boundary is unchanged.

With object tracking on (param `object_tracking`, the deploy's
`DeployRuntime.object_tracking`, issue #418) the declarations also carry the
named classes as `ObjectQuery`s for the Layer-2 object tracker
(`GraspDeclaration.target_queries`, `PlaceDeclaration.destination_query`),
beside the unchanged search box. A label several live detections carry is then
not refused: it grounds as one `count="all"` query with no search box, and the
tracker grounds every instance (the policy's choice is resolved at binding).
Limits: the label path reads the continuous lift (`locate_in_view`'s one-shot
2D answer is not lifted), and the lift must run in the robot base frame
(`object_lift_map_frame`) for the seed to be usable; a map-frame box is refused,
not transformed.

**Crash-safe ladder resume:** set the `ladder_state_path` ROS parameter to a
writable JSON path and the mission ledger + every replanning-ladder bound
(attempts, subdivision offers, decompose nudges, per-task locate budget) is
snapshotted after each mutation and restored at `on_configure`. Empty
(default) disables persistence.

**Dispatch-phase watchdog:** `_rskill_inflight` (the one-goal-at-a-time busy
latch) is bounded by `dispatch_watchdog_s` (default 30 s; `<= 0` disables): if
neither the VRAM-peer eviction nor the goal response resolves within the
ceiling (a runner/peer died after the readiness probe; rclpy futures never
time out on their own), the watchdog releases the latch, reactivates the
peers, and emits a `KIND_CONTROLLER` FailureTrigger
(`state="dispatch_timeout"`). The expired dispatch generation invalidates its
remaining callbacks; a late accepted goal is canceled and cannot overwrite a
newer dispatch.

**Waiting for the runner:** the runner creates `/openral/execute_rskill` in its
own `on_configure`, which can finish after the first prompt arrives. A dispatch
that finds the server missing holds the busy latch (in-flight state
`dispatching`) and re-probes every 0.25 s for up to `execute_server_wait_s`
(default 30 s; `0` fails on the first miss), then sends the goal. Past that
bound it releases the latch and emits a `KIND_CONTROLLER` FailureTrigger
(`state="unavailable"`). This covers every prompt source — the deploy's
startup prompt, an operator prompt, a dispatch after the runner restarts — not
only the first.

The reasoner **never** publishes `openral_msgs/ActionChunk` — actuation
authority lives behind the F1 action server + the F5 safety boundary
("Holds no authority over actuation").

## Completion check (`completion_check_node`)

A VLA emits no success signal, and Robometer's success head is no completion signal on its
own: on Spark it peaked mid-carry as often as at the placement (R221-R223, 2026-10-22). With
`runtime.enable_completion_check` (plus the reward monitor, the critic leg and the scene VLM)
the head is a cheap trigger and a VLM the judge. On a success score of 0.6 or more, a
`DETACH` grasp event or a newly released track (one check per 10 s), the node:

1. holds the runner's commands (`/openral/rskill/hold`: the policy keeps stepping, nothing
   is sent, the robot keeps its last command);
2. asks `/openral/perception/query_scene` the reasoner's `COMPLETION_QUESTION` once per
   RGB camera;
3. ends the goal as a success (`/openral/rskill/complete`) when every camera answers yes,
   and otherwise releases the hold.

It only withholds or ends motion; every command still crosses the safety kernel. Each check
logs `completion_check <reason>: done=<bool>` with each camera's answer.

## Reasoner model registry

Reasoner selection is **model-first** (ADR-0088). The primary knob is
`OPENRAL_REASONER_MODEL`, a key in the curated
[`openral_core.REASONER_MODELS`](../../python/core/src/openral_core/schemas.py)
registry. Registry membership means the model has cleared OpenRAL's robotics
tool-calling contract; there is no separate compatibility flag.
The registry resolves the wire dialect, served model id, default endpoint,
auth requirement, hosting mode, and local-compute floor.

| `OPENRAL_REASONER_MODEL` | Served model | Client | Hosting / endpoint | Auth |
|---|---|---|---|---|
| `claude-opus-4-8` | `claude-opus-4-8` | `AnthropicToolUseClient` | Anthropic cloud | required |
| `claude-opus-5-5` | `claude-opus-5-5` | `AnthropicToolUseClient` (`tool_choice: auto`) | Anthropic cloud | required |
| `claude-sonnet-5-5` | `claude-sonnet-5-5` | `AnthropicToolUseClient` (`tool_choice: auto`) | Anthropic cloud | required |
| `claude-haiku-5-5` | `claude-haiku-5-5` | `AnthropicToolUseClient` (`tool_choice: auto`) | Anthropic cloud | required |
| `gpt-5.5` | `openai/gpt-5.5` | `OpenAICompatibleToolUseClient` | OpenRouter cloud | required |
| `gpt-5.6` | `openai/gpt-5.6` | `OpenAICompatibleToolUseClient` | OpenRouter cloud | required |
| `cosmos3-edge` | `nvidia/Cosmos3-Edge` | `Cosmos3ToolUseClient` | managed local `http://127.0.0.1:8901/v1` | none |
| `qwen3.5-4b` | `qwen3.5:4b` | `OpenAICompatibleToolUseClient` (`tool_choice: auto`) | your Ollama, `http://localhost:11434/v1` | none |
| `qwen3.5-9b` | `qwen3.5:9b` | `OpenAICompatibleToolUseClient` (`tool_choice: auto`) | your Ollama, `http://localhost:11434/v1` | none |

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
- `OPENRAL_REASONER_MAX_SKILL_TOOLS` — optional per-tick cap on skill tools.
  The reasoner ranks the palette's skills against the active mission task and
  any pending operator prompts by keyword overlap and offers at most `k` (#494).
  Inside those `k` slots it keeps, in order: the skill in flight; the best match
  for the newest top-priority pending operator prompt (with `k` ≥ 2 it is offered
  on the tick the prompt arrives); the task's last dispatched skill (outside a
  mission, the last one dispatched); the best navigate and look-at skill not
  already kept (the workflow's APPROACH / FRAME steps; never the last free slot);
  then the highest scores, so at least one ranked substitute is always offered
  while a slot remains. With no task and no pending prompt the last ranking is
  kept. Equal scores (in every pick) prefer
  skills not yet tried for the active task (so substituting a skill reaches past the
  top `k`), then higher weight precision (fp32/bf16/fp16, then fp8, int8, 4-bit, from
  the manifest's `quantization.dtype`), then the id. `cosmos3-edge` defaults to 4 (its tool
  calls run away with 14 near-identical skills; 19/20 valid with 4); every other
  curated model sees the whole palette. `0` removes the cap; any positive value
  sets it for any model, curated or not. A call to a skill outside the subset is
  refused like an off-palette id.
- `OPENRAL_REASONER_DIALECT=anthropic|openai` — required only for an uncurated
  raw model id.
- `OPENRAL_REASONER_TOOL_CHOICE=auto|required` — for an uncurated model on a
  bare-URL `openai` endpoint or on the `anthropic` dialect (where `required` is
  sent as `{"type": "any"}`). Set `auto` for a model that rejects forced tool use
  with HTTP 400 (a thinking-mode gateway; the Claude 5 generation). Case-
  insensitive, and any other value fails at client build (and in `openral
  doctor`). Curated models and named `openai` endpoints carry their own value,
  so there it is ignored with a `reasoner.tool_choice_ignored` warning.

#### Models that refuse forced tool use (#522)

The reasoner asks for exactly one tool call per tick (`tool_choice` `required`,
Anthropic `any`). Some models refuse that: the Claude 5 generation answers
`400 tool_choice: type "tool" and "any" are not supported for this model`, and
some OpenAI-compatible gateways refuse `required` in thinking mode. The clients
handle this without configuration:

- The curated Claude 5 entries run `tool_choice: auto` from the start, with a
  16000-token cap because they think before calling a tool and thinking counts
  against `max_tokens` (an uncurated `anthropic` model gets the same cap).
- Any other model whose 400 refuses forced tool use is retried once with
  `auto`; if that retry succeeds the client stays on `auto`, logged as
  `reasoner.tool_choice_downgraded`. An unrelated 400 that only mentions
  `tool_choice` does not downgrade it.
- Under `auto`, a reply with no tool call gets one nudge ("respond with exactly
  one tool call"). A second prose reply is a tick error (`reasoner.no_tool_call`)
  that the node feeds back into the next prompt, like any undecodable call — so a
  model stuck in prose is told to call a tool rather than stalling the mission.
- A refusal (`stop_reason: refusal` / `finish_reason: content_filter`) or a
  reply cut off at the token cap is a tick error, not a `wait`.
- OpenAI's reasoning models (`gpt-5.6-*`, `gpt-6*`) refuse function tools on
  Chat Completions while reasoning is on, and some (`gpt-6-astra`,
  `gpt-6.1-sol`) cannot turn reasoning off. So the `openai` named endpoint (or
  OpenAI's URL spelled out) uses the **Responses API**
  (`OpenAIResponsesToolUseClient`): reasoning stays at the model's default,
  requests set `store: false`, and the bare model id is sent (`gpt-5.5`, not
  OpenRouter's `openai/gpt-5.5`). Every other OpenAI-compatible endpoint stays on
  Chat Completions; one that asks for `max_completion_tokens` gets it after one
  400 (`reasoner.max_tokens_param_switched`).
- Checked live on 2026-11-09 (idle robot → starts the pick skill; skill
  running → `wait`): `claude-opus-5-5`, `claude-sonnet-5-5`, `claude-haiku-5-5`,
  and on `ENDPOINT=openai` `gpt-5.5`, `gpt-5.6`, `gpt-5.6-luna`/`-sol`/`-terra`,
  `gpt-6-luna`/`-sol`/`-astra`, `gpt-6.1-sol`/`-terra` — all correct, 1–5 s a tick.

```bash
# Curated cloud model
export OPENRAL_REASONER_MODEL=gpt-5.5
export OPENRAL_REASONER_API_KEY=sk-or-...

# Curated managed-local model
export OPENRAL_REASONER_MODEL=cosmos3-edge
```

For the uncurated escape hatch see "Uncurated local endpoint" below.

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
| `anthropic` | curated `claude-opus-4-8` / `claude-opus-5-5` / `claude-sonnet-5-5` / `claude-haiku-5-5`, or `ENDPOINT=anthropic` + a raw model id |
| `openrouter` / `gemini` / `xai` / `deepseek` / `huggingface` | `ENDPOINT=<name>` + `API_KEY` (an OpenAI key: `ENDPOINT=openai`) |
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
(`resolve_place` / Nav2 navigation skills), per-tick progress evaluation,
and observe-but-never-bypass safety/e-stop handling ("Python proposes,
C++ disposes").

At `on_configure` the node calls
[`resolve_reasoner_system_prompt`](../../python/reasoner/src/openral_reasoner/tool_use.py),
which composes:

1. **Base brief** — `DEFAULT_SYSTEM_PROMPT`, unless
   `OPENRAL_REASONER_SYSTEM_PROMPT` is set to a non-empty value (whitespace-only
   counts as unset), which replaces it.
2. **`## THIS ROBOT` block** — appended by `render_robot_context_prompt` from
   the active robot's `RobotCapabilities` (`robot_yaml` ROS parameter, or the
   `robot_capabilities` constructor arg): embodiment tags, whether it can
   locomote (gates the navigate-to-approach rule), manipulation/sensing
   hardware, payload, and control modes.

With no robot wired the prompt stays at the (possibly overridden) base brief
alone.

## Curated reasoner models

The library factory has no default and refuses to guess. `openral deploy sim`
does default to `OPENRAL_REASONER_MODEL=gpt-5.5`, with
`OPENRAL_REASONER_MAX_TOKENS=16384`; it was the most reliable model in live
collective-goal decomposition tests. The default needs
`OPENRAL_REASONER_API_KEY` and fails loudly without it.

### Cloud — GPT-5.5 / GPT-5.6 via OpenRouter or OpenAI

```bash
export OPENRAL_REASONER_MODEL=gpt-5.5  # or gpt-5.6
export OPENRAL_REASONER_API_KEY=sk-or-...
# With an OpenAI key instead, point the same model at OpenAI's own API
# (Responses API; uncurated ids such as gpt-6-sol work the same way):
# export OPENRAL_REASONER_ENDPOINT=openai
# export OPENRAL_REASONER_API_KEY=sk-...
```

### Cloud — Claude Opus 4.8 / Claude 5

```bash
export OPENRAL_REASONER_MODEL=claude-opus-4-8  # or claude-opus-5-5 / claude-sonnet-5-5 / claude-haiku-5-5
export OPENRAL_REASONER_API_KEY=sk-ant-...
```

The Claude 5 entries run `tool_choice: auto` (they refuse forced tool use) and
think before they act, so a tick can take longer than on Opus 4.8; raise
`OPENRAL_REASONER_TIMEOUT_S` if ticks time out at the 10 s default.

### Managed local — NVIDIA Cosmos 3 Edge

[Cosmos 3 Edge](../../docs/reference/cosmos3-edge-reasoner.md) (released
2026-07-20) is the 4B on-device tier of NVIDIA's Cosmos 3 omnimodal
world-model family, built for physical reasoning, task planning, and embodied
decision making on Jetson Thor / RTX-class GPUs. Unlike the general-purpose
LLMs above it is trained on robotics/AV/warehouse data, with spatial grounding
and physical-plausibility judgment, and its `describe_image` completion gate
runs on the same local model (no separate cloud VLM). Weights are
**OpenMDW-1.1** (commercial use OK). One env var is enough; the managed vLLM
server auto-starts on the first tick (first boot downloads ~8 GB):

```bash
export OPENRAL_REASONER_MODEL=cosmos3-edge
```

Requires an NVIDIA GPU (Ampere+; BF16 is the only officially-tested
precision; **≥12 GB recommended**). An 8 GB 4070 was enough for validation,
but the knobs differ by serving stack: the pinned stable stack needed
`OPENRAL_COSMOS3_GPU_MEM_UTIL=0.95` merely to boot (inference remains blocked);
the working vLLM-main native implementation needed expandable CUDA segments
plus `--kv-cache-dtype fp8`. Pre-warm the pinned sidecar with
`python tools/cosmos3_reasoner_sidecar.py`, or point
`OPENRAL_REASONER_ENDPOINT` at a compatible self-managed server.

> ⚠️ **Live status (2026-09-07): works on linux-aarch64; still blocked on
> x86_64.** Read the two platforms separately — the lock resolves a *different
> vLLM* for each, and only one carries the native Edge model
> ([vllm#48291](https://github.com/vllm-project/vllm/pull/48291)).
>
> * **aarch64 (Jetson AGX Thor, vLLM 0.28.0): working.** The architecture
>   resolves natively, the server serves, and a `tool_choice="required"` tick
>   returned a **validated tool call at 1.26–1.36 s warm** — inside the 0.2 Hz
>   budget. The first tool-call request costs ~64 s while xgrammar builds the
>   grammar, so the first tick after boot blows a 5 s budget; later ones do not.
> * **x86_64 (vLLM 0.24.0): still blocked.** That release predates the native
>   model and crashes on inference via the Transformers-fallback
>   `get_rope_index` bug. On a nightly it worked (validated tool call ~1.5 s on
>   an 8 GB 4070, `--kv-cache-dtype fp8`). **Use a curated cloud model there
>   until the x86 branch resolves a vLLM with #48291.**
>
> Getting there needed three sidecar fixes, all landed: the `--universal` lock
> was unsatisfiable on aarch64 (`nvidia-nccl-cu13` pinned across all of linux
> against a torch that needs a newer one); the flattened reasoner view *breaks*
> the native loader, which reads the diffusers layout by path; and FlashInfer's
> JIT sampler needs CUDA toolkit headers a JetPack image does not ship, so the
> sidecar now defaults `VLLM_USE_FLASHINFER_SAMPLER=0`. Full findings in the
> [assessment page](../../docs/reference/cosmos3-edge-reasoner.md).

### On the robot — Qwen 3.5 4B / 9B via Ollama

`qwen3.5-4b` and `qwen3.5-9b` run on the robot's own [Ollama](https://ollama.com)
(vision, tools and thinking; Apache-2.0 weights; the Q4_K_M `qwen3.5:4b`
(3.4 GB) and `qwen3.5:9b` (6.6 GB) tags). They need no key and nothing but the
model (`just bootstrap-ollama --model qwen3.5:9b` for the 9B):

```bash
just bootstrap-ollama            # installs Ollama, pulls qwen3.5:4b, serves a 32K window
export OPENRAL_REASONER_MODEL=qwen3.5-4b
```

**The context window is the one setting that matters.** The reasoner's prompt
is 9–26K tokens (system prompt plus one tool per skill); Ollama loads a model
with a 4096-token window on a GPU under 24 GiB
([docs](https://docs.ollama.com/context-length)) and **silently truncates** a
longer prompt, and its OpenAI-compatible API has no field to raise it. Start
the daemon with `OLLAMA_CONTEXT_LENGTH=32768` (`just bootstrap-ollama` sets it on
the daemon it starts and on the systemd unit of an Ollama it installs; an
existing systemd install: `sudo systemctl edit ollama` →
`Environment=OLLAMA_CONTEXT_LENGTH=32768`). The
client compares the prompt tokens Ollama reports with the request it sent
(confirming a suspected cut against the window `/api/ps` reports, so densely
packed text that fits is not refused) and fails the tick (`ROSPlanningError: … cut the prompt to fit`) rather than act
on a truncated prompt.

Ollama ignores `tool_choice`
([OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility)),
so the `ollama` endpoint sends `auto`: a prose reply gets the one nudge, then
counts as an invalid plan.

The 4B, measured through the real client (Ollama 0.32.14, RTX 5070 Laptop 8 GB,
32K window, thinking on, 20 ticks each; the curation bar is ≥ 17/20, #494):

| prompt | skills | tokens | valid | median tick |
|---|---|---|---|---|
| `so101_follower` + query tools | 3 | 9,229 | 18/20 | 9.2 s |
| `franka_panda` + query tools, full palette | 14 | 20,379 | 20/20 | 9.7 s |

The 9B, Ollama 0.40.16, same laptop (12% of the model on the CPU):
`so101_follower` 20/20 (18.8 s), `franka_panda` full palette 20/20 (16.3 s),
frames 2/2.

`describe_image` answered both SO-101 eraser frames correctly (2/2). Re-run on
Ollama 0.40.16: `so101_follower` 18/20 again, frames 2/2 (ticks slower there
because a CI job held 3.4 GB of the same GPU). With
thinking off (`reasoning_effort: none`) ticks drop to ~2 s but validity to
16/20, under the bar, so the entry keeps thinking on.
Failures were a nested `place_target` sent as a JSON string and a
hallucinated tool name; both come back to the model as invalid-plan feedback.

Not curated, same harness: `qwen3-vl:4b-instruct` 17/20 but 1/2 frames and
24 s ticks; `gemma4:e4b` 17/20, 1/2 frames, and it re-dispatched a skill the
context reported done; `ministral-3:3b` 15/20 (`franka_panda` 9/20);
`granite3.2-vision:2b` 0/20 (a 16K model under a 21K prompt); `qwen3-vl:4b` is
a thinking-only tag (165 s first tick). `lfm3-vl` needs a newer Ollama than
0.32.

Ollama Cloud models are not curated: none could be run here without an
`OLLAMA_API_KEY`. One works through the escape hatch below with
`OPENRAL_REASONER_ENDPOINT=https://ollama.com/v1`, `OPENRAL_REASONER_DIALECT=openai`
and the key as `OPENRAL_REASONER_API_KEY` (sent as `Authorization: Bearer`).

### Uncurated local endpoint — explicit escape hatch

Models outside `REASONER_MODELS` are not claimed compatible. They work when the
model id, endpoint, and dialect are explicit; the factory and `openral doctor`
warn that robotics tool-calling reliability is unverified.

```bash
# Linux: runs ollama's release-tag install.sh only when its sha256 matches
# OLLAMA_INSTALL_SH_SHA256 (scripts/bootstrap_ollama.sh); refuses otherwise.
just bootstrap-ollama --model qwen3:8b
export OPENRAL_REASONER_MODEL=qwen3:8b
export OPENRAL_REASONER_ENDPOINT=ollama
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
| `reasoner.model` | When the client has a `model_id` | LLM model identifier (e.g. `claude-opus-4-8`). |
| `reasoner.force` | Always | `True` when the tick was preempted by `FailureTrigger.severity ≥ FAIL` or a new operator prompt. |
| `reasoner.tool` | Successful + retry-cap suppressed ticks | The `tool` discriminator of the `ReasonerToolCall` variant the LLM picked (e.g. `execute_rskill`). |
| `reasoner.rskill_id` | When the call carries a `rskill_id` (`execute_rskill`) | Skill id the LLM chose. |
| `reasoner.suppressed_reason` | Suppressed ticks | `palette_empty` or `retry_cap`. The `min_interval`, `retry_cap_hold`, `mission_finished` and `heartbeat_idle` gates fire BEFORE the span opens (so dashboards don't show noise). |
| `reasoner.llm_s` | Ticks that reached the LLM | Wall-clock of the LLM round-trip alone. |
| `reasoner.prompt_tokens` | When the client surfaces usage | Provider-reported prompt tokens, cache reads included. |
| `reasoner.tier` | Always | Trigger tier that drove this call: `A` (safety), `B` (replan: hal/sensor/rskill/wam), `C` (critic), `D` (operator/perception), or `heartbeat`. |
| `reasoner.offered_skills` | When a skill-tool cap is set | Skill ids offered this tick (`ReasonerCore.offered_palette`). The ranking (every skill's score and the query) is on the `reasoner.skill_subset` log, emitted when the subset changes. |
| `reasoner.mission_json` | When a mission is active | `MissionState.to_summary()` JSON — the ordered task queue (id/text/status/attempts/verdict) the live dashboard renders as the Mission card checklist. Absent on bare-goal deploys. |
| `reasoner.error_kind` | Provider failure | `ROSPlanningError` subclass name; an `exception` event is added to the span. |
| `reasoner.llm_s` | Every tick that reached the LLM | Wall-clock of the provider round-trip **alone**. `ReasonerTickResult.elapsed_s` is end-to-end tick time (context render + call + bookkeeping), so `elapsed_s - llm_s` is the reasoner-side overhead — without the split a tick that grew from 6 s to 99 s cannot be attributed. |
| `reasoner.prompt_tokens` | When the client reports usage | Provider-reported prompt tokens (Anthropic cache reads included). Tick latency that grows in step with this is a context-size problem — window or summarize the history rather than raising `OPENRAL_REASONER_TIMEOUT_S`. |

`reasoner.llm_s` and `reasoner.prompt_tokens` also appear on the
`reasoner.tick.selected` structured log, readable without opening Jaeger.

The active W3C `traceparent` captured inside this span is threaded onto the
outbound `EmitPromptTool` `PromptStamped.metadata_json` so the F7 bag↔OTel
correlator can join the published prompt back to the producing tick.

Spans are emitted via `opentelemetry-sdk`; a no-op (cost <1 µs) when
`configure_observability` was not called (no provider installed). The
`just docker-smoke-x86-reasoner` smoke installs a real provider so the
round-trip can be observed end-to-end inside the deploy image.

## See also

- [`openral_reasoner.core`](../../python/reasoner/src/openral_reasoner/core.py) — transport-agnostic orchestrator.
- [`packages/openral_prompt_router`](../openral_prompt_router/) — F10 prompt fan-in.

### Collision diagnostics

Typed collision evidence on `/openral/failure/safety` preserves the kernel's
link/object IDs and adjudication configuration. Prompt summaries retain both
actors, signed distance and whether the check was measured or predicted; joint
vectors remain available in the original ROS message for replay. The robot
context lists configured joint names, simulator aliases and parent/child links
without inventing anatomical labels. A predicted collision is not a measurement
of physical contact, and voxel IDs do not establish an obstacle's identity.
