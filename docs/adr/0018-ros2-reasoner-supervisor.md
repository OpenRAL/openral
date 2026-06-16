# ADR-0018: ROS 2 reasoner + supervisor graph

- Status: **Accepted** (incremental landing — see Amendments)
- Date: 2026-05-24
- Amended: 2026-05-24 (see Amendments below)
- Related: CLAUDE.md §6.1 (Layers 2, 3, 4, 6, 7), §6.2 (dual-system
  pattern, **amendment required**), §7.6 (replanning ladder,
  **amendment required**); [ADR-0010](0010-inference-runner.md)
  (Inference Runner), [ADR-0011](0011-nvmm-handoff.md) (NVMM handoff),
  [ADR-0012](0012-open-core-licensing.md) (Licensing),
  [ADR-0017](0017-dashboard-otlp-receiver.md) (Dashboard receiver).

## Context

The OpenRAL runtime today is one Python process — `HardwareRunner` /
`SimRunner` from ADR-0010 — that owns one rSkill, one HAL, and one
WorldState aggregator. There is no graph; there is no fault bus; there is
no way for an out-of-process planner to look at world state and tell the
runtime "stop this skill, start that one." Triggering a skill from outside
the process today means re-running the CLI.

The architecture documented in CLAUDE.md (§6.1 layered design, §6.2
dual-system pattern, §7.6 replanning ladder) assumes a graph that is not
yet on disk: a Reasoner (S2) that consumes world state and emits plans,
HAL nodes that expose joint state and accept commands, sensors that
publish typed topics, a safety kernel that gates every actuation, and an
observability spine that ties it all together. A companion review
inventories what exists
(IDL `ActionChunk`, `PromptStamped`, `FailureTrigger`, action
`ExecuteRskill`; SO-100 HAL lifecycle node; WorldStateAggregator lifecycle
node; OTel spans + dashboard receiver) and what is missing for the graph
to function.

This ADR commits to the shape of that graph. It does **not** commit the
implementation of every component; it commits the topics, actions,
message contracts, and layer ownership so the ten features in the review
can land in independent PRs without each one re-litigating boundaries.

Two non-architectures are explicitly rejected:

**(A) All-Python supervisor.** Replanning, failure handling, and skill
dispatch happen inside one Python process via callbacks. Cheap, but it
collapses §6.1 layers, makes the safety kernel impossible to run as a
separate process (§1.5), and offers no out-of-process tap for the
dashboard or for an operator.

**(B) ROS-only hot path.** Every frame and every action chunk travels
through `sensor_msgs/Image` and `trajectory_msgs/JointTrajectory`. Clean
but throws away NVMM zero-copy (ADR-0011) and forces a serialization per
frame.

The decision below is the hybrid that respects both constraints:
ROS 2 carries control plane, lifecycle, and the chunk-rate safety
boundary; in-process Python (and future C++ for the safety kernel
internals) carries the policy hot path and the camera path.

## Decision

### 1. The graph

OpenRAL adopts a ROS 2 graph with these nodes (all lifecycle):

```
Topics:
  /openral/world_state_fast   (WorldStateStamped, 30 Hz, RELIABLE+VOLATILE+KL=1)
  /openral/world_state_slow   (WorldStateStamped,  5 Hz, RELIABLE+VOLATILE+KL=1)

  /openral/failure/hal        (FailureTrigger, RELIABLE+VOLATILE+KL=50)
  /openral/failure/sensor     (FailureTrigger, ...)
  /openral/failure/rskill     (FailureTrigger, ...)   # renamed from /skill on 2026-05-25
  /openral/failure/safety     (FailureTrigger, ...)
  /openral/failure/wam        (FailureTrigger, ...)
  /openral/failure/critic     (FailureTrigger, ...)

  /openral/perception/motion        (PromptStamped, BEST_EFFORT+VOLATILE+KL=10)
  /openral/perception/objects       (PromptStamped, ...)
  /openral/perception/ocr           (PromptStamped, ...)
  /openral/perception/scene_change  (PromptStamped, ...)

  /openral/prompt             (PromptStamped, RELIABLE+VOLATILE+KL=10)
  /openral/candidate_action   (ActionChunk, RELIABLE+VOLATILE+KL=1)
  /openral/safe_action        (ActionChunk, RELIABLE+VOLATILE+KL=1)
  /openral/estop              (std_msgs/Empty, RELIABLE+VOLATILE+KL=10)
  /openral/human_estop        (std_msgs/Empty, RELIABLE+VOLATILE+KL=10)
  /diagnostics                (diagnostic_msgs/DiagnosticArray, 1 Hz)

Actions:
  /openral/execute_rskill     (ExecuteRskill.action — goal/result/feedback)

Services:
  /openral/estop_reset        (std_srvs/Trigger)
  /openral/skill_registry_changed   (event, not service — fired by ral skill install|remove)
  /openral/sensors/<id>/reload_pipeline   (custom srv carrying gst pipeline YAML)
```

Nodes:

- `world_state_node` — exists today; gains a typed `WorldStateStamped`
  publisher on two topics (fast + slow). JSON publication removed.
- `<robot>_hal_node` — SO-100 exists; Franka / UR5e / UR10e skeletons.
  Subscribes to `/openral/safe_action` and `/openral/estop`.
- `sensors_<vendor>_node` — new package per vendor. Owns the GStreamer
  pipeline with three legs (policy / observability / event).
- `safety_node` — new. Python pass-through with stub envelope checks
  (Day 1); C++ kernel later (separate ADR).
- `rskill_runner_node` — new. Owns the `ExecuteRskill.action` server and
  the in-process `HardwareRunner` from ADR-0010. **One per robot.**
- `reasoner_node` — new. Direct LLM dispatch (no BT in v1).
- `prompt_router_node` — new. Single node with adapter registry
  (CLI + dashboard WebSocket in v1).
- `human_estop_*` adapters — small nodes that forward UI / Slack / voice
  estop intent onto `/openral/human_estop`.
- `hardware_estop_node` — bridges a GPIO relay / USB E-stop pendant onto
  `/openral/estop`.
- `deadman_watchdog_node` — fires `/openral/estop` if no
  `/openral/safe_action` arrives within deadline.

### 2. IDL additions and changes

**New** `openral_msgs/msg/WorldStateStamped`:

```
std_msgs/Header header
string trace_id
# joint state, ee pose, control mode, etc. (full WorldState fields)
# ...
string[]  frame_ids        # tf2 frames referenced (consumers read /tf)
string[]  sensor_ids       # parallel array with staleness_ms
float32[] staleness_ms
```

**Changed** `openral_msgs/msg/FailureTrigger` — promote `kind` and
`severity` from string to `uint8` with constants:

```
std_msgs/Header header

# kind
uint8 KIND_TIMEOUT       = 0
uint8 KIND_FORCE         = 1
uint8 KIND_WORKSPACE     = 2
uint8 KIND_PERCEPTION    = 3
uint8 KIND_CRITIC        = 4
uint8 KIND_CONTROLLER    = 5
uint8 KIND_SELFVERIFY    = 6
uint8 KIND_HUMAN         = 7
uint8 KIND_WAM           = 8
uint8 KIND_REASONER_TIMEOUT = 9
uint8 KIND_SUPPRESSED_SUMMARY = 254
uint8 kind

# severity
uint8 SEVERITY_INFO  = 0
uint8 SEVERITY_WARN  = 1
uint8 SEVERITY_FAIL  = 2
uint8 SEVERITY_ABORT = 3
uint8 severity

string evidence_json     # serialized Pydantic FailureEvidence (discriminated union)
string skill_id
string trace_id
```

Both changes land on the existing pre-publish baseline
(`schema_version: "0.1"`) — no migrator required while we are
pre-publish (CLAUDE.md §1.6).

Existing IDL **unchanged**: `ActionChunk`, `PromptStamped`,
`ExecuteRskill.action`.

### 3. Data-flow contract

- **Images** flow in-process through GStreamer with NVMM/CUDA zero-copy
  on Jetson (ADR-0010, ADR-0011). The ROS observability leg is a `tee`
  branch publishing `image_transport/compressed` at 5 Hz; the policy
  never reads from it.
- **Joint state and transforms** flow over ROS — HAL publishes
  `/joint_states`, `WorldStateAggregator` is the only subscriber and
  bridges them in-process via `.snapshot()` to the skill.
- **`ActionChunk`** flows over ROS at chunk rate (≤30 Hz):
  `Skill → /openral/candidate_action → safety_node → /openral/safe_action →
  HAL`. Two serializations per chunk, acceptable since chunks are not
  per-frame.
- **`/openral/estop`** is subscribed by **both** HAL and skill_runner
  (defense in depth, CLAUDE.md §1.5). If the runner stalls, HAL still
  brakes.

### 4. Reasoner contract (direct LLM dispatch, no BT in v1)

The `reasoner_node`:

- Subscribes to `/openral/world_state_slow` (5 Hz), all
  `/openral/failure/*`, all `/openral/perception/*`, `/openral/prompt`.
- Heartbeat tick at 0.2 Hz (one every 5 s; was 5–10 Hz pre-2026-05-25
  amendment) with event preemption on `FailureTrigger.severity≥FAIL`
  for execution sources or `≥WARN` for `/openral/failure/safety`
  (Tier A) and on new `PromptStamped` (min-interval 100 ms).
  Heartbeat ticks that see no new failure / prompt / perception event
  since the last successful tick are suppressed with
  `reasoner.suppressed_reason="heartbeat_idle"`. See the 2026-05-25
  amendment for the full trigger taxonomy.
- Calls a typed LLM (Anthropic / OpenAI SDK or OpenAI-compatible local
  endpoint, model selected by deployment config) with a structured **text**
  context (WorldState fields + rolling buffer of recent FailureTriggers +
  recent perception events). No pixels in v1.
- Dispatches one of four typed tool calls:
  1. `ExecuteRskill(skill_id, prompt, deadline_s)` → action goal to
     `rskill_runner_node`.
  2. `ReloadGstPipeline(sensor_id, pipeline_yaml)` → service call on the
     sensor node.
  3. `LifecycleTransition(node, transition)` → standard ROS lifecycle.
  4. `EmitPrompt(target_topic, text, metadata_json)` → republish a
     `PromptStamped` onto another topic for cascades.
- Holds **no** authority over actuation: it never publishes
  `ActionChunk`; only the skill_runner does.

The tool palette is built at lifecycle `configure` from the local skill
registry filtered by `RobotCapabilities`, and refreshed on
`/openral/skill_registry_changed`. The LLM cannot dispatch a skill that
isn't installed, isn't capability-matched, or isn't licensed for the
deployment.

Reasoner failures (LLM timeout, malformed tool call) publish
`FailureTrigger` on `/openral/failure/rskill` with `kind=REASONER_TIMEOUT`
or a Pydantic-validation kind. Bounded retry counter per failure kind
prevents storms; min-interval prevents thrash.

BehaviorTree XML output (matching CLAUDE.md §6.2/§7.6 wording) is
deferred to a future iteration. F4's direct-dispatch shape does not
preclude adding a `bt_executor_node` later that consumes
`BehaviorTreeXml` plans alongside direct tool calls.

### 5. Safety contract (Day 1 Python; C++ kernel later)

`safety_node` enforces the intersection of two envelopes:

- The **robot manifest** declares the ceiling (max force, velocity,
  workspace AABB) per `RobotDescription`.
- Each **rSkill manifest** may declare a tighter envelope. Loosening
  beyond the robot ceiling is rejected at goal-acceptance — never
  silently honored.

On envelope violation, safety_node:

1. Drops the candidate ActionChunk (does not republish).
2. Publishes `FailureTrigger` on `/openral/failure/safety` with
   `kind=KIND_FORCE` or `KIND_WORKSPACE`, `severity=SEVERITY_ABORT`.
3. Publishes `std_msgs/Empty` on `/openral/estop`.

Four E-stop sources are wired (defense in depth): safety_node itself,
`/openral/human_estop` (UI / Slack / voice via adapter nodes),
`hardware_estop_node` (GPIO / USB pendant), and `deadman_watchdog_node`
(no `/openral/safe_action` within deadline).

Recovery is manual: after estop, `safety_node`, HAL, and skill_runner
latch into faulted state; the `/openral/estop_reset` service triggers
the explicit recovery sequence. `ROSEStopRequested` (CLAUDE.md §10) is
never auto-cleared.

The Day-1 implementation is a Python lifecycle node with stub envelope
checks. The C++ kernel that replaces internals later (separate ADR) is a
process swap behind the same topic contract — not a refactor of the
graph.

### 6. trace_id is the join key

Every typed message that crosses the graph (`ActionChunk`,
`FailureTrigger`, `WorldStateStamped`, `PromptStamped`,
`ExecuteRskill.action` goal/feedback/result) carries an OTel `trace_id`.
The `rskill_runner_node` sets it from the active OTel context
(`rskill.execute` parent span). `rosbag2` recordings are therefore
replayable end-to-end against the dashboard receiver of ADR-0017 via the
F7 query-time correlator and `openral replay`.

### 7. GStreamer interop

`GStreamerSensorReader` (ADR-0010, ADR-0011) continues to feed the
in-process runner via NVMM/CUDA tensors on Jetson and via CPU
`numpy.ndarray` elsewhere. F6 generalizes the pipeline to a `tee` with
three legs: policy (NVMM, unchanged), observability (low-rate
`image_transport`/`compressed`), event (gst-element ML — `nvinfer` on
Jetson, `tflite` on CPU, motion / scene-change / OCR) publishing
`PromptStamped` onto `/openral/perception/{kind}`.

All three legs share a single `get_shared_cuda_context()` singleton
(ADR-0011) to avoid `CUDA_ERROR_INVALID_CONTEXT`. Event-leg inference
respects the policy leg's CUDA stream priority.

### 8. Licensing

Every package introduced by this ADR (`openral_rskill_ros`,
`openral_safety_ros`, `openral_reasoner_ros`, `openral_prompt_router`,
`openral_sensors_<vendor>`, IDL changes) ships under Apache-2.0 as part
of the open core (ADR-0012). Vendor-specific safety words, closed SDK
shims, and any cloud dispatcher coupling stay out of these packages and
live in `contrib-closed-shims` / `openral/cloud`.

### 9. CLAUDE.md amendments required (must land with the F4 PR)

- **§6.2:** change "S2 — slow reasoning (5–10 Hz), runs as the Reasoner
  and produces BT XML or latent goals" → "produces typed tool calls
  (`ExecuteRskill`, `ReloadGstPipeline`, `LifecycleTransition`,
  `EmitPrompt`); BT v4 XML is a future option."
- **§7.6:** change "The Reasoner emits BT v4 XML, not Python" → "The
  Reasoner emits typed tool calls (`openral_msgs`-typed,
  Pydantic-validated); BT v4 XML output is a future option. LLM tool
  calls are typed (Pydantic structured output). No free-form JSON
  parsing."
- **§2:** add the new packages to the repo map.
- **§7.10:** flip statuses on the repo state map as each PR lands.

## Consequences

### Positive

- The graph in CLAUDE.md §6.1 stops being aspirational. Every layer has
  at least one ROS 2 node owning its responsibility.
- Failure becomes a typed namespaced event. Forensics: "show me all
  failures around trace_id X" is one rosbag2 + one dashboard query.
- The chunk-rate safety boundary is a real topic; the eventual C++
  kernel is a process swap, not a rewrite.
- The dashboard (ADR-0017) becomes more useful: with `trace_id` in every
  typed message and the F7 correlator, `openral replay --bag --trace` is a
  scrubbable timeline.
- The hot path stays in-process with NVMM zero-copy on Jetson
  (ADR-0011). The graph is control plane.
- The LLM in the reasoner sees text only — cheap, fast, no per-tick
  multimodal cost. Pixels enter the loop through gst-element detectors,
  not the LLM.

### Negative

- New IDL lands on the existing pre-publish baseline
  (`schema_version: "0.1"`); two schema changes in this ADR
  (`WorldStateStamped`, `FailureTrigger` enums) — no migrator required
  while we are pre-publish (CLAUDE.md §1.6).
- More moving parts: ~10 lifecycle nodes instead of one Python process.
  Bootstrapping a single robot demo gains a launch file.
- Bag sizes grow even with the slim default. The F7 profile mitigates,
  but full-fidelity bags need real disk.
- Two `trace_id` worlds (OTel context propagation + ROS message field)
  must stay in sync. Contract: OTel context is the truth; ROS fields are
  set from it.
- CLAUDE.md §6.2 / §7.6 wording must change with F4 to accept direct
  dispatch.

### Neutral / out-of-scope

- C++ safety kernel internals — a future ADR replaces F5's pass-through.
- Cloud dispatcher (Apache-2.0 like the rest of the repo, ADR-0012) is not specified here. F4 selects
  the LLM endpoint via config; a cloud endpoint is just another endpoint
  for the LLM. Offloading heavy skills to cloud is a separate ADR.
- Voice / Slack prompt adapters are post-v1 in separate packages.
- BT executor — `bt_executor_node` consuming `BehaviorTreeXml` is a
  future option; F4's contract does not preclude it.

## Rollout plan

Six landing steps, each ≤ ~800 lines per CLAUDE.md §7.2:

1. **F1 + F8 + F5 (pass-through)** — locks the ROS topic contract
   end-to-end (`/openral/candidate_action`, `/openral/safe_action`,
   `/openral/estop`, `/openral/estop_reset`, `/diagnostics`) with zero
   new IDL.
2. **F2 + F3** — typed `WorldStateStamped` + namespaced
   `FailureTrigger.*` bus. Both extend the pre-publish IDL surface in
   place (no migrator); land together.
3. **F6** — GStreamer perception/event tee with shared CUDA context.
4. **F4 + F10** — reasoner with direct LLM dispatch + prompt router.
   F10 is cheap once F4 is in. **CLAUDE.md §6.2 / §7.6 amendment in the
   same PR as F4.**
5. **F7 + F9** — bag↔OTel correlator + LTTng opt-in.
6. **C++ safety kernel** — replaces F5 internals behind the same topic
   contract. Separate ADR, separate timeline.

Each step ships its own tests (real schemas, real manifests, real
launch files; no mocks per CLAUDE.md §1.11), updates `docs/METHODS.md`
(§1.13), and updates the repo state map (§7.10) in the same PR.

## Resolved open questions

| Question | Resolution |
| --- | --- |
| BT.cpp vs `py_trees_ros`? | Neither in v1; F4 uses direct LLM dispatch via typed tool calls. CLAUDE.md §6.2 / §7.6 amended accordingly. BT can be added later behind a `bt_executor_node` without breaking F4's contract. |
| `WorldStateStamped` — inline tf2 frames or reference? | Reference only. `frame_ids[]` list; consumers read `/tf`. |
| Topology of `/openral/perception/events`? | Per-kind topics (`motion`, `objects`, `ocr`, `scene_change`, ...), symmetric with the namespaced `/openral/failure/*` pattern. Reuse `PromptStamped` IDL. |
| Cloud-dispatch hook — reasoner or action server? | The reasoner. F4 selects the LLM endpoint via config; a cloud endpoint is just another endpoint. Skill action server stays local; cloud-offloading heavy skills is a separate ADR. |

## Amendments

### 2026-05-19 — F7 + F9 landed

Step 5 of the rollout plan landed in PR `hera` (branch
`hera`):

- **F7 — bag↔OTel correlator + `openral replay` / `openral record`.** Added
  `openral_observability.replay` (originally `tools/bag_otel_correlator/`
  — relocated post-live-validation since the `tools/` directory is not
  an installable Python package and `openral replay` / `openral record` from
  the installed CLI could not import it) with `read_bag` (mcap reader
  recovering
  the W3C `traceparent` from both `jsonschema`-encoded and CDR/`ros2msg`
  payloads, no `rosbag2_py` dep), `DashboardTraceClient`, and
  `build_timeline` (pure query-time join). The dashboard receiver
  grew a bounded per-trace span index (64 traces × 2048 spans,
  FIFO eviction) plus two routes — `GET /api/traces` and
  `GET /api/spans/{trace_id}`. `openral replay BAG [--trace] [--dashboard]
  [--out]` emits a chronological JSON timeline; `openral record --profile
  slim|full [--dry-run]` wraps `ros2 bag record` with the topic
  presets specified in §F7 of the capability review (slim records
  the action-chunk path, the failure bus, the slow world state, one
  compressed image stream, and diagnostics; full additionally records
  the fast world state, every camera, and every `/openral/perception/*`
  event).
- **F9 — ros2_tracing (LTTng) opt-in.** Added
  `python/observability/.../tracing_lttng.py` with the
  `OPENRAL_ROS2_TRACING` gate, the `lttng_tracepoint` context manager
  (no-op when the gate is off; flows through `lttngust` when
  available or to a per-process JSONL fallback otherwise), and
  the `LttngSession` subprocess wrappers (`lttng create / enable-event
  openral:* / start / stop / destroy`). Tracepoints now bracket the
  runner tick, the HAL state-read / action-write, and the safety
  boundary. `openral profile session {start|stop|view}` is the CLI
  driver. The active OTel `trace_id` is attached on every tracepoint
  as `otel_trace_id` so CTF dumps join to F7's timeline.

Step 5 closes the observability spine. The next step in the rollout
plan is the C++ safety kernel replacing F5's internals.

### 2026-05-25 — F4 tick model: event-driven + 0.2 Hz heartbeat; `/openral/failure/skill` → `/openral/failure/rskill`

The original §4 specified the reasoner ticks at "5–10 Hz, with event
preemption on `FailureTrigger.severity≥FAIL` or new `PromptStamped`".
That made the LLM call the dominant cost in idle-state — at 5 Hz the
reasoner pays one structured tool-use call per 200 ms even when no
event has arrived, no prompt is queued, and the world state hasn't
moved. This amendment downgrades the periodic tick to a slow heartbeat
and makes the event bus the primary trigger, while widening the
preemption contract for the safety source.

**1. Tick model.** The default `tick_hz` drops from `5.0` to `0.2` (one
heartbeat every 5 s). The heartbeat is the safety net for "task is not
making progress but nothing has fired"; the event bus is the primary
trigger. The `ReasonerCore.tick()` `min_interval_s` of 100 ms is
retained — it gates LLM thrash regardless of trigger source.

**2. Heartbeat-idle suppression.** `ReasonerCore.tick(force=False)`
now short-circuits with `suppressed_reason="heartbeat_idle"` when the
`ContextRenderer` has not received a new failure, prompt, or
perception event since the last successful tick (tracked via a
monotonic `ContextRenderer.seq` counter). A forced tick (event
preemption) bypasses this gate per the existing `force=True` contract.

**3. Trigger taxonomy.** The reasoner formalises four trigger tiers:

| Tier | Source | Preempts on | Notes |
|---|---|---|---|
| A — hard safety | `/openral/failure/safety` | `severity ≥ WARN` (=1) | Was `≥ FAIL`. Safety WARN means the C++ kernel (or F5 pass-through) saw a near-miss; the reasoner needs the LLM in the loop before the next chunk lands. `ROSSafetyViolation` remains never-caught (CLAUDE.md §10). |
| B — execution failure | `/openral/failure/{hal,sensor,rskill,wam}` | `severity ≥ FAIL` (=2) | Existing behaviour. Subject to per-kind retry-cap (default 3). |
| C — progress concern | `/openral/failure/critic` | `severity ≥ FAIL` (=2) | **Contract-only — no producer ships in this PR.** Slot for a SARM (state-action reward model) / VLA-as-judge / heuristic watchdog node. Payload is `CriticEvidence(critic_id, score, threshold)` from `openral_core.FailureEvidence` (already on disk). |
| D — operator / world | `/openral/prompt`, `/openral/perception/*` | Operator prompt forces; perception is informational | Existing behaviour. Perception novelty filtering is future work. |

**4. Tier-C `/openral/failure/critic` producer contract.** Any node
publishing on `/openral/failure/critic` MUST:

- Use `FailureTrigger.kind = KIND_CRITIC` (=4) (the IDL constant
  already reserved in §2 of this ADR).
- Carry a `CriticEvidence` payload (`critic_id` identifies the
  judge — heuristic name or model id; `score`/`threshold` in the
  critic's native range).
- Set `severity = SEVERITY_FAIL` when the score crosses threshold
  with sustained debounce; `SEVERITY_WARN` for transient dips
  (which the reasoner buffers without preempting).
- Stamp the OTel `trace_id` from the active `rskill.execute` parent
  span when judging a running skill (per §6 — `trace_id` is the join
  key). When judging absence-of-progress (no active skill), leave
  `trace_id = ""`.

The producer node itself is out of scope for this PR. A future
hand-rolled watchdog (e.g. "no `WorldState` delta beyond ε over a
window") is the easy first version; a SARM-class model loaded as an
rSkill `role: s2-critic` is the long-term answer.

**5. Topic rename `/openral/failure/skill` → `/openral/failure/rskill`.**
The original §1 named the rSkill-execution failure topic
`/openral/failure/skill`, but the carried field on `FailureTrigger`
has always been `rskill_id` and the package format is `rskill.yaml`.
The original `skill` suffix was the lone outlier. Renamed for
consistency with `rskill_id`, `RSkillManifest`, and the `rSkill` class.
`FailureSource.SKILL = "skill"` becomes `FailureSource.RSKILL = "rskill"`.
Hard wire-break per CLAUDE.md §1.6 (pre-publish; no migrator). No
bag-recorded producers existed for the original topic; live consumers
must rebuild.

The §1 topic table entry, the §4 reasoner contract reference, and the
F3 publisher helper documentation update in place — outside this
amendment's own historical narrative, `git grep /openral/failure/skill`
returns no hits after this amendment lands.

**6. CLAUDE.md §6.2 amendment (lands with this PR).** Change "**S2**
slow reasoning (5–10 Hz)" → "**S2** slow reasoning (event-driven; ~0.2
Hz heartbeat)". The Reasoner is no longer pinned to a periodic-LLM
budget.

**7. Observability.** `reasoner.tick` spans gain
`reasoner.suppressed_reason="heartbeat_idle"` when the new gate fires.
Suppressed-tick counts are visible on the dashboard via the existing
`reasoner.suppressed_reason` attribute (ADR-0017 / F7) so the operator
can verify "LLM call rate dropped" empirically.
