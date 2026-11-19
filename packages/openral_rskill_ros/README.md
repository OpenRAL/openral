# openral_rskill_ros

`rskill_runner_node` lifecycle node + `ExecuteRskill` action server: Layer 3
(rSkill) of the ROS 2 reasoner + supervisor graph. One node per robot.

## Layer

CLAUDE.md §3 Layer 3 (rSkill). The runtime path is the in-process
[`openral_runner.DeployRunner`](../../python/runner/src/openral_runner/deploy_runner.py);
this package is the ROS-side surface that exposes a typed action
goal to external clients (CLI, reasoner, dashboard) and routes
chunks to the safety boundary.

## Topic surface (locked)

| Direction | Topic / Service / Action | Type |
|---|---|---|
| pub | `/openral/candidate_action` | `openral_msgs/ActionChunk` (via `ROSPublishingHAL`) |
| pub | `/diagnostics` | `diagnostic_msgs/DiagnosticArray` (1 Hz) |
| pub | `/openral/place_declaration` | `openral_msgs/PlaceDeclaration` (RELIABLE · TRANSIENT_LOCAL) |
| pub | `/openral/grasp_declaration` | `openral_msgs/GraspDeclaration` (RELIABLE · TRANSIENT_LOCAL) |
| pub | `/openral/reward/active_task` | `std_msgs/String` (executing instruction, for the reward monitor) |
| pub | `/openral/episode` | `openral_msgs/Episode` |
| sub | `/openral/estop` | `std_msgs/Empty` (defense in depth alongside HAL) |
| sub | `/openral/estop_cleared` | `std_msgs/Empty` (clears the runner latch after a reset) |
| sub | `/openral/safety_status` | `openral_msgs/SafetyStatus` (latched; RELIABLE · TRANSIENT_LOCAL · KL=1) |
| action | `/openral/execute_rskill` | `openral_msgs/action/ExecuteRskill` |

`/openral/safety_status` (ADR-0096) is read-only and feeds the existing
`safety_abort_getter` seam handed to `ROSPublishingHAL`: when an
apply-wait blocks, the reported reason names the actual fault
(`kind_collision`, `drop_envelope_unconfigured`, …) plus the publisher's
`detail`, instead of collapsing every abort to `/openral/estop`. The
`/openral/estop` latch is unchanged and still reported alongside it.
Because the topic is latched, a runner that reconnects
mid-mission reads the current state immediately. Once a `SafetyStatus`
has been seen, one that goes silent for more than 3 s (three missed 1 Hz
liveness refreshes) is reported as *unknown, not safe* — hazard-log
HZ-0096-1 mitigation 2, failing toward "assume unsafe".

### The seam is read from every blocking wait

`ROSPublishingHAL` waits for `/openral/action_applied` after every active single-slot tick
and after the final slot of an atomic group. The same bounded wait therefore catches a dead
HAL for SmolVLA, ACT, diffusion, and multi-surface policies. The loop-level guard remains
necessary before inference and for pre-goal waits that have no tick id to acknowledge.

`_raise_if_safety_aborted(where)` is the one guard every blocking wait on the
dispatch path calls, all reading the same `_safety_abort_reason()` the HAL polls:

| wait | what a stopped safety layer starves | used to report |
|---|---|---|
| `ROSPublishingHAL._wait_for_group_applied` | `/openral/action_applied` goes silent | `action tick N…` (single) or `action group tick N…` |
| rollout loop, before each inference tick | every chunk dropped, nothing to notice | nothing, until `deadline_exceeded` |
| `_wait_for_starting_pose` | a latched HAL stops publishing `/joint_states` | a timeout warning, then the policy started anyway |
| `_run_approach_skill`, per MoveIt waypoint | every replayed waypoint dropped | approach "succeeded"; policy started from a pose the arm never reached |

All of them raise `ROSEStopRequested` naming the reason and the wait, and every
such raise lands on a branch that stamps `failure_reason="safety_estop:…"` +
`failure_kind=FAILURE_SAFETY_ESTOP`. The MoveIt approach needs an explicit
re-raise: `ROSEStopRequested` is a `ROSError`, so without it it fell into
`_dispatch_moveit_approach`'s planning branch and the goal reported
`FAILURE_PLANNING_ERROR` — the ladder replanning around an "unreachable"
pose while the kernel was in fact latched.

### Execution budgets run on the clock the robot moves on

A goal's budget (`deadline_s`, else the manifest's `latency_budget.max_execution_s`, else
45 s) is measured by `openral_rskill.execution_budget.ExecutionBudget`. Under
`openral deploy sim` the graph runs with `use_sim_time=true` and Nav2 / MoveIt / the
controllers time themselves in sim seconds, so the budget does too — measured on
`time.monotonic()` it fired while a slow simulator was still mid-motion (a rendering
graph at 0.09x real time lost a 300 s Nav2 budget after 28 s of sim time; adapted from
DsslRobot/openral). `ROSActionRskill`'s result wait in result-only mode (Nav2: the server
is moving the robot) follows the same clock; goal acceptance, service responses and a
MoveIt plan (trajectory mode, CPU planning) stay on the wall clock. Giving up on a
result wait cancels the wrapped goal, and the wait raises `ROSDeadlineMissed`
(`FAILURE_DEADLINE_MISSED`, like the runner's own budget). With `use_sim_time=false`
(every real robot) budgets are `time.monotonic()` exactly as before — never the
NTP-stepped ROS system clock; `use_sim_time` is re-read per goal, not cached.

Two wall-clock backstops keep the goal bounded when `/clock` stops (sim pause, HAL
crash, e-stop) or crawls; both abort with `FAILURE_DEADLINE_MISSED`. They are read
from this node by `openral_rskill.execution_budget.node_execution_budget` for the
runner's goals and for every wrapped skill's result wait, cannot be disabled (a
non-positive / non-finite value fails `on_configure`), and a NaN `deadline_s` on a
goal is rejected as `FAILURE_CONFIG_ERROR` rather than never lapsing:

| ROS parameter | default | fires as |
|---|---|---|
| `graph_clock_stall_s` | 10 s | `failure_reason="clock_stalled: graph clock has not advanced for N s of wall time …"` (or `never started` when no `/clock` ever arrived — a sim node reads 0 until then, and the budget does not start counting before its first message). A backward `/clock` jump (sim reset) re-anchors the budget and keeps the elapsed time banked so far. |
| `execution_wall_cap_factor` | 20 | `failure_reason="deadline_exceeded: wall elapsed=… exceeds 20x budget=…"` — a budget survives intact down to 1/20 = 0.05x real time (the slowest measured rendering graph, 0.09x, needs 11.1x); a frozen clock is caught by the stall guard long before. |

Every miss is logged (`rskill_runner.deadline_exceeded` / `rskill_runner.clock_stalled`)
and recorded as the `deadline_missed` span event with `miss.kind`.

The goal's grasp declaration is a second deadline. The kernel drops its contact exemption at
`stamp_ns + timeout_s`, and the declaration is armed before the start-pose move, so it can
expire inside the goal's own budget. A goal still running then has its fingers in contact
with nothing exempting them, and it ends as an unexplained world-collision e-stop (pro#51:
six of eight restock timeouts). The runner ends the goal
`DECLARATION_EXPIRY_MARGIN_S` (2 s) before that expiry instead: `failure_reason=
"deadline_exceeded: grasp declaration '<target>' expires (armed … ago, timeout_s=…)"`,
`TimeoutEvidence.operation="rskill.execute:grasp_declaration:deadline_exceeded"`. A wrapped
`ros_action` goal checks it too, in the `stop_check` it polls while blocked. It only ever ends
a goal earlier.

`_drain_and_idle_hold`'s fixed 100 ms sleep and `_pace_tick`'s period sleep wait on the
clock, not the safety layer. `ROSActionRskill._poll_future` (a wrapped `ros_action`/
`ros_service` skill blocking on its server's result) reads it through the `stop_check` seam
the runner already injects: a latched `/openral/estop` or `/openral/safety_status` (or a
silent status publisher, the same `_safety_abort_reason`) makes `stop_check` raise
`ROSEStopRequested` within one 20 ms poll. An action skill then cancels the wrapped goal and
waits up to 5 s each for the cancel response and the terminal result, and re-raises the stop,
so the goal ends `FAILURE_SAFETY_ESTOP` whatever the latch reads by then (an operator reset
during the handshake does not turn it into a deadline miss). A server that rejects the cancel
or never confirms it is named in the reason ("may still be running"), and the runner then
**rejects every new goal**, through an e-stop reset too, until that wrapped goal's result
arrives. A service skill cannot be cancelled: the stop ends the goal and says the call "may
still be running". This matters most where the wrapped server drives the robot itself: in a
standalone deploy the HAL already drops Nav2's `/cmd_vel` once latched, but the Nav2 goal
stayed alive and resumed after a reset. A latched stop outranks a cancel and a completion:
a step that completes, or a goal cancelled, while a stop is latched ends `FAILURE_SAFETY_ESTOP`
("goal ended (…) while a safety stop was latched"), never as a success.

## Station mode: perception per skill on a mobile base

On a mobile base with object tracking on, `deploy_e2e` launches `openral_segmenter`
configured but **inactive** (`station_perception_nodes`) and the object tracker ACTIVE with
its station perception off (`station_perception_service` =
`/openral/object_tracker/station_perception`); everywhere else (every fixed-base cell,
tracking off) both parameters are empty and the nodes autostart as before. Per goal the runner
reads the skill's manifest (`station_perception_wanted`, from `actuators_required`, not an
LLM decision): a skill that actuates an arm / gripper (pi0.5, MoveIt) gets the segmenter
activated and the tracker's station perception switched on before its declarations are
published (the tracker's search timers start when a declaration arrives); a base-only one
(`body_twist` only: Nav2, a VLN policy) gets them off before its goal is sent; a skill that
actuates nothing, and the end of a goal, leave them as they are, so a pick and the place
after it at one station share one segmenter load. Each transition is logged
(`rskill_runner.station_perception*`); an activation that fails refuses the goal with
`ROSPerceptionStale` naming the node or service, a deactivation that fails is only logged (a
crashed segmenter must not stop every Nav2 goal).

Why the tracker is switched, not deactivated: Nav2's `/cmd_vel` reaches the base through the
HAL's `MobileBaseBridge` (`_send_action_traced`, never `/openral/candidate_action`), so it
never meets the kernel; but a base *policy*'s `BODY_TWIST` chunk does, and the kernel refuses
every candidate once the attachment stream goes stale. Off, the tracker keeps that stream fresh
with the hands' held payloads only (no declaration, no region, no tracked key: nothing the
kernel can read as an allowance), so the base can drive and a payload in a hand stays known
(the kernel, and Nav2's payload scan filter, which fails open on a stale set). The reasoner's
`vram_lifecycle_peers` (detector / locators) is a disjoint set.

**Object grounding is for manipulation goals only.** With `object_tracking` `tracking` or
`on` (station mode or not) the same manifest predicate decides what a goal declares: a
base-only skill arms no grasp or place declaration, so no standing `object_queries_json`
query rides it and there is no grounding wait — it runs as with tracking off
(`rskill_runner.declarations_skipped why=base_only`). A skill that may grasp (a `vla`, or
one whose `actuators_required` names a gripper / dex-hand mode; not a MoveIt look-at, joint or
end-effector-pose motion) whose grasp declaration (the goal's own, else
`grasp_declaration_json`, else the approach one; plus the standing queries) carries no
`target_queries`, or that has none in force at all, is refused before any declaration is
published and before anything moves, with `ROSReasonerInvalidPlan` (`FAILURE_PLANNING_ERROR`): the
reasoner must re-send the call with `grasp_target` naming the object. Without a query the
tracker never tracks the object and the kernel checks the held payload as an untracked
stand-in (Isaac shelf aisle, 2026-11-16). No `place_target` is required: under `tracking`, and under `on`
when the deploy has no DESTINATION standing query (a reasoner-driven goal; with one, `on`
places on the tracked destination and names no zone), the runner names the place zone (or
the station's zone candidates) from the prompt. The tracker resolves a `zone:` target in any
mode (`TrackerSession.place_region`), and the kernel ingests that place region from the
attachment snapshot whether or not an exemption provider is loaded (the provider seam has no
place-region hook). A goal's
own grasp declaration is completed from the scene's (`merge_grasp_declarations`, logged as
`rskill_runner.grasp_declaration_merged` with each field's source): the goal keeps its
`target_id`, queries and any seeded `search_box`; the scene fills `search_box`,
`workspace_box` and `contact_links` the goal leaves unset, only when the goal names the
scene's hand (another hand: no scene box or lifetime, a warning, and a `(note: ...)` naming
both hands appended to that goal's failure reason when the hand can explain it — a safety
stop, a target not grounded or not found — so the reasoner corrects the hand on its retry;
the reasoner takes the scene's hand when the LLM names none, `grasp_contact_links`). A scene
declaration that names its own object (TARGET `target_queries`) lends its boxes only to a
goal naming that label. The deploy's standing CONTEXT queries ride a goal that names its own
target.
A merged non-`sim:` search box seeds the TARGET query, and under `tracking` the tracker counts only a track centred in the
declared search box as the goal's target (AMR aisle, 2026-11-16: a shelf carton grounded
"box" while the one to grasp was in the tote). `timeout_s` is the largest of the goal's, the
scene's and the goal's execution budget + 10 s, capped at 120 s (the reasoner's
`patience_s=0` declaration got 10 s and ended a 68 s skill 8 s in); the budget counts from
the run, so a goal declaration the pre-execution waits ate into is republished (same key)
just before the policy starts. Order per goal: refusals, station perception on, declarations
armed, grounding wait, skill load, starting pose, policy.

Place zones at several stations: when the prompt matches the same slot declared at several
map / odom-frame stations (`s<k>_<slot>` sharing an alias), the goal-scope place declaration
names all of them (`zone_candidates_target_id`, `zone:<a>|<b>|...`) and the tracker keeps the
one in reach at the base's pose; one match is named as before, and a fixed-base scene is
unchanged. A zone named by its own name (`s2_lower_slot_1`) wins over the shared alias; a
prompt naming two slots, or several base-frame zones, names none
(`place_zones_for_prompt`).

Starting pose on an Isaac twin: the OpenAMRobot 2.0 and OpenArm Isaac scenes run their arms on
position gains without gravity feed-forward, so joint4 settles ~0.065 rad short of a stocking
starting pose. `deploy_e2e` sets the runner's `starting_pose_tolerance` to 0.08 rad for those
two robots on an Isaac Sim deploy only (`_runner_starting_pose_tolerance`); real hardware and
MuJoCo keep the manifest's 0.05. The param widens the arrival check only: the ramp is skipped
only when every joint is within the manifest's tolerance.

## Composition (one shared `WorldStateAggregator`)

Per the single-aggregator contract the world_state node is the **only** subscriber of
`/joint_states`. The compose factory in this package builds a single
`WorldStateAggregator`, hands the same reference to a colocated
`_WorldStateLifecycleNode`, and lets `RskillRunnerNode` call
`aggregator.snapshot()` in-process — no ROS topic boundary between
the aggregator and the skill.

```python
from openral_rskill_ros import compose_so100_runtime

runtime = compose_so100_runtime(robot_name="so100")
# runtime.aggregator is runtime.world_state_node._aggregator
# runtime.aggregator is runtime.rskill_runner_node._aggregator
```

One generic launch file ships with this package:

* `launch/deploy_e2e.launch.py` — the robot-agnostic
  graph: `runtime_node` (composed `world_state` + `skill_runner`) +
  C++ `safety_kernel_node` + reasoner + prompt router + HAL. Every
  robot-specific bit is a launch argument resolved at startup by an
  `OpaqueFunction`: `robot_yaml`, `envelope_file`, `hal_package`,
  `hal_executable`, `hal_node_name`, `hal_params_file`.

  Normally you use
  `openral deploy sim --config <DeployScene.yaml>` (see
  `python/cli/src/openral_cli/deploy_sim.py`). The CLI:
  1. resolves the robot via the DeployScene's `robot_id` (or
     `--robot` override) → `$OPENRAL_ROBOTS_DIR/<id>/robot.yaml` or
     `robots/<id>/robot.yaml`, and derives the HAL package/exec from it
     (`_derive_hal_spec`: always the one generic `openral_hal_node`
     package, run under the node name `openral_hal_<id>`);
  2. validates the manifest via
     `RobotDescription.validate_for_e2e_pipeline()`;
  3. runs `ros2 launch openral_rskill_ros deploy_e2e.launch.py …`.

  The launch's `OpaqueFunction` then loads `robot.yaml`, calls
  `openral_safety.envelope_loader.compute_intersection(robot, None)`,
  and forwards each `EnvelopeIntersection` field as a ROS parameter
  on the C++ safety_kernel node. **No envelope YAML file is written
  or read**; the kernel's legacy `envelope_file:=PATH` loader is kept
  for HIL safety tests + `kernel_only.launch.py`.

  Direct invocation (for debugging the launch itself):

  ```bash
  ros2 launch openral_rskill_ros deploy_e2e.launch.py \
      robot_yaml:=$PWD/robots/openarm/robot.yaml \
      hal_package:=openral_hal_node \
      hal_executable:=lifecycle_node.py \
      hal_node_name:=openral_hal_openarm \
      hal_params_file:=/tmp/openral-hal-params-openarm.yaml
  ```

The launch keeps each piece in its own OS process — CLAUDE.md §1.5
forbids collapsing the safety boundary into the runner. A
composable-node container that runs the compose factory inside a
single OS process is a small follow-up — see the single-aggregator
contract for the constraint and the integration test
(`test/test_rskill_runner_node.py::test_compose_factory_shares_one_aggregator`)
for the assertion that the production path satisfies it.

## License gating

The rSkill lifecycle-node contract mandates two gates:

1. **Install-time** — `openral rskill install` refuses non-commercial weights
   in a commercial deployment.
2. **Goal-acceptance** — `rskill_runner_node` re-checks the
   `RSkillLicensePosture` against the
   `OPENRAL_COMMERCIAL_DEPLOYMENT` env var. The same skill cannot
   reach a commercial deployment via a CLI bypass.

## Tests

`test/test_rskill_runner_node.py` is a real `launch_testing`-equivalent
integration test (CLAUDE.md §1.11 / §2: no mocks). It composes the
runtime via `compose_so100_runtime`, brings up a real
`SafetyPassthroughNode`, and asserts:

1. Single-aggregator contract (identity check).
2. End-to-end `ExecuteRskill` goal → `/openral/candidate_action` →
   `safety_node` → `/openral/safe_action` round trip with the right
   `rskill_id` / `flat` / `n_dof` fields.
3. `/openral/estop` aborts the in-flight goal with
   `failure_reason="safety_estop:…"` and
   `failure_kind=FAILURE_SAFETY_ESTOP`.
4. A safety latch that lands **while the HAL is blocked** waiting for an
   active action tick to be applied is still reported as
   `safety_estop:…` (naming the typed fault from
   `/openral/safety_status`, `/openral/estop`, and the unapplied tick),
   not as the `ROSPublishingHAL: action tick N…` / `action group tick N…`
   timeout that a silenced `/openral/action_applied`
   produces on its own. The real `SafetyPassthroughNode` decides the
   violation and fires the estop itself in that test.
4b. The same is true of pre-goal waits that carry no apply-ack. With the real
   `SafetyPassthroughNode` taken down mid-goal by its own lifecycle
   `deactivate` — no `/openral/estop` published, so the estop latch stays
   `False` — a single-slot policy aborts as
   `safety_estop:…/openral/safety_status:stale…while about to dispatch
   inference tick N` instead of running out its budget, and a goal
   dispatched after the supervisor has gone aborts naming the post-reset
   joint-state wait rather than starting the policy.
5. Every other terminal branch carries the matching typed
   `ExecuteRskill.Result.failure_kind` — capability mismatch, config
   error, a `ROSRuntimeError` / `ROSPerceptionStale` /
   `ROSPlanningError` out of `skill.step`, a raw non-`ROSError` escape
   (`FAILURE_UNKNOWN`), the lapsed execution budget
   (`FAILURE_DEADLINE_MISSED`), and `FAILURE_NONE` on success.

`test/test_rskill_runner_failure_reason.py` covers the helpers directly:
the colcon-generated `FAILURE_*` constant set, `_failure_kind_for_exception`
(the CLAUDE.md §5 hierarchy → uint8 map every failure branch calls), and
`_classify_runtime_failure` / `_finalize_goal`.

### `failure_kind` — the typed dispatch outcome

`ExecuteRskill.Result` carries `failure_kind` (uint8) alongside the
free-text `failure_reason`. Machine consumers (the reasoner's replanning
ladder) classify on it, never by substring-matching operator prose. The field is **additive** (CLAUDE.md §1.6): a consumer
reading only `success` / `failure_reason` is untouched, and a Result from a
producer that predates the field decodes as `FAILURE_NONE`. A consumer
must therefore read `failure_kind == FAILURE_NONE` on a failed result as
"unknown — fall back to the string", never as "succeeded".

| Kind | Value | Raised by / set on |
| --- | --- | --- |
| `FAILURE_NONE` | 0 | Success (also the pre-field default) |
| `FAILURE_CONFIG_ERROR` | 1 | `ROSConfigError` — bad manifest, non-ACTIVE resolver result, unusable MoveIt approach manifest |
| `FAILURE_CAPABILITY_MISMATCH` | 2 | `ROSCapabilityMismatch` — a dispatch gate (`check_manifest_gates`): not an ExecuteRskill kind, an actuating skill tagged `any`, embodiment, capability flags, sensors, gripper encoding; or a manifest-less skill not declaring `role: s1` |
| `FAILURE_RUNTIME_ERROR` | 3 | `ROSRuntimeError` (incl. `ROSQuantizationError` / `ROSGPUMemoryError`), plus a labelled torch OOM / dtype failure; `starting_pose not reached within … s: (joint, target - measured, tolerance) [...]` when a fresh joint state measured after the starting-pose move still misses it |
| `FAILURE_SAFETY_ESTOP` | 4 | `ROSEStopRequested` |
| `FAILURE_PERCEPTION_STALE` | 5 | `ROSPerceptionStale`, incl. `no fresh joint state reached starting_pose within … s` (no measured joint state after the move, or the stream stalled) |
| `FAILURE_PLANNING_ERROR` | 6 | `ROSPlanningError` — the MoveIt starting-pose approach |
| `FAILURE_DEADLINE_MISSED` | 7 | The execution budget lapsed (`deadline_exceeded: …`) or the graph clock stalled under `use_sim_time` (`clock_stalled: …`); `ROSDeadlineMissed` |
| `FAILURE_CANCELLED` | 8 | Cancel honoured (no exception) |
| `FAILURE_UNKNOWN` | 255 | A non-`ROSError` escaped the OpenRAL exception surface |

`ROSSafetyViolation` other than `ROSEStopRequested` has no kind by design:
it is re-raised to the safety supervisor, never folded into a Result
(CLAUDE.md §1.1).

An action the kernel never applied (the apply-wait times out) is typed by
the kernel's drop: a stale or missing kernel input (`DROP_VOXEL_UNAVAILABLE`,
`DROP_ATTACHED_UNAVAILABLE`, `DROP_STATE_UNAVAILABLE`) raises
`ROSPerceptionStale` → `FAILURE_PERCEPTION_STALE`; any other drop, or none,
stays `FAILURE_RUNTIME_ERROR`.

### `evidence_json` — the typed detail behind the kind

`ExecuteRskill.Result.evidence_json` carries one
`openral_core.FailureEvidence` variant, so the reasoner reads *what* failed,
not just the kind (`_attach_failure_evidence`, stamped on every failed
result):

| Kind | `evidence_json` |
| --- | --- |
| `FAILURE_SAFETY_ESTOP` | The newest report on `/openral/failure/safety` during this goal: the kernel's (`CollisionEvidence`: which link, against what, how deep), the operator / hardware e-stop's (`HumanEvidence`) or the deadman watchdog's (`TimeoutEvidence`). Empty when no publisher explained the stop |
| `FAILURE_PERCEPTION_STALE` | The kernel's `PerceptionStaleEvidence` (which input, its age, the limit) when a stale input stalled the action |
| `FAILURE_DEADLINE_MISSED` | `TimeoutEvidence` (`operation="rskill.execute:<deadline_exceeded\|clock_stalled>"`, or `"rskill.execute:grasp_declaration:deadline_exceeded"` when the grasp declaration expires first; budget, elapsed) |
| `FAILURE_OBJECT_NOT_FOUND` | The tracker's exhausted `ObjectQueryEvidence` |
| `FAILURE_RUNTIME_ERROR` | `ControllerEvidence` when the cause is known: `state=gpu_out_of_memory` / `quantization_mismatch` (the skill's own typed fault), or the kernel drop that left the action unapplied (`drop_envelope_unconfigured`, `drop_voxel_overflow`, `drop_attached_overflow`), or `controller_name=rskill_runner`, `state=starting_pose_not_reached` with `detail` a JSON list of `{joint, residual, tolerance}` for each joint that missed the starting pose. Empty otherwise |
| anything else | Empty: `failure_reason` says it all |

A safety stop's report and its latch travel on two topics, so the latch can wake the goal first. The failed Result waits up to 1 s (`_SAFETY_EVIDENCE_WAIT_S`) for the report of the kind the latched `SafetyStatus` names, and logs `rskill_runner.safety_evidence_missing` when none arrives. The robot is already stopped; only the Result is delayed. The `/openral/failure/safety` subscription has its own callback group, so it never queues behind the stop's own `/openral/estop` callback.

Additive like `failure_kind`: an older producer's Result decodes as `""`.
The reasoner renders it on the skill's execution line and picks its
strategy hint from the kind and the evidence (`reflect_on_failure`).

### `task_outcome` — did the run do what it was asked

`success` says the goal ended without a fault. Whether the task was done is a second,
typed answer on every Result: `task_outcome` (`TASK_UNKNOWN` = 0, `TASK_SUCCEEDED`,
`TASK_FAILED`) and `task_outcome_json`, one `openral_core.TaskOutcome` (verdict, source,
reason, and the tracked objects it was read from). The same verdict is on the
`rskill.execute` span (`rskill.task_outcome`, `.source`, `.reason`) and in the
`rskill_runner.task_outcome` log line.

| Source | When | Verdict |
| --- | --- | --- |
| `success_condition` | The goal carries an `openral_core.SuccessCondition` (`success_condition_valid`), or the deploy scene declares one for the goal's prompt (`DeployScene.success_conditions` → `success_conditions_json`, matched by `success_condition_for_prompt`); the goal's wins | `succeeded` once `min_count` objects of `object_label` have rested released inside `region` for `settle_s`; `failed` when too few rest there and the tracker sees one held or elsewhere; `unknown` when there is no live tracked set, it is in another frame, too few objects of the label are tracked, one is released but not yet re-confirmed, or they have not rested long enough |
| `skill` | The skill raised `ROSRskillGoalSatisfied` (a wrapped action reached its terminal status), and no condition says `failed` | `succeeded` |
| `checker` | A completion check ended the goal through `/openral/rskill/complete`, and no condition says `failed` | `succeeded` (a condition perception could not decide is named in the reason) |
| `none` | None of the above (also a goal refused before its condition was read) | `unknown` |

The condition is judged by one `openral_core.success.SuccessJudge` per goal, fed every
set the tracker publishes on `/openral/tracked_objects` (its 2 Hz heartbeat included,
whatever the policy's tick rate). A goal is judged at the moment it ends: a deadline miss
with the object placed and one with it still in the gripper now read differently. `unknown`
is never reported as a success.

A condition with `end_on_success: true` also ends the run: each tick, while the world-state
snapshot holds a live tracked set, a settled condition ends the goal
(`rskill_runner.success_condition_met`, `success` true) instead of leaving it to its
budget, and that verdict is the one the Result reports. It is off by default because the
verdict is only as good as the tracker's tracks: on Isaac SO-101 PickOrange (q5090,
2026-11-19) the tracker held three confirmed `orange` tracks in the plate with one orange
there, two of them one object 3 cm apart. Tracks whose centre lies in another counted
track's box now count once, but a track of something that is not there still counts, and
ending a run on it would cut the policy short. Off, a wrong `succeeded` costs a label. A condition that does not validate refuses the goal
(`FAILURE_CONFIG_ERROR`); a goal refused later (a manifest gate, a grounding wait) is still
judged by its condition. It is advisory: it ends a goal or labels its result, commands
nothing, and the safety kernel never reads it.

The settle clock is the tracker's evidence, not the runner's patience: `settle_s` is
measured on the tracked sets' own stamps, a set seen twice counts once, and a missing set, a
stamp that goes backwards or a gap longer than `max_evidence_gap_s` restarts it. One set
alone never decides a success (`settle_s` > 0). A `RELEASED` track (its box is frozen at
the let-go pose until the tracker re-grounds it) does not count as resting.

A scene's conditions are keyed by prompt (case and whitespace aside), so only a goal with
that prompt is judged by them; a wrapped Nav2 move or another task is not, and is never
ended early by them (`rskill_runner.success_condition_unmatched` names the prompt). The
reasoner does not put a condition on its goals yet, so a goal it dispatches is judged only
when its prompt is one the scene declares.

Limits today: the region is a box in the tracker's grid frame (the robot base frame; a
map-frame region reads `unknown`), any track of the label counts (an object of that class
already in the region satisfies it, and so does a track the tracker holds for something
that is not there), and with object tracking off there is no evidence, so
every verdict is `unknown` (`openral deploy` says so at launch).

Additive like `failure_kind`: an older producer's Result decodes as `TASK_UNKNOWN` / `""`.

The default resolver is a `ManifestSkillResolver` (`make_default_skill_resolver`):
the runner resolves the goal's manifest (in-tree by name under the search paths,
else the HF Hub manifest alone), runs `check_manifest_gates` against the robot
before evicting the resident skill or loading anything, then builds — a Hub policy's
weights pinned to the commit whose manifest was gated. Tests override it via
`compose_so100_runtime(skill_resolver=...)`; a callable that is not a
`TwoPhaseSkillResolver` is gated only after it has built the skill.

## Related

- `python/runner/src/openral_runner/ros_publishing_hal.py` —
  `ROSPublishingHAL` HAL adapter that turns `Action` into
  `openral_msgs/ActionChunk` published on `/openral/candidate_action`.
- `packages/openral_safety/` — F5 chunk-rate safety boundary
  (`/openral/candidate_action → /openral/safe_action`).
- `packages/world_state/` — the colocated lifecycle node sharing the
  aggregator.
