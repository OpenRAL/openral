# openral_rskill_ros

> **`rskill_runner_node` lifecycle node + `ExecuteRskill`
> action server.**

This package owns Layer 3 (rSkill) of the ROS 2 reasoner + supervisor
graph. One node per robot.

## Layer

CLAUDE.md §6.1 Layer 3 (rSkill). The runtime path is the in-process
[`openral_runner.DeployRunner`](../../python/runner/src/openral_runner/deploy_runner.py);
this package is the ROS-side surface that exposes a typed action
goal to external clients (CLI, reasoner, dashboard) and routes
chunks to the safety boundary.

## Topic surface (locked)

| Direction | Topic / Service / Action | Type |
|---|---|---|
| pub | `/openral/candidate_action` | `openral_msgs/ActionChunk` (via `ROSPublishingHAL`) |
| pub | `/diagnostics` | `diagnostic_msgs/DiagnosticArray` (1 Hz) |
| sub | `/openral/estop` | `std_msgs/Empty` (defense in depth alongside HAL) |
| sub | `/openral/safety_status` | `openral_msgs/SafetyStatus` (latched; RELIABLE · TRANSIENT_LOCAL · KL=1) |
| action | `/openral/execute_rskill` | `openral_msgs/action/ExecuteRskill` |

`/openral/safety_status` (ADR-0096) is read-only and feeds the existing
`safety_abort_getter` seam handed to `ROSPublishingHAL`: when an
apply-wait blocks, the reported reason names the actual fault
(`kind_collision`, `drop_envelope_unconfigured`, …) plus the publisher's
`detail`, instead of collapsing every abort to `/openral/estop`. The
`/openral/estop` latch is unchanged and still reported alongside it —
belt and braces. Because the topic is latched, a runner that reconnects
mid-mission reads the current state immediately. Once a `SafetyStatus`
has been seen, one that goes silent for more than 3 s (three missed 1 Hz
liveness refreshes) is reported as *unknown, not safe* — hazard-log
HZ-0096-1 mitigation 2, failing toward "assume unsafe".

### The seam is read from every blocking wait, not just the apply-wait

`ROSPublishingHAL`'s apply-wait is only entered by an action with
`tick_group_size > 1`. A single-surface policy — SmolVLA, ACT, diffusion,
i.e. most of them — emits one `Action` per tick, so `send_action`
published and returned without ever consulting the seam, and a latched
or dead safety layer stayed invisible until the execution budget lapsed
and the goal aborted as `deadline_exceeded` / `FAILURE_DEADLINE_MISSED`.
The reasoner's ladder then read "too slow" and retried the same skill
into the same stopped safety layer.

`_raise_if_safety_aborted(where)` is now the one guard every blocking
wait on the dispatch path calls, all of them reading the same
`_safety_abort_reason()` the HAL polls:

| wait | what a stopped safety layer starves | used to report |
|---|---|---|
| `ROSPublishingHAL._wait_for_group_applied` | `/openral/action_applied` goes silent | `action group tick N was not applied within 5.0 s` |
| rollout loop, before each inference tick | every chunk dropped, nothing to notice | nothing, until `deadline_exceeded` |
| `_wait_for_post_reset_joint_state` | a latched HAL stops publishing `/joint_states` | a `post_reset_joint_state_timeout` warning, then the policy started anyway |
| `_run_approach_skill`, per MoveIt waypoint | every replayed waypoint dropped | approach "succeeded"; policy started from a pose the arm never reached |

All of them now raise `ROSEStopRequested` naming the reason and the wait,
and every one of those raises lands on a branch that stamps
`failure_reason="safety_estop:…"` + `failure_kind=FAILURE_SAFETY_ESTOP`.
Note the MoveIt approach in particular: `ROSEStopRequested` is a
`ROSError`, so without an explicit re-raise it fell into
`_dispatch_moveit_approach`'s planning branch and the goal reported
`FAILURE_PLANNING_ERROR` — the ladder replanning around an "unreachable"
pose while the kernel was in fact latched.

**Not instrumented, deliberately.** The `ResetToPose` service call's 1 s
discovery wait and 5 s response wait: no in-tree HAL withholds a
`ResetToPose` response while latched (`ManifestHALLifecycleNode` answers
it regardless), so nothing in this repo can starve them, and the abort
surfaces one bounded step later at the post-reset joint-state wait.
`_drain_and_idle_hold`'s fixed 100 ms sleep and `_pace_tick`'s period
sleep wait on the clock, not on the safety layer. `ROSActionRskill.
_poll_future` (a wrapped `ros_action`/`ros_service` skill blocking on its
server's result — Nav2 will never arrive if the robot cannot move) **can**
be starved, but instrumenting it needs a new injected seam across the
rSkill layer boundary and a live surface with a real wrapped action
server; the per-waypoint guard above bounds the exposure to one blocking
`step()`.

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

* `launch/sim_e2e.launch.py` — the robot-agnostic
  graph: `runtime_node` (composed `world_state` + `skill_runner`) +
  C++ `safety_kernel_node` + reasoner + prompt router + HAL. Every
  robot-specific bit is a launch argument resolved at startup by an
  `OpaqueFunction`: `robot_yaml`, `envelope_file`, `hal_package`,
  `hal_executable`, `hal_node_name`, `hal_params_file`.

  In practice you don't invoke this launch directly — use
  `openral deploy sim --config <SceneEnvironment.yaml>` (see
  `python/cli/src/openral_cli/deploy_sim.py`). The CLI:
  1. resolves the robot via the SceneEnvironment's `robot_id` (or
     `--robot` override) → `_ROBOT_HAL_REGISTRY` for HAL package/exec
     + the set of robot manifest names this HAL accepts;
  2. validates `robots/<robot_id>/robot.yaml` via
     `RobotDescription.validate_for_e2e_pipeline()` and asserts the
     manifest's `name` is in the HAL's `supported_robot_names`;
  3. shells `ros2 launch openral_rskill_ros sim_e2e.launch.py …`.

  The launch's `OpaqueFunction` then loads `robot.yaml`, calls
  `openral_safety.envelope_loader.compute_intersection(robot, None)`,
  and forwards each `EnvelopeIntersection` field as a ROS parameter
  on the C++ safety_kernel node. **No envelope YAML file is written
  or read** — the C++ safety kernel grew a parameter-based loader
  alongside the legacy `envelope_file:=PATH` path
  (kept for HIL safety tests + `kernel_only.launch.py`).

  Direct invocation (for debugging the launch itself):

  ```bash
  ros2 launch openral_rskill_ros sim_e2e.launch.py \
      robot_yaml:=$PWD/robots/openarm/robot.yaml \
      hal_package:=openral_hal_openarm \
      hal_executable:=lifecycle_node.py \
      hal_node_name:=openral_hal_openarm \
      hal_params_file:=/tmp/openral-hal-params-openarm.yaml \
      reset_to_pose_service:=/openral/openarm/reset_to_pose
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

1. **Install-time** — `ral skill install` refuses non-commercial weights
   in a commercial deployment.
2. **Goal-acceptance** — `rskill_runner_node` re-checks the
   `RSkillLicensePosture` against the
   `OPENRAL_COMMERCIAL_DEPLOYMENT` env var. The same skill cannot
   reach a commercial deployment via a CLI bypass.

## Tests

`test/test_rskill_runner_node.py` is a real `launch_testing`-equivalent
integration test (CLAUDE.md §1.11 / §5.4: no mocks). It composes the
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
   atomic action group to be applied is still reported as
   `safety_estop:…` (naming the typed fault from
   `/openral/safety_status`, `/openral/estop`, and the unapplied tick),
   not as the `ROSPublishingHAL: action group tick N was not applied
   within 5.0 s` timeout that a silenced `/openral/action_applied`
   produces on its own. The real `SafetyPassthroughNode` decides the
   violation and fires the estop itself in that test.
4b. The same is true of the waits that carry no apply-ack. With the real
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
free-text `failure_reason`. It is what a machine consumer — the
reasoner's replanning ladder — classifies on, so it never substring-matches
operator prose. The field is **additive** (CLAUDE.md §1.6): a consumer
reading only `success` / `failure_reason` is untouched, and a Result from a
producer that predates the field decodes as `FAILURE_NONE`. A consumer
must therefore read `failure_kind == FAILURE_NONE` on a failed result as
"unknown — fall back to the string", never as "succeeded".

| Kind | Value | Raised by / set on |
| --- | --- | --- |
| `FAILURE_NONE` | 0 | Success (also the pre-field default) |
| `FAILURE_CONFIG_ERROR` | 1 | `ROSConfigError` — bad manifest, non-ACTIVE resolver result, unusable MoveIt approach manifest |
| `FAILURE_CAPABILITY_MISMATCH` | 2 | `ROSCapabilityMismatch` — embodiment-tag gate |
| `FAILURE_RUNTIME_ERROR` | 3 | `ROSRuntimeError` (incl. `ROSQuantizationError` / `ROSGPUMemoryError`), plus a labelled torch OOM / dtype failure |
| `FAILURE_SAFETY_ESTOP` | 4 | `ROSEStopRequested` |
| `FAILURE_PERCEPTION_STALE` | 5 | `ROSPerceptionStale` |
| `FAILURE_PLANNING_ERROR` | 6 | `ROSPlanningError` — the MoveIt starting-pose approach |
| `FAILURE_DEADLINE_MISSED` | 7 | The execution budget lapsed; `ROSDeadlineMissed` |
| `FAILURE_CANCELLED` | 8 | Cancel honoured (no exception) |
| `FAILURE_UNKNOWN` | 255 | A non-`ROSError` escaped the OpenRAL exception surface |

`ROSSafetyViolation` other than `ROSEStopRequested` has no kind by design:
it is re-raised to the safety supervisor, never folded into a Result
(CLAUDE.md §1.1).

Production deployments override the skill resolver via
`compose_so100_runtime(skill_resolver=...)`; the default resolver
calls `openral_rskill.rSkill.from_pretrained` (HF Hub fetch).

## Related

- `python/runner/src/openral_runner/ros_publishing_hal.py` —
  `ROSPublishingHAL` HAL adapter that turns `Action` into
  `openral_msgs/ActionChunk` published on `/openral/candidate_action`.
- `packages/openral_safety/` — F5 chunk-rate safety boundary
  (`/openral/candidate_action → /openral/safe_action`).
- `packages/world_state/` — the colocated lifecycle node sharing the
  aggregator.
