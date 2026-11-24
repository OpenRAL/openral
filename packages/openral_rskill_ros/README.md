# openral_rskill_ros

`rskill_runner_node` (lifecycle node) and the `ExecuteRskill` action server: Layer 3
(rSkill) of the ROS 2 graph, one node per robot. It also ships `deploy_e2e.launch.py`, the
robot-agnostic launch that brings up the whole deploy graph.

The runtime path is the in-process
[`openral_runner.DeployRunner`](../../python/runner/src/openral_runner/deploy_runner.py).
This package is the ROS surface that accepts a typed goal from external clients (CLI,
reasoner, dashboard) and routes chunks to the safety boundary. Transition callbacks use
`log_lifecycle_errors`, and the node calls `refuse_unavailable_transitions`, so an
unavailable lifecycle transition is refused instead of killing the node.

## Run it

Normally through the CLI, which resolves the robot, validates it
(`RobotDescription.validate_for_e2e_pipeline()`) and launches the graph:

```bash
uv run openral deploy sim --config <DeployScene.yaml>
```

Direct launch (for debugging the launch itself):

```bash
ros2 launch openral_rskill_ros deploy_e2e.launch.py \
    robot_yaml:=$PWD/robots/openarm/robot.yaml \
    hal_package:=openral_hal_node \
    hal_executable:=lifecycle_node.py \
    hal_node_name:=openral_hal_openarm \
    hal_params_file:=/tmp/openral-hal-params-openarm.yaml
```

The launch starts, each in its own OS process (the safety boundary is never collapsed into
the runner): `runtime_node` (the composed `world_state` + `skill_runner`), the C++
`safety_kernel_node`, the reasoner, the prompt router and the HAL. Optional nodes (object
tracker, segmenter, octomap bridge, SLAM, Nav2, reward monitor, dashboard, Foxglove) are
switched by launch arguments. A scene that lists peer robots (`OPENRAL_SCENE_PEERS`, ADR-0137)
also gets a `peer_voxels` node; the launch fails if the kernel has no world-voxel check to
receive it.

The launch loads `robot.yaml`, calls `openral_safety.envelope_loader.compute_intersection`
and passes each field to the safety kernel as a ROS parameter. No envelope file is written;
the kernel's `envelope_file:=PATH` loader remains for HIL tests and `kernel_only.launch.py`.

Main launch arguments (see the `DeclareLaunchArgument` list for all of them):

| Group | Arguments |
|---|---|
| Robot / HAL | `robot_yaml`, `hal_package`, `hal_executable`, `hal_node_name`, `hal_params_file`, `hal_mode`, `hardware_estop_device`, `gripper_convention`, `reset_to_pose_service` |
| Scene | `deploy_config`, `workcell_json`, `static_geometry_file`, `map_path`, `clock_origin`, `initial_task_prompt` |
| Declarations | `grasp_declaration_json`, `place_declaration_json`, `place_zones_json`, `success_conditions_json`, `object_queries_json`, `nav_goals_json` |
| Object tracking | `object_tracking` (`off` / `tracking` / `on`), `object_tracking_cameras`, `grounding_wait_s`, `grounding_required`, `tracked_release_timeout_s` |
| Perception | `enable_octomap`, `enable_octomap_kernel_check`, `octomap_cloud_topic`, `world_voxel_deadline_s`, `max_octree_age_s`, `robot_self_filter_padding_m`, `enable_object_detector`, `enable_vision_attachment`, `enable_slam`, `enable_nav2` |
| Skills | `approach_skill_id`, `preload_rskill_id`, `preload_rskill_revision`, `preload_prompt`, `release_rskill_after_goal` |
| Reasoner / UI | `enable_reasoner`, `reasoner_model`, `reasoner_endpoint`, `enable_dashboard`, `dashboard_port`, `enable_foxglove`, `foxglove_port` |
| Reward / completion | `enable_reward_monitor`, `enable_completion_check`, `enable_critic` |

## Interfaces

| Direction | Topic / action / service | Type |
|---|---|---|
| action | `/openral/execute_rskill` | `openral_msgs/action/ExecuteRskill` |
| pub | `/openral/candidate_action` | `openral_msgs/ActionChunk` (via `ROSPublishingHAL`) |
| pub | `/openral/place_declaration`, `/openral/grasp_declaration` | `PlaceDeclaration`, `GraspDeclaration` (RELIABLE, TRANSIENT_LOCAL) |
| pub | `/openral/episode` | `openral_msgs/Episode` |
| pub | `/openral/reward/active_task` | `std_msgs/String` (instruction being executed) |
| pub | `/diagnostics` | `diagnostic_msgs/DiagnosticArray` (1 Hz) |
| sub | `/openral/estop`, `/openral/estop_cleared` | `std_msgs/Empty` (defense in depth beside the HAL; the second clears the runner latch) |
| sub | `/openral/safety_status` | `openral_msgs/SafetyStatus` (RELIABLE, TRANSIENT_LOCAL) |
| sub | `/openral/failure/safety` | typed safety failure report (feeds `evidence_json`) |
| sub | `/openral/tracked_objects` | `openral_msgs/TrackedObjectSet` (grounding wait, success judge) |
| sub | `/openral/failure/world_state` | `FailureTrigger` (tracker query escalations; only with tracking on) |
| srv | `/openral/rskill/hold` (`SetBool`), `/openral/rskill/complete` (`Trigger`) | hold the last action; end the goal as completed |

ROS parameters (declared by the node; the launch sets most from its arguments):
`rate_hz`, `action_applied_timeout_s`, `execution_wall_cap_factor`, `graph_clock_stall_s`,
`gripper_convention`, `joint_state_staleness_limit_s`, `starting_pose_max_delta_per_s`,
`starting_pose_tolerance`, `estop_topic`, `reset_to_pose_service`, `joint_states_topic`,
`approach_skill_id` / `approach_skill_revision`, `preload_rskill_id` / `_revision` /
`preload_prompt`, `release_rskill_after_goal`, `place_declaration_json`,
`grasp_declaration_json`, `success_conditions_json`, `grasp_approach_enabled`,
`place_approach_enabled`, `place_zones_json`, `object_tracking`, `object_queries_json`,
`grounding_wait_s`, `grounding_required`, `grounding_wait_wall_s` (60), `grounding_context_wait_s`
(2), `station_perception_nodes`, `station_perception_service`, `station_perception_timeout_s`
(120).

## Safety behaviour

`/openral/safety_status` feeds the `safety_abort_getter` seam handed to `ROSPublishingHAL`
(ADR-0096): a blocked apply-wait reports the actual fault (`kind_collision`,
`drop_envelope_unconfigured`, ...), not just `/openral/estop`. The topic is latched, so a
reconnecting runner reads the current state at once. Once a status has been seen, one silent
for more than 3 s is reported as unknown, not safe.

`_raise_if_safety_aborted(where)` is the one guard every blocking wait calls. It raises
`ROSEStopRequested`, which ends the goal as `FAILURE_SAFETY_ESTOP`.

| Wait | What a stopped safety layer starves |
|---|---|
| `ROSPublishingHAL._wait_for_group_applied` | `/openral/action_applied` goes silent |
| rollout loop, before each inference tick | every chunk dropped |
| `_wait_for_starting_pose` | a latched HAL stops publishing `/joint_states` |
| `_run_approach_skill`, per MoveIt waypoint | every replayed waypoint dropped |

A wrapped `ros_action` / `ros_service` skill blocked on its server polls the same
`stop_check`, so a latch stops it within one 20 ms poll. An action skill cancels the wrapped
goal and waits up to 5 s each for the cancel response and the result. If the server rejects
the cancel or never confirms it, the runner **rejects every new goal**, through an e-stop
reset too, until that goal's result arrives. A service skill cannot be cancelled: the goal
ends and the reason says the call "may still be running". A latched stop outranks a
completion or a cancel; such a goal ends `FAILURE_SAFETY_ESTOP`, never as a success.

## Execution budgets

A goal's budget is `deadline_s`, else the manifest's `latency_budget.max_execution_s`, else
45 s. `openral_rskill.execution_budget.ExecutionBudget` measures it on the clock the robot
moves on: sim time when `use_sim_time=true` (`openral deploy sim`), `time.monotonic()` on a
real robot. A NaN `deadline_s` is refused as `FAILURE_CONFIG_ERROR`.

Two wall-clock backstops keep a goal bounded when `/clock` stops or crawls. Both abort with
`FAILURE_DEADLINE_MISSED`, cannot be disabled (a non-positive or non-finite value fails
`on_configure`), and apply to wrapped skills too:

| Parameter | Default | Fires as |
|---|---|---|
| `graph_clock_stall_s` | 10 s | `clock_stalled: ...` (also when no `/clock` ever arrived) |
| `execution_wall_cap_factor` | 20 | `deadline_exceeded: wall elapsed=... exceeds 20x budget=...` |

The goal's grasp declaration is a second deadline: the kernel drops its contact exemption
at `stamp_ns + timeout_s`. The runner ends the goal `DECLARATION_EXPIRY_MARGIN_S` (2 s)
before that, as `deadline_exceeded: grasp declaration '<target>' expires ...`, rather than
let the fingers meet an unexplained world-collision stop.

## Declarations and station mode

Per goal the runner reads the skill manifest to decide what to declare. Order: refusals,
station perception on, declarations armed, grounding wait, skill load, starting pose, policy.

- **Manipulation goals only.** With `object_tracking` `tracking` or `on`, a base-only skill
  (`body_twist` only: Nav2, a VLN policy) arms no declaration and has no grounding wait. A
  skill that may grasp (a `vla`, or one actuating a gripper / dex hand; not a MoveIt look-at,
  joint or pose motion) whose grasp declaration has no `target_queries` is refused before
  anything moves (`ROSReasonerInvalidPlan`, `FAILURE_PLANNING_ERROR`); the reasoner must
  resend with `grasp_target`. No `place_target` is required: the runner names the place zone
  from the prompt.
- **Merge.** The goal's own grasp declaration is completed from the scene's
  (`merge_grasp_declarations`): the goal keeps `target_id`, queries and `search_box`; the
  scene fills `search_box`, `workspace_box` and `contact_links` the goal leaves unset, and only
  when the goal names the scene's hand. Declaration `timeout_s` is the largest of the goal's,
  the scene's and the execution budget + 10 s, capped at 120 s.
- **Several stations.** When a prompt matches one slot declared at several map-frame stations,
  the place declaration names all of them (`zone:<a>|<b>|...`) and the tracker keeps the one
  in reach (`place_zones_for_prompt`).
- **Station mode (mobile base, tracking on).** `deploy_e2e` launches `openral_segmenter`
  inactive and the tracker active with station perception off. A skill that actuates an arm
  or gripper activates the segmenter and the tracker's station perception
  (`station_perception_service`) before its declarations; a base-only skill switches them
  off. A failed activation refuses the goal with `ROSPerceptionStale`; a failed deactivation
  is only logged. The tracker stays active when off so its held-payload stream keeps the
  kernel's attachment input fresh.
- **Isaac twin starting pose.** For the OpenAMRobot 2.0 and OpenArm Isaac scenes the launch
  sets `starting_pose_tolerance` to 0.08 rad (`_runner_starting_pose_tolerance`) because the
  arm settles short of the pose without gravity feed-forward; real hardware and MuJoCo keep
  the manifest value.

## Results

`ExecuteRskill.Result` carries `failure_kind` (uint8), `failure_reason` (prose),
`evidence_json`, `task_outcome` and `task_outcome_json`. Consumers classify on the typed
fields, never on prose. `FAILURE_NONE` on a failed result means "unknown, read the string".

| Kind | Value | Set by |
|---|---|---|
| `FAILURE_NONE` | 0 | success (or a producer without the field) |
| `FAILURE_CONFIG_ERROR` | 1 | `ROSConfigError`: bad manifest, unusable approach manifest, bad success condition |
| `FAILURE_CAPABILITY_MISMATCH` | 2 | `ROSCapabilityMismatch`: a `check_manifest_gates` refusal (kind, `any` tag on an actuating skill, embodiment, capabilities, sensors, gripper encoding) |
| `FAILURE_RUNTIME_ERROR` | 3 | `ROSRuntimeError` (incl. quantization / GPU OOM); starting pose not reached after the move |
| `FAILURE_SAFETY_ESTOP` | 4 | `ROSEStopRequested` |
| `FAILURE_PERCEPTION_STALE` | 5 | `ROSPerceptionStale`, incl. no fresh joint state at the starting pose |
| `FAILURE_PLANNING_ERROR` | 6 | `ROSPlanningError`: MoveIt approach, invalid grasp target |
| `FAILURE_DEADLINE_MISSED` | 7 | budget lapsed, clock stalled, or declaration expired; `ROSDeadlineMissed` |
| `FAILURE_CANCELLED` | 8 | cancel honoured |
| `FAILURE_OBJECT_NOT_FOUND` | 9 | `ROSObjectNotInMemory`: the tracker exhausted the goal's object search |
| `FAILURE_UNKNOWN` | 255 | a non-`ROSError` escaped |

`ROSSafetyViolation` other than `ROSEStopRequested` has no kind: it goes to the safety
supervisor, never into a Result. An action the kernel never applied is typed by the kernel's
drop: a stale or missing input (`DROP_VOXEL_UNAVAILABLE`, `DROP_ATTACHED_UNAVAILABLE`,
`DROP_STATE_UNAVAILABLE`) gives `FAILURE_PERCEPTION_STALE`; any other drop stays
`FAILURE_RUNTIME_ERROR`.

### `evidence_json`

One `openral_core.FailureEvidence` variant per failed result (`_attach_failure_evidence`):

| Kind | Evidence |
|---|---|
| `FAILURE_SAFETY_ESTOP` | newest `/openral/failure/safety` report this goal: `CollisionEvidence`, `HumanEvidence` or `TimeoutEvidence`. Empty if no publisher explained the stop |
| `FAILURE_PERCEPTION_STALE` | the kernel's `PerceptionStaleEvidence` (input, age, limit) |
| `FAILURE_DEADLINE_MISSED` | `TimeoutEvidence` (`operation="rskill.execute:..."`, budget, elapsed) |
| `FAILURE_OBJECT_NOT_FOUND` | the exhausted `ObjectQueryEvidence` |
| `FAILURE_RUNTIME_ERROR` | `ControllerEvidence` when known (GPU OOM, quantization mismatch, a kernel drop, starting pose not reached with per-joint residuals) |

The safety report and the latch travel on two topics, so the Result waits up to 1 s
(`_SAFETY_EVIDENCE_WAIT_S`) for the report and logs `rskill_runner.safety_evidence_missing`
if none arrives. The robot is already stopped; only the Result is delayed.

### `task_outcome`

`success` says the goal ended without a fault. `task_outcome` (`TASK_UNKNOWN` 0,
`TASK_SUCCEEDED`, `TASK_FAILED`) says whether the task was done; `task_outcome_json` is one
`openral_core.TaskOutcome` (verdict, source, reason, the tracked objects read).

| Source | When | Verdict |
|---|---|---|
| `success_condition` | the goal carries a `SuccessCondition`, or the scene declares one for the goal's prompt (`DeployScene.success_conditions`); the goal's wins | `succeeded` once `min_count` objects of `object_label` rest released inside `region` for `settle_s`; `failed` when too few rest there and one is held or elsewhere; `unknown` otherwise |
| `skill` | the skill raised `ROSRskillGoalSatisfied`, and no condition says `failed` | `succeeded` |
| `checker` | a completion check ended the goal via `/openral/rskill/complete` | `succeeded` |
| `none` | none of the above | `unknown` |

One `SuccessJudge` per goal reads every set on `/openral/tracked_objects`. `settle_s` is
measured on the tracker's own stamps; a missing set, a backwards stamp or a gap beyond
`max_evidence_gap_s` restarts it. A condition with `end_on_success: true` also ends the run
when it settles; it is off by default because the verdict is only as good as the tracks. The
condition is advisory: it labels or ends a goal and the safety kernel never reads it.

Limits: the region is a box in the tracker's grid frame (a map-frame region reads
`unknown`); any track of the label counts; with object tracking off every verdict is
`unknown`. A scene's conditions are keyed by prompt, so a goal with another prompt is not
judged by them.

## Composition (one shared `WorldStateAggregator`)

`world_state` is the only subscriber of `/joint_states`. `compose_runtime` (and the
`compose_so100_runtime` wrapper) builds one `WorldStateAggregator`, hands it to a colocated
`_WorldStateLifecycleNode`, and lets `RskillRunnerNode` call `aggregator.snapshot()`
in-process. The `runtime_node` script is the entry point:
`ros2 run openral_rskill_ros runtime_node --ros-args -p robot_yaml:=/abs/path/robot.yaml`.

```python
from openral_rskill_ros import compose_so100_runtime

runtime = compose_so100_runtime(robot_name="so100")
# runtime.aggregator is runtime.world_state_node._aggregator
# runtime.aggregator is runtime.rskill_runner_node._aggregator
```

The default resolver is a `ManifestSkillResolver` (`make_default_skill_resolver`): it
resolves the goal's manifest (in-tree by name, else the HF Hub manifest alone), runs
`check_manifest_gates` before evicting the resident skill or loading weights, then builds,
pinning a Hub policy's weights to the commit whose manifest was gated. A custom resolver
that is not a `TwoPhaseSkillResolver` is gated only after it has built the skill.

## License gating

1. Install time: `uv run openral rskill install` refuses non-commercial weights in a
   commercial deployment.
2. Goal acceptance: the runner re-checks the manifest's `RSkillLicensePosture` against
   `OPENRAL_COMMERCIAL_DEPLOYMENT=1`, so a CLI bypass cannot reach a commercial deployment.

## Tests

`test/test_rskill_runner_node.py` composes the runtime via `compose_so100_runtime` with a
real `SafetyPassthroughNode`, no mocks. It checks the single-aggregator identity, the
`ExecuteRskill` to `/openral/candidate_action` to `/openral/safe_action` round trip, e-stop
aborts (including a latch while the HAL is blocked and a supervisor taken down mid-goal), and
a typed `failure_kind` for every terminal branch. `test/test_rskill_runner_failure_reason.py`
covers the `FAILURE_*` constants and the exception-to-kind map. The `test/test_deploy_e2e_*`
files check the launch graph.

## Related

- `python/runner/src/openral_runner/ros_publishing_hal.py`: `ROSPublishingHAL`, which turns
  an `Action` into an `ActionChunk` on `/openral/candidate_action`.
- `packages/openral_safety/`: the chunk-rate safety boundary
  (`/openral/candidate_action` to `/openral/safe_action`).
- `packages/world_state/`: the colocated lifecycle node sharing the aggregator.
