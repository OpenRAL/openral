# openral_rskill_ros

> **ADR-0018 F1 — `rskill_runner_node` lifecycle node + `ExecuteRskill`
> action server.**

This package owns Layer 3 (rSkill) of the ROS 2 graph mandated by
ADR-0018. One node per robot.

## Layer

CLAUDE.md §6.1 Layer 3 (rSkill). The runtime path is the in-process
[`openral_runner.HardwareRunner`](../../python/runner/src/openral_runner/hardware.py);
this package is the ROS-side surface that exposes a typed action
goal to external clients (CLI, reasoner, dashboard) and routes
chunks to the safety boundary.

## Topic surface (locked)

| Direction | Topic / Service / Action | Type |
|---|---|---|
| pub | `/openral/candidate_action` | `openral_msgs/ActionChunk` (via `ROSPublishingHAL`) |
| pub | `/diagnostics` | `diagnostic_msgs/DiagnosticArray` (1 Hz) |
| sub | `/openral/estop` | `std_msgs/Empty` (defense in depth alongside HAL) |
| action | `/openral/execute_rskill` | `openral_msgs/action/ExecuteRskill` |

## Composition (one shared `WorldStateAggregator`)

Per ADR-0018 §3 the world_state node is the **only** subscriber of
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

* `launch/sim_e2e.launch.py` — ADR-0018 F1+F4+F5+F10 robot-agnostic
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
  or read** — the C++ kernel grew a parameter-based loader in
  ADR-0020 PR-K alongside the legacy `envelope_file:=PATH` path
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
single OS process is a small follow-up — see ADR-0018 §3 for the
constraint and the integration test
(`test/test_rskill_runner_node.py::test_compose_factory_shares_one_aggregator`)
for the assertion that the production path satisfies it.

## License gating

ADR-0018 §F1 mandates two gates:

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

1. ADR-0018 §3 single-aggregator contract (identity check).
2. End-to-end `ExecuteRskill` goal → `/openral/candidate_action` →
   `safety_node` → `/openral/safe_action` round trip with the right
   `rskill_id` / `flat` / `n_dof` fields.
3. `/openral/estop` aborts the in-flight goal with
   `failure_reason="safety_estop:…"`.

Production deployments override the skill resolver via
`compose_so100_runtime(skill_resolver=...)`; the default resolver
calls `openral_rskill.rSkill.from_pretrained` (HF Hub fetch).

## Related

- ADR-0018 §F1 — normative spec.
- `python/runner/src/openral_runner/ros_publishing_hal.py` —
  `ROSPublishingHAL` HAL adapter that turns `Action` into
  `openral_msgs/ActionChunk` published on `/openral/candidate_action`.
- `packages/openral_safety/` — F5 chunk-rate safety boundary
  (`/openral/candidate_action → /openral/safe_action`).
- `packages/world_state/` — the colocated lifecycle node sharing the
  aggregator.
