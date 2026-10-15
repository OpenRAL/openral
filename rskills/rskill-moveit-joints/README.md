---
language:
- en
license: apache-2.0
pipeline_tag: robotics
tags:
- OpenRAL
- rskill
- ros2
- moveit
- franka_panda
- ur5e
- ur10e
- so100_follower
- openarm
- rizon4
- sawyer
- widowx
inference: false
---

# rskill-moveit-multi-joints

> **OpenRAL rSkill** — wraps the upstream `moveit_msgs/action/MoveGroup`
> action server as an OpenRAL rSkill so the Reasoner can dispatch
> collision-free joint-space motion planning through the same
> `ExecuteRskill` path used by VLA skills. No model weights — the manifest
> is the entire artefact. Renamed from `openral/rskill-moveit-plan-arm`
> as part of the `rskill-moveit-*` rename.

`kind: ros_action` (a discriminator on `RSkillManifest.kind`) selects the
[`ROSActionRskill`](../../python/rskill/src/openral_rskill/ros_action_rskill.py)
engine at resolve time, with `ros_integration.goal_builder: joint` selecting
the [`JointGoalRskill`](../../python/rskill/src/openral_rskill/joint_goal_rskill.py)
goal-lowering adapter. The adapter constructs an
`rclpy.action.ActionClient` on the host `RskillRunnerNode`, sends one goal
built from `ros_integration.default_goal_json`, awaits the result, and replays
the returned `trajectory_msgs/JointTrajectory` one waypoint per `step()` call
onto `/openral/candidate_action` so the C++ safety kernel still checks every
commanded position.

## What this skill does

Plans and executes a collision-free joint-space motion to a target joint
configuration via MoveIt's `MoveGroup` action. The goal is authored as a
`joint` block (`joint_names` + `positions`) — the clean, LLM-facing form
(target angles, not hand-written constraint dicts) — which `JointGoalRskill`
lowers into MoveGroup `joint_constraints` at configure time. The default goal
targets the Franka Panda home pose (`panda_arm` planning group). Other arm
embodiments need their own manifest copy with the correct planning-group name
and joint names.

| Field | Value |
| --- | --- |
| Actions | `reach` |
| Objects | _none_ — no per-object specialisation |
| Scenes  | _none_ — the wrapped planner does its own collision check against the live `/planning_scene` |
| Embodiment | `franka_panda` (default goal); other arm tags listed in the manifest for capability filtering |

## How it works

`ROSActionRskill` is a thin `rSkillBase` shim around an `ActionClient`, and
`goal_builder: joint` lowers the LLM-facing `joint` block into the MoveGroup
goal:

1. `_configure_impl` lazy-imports `moveit_msgs.action.MoveGroup`, opens
   an `ActionClient` on `/move_action` from the
   `RskillRunnerNode`-supplied node handle, and parses
   `ros_integration.default_goal_json` once. `JointGoalRskill` pops the
   `joint` block (`joint_names` + `positions` + tolerance) and lowers it into
   `request.goal_constraints[0].joint_constraints` — one
   `moveit_msgs/JointConstraint` per joint.
2. On the first `_step_impl(world_state)` call the adapter:
   - builds the `MoveGroup.Goal` from the lowered dict (via
     `rosidl_runtime_py.set_message_fields`),
   - sends it and polls the goal-accept + result futures while the host
     node's main rclpy spin keeps servicing callbacks,
   - extracts `result.planned_trajectory.joint_trajectory`, reorders its
     `joint_names` into the host `RobotDescription.joints` order (see
     [`build_joint_permutation_from_names`](../../python/rskill/src/openral_rskill/ros_action_rskill.py)),
   - returns waypoint 0 as a 1-row `Action(JOINT_POSITION, …)`.
3. Each subsequent `_step_impl` returns the next cached waypoint; after the
   last one it raises `ROSRskillGoalSatisfied`, which the runner catches to
   close the `ExecuteRskill` goal with `success=True`.

The LLM overrides only `joint.positions` (one entry per planning-group joint,
in the manifest's `joint_names` order); planner and tolerance defaults are
inherited from `default_goal_json`. `plan_only: true` so MoveGroup never drives
its own controllers — OpenRAL's per-waypoint replay is the only actuation path.

### Observation → action contract

| Direction | Key | Shape | Notes |
| --- | --- | --- | --- |
| in  | `world_state.joint_state` | `(n_dof,)` float | Read for logging only; the wrapped server consumes its own `/joint_states` subscription |
| out | per-waypoint `Action`     | `joint_targets=[[n_dof floats]]`, `horizon=1`, `is_terminal=False` | One chunk per `step()` until completion is signalled by exception |

`chunk_size: 1` is schema-enforced for `kind: ros_action`: each waypoint is
its own `ActionChunk`, so the C++ safety kernel validates (and can reject) it
before the next one is sent.

### GPU-accelerated planning (cuMotion)

On a host that clears the cuMotion GPU floor (`RobotCapabilities.supports_cumotion()`
— Ampere+, CUDA ≥ 13, ~8 GB VRAM), the runner sets
`MotionPlanRequest.pipeline_id = "isaac_ros_cumotion"` so MoveIt plans with
NVIDIA's CUDA-accelerated cuMotion pipeline; otherwise it falls back to OMPL.
Transparent — same skill, no manifest change — and it never bypasses the safety
kernel: the planned trajectory still replays through `/openral/candidate_action`
and is validated waypoint-by-waypoint. Install: see
[`docs/contributing/toolchain.md`](../../docs/contributing/toolchain.md) →
"GPU motion planning — cuMotion".

## How it was trained / Upstream provenance

Nothing is trained — this rSkill wraps the upstream MoveIt motion
planner.

| Field | Value |
| --- | --- |
| Upstream | [`moveit_msgs/action/MoveGroup`](https://github.com/moveit/moveit_msgs/blob/master/action/MoveGroup.action) (BSD-3-Clause) |
| Planner library | [MoveIt 2](https://moveit.picknik.ai/) (BSD-3-Clause) |
| Collision check | FCL via MoveIt's `PlanningScene` (run during planning) |
| Wrapped artefact | rSkill manifest + README — no weights, no preprocessor JSONs |

## Supported robots / embodiments

| Robot | Embodiment tag | Status | Notes |
| --- | --- | --- | --- |
| Franka Panda | `franka_panda` | validated | default goal targets `panda_arm` home pose |
| Universal Robots UR5e | `ur5e` | experimental | needs a `joint` / `request.group_name` override (`"manipulator"`, UR joint names) |
| Universal Robots UR10e | `ur10e` | experimental | same as UR5e |
| SO-100 follower | `so100_follower` | experimental | needs a MoveIt config and joint-name override |
| OpenArm | `openarm` | experimental | bi-manual — choose `left_arm` or `right_arm` group |
| Flexiv Rizon4 | `rizon4` | experimental | upstream MoveIt config exists; manifest override needed |
| Rethink Sawyer | `sawyer` | experimental | upstream MoveIt config exists |
| Trossen WidowX | `widowx` | experimental | upstream MoveIt config exists |

Listed `embodiment_tags` only gate which robots see this skill in the
Reasoner's tool palette; actual resolution depends on `move_group` being
up for that robot with a `joint` block matching its planning group.

## Sensors required / Observation contract

This skill consumes nothing through OpenRAL's sensor pipeline. MoveIt's
own subscriptions handle:

| Source | Topic | Why MoveIt needs it |
| --- | --- | --- |
| Joint state | `/joint_states` | Plan from the live start state |
| Planning scene | `/planning_scene` (or `/monitored_planning_scene`) | Self- and environment-collision check |
| TF | `/tf`, `/tf_static` | Resolve goal pose / link frames |

For a non-default topic remap, set it on the MoveIt node's launch; the wrapped
action's contract is unchanged.

## Manifest summary

| Field | Value |
| --- | --- |
| `name` | `OpenRAL/rskill-moveit-multi-joints` |
| `version` | `0.1.0` |
| `license` | `apache-2.0` |
| `kind` | `ros_action` |
| `role` | `s1` |
| `actions` | `[reach]` |
| `chunk_size` | `1` (schema-enforced for `kind: ros_action`) |
| `latency_budget.per_chunk_ms` | `2000` (planning latency; the adapter waits ×5 of this on the action result) |
| `ros_integration.package` | `moveit_msgs` |
| `ros_integration.interface_type` | `MoveGroup` |
| `ros_integration.interface_name` | `/move_action` |
| `ros_integration.goal_builder` | `joint` (selects `JointGoalRskill`) |
| `ros_integration.result_trajectory_field` | `planned_trajectory.joint_trajectory` |
| `is_commercial_use_allowed` | `true` (apache-2.0) |

Full schema:
[`openral_core.schemas.RSkillManifest`](../../python/core/src/openral_core/schemas.py).

## Quick start

```python
from openral_rskill.loader import rSkill
pkg = rSkill.from_yaml("rskills/rskill-moveit-joints/rskill.yaml")
print(pkg.manifest.name, pkg.manifest.kind, pkg.manifest.ros_integration.goal_builder)
```

End-to-end, with a real MoveIt launch up:

```bash
# 1. Bring up MoveIt for your robot (example: Panda)
ros2 launch moveit_resources_panda_moveit_config demo.launch.py

# 2. Boot the OpenRAL deploy graph against the same robot
openral deploy sim --config scenes/deploy/<deploy-scene>.yaml --robot franka_panda

# 3. From the Reasoner (or by hand via the action CLI), dispatch the goal:
ros2 action send_goal /openral/execute_rskill openral_msgs/action/ExecuteRskill \
    "{rskill_id: 'OpenRAL/rskill-moveit-multi-joints', deadline_s: 30.0, prompt: 'move to home'}"
```

## Limitations / Roadmap

- **Goal defaults live in the manifest.** The LLM overrides `joint.positions`
  via `goal_params_json`; planner settings and joint names are inherited from
  `default_goal_json`. Cross-embodiment retargeting is a manifest copy with the
  correct `joint.joint_names` + `request.group_name`.
- **Kernel collision checks are deployment-configured.** MoveIt plans
  against its own FCL `PlanningScene`; the C++ safety kernel separately checks
  each waypoint against the per-joint position envelope and, when the deploy
  enables them, self-collision and world-voxel collision.
- **No velocity / jerk bound on `JOINT_POSITION` chunks.** The kernel checks
  each waypoint's position limits, not the rate between waypoints, so a planner
  emitting a rough trajectory would actuate.

## License

The rSkill package itself (this manifest + README) is **Apache-2.0**. The
wrapped MoveIt code (`moveit_msgs` IDL, `moveit2` planners) is **BSD-3-Clause**
and lives outside this repository, installed via `ros-${ROS_DISTRO}-moveit`.
Both are commercial-use-permissive.

## See also

- [Wrapped-ROS rSkills](../../docs/reference/rskills.md#wrapped-ros-rskills-kind-ros_action)
- [`openral_rskill.ros_action_rskill`](../../python/rskill/src/openral_rskill/ros_action_rskill.py) — engine source
- [`openral_rskill.joint_goal_rskill`](../../python/rskill/src/openral_rskill/joint_goal_rskill.py) — goal-lowering adapter
- [`rskills/rskill-moveit-eef-pose/`](../rskill-moveit-eef-pose/) — sibling Cartesian end-effector pose wrapper
- [`rskills/rskill-moveit-look-at/`](../rskill-moveit-look-at/) — sibling camera-aiming wrapper
- [`rskills/rskill-nav2-navigate-to-pose/`](../rskill-nav2-navigate-to-pose/) — sibling Nav2 wrapper (result-only mode)
- [CLAUDE.md §3 — Architecture Discipline](../../CLAUDE.md)
