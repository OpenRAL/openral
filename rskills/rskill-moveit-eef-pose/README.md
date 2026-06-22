---
tags:
  - OpenRAL
  - rskill
  - ros2
  - moveit
license: apache-2.0
language:
  - en
---

# rskill-moveit-eef-pose

> **OpenRAL rSkill** — wraps the upstream `moveit_msgs/action/MoveGroup`
> action server so the Reasoner can dispatch a collision-free Cartesian
> end-effector pose goal (position + orientation) through the same
> `ExecuteRskill` path used by VLA skills. No model weights — the manifest
> is the entire artefact. New under
> [ADR-0054](../../docs/adr/0054-moveit-goal-builder-library.md)'s
> `rskill-moveit-*` family.

This package uses `kind: ros_action` (see
[ADR-0024](../../docs/adr/0024-ros-wrapped-rskills.md)) — a discriminator
on `RSkillManifest.kind` that selects the
[`ROSActionRskill`](../../python/rskill/src/openral_rskill/ros_action_rskill.py)
engine at resolve time, with `ros_integration.goal_builder: pose` selecting
the [`PoseGoalRskill`](../../python/rskill/src/openral_rskill/pose_goal_rskill.py)
goal-lowering adapter (ADR-0054). It is the generic Cartesian sibling of
[`rskill-moveit-look-at`](../rskill-moveit-look-at/) (gaze is a computed-pose
specialisation) and of [`rskill-moveit-joints`](../rskill-moveit-joints/)
(joint-space). The engine constructs an `rclpy.action.ActionClient` on the
host `RskillRunnerNode`, sends one goal built from the lowered `pose` block,
awaits the result, and replays the returned `trajectory_msgs/JointTrajectory`
one waypoint per `step()` call onto `/openral/candidate_action` so the safety
supervisor still applies its per-joint envelope check to every commanded
position.

## What this skill does

Plans and executes a collision-free motion that brings a chosen end-effector
link to a target 6-DOF Cartesian pose (position + orientation) via MoveIt's
`MoveGroup` action. The goal is authored as a `pose` block — target `position`
([x, y, z] metres in `pose.frame_id`) plus a quaternion `orientation` in
`pose.quaternion_order` (default `xyzw`) — which `PoseGoalRskill` lowers into
MoveGroup position + orientation constraints. The default goal targets the
Franka demo (`panda_arm` group, `panda_hand` link in `panda_link0`). Use it
when you have a Cartesian end-effector target such as a pre-grasp pose.

| Field | Value |
| --- | --- |
| Actions | `reach` |
| Objects | _none_ — no per-object specialisation |
| Scenes  | _none_ — the wrapped planner does its own collision check against the live `/planning_scene` |
| Embodiment | `franka_panda` (default goal); other arm tags listed in the manifest for capability filtering |

## How it works

`ROSActionRskill` is a thin `rSkillBase` shim around an `ActionClient`, and
`goal_builder: pose` lowers the LLM-facing `pose` block into the MoveGroup
goal:

1. `_configure_impl` lazy-imports `moveit_msgs.action.MoveGroup`, opens
   an `ActionClient` on `/move_action` from the
   `RskillRunnerNode`-supplied node handle, and parses
   `ros_integration.default_goal_json` once.
2. `PoseGoalRskill` pops the `pose` block and lowers it into
   `request.goal_constraints[0]` — a `position_constraints` entry (a small
   bounding region of `pose.position_tolerance_m` around the target point) and
   an `orientation_constraints` entry (the target quaternion with
   `pose.orientation_tolerance_rad`), both attached to `pose.link_name` in
   `pose.frame_id`. The orientation quaternion is interpreted in
   `pose.quaternion_order` (default `xyzw`). If `pose.tool_frame` is set, the
   adapter looks up the `link_name ← tool_frame` offset over TF2 (the only
   source of frames) and composes it so the target is expressed for the TCP /
   tool frame; omit it to constrain `pose.link_name` directly.
3. On the first `_step_impl(world_state)` call the engine builds the
   `MoveGroup.Goal` from the lowered dict, sends it, polls the goal-accept +
   result futures, extracts
   `result.planned_trajectory.joint_trajectory`, reorders its `joint_names`
   into the host `RobotDescription.joints` order, and returns waypoint 0 as a
   1-row `Action(JOINT_POSITION, …)`.
4. Each subsequent `_step_impl` returns the next cached waypoint; after the
   last one it raises `ROSRskillGoalSatisfied`, which the runner catches to
   close the `ExecuteRskill` goal with `success=True`.

The LLM overrides the `pose` block's `position` + `orientation`; planner
settings are inherited from `default_goal_json`. `plan_only: true` so MoveGroup
never drives its own controllers — OpenRAL's per-waypoint replay is the only
actuation path.

### Observation → action contract

Input is the ADR-0026 `goal_params_json` `pose` block; output is a joint
trajectory replayed one waypoint per `step()` as a 1-row `JOINT_POSITION`
`Action` chunk.

```json
{"pose": {"position": [0.4, 0.0, 0.5], "orientation": [0.0, 0.0, 0.0, 1.0]}}
```

| Direction | Key | Shape | Notes |
| --- | --- | --- | --- |
| in  | `pose.position`    | `[x, y, z]` metres in `pose.frame_id` | LLM-overridable |
| in  | `pose.orientation` | unit quaternion in `pose.quaternion_order` (default `xyzw`) | LLM-overridable |
| out | per-waypoint `Action` | `joint_targets=[[n_dof floats]]`, `horizon=1`, `is_terminal=False` | One chunk per `step()` until completion is signalled by exception |

Why one row per chunk (`chunk_size: 1` is schema-enforced for
`kind: ros_action`): the OpenRAL safety supervisor only validates row 0
of every `ActionChunk` today
([`supervisor_node.py`](../../packages/openral_safety/openral_safety/supervisor_node.py)).
Packing the full trajectory as one chunk with `horizon=N` would let
waypoints 1..N actuate unchecked.

### GPU-accelerated planning (cuMotion, ADR-0065)

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

Nothing is trained — this rSkill wraps the upstream MoveIt motion planner and
lowers the target pose into constraints analytically.

| Field | Value |
| --- | --- |
| Upstream | [`moveit_msgs/action/MoveGroup`](https://github.com/moveit/moveit_msgs/blob/master/action/MoveGroup.action) (BSD-3-Clause) |
| Planner library | [MoveIt 2](https://moveit.picknik.ai/) (BSD-3-Clause) |
| Collision check | FCL via MoveIt's `PlanningScene` (run during planning) |
| Wrapped artefact | rSkill manifest + README — no weights, no preprocessor JSONs |

## Supported robots / embodiments

| Robot | Embodiment tag | Status | Notes |
| --- | --- | --- | --- |
| Franka Panda | `franka_panda` | validated | default goal targets `panda_arm`, `panda_hand` link in `panda_link0` |
| Universal Robots UR5e | `ur5e` | experimental | needs a `pose.link_name` / `request.group_name` override (`"manipulator"`, `tool0`) |
| Universal Robots UR10e | `ur10e` | experimental | same as UR5e |
| SO-100 follower | `so100_follower` | experimental | needs a MoveIt config and link/group override |
| OpenArm | `openarm` | experimental | bi-manual — choose `left_arm` or `right_arm` group |
| Flexiv Rizon4 | `rizon4` | experimental | upstream MoveIt config exists; manifest override needed |
| Rethink Sawyer | `sawyer` | experimental | upstream MoveIt config exists |
| Trossen WidowX | `widowx` | experimental | upstream MoveIt config exists |

Listed `embodiment_tags` only gate which robots see this skill in the
Reasoner's tool palette; actual resolution depends on `move_group` being up for
that robot with a `pose` block targeting a valid link in its planning group.

## Sensors required / Observation contract

This skill consumes nothing through OpenRAL's sensor pipeline. MoveIt's own
subscriptions handle planning; when `pose.tool_frame` is set the adapter also
reads the `link_name ← tool_frame` offset from TF:

| Source | Topic / contract | Why it's needed |
| --- | --- | --- |
| Joint state | `/joint_states` | Plan from the live start state |
| Planning scene | `/planning_scene` (or `/monitored_planning_scene`) | Self- and environment-collision check |
| TF | `/tf`, `/tf_static` | Resolve goal pose / link frames; look up `pose.tool_frame` offset |

## Manifest summary

| Field | Value |
| --- | --- |
| `name` | `OpenRAL/rskill-moveit-eef-pose` |
| `version` | `0.1.0` |
| `license` | `apache-2.0` |
| `kind` | `ros_action` |
| `role` | `s1` |
| `actions` | `[reach]` |
| `chunk_size` | `1` (schema-enforced for `kind: ros_action`) |
| `latency_budget.per_chunk_ms` | `2000` (planning latency) |
| `ros_integration.package` | `moveit_msgs` |
| `ros_integration.interface_type` | `MoveGroup` |
| `ros_integration.interface_name` | `/move_action` |
| `ros_integration.goal_builder` | `pose` (selects `PoseGoalRskill`) |
| `ros_integration.result_trajectory_field` | `planned_trajectory.joint_trajectory` |
| `commercial_use_allowed` | `true` (apache-2.0) |

Full schema:
[`openral_core.schemas.RSkillManifest`](../../python/core/src/openral_core/schemas.py).

## Quick start

```python
from openral_rskill.loader import rSkill
pkg = rSkill.from_yaml("rskills/rskill-moveit-eef-pose/rskill.yaml")
print(pkg.manifest.name, pkg.manifest.ros_integration.goal_builder)
```

End-to-end, with a real MoveIt launch up:

```bash
# 1. Bring up MoveIt for your robot (example: Panda)
ros2 launch moveit_resources_panda_moveit_config demo.launch.py

# 2. Dispatch a Cartesian end-effector pose goal:
ros2 action send_goal /openral/execute_rskill openral_msgs/action/ExecuteRskill \
    "{rskill_id: 'OpenRAL/rskill-moveit-eef-pose', deadline_s: 30.0, prompt: 'move to pre-grasp',
      goal_params_json: '{\"pose\": {\"position\": [0.4, 0.0, 0.5], \"orientation\": [0.0, 0.0, 0.0, 1.0]}}'}"
```

## Limitations / Roadmap

- **Reachability is the planner's call.** A pose outside the arm's dexterous
  workspace simply fails to plan — there is no base-repositioning fallback
  here (that is the navigate rung of the ladder, ADR-0044 Phase 4).
- **OpenRAL safety supervisor does not do collision checking.** We trust
  MoveIt's internal FCL pass; the per-joint envelope check still runs per
  waypoint. Kernel-side collision checking is a separate ADR + multi-PR effort.
- **No velocity / jerk bound at the supervisor.** Same posture as
  `rskill-moveit-joints`: the per-joint position envelope runs per
  waypoint; richer bounds are tracked separately.

## License

The rSkill package itself (this manifest + README) is **Apache-2.0**. The
wrapped MoveIt code (`moveit_msgs` IDL, `moveit2` planners) is **BSD-3-Clause**
and lives outside this repository — installed via `ros-${ROS_DISTRO}-moveit`.
Per [ADR-0012](../../docs/adr/0012-open-core-licensing.md) both postures are
commercial-use-permissive.

## See also

- [ADR-0054 — MoveIt goal-builder library + rskill-moveit-* rename](../../docs/adr/0054-moveit-goal-builder-library.md)
- [ADR-0024 — ROS-wrapped rSkills](../../docs/adr/0024-ros-wrapped-rskills.md)
- [`openral_rskill.ros_action_rskill`](../../python/rskill/src/openral_rskill/ros_action_rskill.py) — engine source
- [`openral_rskill.pose_goal_rskill`](../../python/rskill/src/openral_rskill/pose_goal_rskill.py) — goal-lowering adapter
- [`rskills/rskill-moveit-joints/`](../rskill-moveit-joints/) — sibling joint-space MoveIt wrapper
- [`rskills/rskill-moveit-look-at/`](../rskill-moveit-look-at/) — sibling camera-aiming wrapper
- [CLAUDE.md §3 — Architecture Discipline](../../CLAUDE.md)
