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

# rskill-moveit-look-at

Aim a robot-mounted camera at a 3-D point (ADR-0044 Phase 3; renamed from
`openral/rskill-look-at` under [ADR-0054](../../docs/adr/0054-moveit-goal-builder-library.md)).

A `kind: ros_action` rSkill wrapping `moveit_msgs/action/MoveGroup` — like
[`rskill-moveit-joints`](../rskill-moveit-joints/), but the goal is a
`look_at` block instead of raw constraints, and `ros_integration.goal_builder:
look_at` selects the `LookAtRskill` adapter that lowers it into MoveGroup
pose constraints.

## What this skill does

Given a target point and a camera name, plans a collision-aware arm motion
that points the named camera's optical axis at the target — so a later
perception query (`locate_in_view`) or manipulation skill sees the object
framed. It is the "look" rung of the ADR-0044 *recall → navigate → look →
verify → manipulate* ladder.

## How it works

`ros_integration.goal_builder: look_at` selects `LookAtRskill`
(`openral_rskill.look_at_rskill`), which lowers the `look_at` block at dispatch
time:

1. **Resolve the camera** named by `look_at.camera` (default `"wrist"`) from
   the host `RobotDescription.sensors`. No such sensor → `ROSConfigError`
   listing the robot's available sensors — never a silent guess. A sensor
   whose `frame_id` is itself a robot link (franka's LIBERO eye-in-hand on
   `panda_hand`) is constrained directly; a sensor with `parent_frame` +
   `static_transform_xyz_rpy` (so101-style mount) constrains the parent link
   through the declared offset.
2. **Read the camera's current pose over TF2** (the only source of frames) in
   the goal frame.
3. **Place the camera goal** — in place (pure re-aim, the default) or at
   `look_at.standoff_m` from the target along the current line of approach.
4. **Orient it** with `compute_gaze_pose` (ROS optical convention: camera +Z
   hits `look_at.target_xyz`; roll about the optical axis left free at
   tolerance π for planner reachability) and submit MoveGroup
   `position_constraints` + `orientation_constraints`.

`plan_only: true`, deliberately: OpenRAL's actuation path is the per-waypoint
replay through `/openral/candidate_action` (`chunk_size: 1`, so the safety
supervisor's per-joint envelope check sees every aiming step, and the HAL
actuates). Letting `move_group` also execute on its own controllers would
bypass the kernel and double-drive the arm.

### Observation → action contract

Input is the ADR-0026 `goal_params_json` `look_at` block; output is a joint
trajectory replayed one waypoint per `step()` as a 1-row `JOINT_POSITION`
`Action` chunk.

```json
{"look_at": {"target_xyz": [0.5, 0.0, 0.2], "camera": "wrist"}}
```

Planner settings (`request.group_name`, scaling, attempts) are inherited from
`default_goal_json`. Omit `standoff_m` to re-aim in place; set it to also move
the camera to that distance from the target.

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
computes the gaze pose analytically.

| Field | Value |
| --- | --- |
| Upstream | [`moveit_msgs/action/MoveGroup`](https://github.com/moveit/moveit_msgs/blob/master/action/MoveGroup.action) (BSD-3-Clause) |
| Planner library | [MoveIt 2](https://moveit.picknik.ai/) (BSD-3-Clause) |
| Gaze geometry | `openral_world_state.geometry.compute_gaze_pose` (Apache-2.0, in-tree) |
| Collision check | FCL via MoveIt's `PlanningScene` (run during planning) |
| Wrapped artefact | rSkill manifest + README — no weights, no preprocessor JSONs |

## Supported robots / embodiments

| Robot | Embodiment tag | Status | Notes |
| --- | --- | --- | --- |
| Franka Panda | `franka_panda` | validated | default goal targets `panda_arm`; camera `wrist` = LIBERO eye-in-hand on `panda_hand` |
| Universal Robots UR5e | `ur5e` | experimental | needs `request.group_name` + `look_at.camera`/`frame_id` overrides |
| Universal Robots UR10e | `ur10e` | experimental | same as UR5e |
| SO-100 follower | `so100_follower` | experimental | needs a MoveIt config; `wrist` cam declares a static mount |
| OpenArm | `openarm` | experimental | bi-manual — choose `left_arm`/`right_arm` group |
| Flexiv Rizon4 | `rizon4` | experimental | upstream MoveIt config exists; manifest override needed |
| Rethink Sawyer | `sawyer` | experimental | upstream MoveIt config exists |
| Trossen WidowX | `widowx` | experimental | upstream MoveIt config exists |

A robot must declare the named camera (default `wrist`) in its `robot.yaml`
sensors, or the skill fails at configure with the available-sensor list.
Listed `embodiment_tags` only gate palette visibility; actual resolution needs
`move_group` up for that robot.

## Sensors required / Observation contract

This skill consumes no camera frames through OpenRAL's sensor pipeline — it
*aims* a camera, it doesn't read one. It needs the named camera's **frame** to
exist in TF (declared in the robot manifest + published by
`robot_state_publisher`), plus MoveIt's own subscriptions:

| Source | Topic / contract | Why it's needed |
| --- | --- | --- |
| Camera frame | TF `frame_id` of `look_at.camera` (default `wrist`) | Read the camera's current pose; target the gaze |
| Joint state | `/joint_states` | Plan from the live start state |
| Planning scene | `/planning_scene` | Collision check during planning |
| TF | `/tf`, `/tf_static` | Resolve camera + link frames |

## Manifest summary

| Field | Value |
| --- | --- |
| `name` | `OpenRAL/rskill-moveit-look-at` |
| `version` | `0.1.0` |
| `license` | `apache-2.0` |
| `kind` | `ros_action` |
| `role` | `s1` |
| `actions` | `[look]` |
| `chunk_size` | `1` (schema-enforced for `kind: ros_action`) |
| `latency_budget.per_chunk_ms` | `2000` (planning latency) |
| `ros_integration.package` | `moveit_msgs` |
| `ros_integration.interface_type` | `MoveGroup` |
| `ros_integration.interface_name` | `/move_action` |
| `ros_integration.goal_builder` | `look_at` (selects `LookAtRskill`) |
| `ros_integration.result_trajectory_field` | `planned_trajectory.joint_trajectory` |
| `commercial_use_allowed` | `true` (apache-2.0) |

Full schema:
[`openral_core.schemas.RSkillManifest`](../../python/core/src/openral_core/schemas.py).

## Quick start

```python
from openral_rskill.loader import rSkill
pkg = rSkill.from_yaml("rskills/rskill-moveit-look-at/rskill.yaml")
print(pkg.manifest.name, pkg.manifest.ros_integration.goal_builder)
```

End-to-end, with a real MoveIt launch up:

```bash
# 1. Bring up MoveIt for your robot (example: Panda)
ros2 launch moveit_resources_panda_moveit_config demo.launch.py

# 2. Dispatch a look-at goal (aim the wrist camera at a tabletop point):
ros2 action send_goal /openral/execute_rskill openral_msgs/action/ExecuteRskill \
    "{rskill_id: 'OpenRAL/rskill-moveit-look-at', deadline_s: 30.0, prompt: 'look at the mug',
      goal_params_json: '{\"look_at\": {\"target_xyz\": [0.5, 0.0, 0.2], \"camera\": \"wrist\"}}'}"
```

## Limitations / Roadmap

- **Reachability is the planner's call.** Roll about the optical axis is left
  free, but a target outside the arm's dexterous workspace simply fails to
  plan — there's no base-repositioning fallback here (that's the navigate rung
  of the ladder, ADR-0044 Phase 4).
- **Single-camera aim.** One camera per dispatch; multi-camera coverage is a
  reasoner-level concern.
- **No velocity / jerk bound at the supervisor.** Same posture as
  `rskill-moveit-joints`: the per-joint position envelope runs per
  waypoint; richer bounds are tracked separately.

## License

The rSkill package itself (this manifest + README) is **Apache-2.0**. The
wrapped MoveIt code (`moveit_msgs` IDL, `moveit2` planners) is **BSD-3-Clause**
and is installed via `ros-${ROS_DISTRO}-moveit`, outside this repository. Per
[ADR-0012](../../docs/adr/0012-open-core-licensing.md) both postures are
commercial-use-permissive.

## See also

- [ADR-0054 — MoveIt goal-builder library + rskill-moveit-* rename](../../docs/adr/0054-moveit-goal-builder-library.md)
- [ADR-0044 — look_at skill + grid-refined approach](../../docs/adr/0044-look-at-skill-grid-refined-approach.md)
- [`openral_rskill.look_at_rskill`](../../python/rskill/src/openral_rskill/look_at_rskill.py) — adapter source
- [`openral_world_state.geometry`](../../python/world_state/src/openral_world_state/geometry.py) — gaze math
- [`rskills/rskill-moveit-joints/`](../rskill-moveit-joints/) — sibling joint-space MoveIt wrapper
- [`rskills/rskill-moveit-eef-pose/`](../rskill-moveit-eef-pose/) — sibling Cartesian end-effector pose wrapper
