---
language:
- en
license: apache-2.0
pipeline_tag: robotics
tags:
- OpenRAL
- rskill
- ros2
- mobile_base
- nav2
inference: false
---

# rskill-nav2-mobile_base-navigate_to_pose

> **OpenRAL rSkill** — wraps the upstream `nav2_msgs/action/NavigateToPose`
> action server as an OpenRAL rSkill so the Reasoner can dispatch
> mobile-base navigation through the same `ExecuteRskill` path used by
> VLA skills. **Result-only mode** — Nav2's behaviour tree publishes
> `/cmd_vel`, which the HAL's mobile-base bridge applies directly; no
> `Action` chunk flows through OpenRAL's safety kernel.

This package uses `kind: ros_action` with
`ros_integration.result_trajectory_field: null` to put the
[`ROSActionRskill`](../../python/rskill/src/openral_rskill/ros_action_rskill.py)
adapter into result-only mode: it sends the goal, awaits the action
result, and raises `ROSRskillGoalSatisfied` on success. Compare with
the sibling [`rskill-moveit-multi-joints`](../rskill-moveit-joints/)
skill, which sets `result_trajectory_field` and replays a joint
trajectory one waypoint at a time.

## What this skill does

Navigates the mobile base to a `geometry_msgs/PoseStamped` target via Nav2's
`NavigateToPose` action. The Reasoner fills `goal_params_schema` (absolute goal
in `map`, or a relative turn / drive in `base_link`) and sends it as
`goal_params_json`, which the adapter deep-merges over
`ros_integration.default_goal_json`. The default goal is the identity pose in
`base_link`, so a dispatch without parameters is a no-op.

| Field | Value |
| --- | --- |
| Actions | `navigate` |
| Objects | _none_ |
| Scenes  | `indoor` |
| Embodiment | `mobile_base` (any robot with a planar base) |

## How it works

`ROSActionRskill` is a thin `rSkillBase` shim around an `ActionClient`:

1. `_configure_impl` lazy-imports `nav2_msgs.action.NavigateToPose`,
   opens an `ActionClient` on `/navigate_to_pose` from the
   `RskillRunnerNode`-supplied node handle, and parses
   `ros_integration.default_goal_json` once.
2. On the first `_step_impl(world_state)` call the adapter sends the
   goal, polls the goal-accept + result futures while the host node's
   main rclpy spin services callbacks, and — because
   `result_trajectory_field is None` — raises
   `ROSRskillGoalSatisfied` immediately on success. The runner catches
   it specifically and closes the `ExecuteRskill` goal with
   `success=True`.

### Observation → action contract (result-only mode)

| Direction | Key | Shape | Notes |
| --- | --- | --- | --- |
| in  | `world_state.joint_state` | unused | Nav2 consumes its own sensor topics (laser, odom, camera) |
| out | _none via OpenRAL_ | — | Nav2's behaviour tree publishes `/cmd_vel` directly to the base controller |

**Safety implication.** No `ActionChunk` is published on
`/openral/candidate_action`, so the OpenRAL safety supervisor does NOT see
Nav2's velocity commands (neither the C++ kernel nor the Python supervisor
node is in that path): the HAL's
[`MobileBaseBridge`](../../python/hal/src/openral_hal/mobile_base_bridge.py)
maps each `/cmd_vel` message to a `BODY_TWIST` action and applies it directly
(Nav2's `velocity_smoother` caps velocity). Collision avoidance relies on
Nav2's costmap + behaviour tree. The kernel's `BODY_TWIST` base-speed bound
applies only to chunks that reach it on `/openral/candidate_action`, and only
when `robot.yaml` sets `max_base_linear_speed_m_s` / `max_base_angular_speed_rad_s`
(both default to unbounded).

## How it was trained / Upstream provenance

Nothing is trained — this rSkill wraps the upstream Nav2 stack.

| Field | Value |
| --- | --- |
| Upstream | [`nav2_msgs/action/NavigateToPose`](https://github.com/ros-navigation/navigation2/blob/main/nav2_msgs/action/NavigateToPose.action) (Apache-2.0) |
| Planner library | [Nav2](https://docs.nav2.org/) (Apache-2.0) |
| Costmap / behaviour tree | Nav2's own subsystems — see Nav2 docs |
| Wrapped artefact | rSkill manifest + README — no weights |

## Supported robots / embodiments

| Robot | Embodiment tag | Status | Notes |
| --- | --- | --- | --- |
| Any planar mobile base | `mobile_base` | experimental | class tag; the robot's `robot.yaml` must declare it and a Nav2 stack must serve `/navigate_to_pose` |

Mobile-base HALs that accept `BODY_TWIST` ship in-tree (for example
[`PandaMobileHAL`](../../python/hal/src/openral_hal/panda_mobile.py)); any
manifest-driven HAL whose `robot.yaml` declares `base_joints` gets the
`/cmd_vel` bridge. No real wheeled-base HAL ships in-tree.

## Sensors required / Observation contract

This skill consumes nothing through OpenRAL's sensor pipeline. Nav2's
own subscriptions handle:

| Source | Topic | Why Nav2 needs it |
| --- | --- | --- |
| Laser scan | `/scan` (or per-deployment remap) | Costmap obstacle layer |
| Odometry | `/odom` | Localisation + behaviour-tree feedback |
| TF | `/tf`, `/tf_static` | Resolve goal pose in the `map` frame |
| Map | `/map` (or AMCL initial pose) | Global planner |

For a non-default topic remap, set it on the Nav2 launch; the wrapped
action's contract is unchanged.

## Manifest summary

| Field | Value |
| --- | --- |
| `name` | `OpenRAL/rskill-nav2-mobile_base-navigate_to_pose` |
| `version` | `0.1.0` |
| `license` | `apache-2.0` |
| `kind` | `ros_action` |
| `role` | `s1` |
| `actions` | `[navigate]` |
| `chunk_size` | `1` (schema-enforced for `kind: ros_action`) |
| `latency_budget.per_chunk_ms` | `60000` (navigation is long-horizon; the adapter waits ×5 of this on the action result) |
| `ros_integration.package` | `nav2_msgs` |
| `ros_integration.interface_type` | `NavigateToPose` |
| `ros_integration.interface_name` | `/navigate_to_pose` |
| `ros_integration.result_trajectory_field` | _omitted → result-only mode_ |
| `is_commercial_use_allowed` | `true` (apache-2.0) |

Full schema:
[`openral_core.schemas.RSkillManifest`](../../python/core/src/openral_core/schemas.py).

## Quick start

```python
from openral_rskill.loader import rSkill
pkg = rSkill.from_yaml("rskills/rskill-nav2-navigate-to-pose/rskill.yaml")
print(pkg.manifest.name, pkg.manifest.kind, pkg.manifest.ros_integration.interface_name)
```

End-to-end, against the `panda_mobile` twin:

```bash
# 1. Boot the deploy graph against a mobile-base twin; Nav2 comes up by
#    default for a robot with a lidar (runtime.enable_nav2 overrides)
openral deploy sim --config scenes/deploy/<deploy-scene>.yaml --robot panda_mobile

# 2. From the Reasoner (or by hand), dispatch the goal:
ros2 action send_goal /openral/execute_rskill openral_msgs/action/ExecuteRskill \
    "{rskill_id: 'OpenRAL/rskill-nav2-mobile_base-navigate_to_pose', deadline_s: 120.0, prompt: 'go to map origin', goal_params_json: '{\"pose\": {\"header\": {\"frame_id\": \"map\"}, \"pose\": {\"position\": {\"x\": 0.0, \"y\": 0.0}, \"orientation\": {\"z\": 0.0, \"w\": 1.0}}}}'}"
```

## Limitations / Roadmap

- **Velocity stream bypasses the OpenRAL safety kernel** (see
  *Safety implication* above).
- **No real wheeled-base HAL in-tree.** On hardware the skill resolves only
  inside deployments that ship their own mobile-base wiring.

## License

The rSkill package itself (this manifest + README) is **Apache-2.0**. The
wrapped Nav2 code (`nav2_msgs` IDL, `navigation2` planners) is **Apache-2.0**
and lives outside this repository, installed via
`ros-${ROS_DISTRO}-nav2-bringup`. Both are commercial-use-permissive.

## See also

- [Wrapped-ROS rSkills](../../docs/reference/rskills.md#wrapped-ros-rskills-kind-ros_action)
- [`openral_rskill.ros_action_rskill`](../../python/rskill/src/openral_rskill/ros_action_rskill.py) — adapter source
- [`rskills/rskill-moveit-joints/`](../rskill-moveit-joints/) — sibling MoveIt wrapper (trajectory mode)
- [CLAUDE.md §3 — Architecture Discipline](../../CLAUDE.md)
