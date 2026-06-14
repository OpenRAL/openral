---
tags:
  - OpenRAL
  - rskill
  - ros2
  - nav2
license: apache-2.0
language:
  - en
---

# rskill-nav2-navigate-to-pose

> **OpenRAL rSkill** — wraps the upstream `nav2_msgs/action/NavigateToPose`
> action server as an OpenRAL rSkill so the Reasoner can dispatch
> mobile-base navigation through the same `ExecuteSkill` path used by
> VLA skills. **Result-only mode** — Nav2's behaviour tree publishes
> `/cmd_vel` directly to the base controller; no `Action` chunk flows
> through OpenRAL's safety supervisor.

This package uses `kind: ros_action` (see
[ADR-0024](../../docs/adr/0024-ros-wrapped-rskills.md)) with
`ros_integration.result_trajectory_field: null` to put the
[`ROSActionRskill`](../../python/rskill/src/openral_rskill/ros_action_rskill.py)
adapter into result-only mode: it sends the goal, awaits the action
result, and raises `ROSRskillGoalSatisfied` on success. Compare with
the sibling [`rskill-moveit-joints`](../rskill-moveit-joints/)
skill, which sets `result_trajectory_field` and replays a joint
trajectory one waypoint at a time.

## What this skill does

Navigates the mobile base to a fixed `geometry_msgs/PoseStamped` target
in the `map` frame via Nav2's `NavigateToPose` action. The default goal
points at the map origin; override `ros_integration.default_goal_json`
in a per-deployment copy for real targets.

| Field | Value |
| --- | --- |
| Actions | `navigate` |
| Objects | _none_ |
| Scenes  | `indoor` |
| Embodiment | mobile-manipulator and similar mobile-base embodiments |

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
   it specifically and closes the `ExecuteSkill` goal with
   `success=True`.

### Observation → action contract (result-only mode)

| Direction | Key | Shape | Notes |
| --- | --- | --- | --- |
| in  | `world_state.joint_state` | unused | Nav2 consumes its own sensor topics (laser, odom, camera) |
| out | _none via OpenRAL_ | — | Nav2's behaviour tree publishes `/cmd_vel` directly to the base controller |

**Safety implication.** No `ActionChunk` is published on
`/openral/candidate_action`, so the OpenRAL safety supervisor does NOT
see Nav2's velocity commands. Collision avoidance relies entirely on
Nav2's costmap + behaviour tree. The follow-up that brings velocity
streams under the supervisor's envelope is tracked in ADR-0024's
§Out-of-scope and depends on (a) a mobile-base HAL declaring
`body_twist` in `supported_control_modes` (none exist in-tree today),
and (b) a velocity / jerk envelope landing in the supervisor (it
currently checks per-joint position only).

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
| Panda Mobile | `panda_mobile` | experimental | no HAL accepting `body_twist` ships in-tree yet; resolution depends on the deployment wiring its own mobile-base HAL or running Nav2's controllers against a sim |
| Generic mobile-manipulator | `mobile_manipulator` | experimental | union tag — palette filter only |

A real mobile-base HAL (Turtlebot 4, Clearpath Jackal, etc.) is the
prerequisite for full end-to-end execution — tracked as a separate
issue.

## Sensors required / Observation contract

This skill consumes nothing through OpenRAL's sensor pipeline. Nav2's
own subscriptions handle:

| Source | Topic | Why Nav2 needs it |
| --- | --- | --- |
| Laser scan | `/scan` (or per-deployment remap) | Costmap obstacle layer |
| Odometry | `/odom` | Localisation + behaviour-tree feedback |
| TF | `/tf`, `/tf_static` | Resolve goal pose in the `map` frame |
| Map | `/map` (or AMCL initial pose) | Global planner |

If your deployment uses a non-default topic remap, surface it on the
Nav2 launch — the wrapped action's contract is intact.

## Manifest summary

| Field | Value |
| --- | --- |
| `name` | `OpenRAL/rskill-nav2-navigate-to-pose` |
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
| `commercial_use_allowed` | `true` (apache-2.0) |

Full schema:
[`openral_core.schemas.RSkillManifest`](../../python/core/src/openral_core/schemas.py).

## Quick start

```python
from openral_rskill.loader import rSkill
pkg = rSkill.from_yaml("rskills/rskill-nav2-navigate-to-pose/rskill.yaml")
print(pkg.manifest.name, pkg.manifest.kind, pkg.manifest.ros_integration.interface_name)
```

End-to-end, with a real Nav2 launch up (e.g. against a Gazebo /
Turtlebot4 sim):

```bash
# 1. Bring up Nav2 for your mobile-base sim or robot
ros2 launch nav2_bringup tb4_simulation_launch.py

# 2. Bring up the OpenRAL runner against that embodiment
ros2 launch openral_rskill_ros skill_runner.launch.py robot:=panda_mobile

# 3. From the Reasoner (or by hand), dispatch the goal:
ros2 action send_goal /openral/execute_skill openral_msgs/action/ExecuteSkill \
    "{rskill_id: 'OpenRAL/rskill-nav2-navigate-to-pose', deadline_s: 120.0, prompt: 'go to map origin'}"
```

## Limitations / Roadmap

- **Velocity stream bypasses the OpenRAL safety supervisor.** Nav2
  publishes `/cmd_vel` directly. See ADR-0024 §Out-of-scope.
- **Goal hard-coded in the manifest.** v1 ships one goal per
  manifest; structured-prompt support is the next ADR.
- **No mobile-base HAL in-tree.** Until one lands, the skill resolves
  only inside deployments that ship their own mobile-base wiring.

## License

The rSkill package itself (this manifest + README) is **Apache-2.0**.
The wrapped Nav2 code (`nav2_msgs` IDL, `navigation2` planners) is
**Apache-2.0** and lives outside this repository — installed via
`ros-${ROS_DISTRO}-nav2-bringup`. Per
[ADR-0012](../../docs/adr/0012-open-core-licensing.md) both postures
are commercial-use-permissive.

## See also

- [ADR-0024 — ROS-wrapped rSkills](../../docs/adr/0024-ros-wrapped-rskills.md)
- [`openral_rskill.ros_action_rskill`](../../python/rskill/src/openral_rskill/ros_action_rskill.py) — adapter source
- [`rskills/rskill-moveit-joints/`](../rskill-moveit-joints/) — sibling MoveIt wrapper (trajectory mode)
- [CLAUDE.md §3 — Architecture Discipline](../../CLAUDE.md)
