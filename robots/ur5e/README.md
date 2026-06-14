# `ur5e` — Robot description

Canonical `RobotDescription` manifest for the **Universal Robots UR5e**
6-DoF cobot — 5 kg payload, 0.85 m reach, BLDC-driven joints with
on-board torque feedback. Two HAL paths are supported: a MuJoCo-backed
simulation adapter and a real-hardware adapter that wraps
[`ur_robot_driver`](https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver)
(URCap / RTDE) under the same `HAL` Protocol.

## At a glance

| Field | Value |
| --- | --- |
| `name` | `ur5e` |
| `embodiment_kind` | `manipulator` |
| Joints | 6 revolute (`shoulder_pan_joint` → `wrist_3_joint`) |
| End-effector | `tool0` (mounting flange; 5 kg payload, 0.85 m workspace radius) |
| Embodiment tags | `ur5e`, `ur` |
| Supported control modes | `joint_position` |
| `sdk_kind` (production) | `closed` — runtime path requires real UR + URCap on the teach pendant |
| `hal.real` | `openral_hal.ur_real:UR5eRealHAL` (`deploy run`) |
| `hal.sim` | `openral_hal.ur:UR5eHAL` (`deploy sim` / `sim run`; MuJoCo via `mujoco_menagerie`) |
| Driver license | BSD-3-Clause (`ur_robot_driver`; CLAUDE.md §7.4 compatible) |

Per-joint velocity = π rad/s (180°/s) for every joint. Effort limits
match the UR5e datasheet: 150 Nm on shoulder/lift/elbow, 28 Nm on the
three wrists. Position limits use the `mujoco_menagerie` ranges (the
elbow is constrained to ±π by mechanical stop on real hardware).

The production manifest (`robot.yaml`) pins the **real-hardware** entry
point — the YAML's `sdk_kind: closed` flags that the runtime path needs
the URCap `external_control` program on the teach pendant, not that the
adapter or driver carry a restrictive license. The sim entry point
(`UR5e_DESCRIPTION` / `openral_hal.ur:UR5eHAL`) remains in-code as
the MuJoCo-backed sibling — kinematics, safety envelope, and capabilities
are identical between the two; only `sdk_kind` differs (the `hal` block is shared, ADR-0031).

## Wiring

| Layer | Where |
| --- | --- |
| Python HAL adapter (real HW) | `openral_hal.ur_real:UR5eRealHAL` (`python/hal/src/openral_hal/ur_real.py`) |
| Python HAL adapter (sim) | `openral_hal.ur:UR5eHAL` (`python/hal/src/openral_hal/ur.py:300`) |
| ROS 2 lifecycle node | `packages/openral_hal_ur5e/` |
| Conformance test | `tests/unit/test_hal_protocol_conformance.py::HAL_BUILDERS["UR5eRealHAL+SimTransport"]` |
| Unit test | `tests/unit/test_ur_real_hal.py` |
| HIL test | `tests/hil/test_ur5e.py` (gated by `UR5E_HOST` env var + `[self-hosted, lab-ur5e]` runner label) |
| Sim test | `tests/sim/test_ur5e_hal_mujoco.py` |

## Real-hardware deployment

The real path uses `ros2_control` + `ur_robot_driver` (URCap / RTDE):

1. Install the [URCap `external_control`](https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver/blob/main/ur_robot_driver/doc/install_urcap_e_series.rst)
   program on the UR teach pendant.
2. Bring up the driver with the controller's static IP:
   `ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5e robot_ip:=$UR5E_HOST`
3. Construct `UR5eRealHAL(robot_ip=$UR5E_HOST)` from upper layers — the
   adapter speaks `trajectory_msgs/JointTrajectory` on
   `/scaled_joint_trajectory_controller/joint_trajectory` and reads
   `sensor_msgs/JointState` on `/joint_states`.
4. The safety supervisor subscribes to
   `/io_and_status_controller/safety_mode`
   (`ur_msgs/msg/SafetyMode`) for deadman / E-stop transitions; the HAL
   exposes the topic name via `hal.deadman_topic` for launch-file pickup.

## See also

- [`python/hal/README.md`](../../python/hal/README.md) — HAL Protocol + per-robot adapters.
- [`packages/openral_hal_ur5e/README.md`](../../packages/openral_hal_ur5e/README.md) — ROS 2 lifecycle node.
- `UR5e_DESCRIPTION` constant (sim): `python/hal/src/openral_hal/ur.py:154`.
- `UR5e_REAL_DESCRIPTION` constant (real HW): `python/hal/src/openral_hal/ur_real.py`.
