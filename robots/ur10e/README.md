# `ur10e` — Robot description

Canonical `RobotDescription` manifest for the **Universal Robots UR10e**
6-DoF cobot — 12.5 kg payload, 1.30 m reach, larger torques and slower
shoulder/lift slew rates than the UR5e. Same kinematic chain naming and
the same dual-HAL story: MuJoCo-backed sim plus a real-hardware adapter
on top of [`ur_robot_driver`](https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver)
(URCap / RTDE).

## At a glance

| Field | Value |
| --- | --- |
| `name` | `ur10e` |
| `embodiment_kind` | `manipulator` |
| Joints | 6 revolute (`shoulder_pan_joint` → `wrist_3_joint`) |
| End-effector | `tool0` (mounting flange; 12.5 kg payload, 1.30 m workspace radius) |
| Embodiment tags | `ur10e`, `ur` |
| Supported control modes | `joint_position` |
| `sdk_kind` (production) | `closed` — runtime path requires real UR + URCap on the teach pendant |
| `hal.real` | `openral_hal.ur_real:UR10eRealHAL` (`deploy run`) |
| `hal.sim` | `openral_hal.ur:UR10eHAL` (`deploy sim` / `sim run`; MuJoCo via `mujoco_menagerie`) |
| Driver license | BSD-3-Clause (`ur_robot_driver`; CLAUDE.md §7.4 compatible) |

Velocity caps (rad/s): shoulder/lift = 2.094 (120°/s), elbow = 3.142
(180°/s), wrists = π. Effort caps (Nm): shoulder/lift = 330, elbow =
150, wrists = 56.

The production manifest (`robot.yaml`) pins the **real-hardware** entry
point — same convention as the UR5e sibling. The MuJoCo sim entry point
(`UR10e_DESCRIPTION` / `openral_hal.ur:UR10eHAL`) shares the same
kinematics and safety envelope; only `sdk_kind` differs (the `hal` block is shared, ADR-0031).

## Wiring

| Layer | Where |
| --- | --- |
| Python HAL adapter (real HW) | `openral_hal.ur_real:UR10eRealHAL` (`python/hal/src/openral_hal/ur_real.py`) |
| Python HAL adapter (sim) | `openral_hal.ur:UR10eHAL` (`python/hal/src/openral_hal/ur.py:342`) |
| ROS 2 lifecycle node | `packages/openral_hal_ur10e/` |
| Conformance test | `tests/unit/test_hal_protocol_conformance.py::HAL_BUILDERS["UR10eRealHAL+SimTransport"]` |
| Unit test | `tests/unit/test_ur_real_hal.py` |
| HIL test | `tests/hil/test_ur10e.py` (gated by `UR10E_HOST` env var + `[self-hosted, lab-ur10e]` runner label) |
| Sim test | `tests/sim/test_ur10e_hal_mujoco.py` |

## Real-hardware deployment

Same recipe as the UR5e — the `ur_robot_driver` binary is the same for
both arms; only the URDF / per-joint envelope changes. Bring the driver
up with `ur_type:=ur10e robot_ip:=$UR10E_HOST` and construct
`UR10eRealHAL(robot_ip=$UR10E_HOST)`.

## See also

- [`python/hal/README.md`](../../python/hal/README.md) — HAL Protocol + per-robot adapters.
- [`robots/ur5e/README.md`](../ur5e/README.md) — sister manifest with the
  full deployment recipe.
- [`packages/openral_hal_ur10e/README.md`](../../packages/openral_hal_ur10e/README.md) — ROS 2 lifecycle node.
- `UR10e_DESCRIPTION` constant (sim): `python/hal/src/openral_hal/ur.py:188`.
- `UR10e_REAL_DESCRIPTION` constant (real HW): `python/hal/src/openral_hal/ur_real.py`.
