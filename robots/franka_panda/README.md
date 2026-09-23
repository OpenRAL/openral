# `franka_panda` — Robot description

Canonical `RobotDescription` manifest for the **Franka Emika Panda**
7-DoF cobot (3 kg payload, 0.855 m reach, joint torque sensors,
parallel gripper). Mirrors the in-code `FRANKA_PANDA_DESCRIPTION`
(`python/hal/src/openral_hal/franka_panda.py:168`); drift between
the two is guarded by
[`tests/unit/test_robot_manifests_match_hal_constants.py`](../../tests/unit/test_robot_manifests_match_hal_constants.py).

## At a glance

| Field | Value |
| --- | --- |
| `name` | `franka_panda` |
| `embodiment_kind` | `manipulator` |
| Joints | 7 revolute + 1 synthetic gripper (`panda_joint1`–`panda_joint7`, `panda_gripper`) |
| End-effector | `panda_hand` parallel gripper (1 DoF, 70 N max grip force, 3 kg payload) |
| Embodiment tags | `franka_panda`, `franka`, `panda` |
| Supported control modes | `joint_position` |
| `sdk_kind` | `closed_with_api` (MuJoCo via `mujoco_menagerie`); real-HW (FCI) tracked by [#56](https://github.com/OpenRAL/openral/issues/56) |
| `hal.sim` | `openral_hal.franka_panda:FrankaPandaHAL` (`deploy sim`) |
| `hal.real` | `openral_hal.franka_panda_real:FrankaPandaRealHAL` (`deploy run`) |

The synthetic `panda_gripper` joint is reported as a normalised value in
`[0, 1]` (0 = fully closed, 1 = fully open). The HAL's
`MujocoArmHAL` translates to/from the underlying MJCF tendon actuator.

## Why "physical robot only"

This manifest describes the **physical Franka Panda only** — kinematic
chain, actuator limits, end-effector, capabilities, safety envelope.
Sim-imposed observation/action contracts (LIBERO's 8-D
`eef_pos+axisangle+gripper_qpos` state, 7-D delta-EEF action, 180°
image flip; RoboCasa's variants; etc.) live in the matching scene
adapter under
[`python/sim/src/openral_sim/backends/`](../../python/sim/src/openral_sim/backends/),
not here. This follows the robot/sim split convention —
the previous `robots/libero_franka/` manifest conflated the two and
has been retired.

## Wiring

| Layer | Where |
| --- | --- |
| Python HAL adapter (sim) | `openral_hal.franka_panda:FrankaPandaHAL` |
| Real-HW adapter | `openral_hal.franka_panda_real:FrankaPandaRealHAL` |
| ROS 2 lifecycle node | [`packages/openral_hal_node/`](../../packages/openral_hal_node/README.md) (generic; node name `openral_hal_<robot_id>`) |
| Sim test (HAL) | `tests/sim/test_franka_panda_hal_mujoco.py` |
| Sim test (LIBERO + VLA) | `tests/sim/test_franka_panda_smolvla_libero.py`, `test_xvla_libero.py` (and skill-level π0.5 LIBERO test) |
| Example configs | `scenes/{smolvla,xvla,pi05}_libero_spatial.yaml` |

## E-stop and recovery

`/openral/estop` reaches `FrankaPandaRealHAL.estop()` through the lifecycle
node (issue #295). The stop is the acknowledged deactivation of
`franka_arm_controller` through `controller_manager/switch_controller`
(STRICT, read back via `list_controllers`): `franka_hardware`'s
`on_deactivate` calls `libfranka`'s `stopRobot()`, so the FCI control loop
ends and the arm holds. There is no separate vendor stop, and the
`/error_recovery` action is deliberately **not** invoked on e-stop — it clears
a reflex, which is a recovery step.

The outcome is on `/diagnostics` (`downstream_stop=acknowledged|unacknowledged`)
and in the lifecycle log; an unacknowledged stop is logged FATAL and the node
stays latched either way.

**Recovery is `RESTART_REQUIRED`.** `/openral/estop_cleared` is rejected by
this HAL. To resume: release the Franka user stop, run `franka_ros2`'s
`/error_recovery` action (`franka_msgs/action/ErrorRecovery`), re-activate
the controller (`ros2 control set_controller_state franka_arm_controller
active`), then relaunch the HAL lifecycle node and re-align.

Evidence: `tests/unit/test_franka_panda_real.py::TestSafety`,
`tests/integration/test_real_hal_estop_ros2_control_live.py`, and the attended
`tests/hil/test_franka_panda.py::TestFrankaDownstreamEStop`.

## See also

- [`python/hal/README.md`](../../python/hal/README.md) — HAL Protocol + per-robot adapters.
- [`packages/openral_hal_node/README.md`](../../packages/openral_hal_node/README.md) — the generic manifest-driven ROS 2 lifecycle node.
- The robot/sim split convention — design decision behind this manifest.
- `FRANKA_PANDA_DESCRIPTION` constant: [`python/hal/src/openral_hal/franka_panda.py:168`](../../python/hal/src/openral_hal/franka_panda.py).
