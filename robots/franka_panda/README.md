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
| `sdk_kind` | `open` (MuJoCo via `mujoco_menagerie`); real-HW (FCI) tracked by [#56](https://github.com/OpenRAL/openral/issues/56) |
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
[`python/eval/src/openral_sim/adapters/`](../../python/eval/src/openral_sim/adapters/),
not here. See
[ADR-0007](../../docs/adr/0007-robot-sim-split.md) for the rationale —
the previous `robots/libero_franka/` manifest conflated the two and
has been retired.

## Wiring

| Layer | Where |
| --- | --- |
| Python HAL adapter (sim) | `openral_hal.franka_panda:FrankaPandaHAL` |
| Real-HW adapter | _planned_ — see [#56](https://github.com/OpenRAL/openral/issues/56) (`franka_ros2` / FCI) |
| ROS 2 lifecycle node | `packages/openral_hal_franka/` |
| Sim test (HAL) | `tests/sim/test_franka_panda_hal_mujoco.py` |
| Sim test (LIBERO + VLA) | `tests/sim/test_franka_panda_smolvla_libero.py`, `test_xvla_libero.py` (and skill-level π0.5 LIBERO test) |
| Example configs | `scenes/{smolvla,xvla,pi05}_libero_spatial.yaml` |

## See also

- [`python/hal/README.md`](../../python/hal/README.md) — HAL Protocol + per-robot adapters.
- [`packages/openral_hal_franka/README.md`](../../packages/openral_hal_franka/README.md) — ROS 2 lifecycle node.
- [`docs/adr/0007-robot-sim-split.md`](../../docs/adr/0007-robot-sim-split.md) — design decision behind this manifest.
- `FRANKA_PANDA_DESCRIPTION` constant: [`python/hal/src/openral_hal/franka_panda.py:168`](../../python/hal/src/openral_hal/franka_panda.py).
