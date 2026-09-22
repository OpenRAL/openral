# `sawyer` — Robot description

Canonical `RobotDescription` manifest for the **Rethink Robotics Sawyer**
7-DoF arm (4 kg payload, 1.26 m reach, joint torque sensors, parallel
gripper). The MetaWorld MT50 benchmark simulates this same robot through
`metaworld` / `robosuite` MuJoCo wrappers; the sim-imposed observation /
action contract (4-D `agent_pos` state, 4-D delta-XYZ-plus-gripper
action) lives in the matching scene adapter at
[`python/sim/src/openral_sim/backends/metaworld.py`](../../python/sim/src/openral_sim/backends/metaworld.py),
not in this manifest. This follows the
robot/sim split convention.

## At a glance

| Field | Value |
| --- | --- |
| `name` | `sawyer` |
| `embodiment_kind` | `manipulator` |
| Joints | 7 revolute (`right_j0`–`right_j6`) |
| End-effector | `right_hand` parallel gripper (1 DoF, 35 N max grip force, 4 kg payload) |
| Embodiment tags | `sawyer`, `rethink` |
| Supported control modes | `joint_position` |
| `sdk_kind` | `closed_with_api` (Rethink's `intera_sdk`; vendor dissolved in 2018; community forks remain the reference) |
| `hal.sim` | _none_ — no MuJoCo twin (real-only; `deploy sim` raises `ROSCapabilityMismatch`) |
| `hal.real` | `openral_hal.sawyer_real:SawyerRealHAL` ([#57](https://github.com/OpenRAL/openral/issues/57)) |

The MetaWorld MuJoCo backend simulates this Sawyer through the
`openral_sim` MetaWorld scene adapter; no Sawyer-specific HAL
ships in tree today.

## Why "physical robot only"

Same rationale as `franka_panda` — the
robot/sim split convention. The manifest
describes the physical Sawyer; the MetaWorld scene adapter translates
to/from MetaWorld's 4-D action / observation conventions inside the
runner.

## Wiring

| Layer | Where |
| --- | --- |
| Python HAL adapter (sim) | _planned_ — currently driven by `openral_sim.backends.metaworld` |
| Real-HW adapter | `openral_hal.sawyer_real:SawyerRealHAL` |
| ROS 2 lifecycle node | _planned_ — covered by PR 7 of the refinement plan |
| Sim test | none yet — `scenes/benchmark/metaworld_push.yaml` is referenced only by unit-level guard tests (`tests/unit/test_benchmark_scene_writeback_guard.py`, `tests/unit/test_sim_run_fixed_robot_guard.py`), not a closed-loop sim rollout test |
| Example configs | [`scenes/benchmark/metaworld_push.yaml`](../../scenes/benchmark/metaworld_push.yaml) (pass `--rskill rskills/smolvla-metaworld`) |

## E-stop and recovery

`/openral/estop` reaches `SawyerRealHAL.estop()` through the lifecycle node
(issue #295). Two steps, in order:

1. `controller_manager/switch_controller` deactivates
   `sawyer_arm_controller` (STRICT), read back via `list_controllers` — the
   acknowledged half.
2. `std_msgs/Empty` on `/robot/set_super_stop` — intera's super stop, the
   software equivalent of the pendant e-stop ("Robot must be reset to clear
   the stopped state"). The topic carries no acknowledgement; the stopped
   state is observable on `/robot/state` (`RobotAssemblyState.stopped`).

The outcome is on `/diagnostics` (`downstream_stop=acknowledged|unacknowledged`)
and in the lifecycle log.

**Recovery is `RESTART_REQUIRED`.** `/openral/estop_cleared` is rejected by
this HAL. To resume: publish `/robot/set_super_reset`, re-enable the robot
(`/robot/set_super_enable`), re-activate the controller, then relaunch the
HAL lifecycle node and re-align.

Evidence: `tests/unit/test_sawyer_real.py::TestSafety`,
`tests/integration/test_real_hal_estop_ros2_control_live.py`, and the attended
`tests/hil/test_sawyer.py::TestSawyerDownstreamEStop`.

## See also

- The robot/sim split convention — robot-vs-sim split rationale.
- [`docs/reference/vla_compatibility.md`](../../docs/reference/vla_compatibility.md) §3.2 — MetaWorld VLA matrix.
- [`python/sim/src/openral_sim/backends/metaworld.py`](../../python/sim/src/openral_sim/backends/metaworld.py) — sim-side IO contract.
