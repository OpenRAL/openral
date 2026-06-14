# `sawyer` — Robot description

Canonical `RobotDescription` manifest for the **Rethink Robotics Sawyer**
7-DoF arm (4 kg payload, 1.26 m reach, joint torque sensors, parallel
gripper). The MetaWorld MT50 benchmark simulates this same robot through
`metaworld` / `robosuite` MuJoCo wrappers; the sim-imposed observation /
action contract (4-D `agent_pos` state, 4-D delta-XYZ-plus-gripper
action) lives in the matching scene adapter at
[`python/sim/src/openral_sim/adapters/metaworld.py`](../../python/sim/src/openral_sim/adapters/metaworld.py),
not in this manifest. See
[ADR-0007](../../docs/adr/0007-robot-sim-split.md) for the
robot-vs-sim split rationale.

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

Same rationale as `franka_panda` — see
[ADR-0007](../../docs/adr/0007-robot-sim-split.md). The manifest
describes the physical Sawyer; the MetaWorld scene adapter translates
to/from MetaWorld's 4-D action / observation conventions inside the
runner.

## Wiring

| Layer | Where |
| --- | --- |
| Python HAL adapter (sim) | _planned_ — currently driven by `openral_sim.adapters.metaworld` |
| Real-HW adapter | _planned_ — see [#57](https://github.com/OpenRAL/openral/issues/57) (`intera_sdk` / `sawyer_robot`) |
| ROS 2 lifecycle node | _planned_ — covered by PR 7 of the refinement plan |
| Sim test | `tests/sim/test_smolvla_metaworld_push.py` (or wherever the metaworld_push config gets exercised) |
| Example configs | [`scenes/benchmark/metaworld_push.yaml`](../../scenes/benchmark/metaworld_push.yaml) (pass `--rskill rskills/smolvla-metaworld`) |

## See also

- [`docs/adr/0007-robot-sim-split.md`](../../docs/adr/0007-robot-sim-split.md) — robot-vs-sim split rationale.
- [`docs/reference/vla_compatibility.md`](../../docs/reference/vla_compatibility.md) §3.2 — MetaWorld VLA matrix.
- [`python/sim/src/openral_sim/adapters/metaworld.py`](../../python/sim/src/openral_sim/adapters/metaworld.py) — sim-side IO contract.
