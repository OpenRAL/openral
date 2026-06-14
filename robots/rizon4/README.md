# `rizon4` — Robot description

Canonical `RobotDescription` manifest for the **Flexiv Rizon 4** — a
7-DoF collaborative arm with whole-body force sensitivity (0.1 N
resolution), 4 kg payload, and 780 mm reach. The same manifest covers
the real Rizon 4 (wrapping `flexiv_rdk`, planned follow-up) and the
real-physics MuJoCo digital twin (`Rizon4MujocoHAL` on the
`mujoco_menagerie` MJCF).

## At a glance

| Field | Value |
| --- | --- |
| `name` | `rizon4` |
| `embodiment_kind` | `manipulator` |
| Joints | 7 revolute (`joint1` … `joint7`) |
| End-effector | tool flange (no gripper on the base manifest) |
| Payload | 4 kg |
| Reach | 780 mm |
| Force sensitivity | 0.1 N (whole-body) — `has_force_control: true` |
| Embodiment tags | `rizon4`, `flexiv` |
| Supported VLA embodiments | `rizon4` |
| Supported control modes | `joint_position` |
| `sdk_kind` | `open` (menagerie MJCF, Apache-2.0) |
| `hal.sim` | `openral_hal.flexiv_rizon4:Rizon4MujocoHAL` (`deploy sim`) |
| `hal.real` | _null_ — sim-only until a `flexiv_rdk` HAL lands (`deploy run` raises `ROSCapabilityMismatch`) |

## Pair with

| Component | Path |
| --- | --- |
| Python HAL adapter | `openral_hal.flexiv_rizon4.Rizon4MujocoHAL` (MuJoCo digital twin) |
| Python description | `openral_hal.RIZON4_DESCRIPTION` |
| Sim test | `tests/sim/test_rizon4_hal_mujoco.py` |
| Future real-HW HAL | planned wrapper around `flexivrobotics/flexiv_rdk` (Python + C++ SDK, BSD-style but vendor-distributed → `sdk_kind: closed_with_api`) |

## Joints

| Index | Name | Range (rad) | Effort (N·m) |
| ---: | --- | --- | ---: |
| 0 | `joint1` | ±2.88 | 123 |
| 1 | `joint2` | ±2.356 | 123 |
| 2 | `joint3` | ±3.054 | 64 |
| 3 | `joint4` | -1.955, 2.775 | 64 |
| 4 | `joint5` | ±3.054 | 39 |
| 5 | `joint6` | -1.484, 4.625 | 39 |
| 6 | `joint7` | ±3.054 | 39 |

Position limits come from the upstream `mujoco_menagerie` MJCF
verbatim (which pins them to the published Flexiv Rizon 4 spec).
Velocity (2 rad/s) and effort limits come from the spec sheet.

## Tests

- Sim: `tests/sim/test_rizon4_hal_mujoco.py` exercises
  `Rizon4MujocoHAL` end-to-end against real MuJoCo physics (lifecycle,
  schema-drift guard including a check that the menagerie actuators
  stay in **position** mode — H1-style torque actuators would need
  the PD-loop path the H1 HAL uses), closed-loop convergence including
  a per-joint-identity wiring test, and a multi-step action chunk.
- HIL: planned alongside the real-HW HAL.

## See also

- [`python/hal/README.md`](../../python/hal/README.md) — `Rizon4MujocoHAL`, supported robots.
- [Flexiv Rizon product page](https://www.flexiv.com/products/rizon) — vendor specs.
- [flexivrobotics/flexiv_description](https://github.com/flexivrobotics/flexiv_description) — upstream URDF.
- [flexivrobotics/flexiv_rdk](https://github.com/flexivrobotics/flexiv_rdk) — vendor SDK (future real-HW path).
