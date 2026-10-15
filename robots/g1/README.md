# `g1` — Robot description

Canonical `RobotDescription` manifest for the **Unitree G1** humanoid, a
29-DoF biped (2 × 6 legs + 3 waist + 2 × 7 arms; no end-effectors on this
menagerie variant). One manifest covers the real G1 (over `unitree_sdk2`, M2
milestone) and the MuJoCo digital twin (`G1MujocoHAL` on the `mujoco_menagerie`
MJCF).

## What this is — and what it isn't

The MuJoCo twin is a **HAL contract validator**, not a useful humanoid sim:
with a floating base and no S0 cerebellar controller it falls over under
gravity, so the closed-loop sim tests run with `gravity_enabled=False`. It
verifies:

- 29-DoF joint-position action layout,
- lifecycle wiring (`connect → read_state → send_action → estop`),
- joint indexing and ordering,
- `RobotDescription` round-trip,
- embodiment / VLA tag plumbing.

- the `body_twist` path: by default ADR-0087's upright kinematic glide (no
  balance); with `walking_enabled=True`, ADR-0089's pinned MuJoCo Playground
  ONNX walking controller for gravity-on sim walking — a sim-only learned
  controller that a real HAL must never reuse.

Real balance and walking belong to the C++ S0 cerebellum (CLAUDE.md §3, M2
milestone) and are out of scope here; see the "HAL · Unitree G1 (real-HW)" planned block in
`docs/architecture/repo-state-map.html`.

## At a glance

| Field | Value |
| --- | --- |
| `name` | `g1` |
| `embodiment_kind` | `humanoid` |
| Joints | 29 actuated (2 × 6 leg + 3 waist + 2 × 7 arm). The MJCF's `floating_base_joint` is implicit world state and is NOT enumerated in `joints`. |
| End-effectors | none — wrist endpoints are bare on this menagerie variant. A future `g1_with_hands` rev would add Inspire / Dex-3 entries. |
| Embodiment tags | `g1`, `unitree_g1`, `humanoid`, `mobile_base` |
| Supported VLA embodiments | `g1`, `humanoid_everyday_g1` |
| Supported control modes | `joint_position`, `body_twist` |
| Locomotion | `bipedal` |
| Bimanual | yes |
| `sdk_kind` | `open` (menagerie MJCF + `mujoco` Python package, Apache-2.0) |
| `hal.sim` | `openral_hal.g1:G1MujocoHAL` (`deploy sim`) |
| `hal.real` | _null_ — sim-only until the M2 C++ S0 cerebellum (`deploy run` raises `ROSCapabilityMismatch`) |

## Pair with

| Component | Path |
| --- | --- |
| Python HAL adapter | `openral_hal.g1.G1MujocoHAL` (MuJoCo digital twin) |
| Python description | `openral_hal.G1_DESCRIPTION` |
| Sim test | `tests/sim/test_g1_hal_mujoco.py` |
| Future real-HW HAL | planned under M2 (CLAUDE.md §3, S0), `unitree_sdk2` + C++ S0 cerebellum |

## Joints

The 29 actuated joints in canonical order (which matches both the
menagerie MJCF and the `Action.joint_targets[i]` slot order):

| Index | Name | Range (rad) | Group |
| ---: | --- | --- | --- |
| 0 | `left_hip_pitch_joint` | -2.531, 2.880 | left leg |
| 1 | `left_hip_roll_joint` | -0.524, 2.967 | left leg |
| 2 | `left_hip_yaw_joint` | -2.758, 2.758 | left leg |
| 3 | `left_knee_joint` | -0.087, 2.880 | left leg |
| 4 | `left_ankle_pitch_joint` | -0.873, 0.524 | left leg |
| 5 | `left_ankle_roll_joint` | -0.262, 0.262 | left leg |
| 6 | `right_hip_pitch_joint` | -2.531, 2.880 | right leg |
| 7 | `right_hip_roll_joint` | -2.967, 0.524 | right leg |
| 8 | `right_hip_yaw_joint` | -2.758, 2.758 | right leg |
| 9 | `right_knee_joint` | -0.087, 2.880 | right leg |
| 10 | `right_ankle_pitch_joint` | -0.873, 0.524 | right leg |
| 11 | `right_ankle_roll_joint` | -0.262, 0.262 | right leg |
| 12 | `waist_yaw_joint` | -2.618, 2.618 | waist |
| 13 | `waist_roll_joint` | -0.52, 0.52 | waist |
| 14 | `waist_pitch_joint` | -0.52, 0.52 | waist |
| 15 | `left_shoulder_pitch_joint` | -3.089, 2.670 | left arm |
| 16 | `left_shoulder_roll_joint` | -1.588, 2.252 | left arm |
| 17 | `left_shoulder_yaw_joint` | -2.618, 2.618 | left arm |
| 18 | `left_elbow_joint` | -1.047, 2.094 | left arm |
| 19 | `left_wrist_roll_joint` | -1.972, 1.972 | left arm |
| 20 | `left_wrist_pitch_joint` | -1.614, 1.614 | left arm |
| 21 | `left_wrist_yaw_joint` | -1.614, 1.614 | left arm |
| 22 | `right_shoulder_pitch_joint` | -3.089, 2.670 | right arm |
| 23 | `right_shoulder_roll_joint` | -2.252, 1.588 | right arm |
| 24 | `right_shoulder_yaw_joint` | -2.618, 2.618 | right arm |
| 25 | `right_elbow_joint` | -1.047, 2.094 | right arm |
| 26 | `right_wrist_roll_joint` | -1.972, 1.972 | right arm |
| 27 | `right_wrist_pitch_joint` | -1.614, 1.614 | right arm |
| 28 | `right_wrist_yaw_joint` | -1.614, 1.614 | right arm |

Position limits come from the upstream `mujoco_menagerie` MJCF
(which the menagerie pins to the published Unitree G1 spec sheet).
Velocity and effort limits in the manifest are conservative
published-spec values used for capability matching only — MuJoCo's
`mj_step` honours the MJCF's own `ctrlrange`, not these.

## Tests

- Sim: `tests/sim/test_g1_hal_mujoco.py` (32 tests) runs `G1MujocoHAL`
  against real MuJoCo physics on the Menagerie MJCF: lifecycle, schema-drift
  guard, per-section convergence on legs, waist, left arm and right arm, the
  head camera, and the `body_twist` glide and walking paths.
- HIL: planned alongside the real-HW HAL under M2.

## Asymmetric joint conventions

The G1 mirrors its arm conventions left vs right: positive
`left_shoulder_roll_joint` rolls the arm AWAY from the torso, positive
`right_shoulder_roll_joint` rolls it INTO the torso. One uniform target for all
joints is therefore self-colliding on one side: physical reality, not a HAL
bug. Tests validate each section independently.

## See also

- [`python/hal/README.md`](../../python/hal/README.md) — `G1MujocoHAL`, supported robots.
- CLAUDE.md §3 (the seven layers; dual-system pattern, S0 cerebellum for humanoids).
- [`docs/architecture/repo-state-map.html`](../../docs/architecture/repo-state-map.html) — HAL · Unitree G1 (sim, MuJoCo) green block.
