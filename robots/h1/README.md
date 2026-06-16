# `h1` — Robot description

Canonical `RobotDescription` manifest for the **Unitree H1** humanoid —
a 19-DoF bipedal robot (2 × 5 legs + 1 torso + 2 × 4 arms; no
end-effectors on this menagerie variant). The same manifest covers
the real Unitree H1 (over `unitree_sdk2`, M2 milestone) and the
real-physics MuJoCo digital twin (`H1MujocoHAL` on the
`mujoco_menagerie` MJCF).

The H1 is the **predecessor** to the Unitree G1 — bigger, taller,
coarser-DoF (no wrists, no waist roll/pitch, single-DoF ankles). See
the [`g1`](../g1/README.md) sibling for the newer humanoid.

## Vendored URDF (`h1.urdf`)

`assets.urdf` points at the committed `robots/h1/h1.urdf`, a vendored copy of
`robot_descriptions`' `h1_description` (Unitree `unitree_ros`, **BSD-3-Clause**)
with **only the joint names patched**: the upstream suffixes every joint with
`_joint` (`torso_joint`, `left_knee_joint`, …), which would make
`robot_state_publisher`'s `/tf` use names that don't match the HAL's
`/joint_states` (the manifest's bare `torso`, `left_knee`, …). `openral robot
vendor-urdf h1 --upstream rd:h1_description --out robots/h1 --raw-text` strips
the `_joint` suffix from joint names **on the raw URDF text** — link names,
geometry, inertials and the `package://h1_description/...` mesh paths are
byte-identical to upstream (ADR-0057 §4). End users need no xacro tooling and
no joint-name reconciliation at runtime.

## What this is — and what it isn't

The MuJoCo digital twin is a **HAL contract validator**, not a useful
humanoid sim. The H1 has a floating base and no S0 cerebellar
controller; left to its own devices it falls over under gravity. The
closed-loop sim tests therefore run with `gravity_enabled=False`.

The twin is the right tool for verifying:

- 19-DoF joint-position action layout,
- lifecycle wiring (`connect → read_state → send_action → estop`),
- joint indexing and ordering,
- `RobotDescription` round-trip,
- embodiment / VLA tag plumbing.

It is **not** the tool for rolling out a humanoid policy that walks or
balances. Balance + walking live under CLAUDE.md §6.2 — the C++ S0
cerebellum tracked under the M2 milestone — and are explicitly out of
scope here.

## Torque vs position actuators

The H1 menagerie MJCF ships **torque** (`motor`) actuators rather
than the position actuators the G1 / UR / Franka / SO-100 MJCFs use.
Writing `ctrl[i] = x` applies `x` N·m directly, not "drive joint i to
position x". `H1MujocoHAL` therefore runs a software PD position loop
every physics step (`tau = kp * (target - q) - kv * dq` clamped to
`ctrlrange`) so the public action contract stays "position targets in
radians" — the same as every other `MujocoArmHAL` subclass. This
mirrors how the real `unitree_sdk2` driver works on hardware: the
bus is torque-mode but the user-facing API takes positions and runs
Kp/Kd in the driver layer.

The PD gains are sized so a 1-rad position error roughly saturates
each actuator at its peak torque (e.g. 300 N·m for the knees). These
are **contract-validation gains**, not balance / production gains —
the latter live in the future C++ S0 cerebellum.

## At a glance

| Field | Value |
| --- | --- |
| `name` | `h1` |
| `embodiment_kind` | `humanoid` |
| Joints | 19 actuated (2 × 5 leg + 1 torso + 2 × 4 arm). The MJCF's floating-base joint is implicit world state and is NOT enumerated in `joints`. |
| End-effectors | none — wrists aren't actuated on this menagerie variant. |
| Embodiment tags | `h1`, `unitree_h1`, `humanoid` |
| Supported VLA embodiments | `h1`, `humanoid_everyday_h1` |
| Supported control modes | `joint_position` |
| Locomotion | `bipedal` |
| Bimanual | yes |
| `sdk_kind` | `open` (menagerie MJCF + `mujoco` Python package, Apache-2.0) |
| `hal.sim` | `openral_hal.h1:H1MujocoHAL` (`deploy sim`) |
| `hal.real` | _null_ — sim-only until the M2 C++ S0 cerebellum (`deploy run` raises `ROSCapabilityMismatch`) |

## Pair with

| Component | Path |
| --- | --- |
| Python HAL adapter | `openral_hal.h1.H1MujocoHAL` (MuJoCo digital twin) |
| Python description | `openral_hal.H1_DESCRIPTION` |
| Sim test | `tests/sim/test_h1_hal_mujoco.py` |
| Sibling humanoid (newer) | [`robots/g1/`](../g1/README.md) — Unitree G1, 29-DoF |
| Future real-HW HAL | planned under M2 (CLAUDE.md §6.2), `unitree_sdk2` + C++ S0 cerebellum |

## Joints

The 19 actuated joints in canonical order (which matches both the
menagerie MJCF and the `Action.joint_targets[i]` slot order):

| Index | Name | Range (rad) | Effort (N·m) | Group |
| ---: | --- | --- | ---: | --- |
| 0 | `left_hip_yaw` | -0.43, 0.43 | 200 | left leg |
| 1 | `left_hip_roll` | -0.43, 0.43 | 200 | left leg |
| 2 | `left_hip_pitch` | -1.57, 1.57 | 200 | left leg |
| 3 | `left_knee` | -0.26, 2.05 | 300 | left leg |
| 4 | `left_ankle` | -0.87, 0.52 | 40 | left leg |
| 5 | `right_hip_yaw` | -0.43, 0.43 | 200 | right leg |
| 6 | `right_hip_roll` | -0.43, 0.43 | 200 | right leg |
| 7 | `right_hip_pitch` | -1.57, 1.57 | 200 | right leg |
| 8 | `right_knee` | -0.26, 2.05 | 300 | right leg |
| 9 | `right_ankle` | -0.87, 0.52 | 40 | right leg |
| 10 | `torso` | -2.35, 2.35 | 200 | torso (yaw only) |
| 11 | `left_shoulder_pitch` | -2.87, 2.87 | 40 | left arm |
| 12 | `left_shoulder_roll` | -0.34, 3.11 | 40 | left arm |
| 13 | `left_shoulder_yaw` | -1.3, 4.45 | 18 | left arm |
| 14 | `left_elbow` | -1.25, 2.61 | 18 | left arm |
| 15 | `right_shoulder_pitch` | -2.87, 2.87 | 40 | right arm |
| 16 | `right_shoulder_roll` | -3.11, 0.34 | 40 | right arm |
| 17 | `right_shoulder_yaw` | -4.45, 1.3 | 18 | right arm |
| 18 | `right_elbow` | -1.25, 2.61 | 18 | right arm |

Note the H1's H1-specific naming convention — no `_joint` suffix
(unlike the G1's `left_hip_pitch_joint`). The H1 menagerie ships
joints as `left_hip_pitch`, `left_knee`, etc.

Position limits + effort limits come from the upstream
`mujoco_menagerie` MJCF verbatim (the menagerie pins them to the
published Unitree H1 spec sheet).

## Tests

- Sim: `tests/sim/test_h1_hal_mujoco.py` exercises `H1MujocoHAL`
  end-to-end against real MuJoCo physics on the Menagerie MJCF:
  lifecycle, schema-drift guard (verifies the 19-joint order + the
  motor-actuator alignment), per-section convergence on the legs,
  torso, left arm, and right arm.
- HIL: planned alongside the real-HW HAL under M2.

## Asymmetric joint conventions

Like the G1, the H1 mirrors its arm conventions left vs right:
`left_shoulder_roll` ∈ [-0.34, 3.11] but `right_shoulder_roll` ∈
[-3.11, 0.34]. This means a "all joints at the same target value"
command is invalid for one side or the other — physical reality,
not a HAL bug. Tests deliberately validate each section
independently rather than driving every joint to a uniform target.

## See also

- [`python/hal/README.md`](../../python/hal/README.md) — `H1MujocoHAL`, supported robots.
- [`robots/g1/README.md`](../g1/README.md) — Unitree G1 (newer / more DoF humanoid sibling).
- CLAUDE.md §6.1 (8 layers) and §6.2 (dual-system pattern; S0 cerebellum for humanoids).
- [`docs/architecture/repo-state-map.html`](../../docs/architecture/repo-state-map.html) — HAL · Unitree H1 (sim, MuJoCo) green block.
