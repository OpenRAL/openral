# `g1` — Robot description

Canonical `RobotDescription` manifest for the **Unitree G1** humanoid, a
29-DoF biped (2 × 6 legs + 3 waist + 2 × 7 arms; no end-effectors on this
menagerie variant). One manifest covers the real G1 (`UnitreeHgRealHAL` over
NVIDIA's Isaac ROS `ros2_control` stack, bringup in
[`packages/openral_hal_unitree`](../../packages/openral_hal_unitree/README.md)) and
the MuJoCo digital twin (`G1MujocoHAL` on the `mujoco_menagerie` MJCF).

## What this is — and what it isn't

The MuJoCo twin is a **HAL contract validator** with three sim-only base modes,
all with gravity on, picked by the `s0_controller` HAL parameter
(`hal: {defaults: {s0_controller: …}}` in a deploy scene, or `--hal`):

| `s0_controller` | What keeps the twin up | MJCF | Base |
| --- | --- | --- | --- |
| `none` (default) | ADR-0087 kinematic glide — the free base is re-pinned upright every physics step; no balance | Menagerie `g1.xml` | pinned |
| `playground_walk` (= legacy `walking_enabled: true`, used by `scenes/deploy/g1_vln.yaml`) | ADR-0089 pinned MuJoCo Playground ONNX walker (Apache-2.0) | its own policy-tuned MJCF | free while walking, pinned while idle |
| `gr00t_wbc` | NVIDIA GR00T Decoupled WBC Balance + Walk ONNX policies own legs + waist; S1 keeps the arms | Menagerie `g1.xml` | **free, never pinned** |

The joint-convergence sim tests zero gravity (`gravity_enabled=False`) only to
keep those assertions deterministic. It verifies:

- 29-DoF joint-position action layout,
- lifecycle wiring (`connect → read_state → send_action → estop`),
- joint indexing and ordering,
- `RobotDescription` round-trip,
- embodiment / VLA tag plumbing,
- the `body_twist` path under each `s0_controller` — every one a sim-only
  controller that a real HAL must never reuse.

### `gr00t_wbc`: GR00T Decoupled WBC as the sim S0

A Python port of the upstream MuJoCo runner (`decoupled_wbc/sim2mujoco/scripts/run_mujoco_gear_wbc.py`,
[`NVlabs/GR00T-WholeBodyControl`](https://github.com/NVlabs/GR00T-WholeBodyControl) @ `1e1ec4f5`)
in `python/hal/src/openral_hal/_g1_gr00t_wbc.py`:

- **Weights / licence.** `GR00T-WholeBodyControl-{Walk,Balance}.onnx` and the
  `NVIDIA Open Model License` text beside them are downloaded on first use from
  that pinned commit into `$OPENRAL_CACHE_DIR/g1_gr00t_wbc/<sha>/`, each
  SHA-256-verified, fail-closed (`ROSConfigError`). Code is Apache-2.0; the
  weights are **NVIDIA Open Model License** (commercial use permitted, posture
  `nvidia_open_model`) and are not bundled here. The HAL logs the posture and
  the licence-text path (`hal.g1.s0_controller`) at construction.
- **Policy.** 50 Hz; owns joints 0–14 (12 legs + 3 waist). Observation: one
  86-D frame (`[vx·2, vy·2, wz·0.5, height 0.74, torso rpy 0, pelvis ω·0.5,
  projected gravity, q − default, dq·0.05, last action]`), 6-frame history,
  516-D input; Balance runs when `‖(vx, vy, wz)‖ ≤ 0.05`, Walk otherwise.
  Targets `0.25·a + default` (legs `[-0.1, 0, 0, 0.3, -0.2, 0]`, waist 0).
  Base height and torso orientation stay at the upstream defaults: the
  `Action` schema has no field for them.
- **Actuation.** The Menagerie `<position kp=500>` actuators are turned into
  torque motors **in the loaded model** (the MJCF on disk is untouched); a PD
  law `τ = kp(q* − q) − kd·dq` runs every 2 ms physics step (implicitfast),
  clamped by the MJCF `actuatorfrcrange`. Legs kp `[150,150,150,200,40,40]` kd
  `[2,2,2,4,2,2]`, waist kp 250 kd 5 (upstream sim gains); arms GR00T's
  real-robot `MOTOR_KP [100,100,40,40,20,20,20]` / `MOTOR_KD [5,5,2,2,2,2,2]`
  (stable at 2 ms and as accurate as the upstream sim runner's kp 100 / kd 0.5,
  with the damping the hardware uses). The camera rig's visual-only floor is
  made to collide in the loaded model so the free base has ground to stand on.
- **S1 contract.** `body_twist` sets the held `(vx, vy, wz)` command (arm
  actions do not change it; an idle tick zeroes it). `joint_position` must name
  its joints (`Action.joint_names`, ADR-0102) and may name only the 14 arm
  joints; a command on a leg / waist joint, or a whole-vector 29-D action, is
  refused with `ROSCapabilityMismatch`. `idle_step` keeps the Balance policy
  running instead of pinning.
- **Fall latch.** Pelvis below 0.5 m or torso tilt above 45° cuts every joint
  torque and raises `ROSSafetyViolation` from `send_action` until
  `reset_to_pose` (which re-seats the base upright) or a reconnect.
- **Measured** (Menagerie MJCF; bake-off, 10 seeds per row, 5 ms physics;
  `tests/sim/test_g1_gr00twbc_mujoco.py` re-checks the headline rows at the
  HAL's 2 ms): stand 20 s 0/10 falls (drift 0.06 m); `vx 0.6` tracked at
  0.51 m/s; lateral pushes on the torso (100 ms) 0/10 falls up to 400 N
  standing and 300 N walking, 10/10 at 500 N; random / aggressive arm motion
  with 2 kg per hand while walking 0/10 falls. Policy latency ~0.2 ms median
  on one CPU thread.

`gr00t_wbc` stays **opt-in**: it fetches third-party weights under a licence
other than Apache-2.0, and the contract-validator default must stay
dependency-free and deterministic. `scenes/deploy/g1_vln.yaml` keeps
`playground_walk` until a `deploy sim` run validates `gr00t_wbc` end to end
behind InternVLA-N1.

On real hardware, balance and walking are NVIDIA's C++ S0 layer: a whole-body
ONNX policy in isaac_ros_deploy's `InferenceController` (GR00T Decoupled WBC by
default — `hal.parameters.defaults.s0_policy`; WBC-AGILE selectable) owns the legs
(GR00T: and the whole waist). `UnitreeHgRealHAL` sends it a base twist and commands
only the arms, refusing any action that targets a policy-owned joint; it never
reuses the sim walking controller. Its e-stop hands every joint to
`freeze_controller` (a position hold, **not** balance). Not yet run on a G1 — see
the package README for bring-up, rates (200 Hz manager / 50 Hz policy) and risks.

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
| Control rate | 50 Hz (`action_spec.control_freq_hz`; the sim S0 policy rate — one `send_action` = 10 × 2 ms physics steps) |
| Locomotion | `bipedal` |
| Bimanual | yes |
| `sdk_kind` | `closed_with_api` — the real path links `unitree_sdk2`'s prebuilt core; the sim path itself is fully open (menagerie MJCF + `mujoco`, Apache-2.0) |
| `hal.sim` | `openral_hal.g1:G1MujocoHAL` (`deploy sim`) |
| `hal.real` | `openral_hal.unitree_real:UnitreeHgRealHAL` (`deploy run`) |
| `hal.real_bringup` | `openral_hal_unitree:real_bringup.launch.py` |
| `hal.parameters.defaults` | `robot_model: g1`, `s0_policy: gr00t_decoupled_wbc` |
| `action_spec.control_freq_hz` | 20 Hz (= the twin's `body_twist_dt_s`; the real HAL's trajectory step) |

## Pair with

| Component | Path |
| --- | --- |
| Python HAL adapter | `openral_hal.g1.G1MujocoHAL` (MuJoCo digital twin) |
| Python description | `openral_hal.G1_DESCRIPTION` |
| Sim test | `tests/sim/test_g1_hal_mujoco.py`, `tests/sim/test_g1_gr00twbc_mujoco.py` (GR00T WBC S0) |
| Real-HW HAL | `openral_hal.unitree_real.UnitreeHgRealHAL` / `G1_REAL_DESCRIPTION` (this manifest's mirror) |
| Real-HW bringup | `packages/openral_hal_unitree` (NVIDIA `isaac_ros_robots` + `isaac_ros_deploy`, pinned) |
| S0 bundle builder | `tools/build_gr00t_wbc_leapp.py` (GR00T WBC → LEAPP bundle) |

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
Effort limits are the vendor's per-joint ratings from Unitree's
`g1_29dof.urdf` (`unitreerobotics/unitree_ros` @ `267182b8`): hip and waist
yaw 88, knee 139, ankle and waist roll/pitch 35, shoulder / elbow / wrist roll
25, wrist pitch/yaw 5 N·m (the menagerie MJCF rates a few joints higher; the
lower vendor value is used). Where the vendor rating is above the previous
OpenRAL cap, the previous cap is kept so this correction never loosens the
safety envelope: knee 88 (vendor 139) and ankles 30 (vendor 35) N·m. Velocity limits are deliberately far below the
URDF's 20–37 rad/s ratings. Both reach the safety kernel as per-joint caps;
MuJoCo's `mj_step` honours the MJCF's own `ctrlrange` / `actuatorfrcrange`, not
these.

## Tests

- Sim: `tests/sim/test_g1_hal_mujoco.py` (32 tests) runs `G1MujocoHAL`
  against real MuJoCo physics on the Menagerie MJCF: lifecycle, schema-drift
  guard, per-section convergence on legs, waist, left arm and right arm, the
  head camera, and the `body_twist` glide and walking paths.
- Sim: `tests/sim/test_g1_gr00twbc_mujoco.py` (13 tests) runs the
  `gr00t_wbc` S0 on the real pinned weights: observation layout and history,
  ONNX parity with a reference port of the upstream runner, 20 s stand, 0.5 m/s
  walk, arm tracking while walking, S0-joint refusal, 300 N push recovery, and
  the fall latch.
- Unit: `tests/unit/test_unitree_real_hal.py`, `tests/unit/test_unitree_real_bringup_config.py`
  (joint-name parity with NVIDIA's xacro, policy split, launch), and
  `tests/unit/test_gr00t_wbc_leapp_bundle.py` (bundle vs GR00T's runner, tick for tick).
- Integration: `tests/integration/test_unitree_real_hal_ros2_control_live.py` (real
  NVIDIA controllers over mock hardware; skips without the NVIDIA stack).
- HIL: none yet — the first real bring-up is a validation run on a gantry.

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
