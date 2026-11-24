# `h2` — Robot description

Canonical `RobotDescription` manifest for the **Unitree H2** humanoid: 31
actuated joints (2 × 6 legs + 3 waist + 2 × 7 arms + 2 head), bare wrists, no
end-effectors. One manifest covers the MuJoCo digital twin (`H2MujocoHAL`) and
the real H2 (`UnitreeHgRealHAL` over NVIDIA's ros2_control stack, the G1's path —
see [Real hardware](#real-hardware)).

## Sources and licences

The H2 is in neither `mujoco_menagerie` nor `robot_descriptions`. The manifest
uses Unitree's official assets, both **BSD-3-Clause**:

| Asset | Upstream (pinned commit) | Path |
| --- | --- | --- |
| URDF | `unitreerobotics/unitree_ros` @ `41dec58c7a3ccb3050243ef16ac2ca74b2b0f9ca` | `robots/h2_description/H2.urdf` + 34 STL meshes |
| MJCF | `unitreerobotics/unitree_mujoco` @ `1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d` | `unitree_robots/h2/{h2_mujoco.xml,scene.xml}` + 32 STL meshes |
| Motor order, head gains | `unitreerobotics/unitree_sdk2` @ `8e1b927586712edd6a4c29d50dd52dff1e163439` | `example/h2/low_level/h2_ankle_swing_example.cpp` |
| Body PD gains, BODY_TWIST box | `unitreerobotics/unitree_rl_mjlab` @ `1425b15f73bd4095f0df53709d7c389c3eb9e790` (Apache-2.0) | `src/assets/robots/unitree_h2/h2_constants.py`, `src/tasks/velocity/config/h2/env_cfgs.py` |

### Fetched, not vendored

`assets.urdf.ref` and `assets.mjcf` are both `unitree:h2`
(`openral_core.assets`). `openral_hal._unitree_h2_assets` downloads every file at
the pinned commit from `raw.githubusercontent.com`, checks its **SHA-256**
against a table in the module, and installs it only on a match; a mismatch or a
failed download raises `ROSConfigError` and installs nothing (fail-closed). Cache:
`$OPENRAL_CACHE_DIR/unitree_h2/<repo>/<sha>/...` (default `~/.cache/openral`).

Why not vendor under `robots/h2/` like `robots/h1/h1.urdf`:

- Each mesh set is about 13 MB. The two sets are **not** byte-identical: the head
  and wrist meshes differ, and the URDF adds hand meshes.
- Both description files name their meshes by a path relative to the file
  (`meshes/*.stl`). With each file kept next to its own meshes in the cache, the
  upstream references resolve unchanged. MuJoCo's `meshdir` and yourdfpy's mesh
  lookup (collision lowering) both work, and no file is patched.
- No joint rename is needed. The upstream joint names already match the SDK
  (`left_hip_pitch_joint` and so on), so the H1's reason for a patched copy does
  not apply.

## Joints (unitree_sdk2 motor order)

`Action.joint_targets[i]` slot `i` is SDK motor index `i` (`H2_NUM_MOTOR = 31`).
Origins, axes and limits are verbatim from `H2.urdf`. Velocity and effort are the
URDF's hardware peaks. The safety envelope's `max_joint_speed_factor: 0.3`
applies the derating.

| Index | Name | Range (rad) | Effort (N·m) | Velocity (rad/s) |
| ---: | --- | --- | ---: | ---: |
| 0 / 6 | `{left,right}_hip_pitch_joint` | -2.4526, 2.7754 | 360 | 20 |
| 1 / 7 | `{left,right}_hip_roll_joint` | L -0.4674, 2.1689 · R -2.1689, 0.4674 | 360 | 20 |
| 2 / 8 | `{left,right}_hip_yaw_joint` | -2.827, 2.827 | 360 | 20 |
| 3 / 9 | `{left,right}_knee_joint` | -0.0873, 2.5303 | 360 | 20 |
| 4 / 10 | `{left,right}_ankle_roll_joint` | L -0.3491, 0.2967 · R -0.2967, 0.3491 | 19 | 100.7 |
| 5 / 11 | `{left,right}_ankle_pitch_joint` | -1.1345, 0.6109 | 66.88 | 28.61 |
| 12 | `waist_yaw_joint` | -1.7453, 1.7453 | 120 | 28.375 |
| 13 | `waist_roll_joint` | -0.5236, 0.5236 | 180 | 28.375 |
| 14 | `waist_pitch_joint` | -0.4363, 0.5236 | 180 | 28.375 |
| 15 / 22 | `{left,right}_shoulder_pitch_joint` | -2.618, 1.833 | 130 | 21.9 |
| 16 / 23 | `{left,right}_shoulder_roll_joint` | L -0.517, 2.494 · R -2.494, 0.517 | 60 | 18.7 |
| 17 / 24 | `{left,right}_shoulder_yaw_joint` | -2.618, 2.618 | 60 | 18.7 |
| 18 / 25 | `{left,right}_elbow_joint` | -0.986, 3.071 | 60 | 18.7 |
| 19 / 26 | `{left,right}_wrist_roll_joint` | -2.618, 2.618 | 60 | 18.7 |
| 20 / 27 | `{left,right}_wrist_pitch_joint` | -0.576, 0.576 | 10 | 37.7 |
| 21 / 28 | `{left,right}_wrist_yaw_joint` | -1.22, 1.22 | 10 | 37.7 |
| 29 | `head_pitch_joint` | -0.5236, 0.8378 | 50 | 10 |
| 30 | `head_yaw_joint` | -1.7453, 1.7453 | 50 | 10 |

Things that differ from the G1:

- **The ankle roll comes before the pitch** (indices 4/5).
- **There is a 2-DoF head.**
- **The MJCF tree order is not the SDK order.** In the tree, the head joints sit
  between the waist and the arms (qpos 22–23), so the manifest's `sim:` block
  carries explicit `joint_qpos_addr` / `joint_qvel_addr` maps. With the default
  `7 + i` mapping, the arm targets would drive the head. The actuator order is
  the SDK order, so the default actuator map holds.
- **Some joints are parallel linkages.** The ankles and the waist roll/pitch are
  parallel linkages on the hardware (`H2_loop.urdf` models the loops). The
  description and the twin model them as serial joints, which is the interface
  the SDK's `Mode::PR` (series pitch/roll) exposes.

## The MuJoCo twin

The MJCF ships **torque** (`motor`) actuators: `ctrl` is N·m. `H2MujocoHAL` runs
a software PD loop, `tau = kp (q* - q) - kd dq` clamped to `ctrlrange`, before
every `mj_step`. It also runs during `idle_step`, so a held pose is actively
held. The public contract stays "position targets in radians".

By default (`s0_controller: none`) the free joint is **pinned upright** every
step (the ADR-0087 kinematic glide, as on the G1). `BODY_TWIST` slides the pinned base in the plane, and the joints
keep their hold targets. The upstream MJCF spawns the pelvis at 1.2 m with the
feet 0.157 m in the air. On `connect`, the HAL lowers the pin so the lowest
foot-sphere surface sits on `z = 0`, which puts the pelvis at 1.0425 m.

**PD gains.** The body joints use the `unitree_rl_mjlab` H2 actuator config: legs
200/4, ankles 40/2, waist 150/3, shoulders + elbow + wrist roll 40/2, wrist
pitch/yaw 20/1. That config covers 29 joints and has no head, so the head uses
Kp 30 / Kd 1 from the `unitree_sdk2` H2 example. These are twin gains for
tracking position targets. They are not balance gains.

**Gravity under the pin (known limitation).** The pin teleports the base back to
its pose before each step. During each step the base is in free fall, so the
joints feel almost no gravity load. The G1 glide behaves the same way. The joint
convergence tests prove indexing and PD tracking. They do **not** prove that the
gains hold a gravity-loaded pose.

### Sim S0: `s0_controller: agile_velocity`

`H2MujocoHAL(s0_controller="agile_velocity")` (scene
[`scenes/deploy/h2_walk.yaml`](../../scenes/deploy/h2_walk.yaml)) runs the real H2's
S0 export — WBC-AGILE, the pinned and SHA-256-checked Hub bundle below — as a
**sim-only** Python S0 on the **unpinned** free base, mirroring the G1's
`gr00t_wbc` mode. Download the bundle first (`hf download nvidia/agile_wbc_unitree_h2
--revision 222da1df… --local-dir ~/.cache/openral/unitree/h2/agile_velocity`);
without it the HAL refuses with that command.

- The ONNX steps at its 50 Hz (every 20th 1 ms physics step) with its feedback
  tensors looped as the C++ `InferenceController` does
  (`openral_hal._h2_agile_velocity.LeappVelocityPolicy`, shared with the unit
  rollout test). Inputs are mapped by the export's `element_names`: pelvis angular
  velocity (MuJoCo free-joint `qvel[3:6]`, body frame), quaternion as
  `(qx, qy, qz, qw)`, `(vx, vy, wz)`, 31 joint positions / velocities in Isaac Lab
  breadth-first order. Verified each run in `tests/sim/test_h2_agile_s0_mujoco.py`:
  the export's own history outputs (projected gravity, 0.2 × angular velocity,
  joint-pos − default, 0.05 × joint vel) match the twin state to 1e-5.
- Each policy tick writes the 14 targets **and** kp/kd into the software PD of
  legs + waist roll/pitch; targets past the MJCF joint ranges are clamped, counted
  (`H2MujocoHAL.s0_clamp_counts`) and logged once per joint
  (`hal.h2.s0_controller.target_clamped`). The other 17 joints keep the policy's
  training gains (`agile.articulations.robot.default_joint_stiffness/_damping`).
- `BODY_TWIST` is the velocity command (zero = stand); a `JOINT_POSITION` on a
  policy joint, or a whole-vector one, is refused. A fall (pelvis < 0.6 m — the
  training termination is 0.615 m — or torso tilt > 45°) cuts every torque and
  refuses commands until `reset_to_pose` / reconnect.
- The twin starts in the policy's default pose (hip pitch −0.1, knee 0.3, ankle
  pitch −0.2) with the feet on a floor the twin adds (the bare MJCF has none).

**Measured** (2026-11-24, MuJoCo 3.8.0, stock MJCF dynamics): stands 10 s with
~0.01 m drift and < 1° torso tilt, pelvis settling from 1.04 to ~0.94 m (training
target 0.915 m). At `vx = 0.4` it walks ~2.1–2.5 m in 10 s (**~0.21 m/s, ~50 %
tracking**), < 3.5° tilt; `vy = 0.3` gives ~0.23 m/s; `wz = 0.5` turns ~4.3 rad in
10 s (~87 %). The policy stands still below ~0.2 m/s (the training's null-command
band). While walking it asks for ankle pitch up to ~1.1 rad against the MJCF's
0.611 rad limit (clamped on ~20 % of ticks at 0.4 m/s) and ankle roll to −0.76 rad
against −0.349 rad. The forward under-tracking did not move with any sim knob tried:
armature 0.01–0.1 (training uses 0.02), timestep 1 ms vs 5 ms (training: 200 Hz,
decimation 4), joint frictionloss 0.2 / 0.1 / 0.003 and damping 0 (WBC-AGILE's own
sim2mujoco recipe), widened ankle limits, feet-only collision (training's
`feet_only` profile), floor friction 2.0, stiffer contacts, elliptic cones, noslip.
The H2 Plus USD it was trained on is a conversion of this same `unitree_mujoco`
MJCF (with the same wrist fix), so the kinematics match; the gap is the
PhysX→MuJoCo contact/actuator model, not a mapping bug. The policy's upstream
sim2mujoco validation itself only gates on balance.

## Upstream disagreements (not papered over)

- **Mass.** The `H2.urdf` links sum to **75.59 kg**. The MJCF's explicit
  `<inertial>` masses sum to **67.04 kg**. The MJCF pelvis has no `<inertial>`,
  so MuJoCo infers about 4.25 kg from its mesh and the compiled model weighs
  **71.29 kg**. The real robot's mass is not verified here.
  `tests/sim/test_h2_hal_mujoco.py` pins all three numbers.
- **Wrist kinematics (resolved).** The pinned `H2.urdf` (`unitree_ros` @
  `41dec58c`) puts the wrist-pitch axis 0.079007 m past wrist roll; the pinned
  `h2_mujoco.xml` (`unitree_mujoco` @ `1eb6642e`) puts it at 0.037994 m. Unitree
  fixed the URDF in `unitree_ros` `25746e8c` / `2ded279c` (2026-11-03/04, "fix h2
  wrist_pitch_joint location"), moving the axis to 0.079007 m and offsetting the
  pitch link and yaw joint so every link stays put at q = 0. `unitree_mujoco` and
  `xr_teleoperate` still carry the pre-fix value; `unitree_rl_mjlab`'s independent
  H2 export (0.0806 m) and the roll link's clevis geometry (x 0.051–0.093 m) agree
  with the fix. `ensure_h2_mjcf` therefore returns a derived twin
  (`h2_mujoco.openral-<hash>.xml`) with Unitree's URDF fix applied, so the twin,
  the manifest and the safety kernel share one wrist kinematics. The two
  placements agree at q = 0 and differ by up to 23 mm at the hand at the
  ±0.576 rad pitch limit. `tests/sim/test_h2_hal_mujoco.py` pins this and fails
  if a re-pin of `unitree_mujoco` picks up the fix upstream.
- **Head and wrist meshes** differ between the two repos (see above).

## Sensors

`sensors: []`. The assets prove only frames:

- URDF fixed links `camera` (on `head_yaw_link`), `imu_in_torso` and
  `imu_in_pelvis`.
- MJCF sites `imu` (pelvis) and `secondary_imu` (torso). These match the SDK's
  `rt/secondary_imu` topic.

The camera and LiDAR model, intrinsics, extrinsic calibration and rates are
**unverified**: Unitree's product page could not be reached when this manifest
was written. A `SensorSpec` needs a rate and, for a camera, intrinsics, so none
is declared. `capabilities.has_vision` and `has_lidar` are `false` until a
datasheet or a calibrated unit pins them.

## Collision model and footprint

- `collision_geometry` and `allowed_collision_pairs` come from
  `openral collision lower --robot robots/h2/robot.yaml --write` with tight links
  `torso_link`, `{left,right}_shoulder_pitch_link`,
  `{left,right}_shoulder_yaw_link`, `{left,right}_elbow_link` and
  `{left,right}_wrist_yaw_link`. These are the G1's tight set minus
  `wrist_roll`, which has no collision mesh in `H2.urdf`. Without them the
  capsules around the pelvis/torso and arm/torso overlap at q = 0. The lowering
  uses URDF sampling.
- The URDF's fixed `left_hand_link` / `right_hand_link` carry collision meshes,
  so they are declared as `fixed_attachments` and are part of the kernel model.
- **Visual-only links are covered.** `H2.urdf` has `<visual>` but no
  `<collision>` on 11 links (`{left,right}_hip_pitch_link`,
  `{left,right}_ankle_roll_link`, `waist_yaw_link`, `waist_roll_link`,
  `{left,right}_wrist_roll_link`, `{left,right}_wrist_pitch_link`,
  `head_pitch_link`). With the twin's wrist frames fixed (above), the lowering
  folds the twin's meshes in and fits those links from their `<visual>` plus
  twin CAD, so the kernel sees the forearms. The enclosure test now passes for
  H2 (it was a strict xfail). `wrist_roll`↔`wrist_yaw` is exempt as always-
  colliding under the primitives, though the real meshes are 10.5 mm apart at
  rest and 0.5–0.8 mm apart at the pitch stops — flagged for Safety-WG review.
- **The ACM regeneration is a safety input and needs safety-WG review**
  (CLAUDE.md §3) before anything relies on it.
- `footprint_polygon` / `footprint_radius` (0.50 m) cover the whole body at the
  twin's default all-zero pose. With the elbows at zero, the forearms point
  forward, so the front edge reaches x = 0.47 m. Without the arms, x ≤ 0.18 m
  and y = ±0.19 m. The outline was computed from every visual-mesh vertex of
  `h2_mujoco.xml` posed by `mj_forward`, in the pelvis frame; see the comment in
  `robot.yaml`. It is pose-dependent.

## Real hardware

> **Status: yellow — untested on hardware.** Wired and unit/integration-tested;
> not yet run on an H2.

`hal.real` is `openral_hal.unitree_real:UnitreeHgRealHAL` and `hal.real_bringup`
is `openral_hal_unitree:real_bringup.launch.py`, as for the G1
([package README](../../packages/openral_hal_unitree/README.md#h2)). NVIDIA's
`unitree_hg_ros2_control` drives the 31 motors (`robot_variant: h2`); OpenRAL
supplies the robot_description NVIDIA does not ship
(`urdf/h2_real_hardware.urdf.xacro` around the pinned `H2.urdf`).

**S0 policy:** NVIDIA WBC-AGILE, `nvidia/agile_wbc_unitree_h2` @ `222da1df`
(NVIDIA Open Model License, commercial use permitted; trained on the **H2 Plus**),
in C++ as an `InferenceController` at 50 Hz under a 200 Hz controller manager. It
owns the 12 legs + waist roll/pitch; OpenRAL commands waist yaw, both arms and
the head (17 joints) and a planar base twist. A `JOINT_POSITION` action on a
policy-owned joint is refused. `deploy run` with `robot_id: h2` builds this HAL
from `hal.parameters.defaults` (`robot_model: h2`, `s0_policy: agile_velocity`),
and the bringup launch reads the same two values from this manifest.

`action_spec.control_freq_hz: 50` is the policy's rate; the twin now also takes
one 20 ms control period (20 physics steps) per action, and a `BODY_TWIST`
integrates over that period. The real-path safety fields (`max_ee_accel_m_s2`,
`contact_force_threshold_n`, `self_collision_margin_m`, the starting-pose ramp,
`joint_state_staleness_limit_s`) are the G1's provisional values — none was
measured on an H2.

**Open hardware gates** (issue #513; hazard-log Entry 074 and its Safety-WG
review): stay on a gantry until `freeze_controller` (a stiff hold, not balance)
is characterised; check the IMU frame (`imu_in_pelvis`) on hardware; the S0 loop
runs at 200 Hz; confirm the C++ chain clamps the policy's targets to the joint
limits (an open-loop rollout asks for a left ankle roll of −0.371 rad against the
−0.349 rad limit; walking in the MuJoCo twin, ankle pitch up to ~1.1 rad against
0.611 rad); Safety-WG review of the allowed-collision list and of this
bring-up.

## Policies

- **Real S0:** WBC-AGILE (above) — the only public H2 checkpoint known.
  `unitree_rl_mjlab` has `Unitree-H2-Flat` / `-Rough` training tasks but ships
  no checkpoint; NVIDIA GEAR-SONIC has an H2 config but no H2 weights.
- **Sim:** the kinematic glide by default; `s0_controller: agile_velocity` runs
  the real S0 export as a sim-only walking S0 (above). `BODY_TWIST` kernel bounds, `max_base_linear_speed_m_s: 0.5` and
  `max_base_angular_speed_rad_s: 0.5`, come from the `unitree_rl_mjlab` H2 play
  command box.

## At a glance

| Field | Value |
| --- | --- |
| `name` | `h2` |
| `embodiment_kind` | `humanoid` |
| Joints | 31 actuated; the floating base is implicit world state |
| End-effectors | none (bare wrists) |
| Embodiment tags | `h2`, `unitree_h2`, `humanoid`, `mobile_base` |
| Control modes | `joint_position`, `body_twist` (sim glide or sim AGILE S0; real: S0 velocity command) |
| Control rate | 50 Hz (`action_spec.control_freq_hz`) |
| `sdk_kind` | `closed_with_api` (real path links `unitree_sdk2`'s prebuilt core) |
| `hal.sim` | `openral_hal.h2:H2MujocoHAL` |
| `hal.real` | `openral_hal.unitree_real:UnitreeHgRealHAL` (yellow, untested on hardware) |

## Pair with

| Component | Path |
| --- | --- |
| Python HAL adapter | `openral_hal.h2.H2MujocoHAL` (sim), `openral_hal.unitree_real.UnitreeHgRealHAL` (real) |
| Python description | `openral_hal.H2_REAL_DESCRIPTION` (in-code mirror of this manifest; `H2_DESCRIPTION` is its sim baseline) |
| Real bringup | [`packages/openral_hal_unitree/`](../../packages/openral_hal_unitree/README.md) (`config/h2/`, `urdf/h2_real_hardware.urdf.xacro`) |
| Asset fetcher | `openral_hal._unitree_h2_assets` (`unitree:h2`) |
| Sim tests | `tests/sim/test_h2_hal_mujoco.py`, `tests/sim/test_h2_agile_s0_mujoco.py` (AGILE S0) |
| Real-HAL tests | `tests/unit/test_unitree_real_hal.py`, `tests/unit/test_unitree_real_bringup_config.py`, `tests/unit/test_h2_agile_velocity_rollout.py`, `tests/integration/test_unitree_real_hal_ros2_control_live.py` |
| Sibling humanoids | [`robots/g1/`](../g1/README.md), [`robots/h1/`](../h1/README.md) |

## See also

- [`python/hal/README.md`](../../python/hal/README.md): supported robots.
- [`docs/reference/robots.md`](../../docs/reference/robots.md): the robot catalogue.
- [`docs/architecture/repo-state-map.html`](../../docs/architecture/repo-state-map.html): HAL · Unitree H2 (sim, MuJoCo) block.
