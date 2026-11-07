# `h2` — Robot description

Canonical `RobotDescription` manifest for the **Unitree H2** humanoid: 31
actuated joints (2 × 6 legs + 3 waist + 2 × 7 arms + 2 head), bare wrists, no
end-effectors. One manifest covers the MuJoCo digital twin (`H2MujocoHAL`) today
and the real H2 over `unitree_sdk2` later.

**Phase 1 scope:** description + MuJoCo twin + tests. `hal.real` is `null`: the
real-hardware HAL (`unitree_sdk2` low-level, `rt/lowcmd` / `rt/lowstate`) and the
C++ S0 whole-body controller are a later phase, owned separately, so
`deploy run --robot h2` raises `ROSCapabilityMismatch` until then.

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

The free joint is **pinned upright** every step (the ADR-0087 kinematic glide,
as on the G1). `BODY_TWIST` slides the pinned base in the plane, and the joints
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

## Upstream disagreements (not papered over)

- **Mass.** The `H2.urdf` links sum to **75.59 kg**. The MJCF's explicit
  `<inertial>` masses sum to **67.04 kg**. The MJCF pelvis has no `<inertial>`,
  so MuJoCo infers about 4.25 kg from its mesh and the compiled model weighs
  **71.29 kg**. The real robot's mass is not verified here.
  `tests/sim/test_h2_hal_mujoco.py` pins all three numbers.
- **Wrist kinematics.** `H2.urdf` places the wrist-pitch joint 0.079 m past the
  wrist roll. `h2_mujoco.xml` places it at 0.038 m, and its wrist-yaw offset
  absorbs the difference, so the hand frame lands in about the same place. The
  `unitree_rl_mjlab` H2 MJCF agrees with the URDF (0.081 m). The manifest
  follows the URDF, which is what the safety kernel's collision model is lowered
  from. The twin runs the MJCF, so the kernel's wrist-pitch frame is up to
  4.1 cm off the simulated one. A test pins the disagreement so that an upstream
  fix is noticed.
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
- **Known non-conservative gap:** `H2.urdf` has `<visual>` but no `<collision>`
  on 11 links: `{left,right}_hip_pitch_link`, `{left,right}_ankle_roll_link`,
  `waist_yaw_link`, `waist_roll_link`, `{left,right}_wrist_roll_link`,
  `{left,right}_wrist_pitch_link` and `head_pitch_link`. The URDF lowering
  therefore emits no primitive for them, and the kernel does not see the
  forearm (`wrist_roll_link`). `tests/unit/test_collision_geometry_enclosure.py`
  tracks this as a strict xfail. Closing it needs a lowering from visual
  geometry, or upstream collision meshes.
- **The ACM regeneration is a safety input and needs safety-WG review**
  (CLAUDE.md §3) before anything relies on it.
- `footprint_polygon` / `footprint_radius` (0.50 m) cover the whole body at the
  twin's default all-zero pose. With the elbows at zero, the forearms point
  forward, so the front edge reaches x = 0.47 m. Without the arms, x ≤ 0.18 m
  and y = ±0.19 m. The outline was computed from every visual-mesh vertex of
  `h2_mujoco.xml` posed by `mj_forward`, in the pelvis frame; see the comment in
  `robot.yaml`. It is pose-dependent.

## Policies

No pretrained H2 locomotion or whole-body checkpoint is public:

- `unitree_rl_mjlab` has `Unitree-H2-Flat` / `-Rough` training tasks but ships
  no checkpoint.
- NVIDIA GEAR-SONIC (`GR00T-WholeBodyControl`) has an H2 config but no H2 weights.

`BODY_TWIST` is therefore the kinematic glide only. Its kernel bounds,
`max_base_linear_speed_m_s: 0.5` and `max_base_angular_speed_rad_s: 0.5`, come
from the `unitree_rl_mjlab` H2 play command box.

## At a glance

| Field | Value |
| --- | --- |
| `name` | `h2` |
| `embodiment_kind` | `humanoid` |
| Joints | 31 actuated; the floating base is implicit world state |
| End-effectors | none (bare wrists) |
| Embodiment tags | `h2`, `unitree_h2`, `humanoid`, `mobile_base` |
| Control modes | `joint_position`, `body_twist` (sim glide) |
| `sdk_kind` | `open` |
| `hal.sim` | `openral_hal.h2:H2MujocoHAL` |
| `hal.real` | _null_ (later phase) |

## Pair with

| Component | Path |
| --- | --- |
| Python HAL adapter | `openral_hal.h2.H2MujocoHAL` |
| Python description | `openral_hal.H2_DESCRIPTION` (in-code mirror of this manifest) |
| Asset fetcher | `openral_hal._unitree_h2_assets` (`unitree:h2`) |
| Sim test | `tests/sim/test_h2_hal_mujoco.py` |
| Sibling humanoids | [`robots/g1/`](../g1/README.md), [`robots/h1/`](../h1/README.md) |

## See also

- [`python/hal/README.md`](../../python/hal/README.md): supported robots.
- [`docs/reference/robots.md`](../../docs/reference/robots.md): the robot catalogue.
- [`docs/architecture/repo-state-map.html`](../../docs/architecture/repo-state-map.html): HAL · Unitree H2 (sim, MuJoCo) block.
