# `makermods_metal_station` — Robot description

Canonical `RobotDescription` manifest for a **MakerMods Metal Arm Station**:
two [`makermods_metal`](../makermods_metal/README.md) arms side by side on one
base plate, plus a top camera between them looking down at the shared
workspace. Each arm is the single-arm manifest's 6 revolute joints + parallel
gripper (two coupled prismatic fingers), with `left_` / `right_` prefixes on
every joint, link, end effector and wrist camera. The arms use Damiao motors
over CAN (MIT mode) and **have no brakes**.

Per-arm values must stay in lock-step with
[`robots/makermods_metal/robot.yaml`](../makermods_metal/robot.yaml); a unit test
guards it (see the manifest header).

## At a glance

| Field | Value |
| --- | --- |
| `name` | `makermods_metal_station` |
| `embodiment_kind` | `bimanual` |
| `base_frame` | `station_base` |
| Joints | 14 actuated: per arm 6 revolute + 1 gripper channel |
| End effectors | `left_gripper`, `right_gripper` — parallel grippers, normalised `[0 = closed, 1 = open]` |
| Reach | 612.5 mm per arm (vendor figure) |
| Sensors | `top` RGB on `station_base` (`observation.images.top`), `wrist_left` / `wrist_right` RGB on each `*_gripper_base` (`observation.images.wrist_{left,right}`) |
| Embodiment tags | `makermods_metal_station`, `makermods`, `bimanual` |
| Supported control modes | `joint_position`, `gripper_position` |
| Action / observation spec | 14-D joint positions @ 30 Hz (left arm first) / `(14,)` joint state |
| `sdk_kind` | `open` |
| `hal.sim` | `null` — `build_hal` derives `MujocoArmHAL.from_description` from the `sim:` block |
| `hal.real` | `null` — no real-hardware HAL yet |

## Measured geometry

MakerMods publishes no base-to-base spacing and no top-camera mount for the
station. Both are fitted from the real station's recordings in
[`makermods/pen_holder_bi_20260723_161553`](https://huggingface.co/datasets/makermods/pen_holder_bi_20260723_161553)
(41 bimanual pen-into-holder episodes, top + both wrist cameras, joint states in
degrees with the URDF's signs and zeros; fit 2026-11-07).

**Method.** One joint fit of three signals in the top camera, with both arms posed
by forward kinematics of the recorded joint states:

- arm-mesh silhouettes of both arms against dark-pixel masks in 40 random frames;
- 1 897 detections of the left wrist camera's green PCB;
- the pen-at-rest line in 39 episodes, which pins the right arm's base.

The arm bases are never in view; the spacing comes from the links and the pen.

| Quantity | Value | Error / checks | Where it lives |
| --- | --- | --- | --- |
| Arm spacing | **0.612 m**, arms at y = ±0.306 m | ±0.02 m. The held-out silhouette cost is flat over 0.60–0.64 m and rises clearly by 0.55 / 0.68 m. Independent kinematic check (pen axis vs holder at insertion, 40 episodes): 0.584 ± 0.011 m. The previous footage estimate, 0.78 m, gives a 93 px pen-line residual against 1.6 px here. | `STATION_ARM_SPACING_M` in `openral_hal._makermods_metal_assets`, and `fixed_attachments` `left_mount` / `right_mount` |
| Right base relative to left | (0.0069, −0.6119) m, yaw +0.9° | The 7 mm fore-aft offset and the 0.9° yaw are within the spacing error, so the model mounts both arms symmetrically with no yaw | — |
| Top camera position | (0.068, −0.001, 0.587) m in `station_base`: on the mast centred between the arms, ~7 cm **in front of** the base line, 0.587 m above the plate | ±1.5 cm | `sensors[top]` |
| Top camera orientation | Optical axis pitched 49.7° down, roll −2.2°, yaw −1.7° (REP-103 rpy `[-0.0377, 0.867, -0.0302]`); it meets the plate at (0.566, −0.016) m | ±1° | `sensors[top]` (`static_transform_xyz_rpy`; `sim_placement` target + image-up hint) |
| Top camera lens | f = 314.4 px, principal point (321.4, 241.0), 640×480 (VFOV 74.7°, HFOV 91°), no distortion fitted | f ±10 px. Residuals: PCB median 2.9 px (p90 6.1), pen line 1.6 px (p90 6.9), held-out silhouette 0.53 px mean | `sensors[top].intrinsics` |
| Wrist cameras | The single-arm refit, unchanged | FK fingertips and the FK object point land on the real fingertips and objects in both wrists; right-wrist pen-centre column error −5 px median | `sensors[wrist_*]` |

`station_base` is the midpoint of the two bases (the fit's `station_mid`); z = 0 is
the plate / desk top. Rendering the derived MJCF's top camera at recorded joint
states puts both grippers on the real ones in the dataset frames.

The fit assumes zero lens distortion and URDF meshes that match the real links;
cables and the wrist-camera housings are not modelled. The mounts stay
`static_transform_calibrated: false` because they come from a dataset fit, not a
measurement of the mount. If the station is rebuilt, re-fit and update the
asset constant, `fixed_attachments` and the camera together.

## Assets and licensing

Same source and licensing situation as the single arm: the upstream
[`makermods-robotics/metal-python-ros`](https://github.com/makermods-robotics/metal-python-ros)
repository has no LICENSE file (the vendor site says MIT), so **nothing is
vendored**. The `makermods:metal_station` ref makes
`openral_hal._makermods_metal_assets` fetch the pinned clone into
`$OPENRAL_CACHE_DIR` and derive:

- a two-arm URDF rooted at `station_base`, with `left_` / `right_` prefixed
  copies of the arm on fixed `left_mount` / `right_mount` joints;
- its MJCF, compiled by MuJoCo, with OpenRAL's sim-only `<position>` actuators
  and one finger `<equality>` per arm. The gains say nothing about the real
  Damiao loop.

See [`robots/makermods_metal/README.md`](../makermods_metal/README.md#assets-and-licensing)
for the full derivation.

## MJCF layout

| Arm | qpos | Actuators |
| --- | --- | --- |
| left | 0–5 `left_joint1..6`, 6 `left_joint7` (mimic), 7 `left_joint8` (→ `left_gripper`) | 0–5 joints, 6 gripper |
| right | 8–13 `right_joint1..6`, 14 `right_joint7` (mimic), 15 `right_joint8` (→ `right_gripper`) | 7–12 joints, 13 gripper |

Each `<side>_gripper` joint's child is the logical `<side>_finger_pair`; the
fingers `<side>_link7` / `<side>_link8` are `mimic_joints` rows, as in the
[single arm](../makermods_metal/README.md#gripper-kinematics).

## Sim usage

There is no scene or rSkill for the station yet; drive the bare twin through
the HAL:

```python
from openral_core import RobotDescription
from openral_hal import build_hal

desc = RobotDescription.from_yaml("robots/makermods_metal_station/robot.yaml")
hal = build_hal(desc, mode="sim")   # derived MujocoArmHAL, 14 joints
hal.connect()
print(hal.read_state().name)
hal.disconnect()
```

## Safety: self- and inter-arm collision model

`collision_geometry` / `allowed_collision_pairs` are generated from the derived
`makermods:metal_station` MJCF the same way as the single arm's (see
[`robots/makermods_metal/README.md`](../makermods_metal/README.md#safety-self-collision-model)):
every `left_` / `right_` link is a box plus its exact hull, `*_link1` in two slabs,
both fingers per arm as mimic links. Regenerate with

```bash
openral collision lower --robot robots/makermods_metal_station/robot.yaml --write
```

The exempt pairs are each arm's own adjacent pairs and its two fingers; **no pair
across the arms is exempt**, so the kernel refuses an arm-to-arm contact (pinned in
`tests/unit/test_collision_geometry_zero_pose.py`: both arms yawed inward and
reaching to the middle of the plate until the fingertips meet trips `left_link7`
against `right_link8`, where 21 finger mesh vertices are inside the other finger's
hull). The station start pose does not trip
(`tests/unit/test_collision_makermods_metal_poses.py`).

Checked once against all 25 987 recorded states of the pen-holder dataset (not a
committed test; the parquet is not vendored): **no arm-to-arm stop**. Four states
stop on `right_link1` against `right_link5` (−8.5 to −11.8 mm; episode 0 frames
1–3 and episode 8 frame 2). All four are the folded rest with the wrist in
`link1`'s pocket, a few mm from the real mesh, so this is a per-arm
conservatism of the `link1` slabs, unrelated to the spacing.

The arm spacing is measured to ±0.02 m (see [Measured geometry](#measured-geometry)).
At 0.612 m the two 0.6125 m reach spheres overlap by 0.613 m, so the arms share
most of the plate. Re-lower after any change to `STATION_ARM_SPACING_M`.

**Still owed**: Safety-WG
review, a hazard-log entry and a C++ kernel run are owed as for the single arm.

## Real hardware

No HAL exists yet (`hal.real: null`). The upstream driver runs one CAN bus per
arm (`two_arm_control.yaml`: right arm `can0`, left arm `can1`) through its own
`metal_controller` topics and `ArmJointPositionControl` messages, not
`ros2_control`. The integration options and the brakeless-arm safety
requirement are the same as for the single arm; see
[`robots/makermods_metal/README.md`](../makermods_metal/README.md#real-hardware).

> **Safety: neither arm has brakes. A de-energised arm falls.** A real HAL needs
> a safety-WG-reviewed stop behaviour (hold, or a damped descent), not a torque
> cut. The manifest's `safety` block holds sim values that were never measured
> on hardware.

## References

- Product page: <https://www.makermods.ai/metal-arm>
- Upstream description + driver: <https://github.com/makermods-robotics/metal-python-ros>
- Fetch / derive code: [`python/hal/src/openral_hal/_makermods_metal_assets.py`](../../python/hal/src/openral_hal/_makermods_metal_assets.py)
- Single-arm manifest: [`robots/makermods_metal/`](../makermods_metal/README.md)
