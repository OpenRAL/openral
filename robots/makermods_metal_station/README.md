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

## Estimated geometry

**MakerMods publishes no base-to-base spacing and no top-camera mount for the
station.** The values below were triangulated from MakerMods' own footage, not
measured on hardware:

| Quantity | Value | How it was estimated | Where it lives |
| --- | --- | --- | --- |
| Arm spacing | 0.78 m (arms at y = ±0.39 m; plausible range 0.72–0.90) | Parked grippers' positions in the Station app's top-camera feed ([`metal-station/ask-bottle-app.mp4`](https://www.makermods.ai/metal-arm-station)) under the camera pose below | `STATION_ARM_SPACING_M` in `openral_hal._makermods_metal_assets`, and `fixed_attachments` `left_mount` / `right_mount` |
| Top camera position | Mast centred between the arms, ~0.19 m behind the arm-base line, lens ~0.62 m above the plate | Height: Station hero video, arm rest height (0.283 m) as scale. Setback: PnP below | `sensors[top]` (`static_transform_calibrated: false`) |
| Top camera orientation / lens | Facing forward, ~25° down; f ≈ 300 px at 640×480 (~94° HFOV) | PnP of the held cube (orange-blob centroid) against FK of the follower in [`HappyEthan/metal_pick_place`](https://huggingface.co/datasets/HappyEthan/metal_pick_place) episodes 0–16 (same kit stand; 244 points, median 3 px). Footage alone couples focal length and distance; the photo height fixes it | `sensors[top]` |
| Wrist cameras | sim placement on each `*_gripper_base` | `sensors[wrist_*]` (`static_transform_calibrated: false`) |

Measure the real station, then update the asset constant, `fixed_attachments`
and the camera transforms together. Until then, no frame derived from these
values is calibrated.

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
