# OpenAMRobot 2.0 (`openamr_v2`)

OpenAMRobot 2.0 combines an open-source differential-drive mobile base, a 350 mm
DOLD v4 lift and two Enactic OpenArm v2.0 arms with pinch grippers. With the lift
travel the shoulders sit between 1.00 and 1.35 m.

Status: **sim only** (MuJoCo twin and Isaac Sim). There is no real HAL.
Upstream's dual-arm hardware ships from 2027-04, and `OpenArmRealHAL` hard-codes
the OpenArm's 16 joints.

## Provenance

| What | Source | Licence |
| --- | --- | --- |
| Combined URDF | [`openAMRobot/openamr-upperbody-sw`](https://github.com/openAMRobot/openamr-upperbody-sw) @ `17cf693` (main, 2026-11-02), `ros2/src/openamrobot_upperbody_description/urdf/combined_robot.xacro` | MIT (software); CAD-derived geometry CERN-OHL-P-2.0 |
| Base meshes | [`openAMRobot/openamr-platform-sw`](https://github.com/openAMRobot/openamr-platform-sw) @ `131e500f`, fetched on first use (`openral_hal._openamr_platform_assets`) | MIT / CERN-OHL-P-2.0 |
| Arm and gripper meshes | [`enactic/openarm_description`](https://github.com/enactic/openarm_description), fetched on first use (our pin `c103041`; upstream builds against `14ff67b`, and the meshes are byte-identical between the two) | Apache-2.0 |

The OpenAMRobot name and logos are trademarks and are excluded from upstream's
licences.

## Regenerating `openamr_v2.urdf`

The file is `combined_robot.xacro`, expanded with its defaults
(`column=dold_v4_350 base_plate=bp000 head_pitch=hp30 base_tilt=bt10
arm_package=openarm hardware_profile=mock enable_planar_base=false`) and then
normalised:

1. Clone the three repos at the pinned commits (upperbody `config/upstream_pins.yaml`).
2. Build a scratch ament prefix with `share/<pkg>` links and
   `share/ament_index/resource_index/packages/<pkg>` markers for
   `openarm_description`, `openamrobot_description`,
   `openamrobot_upperbody_description` and `openamrobot_upperbody_control`.
   Prepend it to `AMENT_PREFIX_PATH`. A ROS Jazzy `xacro` is enough; no colcon build.
3. `xacro .../combined_robot.xacro > combined.urdf`.
4. Parse it, drop every `<gazebo>`, `<ros2_control>` and `<transmission>`
   element and all comments, and turn the two `continuous` drive-wheel joints
   into `fixed` joints. The base is driven as a planar body (`base_joints`), not
   wheel by wheel. Write the result with `ET.indent`.

## Simulation

- **MuJoCo:** `assets.mjcf: openarm:openamr_v2` resolves to an MJCF that
  `openral_hal._openamr_v2_assets` builds from this URDF and caches.
  `OpenAMRv2MujocoHAL` drives it. The 20-D action is: 3 planar base joints, the
  lift, then for each arm 7 joints and a gripper. `BODY_TWIST` takes `vx` and
  `wz` only. Run `openral deploy sim --config scenes/deploy/openamr_v2_empty.yaml`.
- **Isaac Sim:** run `scenes/deploy/isaac_openamr_v2_warehouse.yaml`, which
  places the robot in the shelf aisle of `warehouse_multiple_shelves.usd`.

## Known gaps

- **No `collision_geometry`.** `openral collision lower` refuses the model.
  Its URDF path fits each link's collision and visual geometry together, but it
  resolves `package://` meshes only on static links. `base_link` pairs a box
  collider with `package://openamrobot_description` visual meshes, so the link
  resolves only partly and the tool stops rather than under-cover it. The safety
  kernel therefore has no self-collision geometry for this robot yet. That needs
  Safety-WG work in `openral_safety.urdf_lowering`.
- **Upstream placeholders.** Upstream marks these as placeholders: the wrist
  camera poses, the 0.400 m wheel track, the lift speed and effort (DOLD
  catalogue, "provisional") and every acceleration limit. Upstream has also not
  accepted any hardware physically yet. Re-sync at v2.0.0 (2026-12-18).
- **Placeholder sensors.** Camera intrinsics here are nominal, and the base
  speed caps (`safety.max_base_*`) are a conservative placeholder envelope.
- **Isaac base is holonomic.** The Isaac kinematic base integrates any twist,
  so it does not refuse `vy`; only the MuJoCo HAL refuses it.
