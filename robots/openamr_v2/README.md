# OpenAMRobot 2.0 (`openamr_v2`)

OpenAMRobot 2.0 combines an open-source differential-drive mobile base, a 350 mm
DOLD v4 lift and two Enactic OpenArm v2.0 arms with pinch grippers. With the lift
travel the shoulders sit between 1.00 and 1.35 m.

Status: **sim** (MuJoCo twin and Isaac Sim); the **real** HAL is composed and has
not yet run on hardware. Upstream's dual-arm units ship from 2027-04.

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
   wheel by wheel.
5. Give the root `base_footprint` a 0.1 kg inertial (`ixx = iyy = izz = 0.001`).
   Upstream's root is massless and empty, and Isaac's URDF converter drops such a
   link. Its two children (`base_link`, `lift_column_base_link`) then become two
   articulations fixed to the world, and the lift and arms fall outside the one the
   scene drives. Write the result with `ET.indent`.

## Simulation

- **MuJoCo:** `assets.mjcf: openarm:openamr_v2` resolves to an MJCF that
  `openral_hal._openamr_v2_assets` builds from this URDF and caches.
  `OpenAMRv2MujocoHAL` drives it. The 20-D action is: 3 planar base joints, the
  lift, then for each arm 7 joints and a gripper. `BODY_TWIST` takes `vx` and
  `wz` only. Run `openral deploy sim --config scenes/deploy/openamr_v2_empty.yaml`.
- **Isaac Sim:** run `scenes/deploy/isaac_openamr_v2_warehouse.yaml`, which
  places the robot in the shelf aisle of `warehouse_multiple_shelves.usd`.

## HALs

- **sim:** `OpenAMRv2MujocoHAL` drives one MJCF of the whole robot. In Isaac,
  `SimAttachedHAL` drives the imported URDF.
- **real:** `ComposedHAL` reuses the OpenArm instead of re-implementing it. Parts
  are routed by joint name:
  - **arm:** `OpenArmRealHAL`, built from `robots/openarm/robot.yaml` and unchanged.
    Its 16 joints carry the same names here. The lifecycle node wires its ros2_control
    transport (the composite delegates the seam), so 16-D OpenArm rSkills reach the
    arms as the same four-slot tick they send to the OpenArm. Upstream's real profile
    runs the same `openarm_hardware` over CAN-FD.
  - **base:** `PlatformBaseHAL` over the platform's own stack. It reads odometry
    from `/platform/odom` and publishes twist on `/platform/cmd_vel`. Nav2 drives the
    base through `MobileBaseBridge` (`/cmd_vel` → `BODY_TWIST` → this part).
  - **lift:** `HeldJointsHAL` at a declared height. The real lift has no
    ros2_control interface (its firmware owns it), so a command that would move it
    is refused.

Open before the first real run:

- **Platform remap.** Remap the platform's `/odom` and `/cmd_vel` onto
  `/platform/*`, and turn its EKF TF broadcast off. `MobileBaseBridge` owns `/cmd_vel`
  and `odom → base_footprint`, and upstream's EKF publishes `odom → base_link`, which
  would give `base_link` two parents.
- **Arm controllers.** The OpenArm bringup (`openral_hal_openarm`) and upstream's
  `arm_controllers.yaml` name the controllers differently
  (`left_joint_trajectory_controller` vs `left_arm_controller`), and upstream's
  grippers are `GripperActionController`s. Run OpenRAL's OpenArm bringup, or pass
  matching controller names in the arm part's `params`.
- **Lift height.** The declared lift height (`positions.lift_joint`, 0.0 = L1000) is
  what the safety kernel places the arms from. It must match the real lift before
  any motion, and a lift moved by its firmware is not seen. Reading the lift's stop
  sensors would replace the declaration.
- **CAN interfaces.** `openral detect` fills CAN interfaces only for top-level
  `hal.parameters.defaults` keys. The arm part's `left_can_interface` /
  `right_can_interface` sit inside `parts[0].params` and stay at their defaults.

## Known gaps

- **Collision model: reused, not refitted; Safety-WG sign-off owed.** The arms
  carry `robots/openarm`'s `collision_geometry` unchanged (box + hull per link,
  only the link names translated); the body is one box per URDF link, bounded
  from the MuJoCo twin's geoms (+1 mm), with `base_link`'s box covering the
  whole chassis subtree. The ACM is `openral collision lower --acm-only`
  (mjcf route: `select_lowering` now treats the partially resolving URDF as
  not usable for lowering) unioned with `openamr_v2.srdf`: OpenArm's six arm
  rows renamed, plus six lift-internal rows (adapter plate on the column top
  at the bottom of the stroke; the position flag and its rod through the stop
  sensors' forks). No arm link is exempted against any body part. Owed: a
  Safety-WG review and a private hazard-log entry for the model and those
  rows. Known holes:
  - The OpenArm hulls were fitted to the OpenArm MJCF's meshes. This robot's
    twin loads openarm_description's meshes, and the hulls miss them by
    0.13 to 0.69 mm per link and by 13.0 mm on link6. The boxes still hold
    them with at least 0.63 mm. `test_collision_geometry_enclosure.py`
    marks this robot as a strict xfail until the hulls are refitted or the
    gap is accepted.
  - Not modelled: the wrist cameras. They are visual-only placeholder cubes
    in the URDF.
- **Upstream placeholders.** Upstream marks these as placeholders: the wrist
  camera poses, the 0.400 m wheel track, the lift speed and effort (DOLD
  catalogue, "provisional") and every acceleration limit. Upstream has also not
  accepted any hardware physically yet. Re-sync at v2.0.0 (2026-12-18).
- **Placeholder sensors.** Camera intrinsics here are nominal, and the base
  speed caps (`safety.max_base_*`) are a conservative placeholder envelope.
- **Isaac base is holonomic.** The Isaac kinematic base integrates any twist,
  so it does not refuse `vy`; only the MuJoCo HAL refuses it.
