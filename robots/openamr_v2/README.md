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
- **MuJoCo, walled room:** `scenes/deploy/openamr_v2_walls.yaml` puts the twin in a
  20 m room with three pillars at known poses and runs SLAM and Nav2. The base is
  kinematic, so odometry is exact and any map→odom drift is SLAM's error. A 4 m
  Nav2 goal: map and odom agree within 0.02 m and Nav2 stops at the goal. Before
  `/scan` published missed beams as `+inf`, the map under-counted travel by ~14 %
  (3.72 m mapped, 4.31 m driven) and Nav2 overshot, as it did in the Isaac aisle.
- **Isaac Sim:** run `scenes/deploy/isaac_openamr_v2_warehouse.yaml`, which
  places the robot in the shelf aisle of `warehouse_multiple_shelves.usd`.

## Head camera

The head camera is a ZED Mini (`head_zed` depth + `top` RGB, the key OpenArm rSkills
read as `observation.images.top`) with OpenArm's intrinsics, on upstream's head mount
(`head_camera_link`, on its bracket above the lift carriage, pitched 30° down). OpenArm's
own measured mount cannot carry over. The arm bases sit on the lift carriage as on the
OpenArm body, so the same mount relative to the arms puts the camera inside upstream's
Getac tablet: their centres are 1.5 cm apart, and every depth ray hit the tablet. The
view is therefore not OpenArm's: about 13 cm higher, and pitched 30° down instead of
65.6°. An OpenArm rSkill sees its workspace from another angle than it was trained on.
`tests/unit/test_openamr_v2_head_zed.py` checks the intrinsics, the TF/sim agreement,
and that no pixel of the view lands on the robot.

## Costmap clearing

`scenes/deploy/openamr_v2_walls_clearing.yaml` adds two boxes that sink out of the
lidar plane about 80 s into the run (sim time).
- **Box 0** has the east wall behind it. Once it was gone, the beam returned off the
  wall at 10 m, and Nav2 cleared its cells in both costmaps (100 → 0).
- **Box 1** was measured with the earlier 12 m range, on a bearing with nothing inside
  it. Its beams read `+inf`, which Nav2 discards, so its cells stayed lethal. The
  manifest now declares the RPLIDAR S3's 40 m, which reaches that corner (~14 m), so
  box 1 is expected to clear like box 0; that has not been re-run.

A real lidar behaves the same way: an obstacle that vanishes in front of open space
(nothing inside the range) is not cleared until a return passes through its cells.

## Lidar range

The base lidar is the RPLIDAR S3 (upstream `real_bringup.launch.py lidar_model:=s3`).
Its datasheet range is 0.05–40 m at 70 % reflectivity (15 m at 10 %, 5 m at 2 %).
slam_toolbox maps out to its `max_laser_range` (12 m). A return between 12 m and 40 m
still clears free space out to 12 m (Karto `AddScan`). A beam with no return at all
(`+inf`) clears nothing. With the 12 m fan clipped at 12 m, the warehouse aisle ahead of
the spawn had no return inside range, so the first map showed only the side wedges.

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
    is refused. When upstream exposes it, the lift becomes a stock `RosControlHAL` on
    `lift_controller` (upstream's mock and Gazebo profiles already have that
    controller, a position `JointTrajectoryController`). The swap is one `parts` entry,
    shown commented in `robot.yaml`. `ComposedHAL` merges it with the arm onto one
    ros2_control transport, and `/openral/estop` deactivates `lift_controller` with the
    arm controllers. `tests/sim/test_openamr_v2_lift_ros2_control.py` runs this against
    a real `controller_manager` with mock hardware. In sim the lift is already
    commanded: the MuJoCo twin and Isaac drive `lift_joint` from `JOINT_POSITION`.
    Upstream's firmware enforces register LIFT (lift motion only with the arms stowed
    and the base stopped). OpenRAL does not enforce it, so the commanded lift waits on
    that interlock and Safety-WG sign-off.

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
- **CAN interfaces.** A top-level `left_can_interface` / `right_can_interface` (a
  `deploy run` transport override, or a top-level `hal.parameters.defaults` key, the
  ones `openral detect` fills) reaches the arm part over its `params`: `ComposedHAL`
  hands every keyword to the parts whose constructor takes it. So does the manifest's
  `safety.joint_state_staleness_limit_s` (0.1 s), which bounds the odometry age too.
- **Platform command timeout.** If the HAL node dies, nothing republishes
  `/platform/cmd_vel`, and the base keeps its last twist unless the platform stops on
  its own. The platform's twist input must have a command timeout (micro-ROS firmware
  or `twist_mux`), as `diff_drive_controller`'s `cmd_vel_timeout` gives. Stale
  odometry no longer kills the node (`MobileBaseBridge` skips the `/odom` tick, and
  Nav2 stops on the stale TF), but the timeout is still the last line.

## Known gaps

- **Collision model: reused, not refitted; Safety-WG sign-off owed.** The arms
  import `robots/openarm`'s `collision_geometry` at load (`collision_geometry_imports`:
  box + hull per link, only the link names translated), so an OpenArm re-lowering
  reaches this robot; the body is one box per URDF link, bounded
  from the MuJoCo twin's geoms (+1 mm), with `base_link`'s box covering the
  whole chassis subtree. The ACM is `openral collision lower --acm-only`
  (mjcf route: `select_lowering` now treats the partially resolving URDF as
  not usable for lowering) unioned with `openamr_v2.srdf`: OpenArm's six arm
  rows renamed, plus six lift-internal rows (adapter plate on the column top
  at the bottom of the stroke; the position flag and its rod through the stop
  sensors' forks). No arm link is exempted against any body part. Owed: a
  Safety-WG review and a private hazard-log entry for the model and those
  rows. Known holes:
  - **link6 hull gap: found here, already on master, fixed for both robots.**
    Enactic ships two different link6 shapes. The `enactic/openarm_mujoco` v2
    mesh, which OpenArm's collision model was fitted to, fit its hull with 0 mm
    to spare. The `enactic/openarm_description` link6, which `robots/openarm`'s
    own URDF uses (robot_state_publisher, Foxglove, the Isaac import), was
    10.6 mm (CAD visual) to 13.0 mm (collision STL) outside it, at one corner
    around (−38, −24 to −32, +1 to +3.5) mm in the link frame. No mirroring
    explains it: the shapes differ. link3 and link5 were 1.8 mm outside, the
    other links 0.1 to 0.4 mm. The MJCF lowering now fits each link to the union
    of its MJCF and URDF meshes (`urdf_lowering._urdf_link_meshes`).
    `robots/openarm` was re-lowered with it and this robot re-copied.
    - Box growth: at most 1.9 mm per half-extent (link6
      [29.9, 32.2, 34.1] → [30.0, 34.1, 35.1] mm, link5 y/z +1.2/+1.0 mm).
    - The 28-pair OpenArm ACM is unchanged.
    - `test_collision_geometry_urdf_enclosure.py` holds both robots' URDF meshes
      inside their primitives.
    - Owed: Safety-WG review (a slightly larger arm can stop earlier near
      obstacles) and a hazard-log update for robots/openarm. Which shape the
      physical link6 has is still worth confirming with Enactic: the union covers
      both.
  - Not modelled: the wrist cameras. They are visual-only placeholder cubes
    in the URDF.
- **Upstream placeholders.** Upstream marks these as placeholders: the wrist
  camera poses, the 0.400 m wheel track, the lift speed and effort (DOLD
  catalogue, "provisional") and every acceleration limit. Upstream has also not
  accepted any hardware physically yet. Re-sync at v2.0.0 (2026-12-18).
- **Placeholder sensors.** Camera intrinsics here are nominal, and the base
  speed caps (`safety.max_base_*`) are a conservative placeholder envelope.
- **Isaac base is kinematic.** The Isaac base integrates any planar twist, so the
  HAL refuses `vy` before it reaches the sidecar (`SimAttachedHAL`, on the manifest's
  `base_kinematics: differential`), as the MuJoCo twin and `PlatformBaseHAL` do.
