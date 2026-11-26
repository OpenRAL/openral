# OpenAMRobot 2.0 (`openamrobot_v2`)

OpenAMRobot 2.0 combines an open-source differential-drive mobile base, a 350 mm
DOLD v4 lift and two Enactic OpenArm v2.0 arms with pinch grippers. With the lift
travel the shoulders sit between 1.00 and 1.35 m.

Status: **sim** (MuJoCo twin and Isaac Sim); the **real** HAL is composed and has
not yet run on hardware.

## Provenance

| What | Source | Licence |
| --- | --- | --- |
| Combined URDF | [`openAMRobot/openamr-upperbody-sw`](https://github.com/openAMRobot/openamr-upperbody-sw) @ `6687f65` (v2.0.0-rc.1 candidate 2, 2026-11-26: the pin of `openamrobot-manifest` branch `release/v2.0.0-rc.1-pins`; not yet tagged upstream), `ros2/src/openamrobot_upperbody_description/urdf/combined_robot.xacro` | MIT (software); CAD-derived geometry CERN-OHL-P-2.0 |
| Base meshes | [`openAMRobot/openamr-platform-sw`](https://github.com/openAMRobot/openamr-platform-sw) @ `2ac4070` (rc.1 candidate 2; base masses from the BOM 7.15 mass export, #126), fetched on first use (`openral_hal._openamrobot_platform_assets`) | MIT / CERN-OHL-P-2.0 |
| Arm and gripper meshes | [`enactic/openarm_description`](https://github.com/enactic/openarm_description), fetched on first use (our pin `c103041`; upstream builds against `14ff67b`, and the meshes are byte-identical between the two) | Apache-2.0 |

The OpenAMRobot name and logos are trademarks and are excluded from upstream's
licences.

## Regenerating `openamrobot_v2.urdf`

`upstream/combined_robot.urdf` is upstream's own expansion of `combined_robot.xacro`,
vendored byte-for-byte: every argument at its default (`column=dold_v4_350
roadmap_masses=true base_plate=bp000 head_pitch=hp30 base_tilt=bt10 arm_package=openarm
hardware_profile=mock enable_planar_base=false`) plus `sim_dynamics:=on`, upstream's
MuJoCo-facing output, in which each OpenArm joint carries the `<dynamics damping friction>`
of upstream's simulator profile. That profile is vendored unchanged as
`sim_dynamics_openarm_v2.yaml`; the MuJoCo twin reads the joint armature from it, since
URDF has no armature element.

`openamrobot_v2.urdf` is that file with OpenRAL's overlay, `tools/openamrobot_v2_urdf.py`,
which changes only the cameras and drops the Gazebo / ros2_control blocks:

- **Wrist cameras.** Upstream's are placeholders: 4 cm marker boxes beside the hand,
  looking along its +x, that never see the fingers
  (`tests/unit/test_openamrobot_v2_wrist_cameras.py`). The overlay drops the boxes and
  puts `left/right_wrist_camera_joint` on the OpenArm twin's `camera_wrist_left/right`
  mount on `openarm_*_ee_base_link` (`openarm_v20_bimanual.xml`: xyz
  `0.023 ±0.0182 -0.10236`, rpy `-2.100862 1.142449 -2.143093` left and
  `2.10087 1.142449 2.143093` right), the view the OpenArm rSkills were trained on.
- **Head camera.** Upstream's ZED assembly moves rigidly onto the head ZED's lens (see
  "Head camera"): `head_camera_mount_joint` xyz `0.258568 0.027277 0.267553`, rpy
  `-0.028403 0.621812 -0.00224` (the lens pose, 2.5 mm back along the view, then
  upstream's 30° pitch undone, so `head_camera_pitch_joint` and the bracket keep their
  upstream origins), and `head_camera_optical_joint` xyz `0.0025 0 0`.
- **`<gazebo>`, `<ros2_control>`, `<transmission>`.** Gazebo and controller_manager
  configuration that OpenRAL never reads; their `<joint>` entries would otherwise look
  like URDF joints to a parser walking the whole tree.

Everything else is upstream's, including the two `continuous` drive-wheel joints (the MuJoCo
twin drives the base through them, see "Simulation") and the
massless root `base_footprint` (the Isaac scene gives a massless branching root a 1 g
inertial at import, `strip_urdf_mimics` in `tools/isaac_manifest_scene.py`).
`tests/unit/test_openamrobot_v2_urdf_overlay.py` checks that the committed file is the
overlay of the vendored one and that nothing else differs.

To re-sync, clone the three repos at the pinned commits (upperbody at the commit in the
provenance table, and the `openarm_description` / `openamr-platform-sw` commits it pins in
its `config/upstream_pins.yaml`), then, with ROS 2 Jazzy sourced (only `xacro` is used, no
colcon build):

```bash
python tools/openamrobot_v2_urdf.py --expand UPPERBODY PLATFORM OPENARM_DESCRIPTION
```

Copy `config/sim_dynamics/openarm_v2.yaml` over `sim_dynamics_openarm_v2.yaml`, bump
`_PINNED_SHA` in `openral_hal._openamrobot_platform_assets`, and re-run
`openral collision lower --robot robots/openamrobot_v2/robot.yaml --acm-only`. Upstream's
own model, unmodified, opens in RViz with its `openamrobot_upperbody_bringup`
`view.launch.py`.

## Mass

The masses are upstream's: the base from its reviewed mass export (BOM 7.15,
`openamr-platform-sw` #126: `base_link` 49.948 kg, complete base 58.163 kg) and every
upper-body link from the roadmap mass export (`roadmap_masses=true`). The whole robot
is 87.11 kg. Before 2026-11-11 the vendored URDF
carried 31.5 kg (a 4.2 kg `base_link`). The MuJoCo twin derives its base gains from
the compiled mass (`openral_hal._openamrobot_v2_assets`, a critically damped tracker at
a fixed natural frequency), so the base tracks a twist identically at either mass
(0.2 m/s ramp: 17.5 mm lag, no overshoot); the lift holds within 2.3 mm under the
new carriage masses.

## Simulation

- **MuJoCo:** `assets.mjcf: openarm:openamrobot_v2` resolves to an MJCF that
  `openral_hal._openamrobot_v2_assets` builds from this URDF and caches.
  `OpenAMRobotV2MujocoHAL` drives it. The 20-D action is: 3 planar base joints, the
  lift, then for each arm 7 joints and a gripper. `BODY_TWIST` takes `vx` and
  `wz` only. Run `openral deploy sim --config scenes/deploy/openamrobot_v2_empty.yaml`.
  The base rolls on upstream's two drive wheels, as upstream's Gazebo `DiffDrive` and
  the real base do: a twist becomes the wheel speeds `(vx ∓ wz·h)/r`, with the radius
  `r` (0.098 m) and half-track `h` (0.200 m, upstream's placeholder) read from the URDF
  the twin is built from, ramped by upstream's own DiffDrive acceleration limits
  (0.5 m/s², 1.0 rad/s², read from `upstream/combined_robot.urdf`), held until the next
  twist and ramped to a stop after 0.5 s without one (`diff_drive_controller`'s
  timeout). An e-stop stops the wheels at once; at zero speed they brake (MuJoCo
  `intvelocity` servo, `_WHEEL_KP`/`_WHEEL_KV`, tuning values rather than motor data).
  The planar base joints (and a `base_z` slide that lets gravity load the wheels) are
  unactuated and measure the body: `/joint_states` carries the ground truth, `/odom` the
  wheel odometry, so wheel slip shows up as odometry drift, as on the real base. The
  four casters are frictionless spheres (MuJoCo has no swivel caster); roll and pitch
  stay rigid, so the drive wheels carry the base's whole weight. Gravity is on, and
  everything on the lift carriage is gravity-compensated so the OpenArm v2 gains hold
  the arms. Command the lift at its rated 0.018 m/s: a step throws the carriage at the
  actuator's 3 kN limit and its reaction lifts the base off its wheels.
  Cameras and viewers draw the visual meshes: the collision geoms sit in group 3,
  which MuJoCo does not render by default (contacts and the lidar ray-cast still hit
  them). Colours are upstream's URDF materials; the upper body is still upstream's
  placeholder boxes.
- **MuJoCo, walled room:** `scenes/deploy/openamrobot_v2_walls.yaml` puts the twin in a
  20 m room with three pillars at known poses and runs SLAM and Nav2. The base rolls
  on its wheels and `/joint_states` carries the ground truth, so wheel-odometry drift
  and SLAM error can each be checked against it. A 4 m Nav2 goal (2026-11-26): Nav2
  succeeded, stopping 3.83 m out, with odometry, map and ground truth within 0.015 m
  of each other (driving straight the wheels do not slip; turning on the spot they
  scrub, about 0.05 rad of odometry yaw drift over a 0.5 rad/s arc). Before
  `/scan` published missed beams as `+inf`, the map under-counted travel by ~14 %
  (3.72 m mapped, 4.31 m driven) and Nav2 overshot, as it did in the Isaac aisle.
- **Isaac Sim:** run `scenes/deploy/isaac_openamrobot_v2_warehouse.yaml`, which
  places the robot in the shelf aisle of `warehouse_multiple_shelves.usd`. The scene
  runs with the voxel map on: octomap maps the head ZED's depth into the safety
  kernel's voxel check and the robot console's 3D view. The camera looks 65.6° down
  from 1.45 m (lift 0.35 m), so its view meets the floor no farther than 2.4 m ahead and
  2.2 m to the side: from the spawn, 3.8 m from either rack, no shelf is mapped; a shelf
  enters the map once the robot stands at it (`shelf_1_props`, 0.8 m off Shelf_1).

## Head camera

The head camera is a ZED Mini (`head_zed` depth + `top` RGB, the key OpenArm rSkills
read as `observation.images.top`) with OpenArm's intrinsics, at the fixed OpenArm shelf
cell's head-camera pose relative to the arm bases: 65.6° down, at the tote the robot
carries on its lift carriage and at the shelf. The robot owner confirmed the real head ZED
looks down at the tote and shelf like the cell's; the pose is a corrected design pose, not
measured on a robot (`static_transform_calibrated: false`). The arm bases sit on the lift
carriage at y = ±0.031 m with OpenArm's orientation, so the cell's `openarm_base` (their
midpoint) mount becomes `openarm_left_base_link` + (0.077587, -0.003761, 0.099276) with
OpenArm's measured rpy, and the camera rides the lift with the arms. The lens then moves
2.013 mm forward (x 0.0796), same view direction: at the cell's x it sat 1 mm inside the
Getac placeholder's collision box, and the octomap self-filter, which casts every depth
pixel from the lens into the robot's collision boxes, entered that box at depth 0 on
every pixel, so the whole head_zed image read as robot. It now clears every box by 1 mm.
Upstream's own ZED assembly (`head_camera_link` body box and `head_camera_bracket_link`,
on the carriage pitched 30° down, a view that saw only the shelf with the tote below the
image) is moved rigidly onto this lens (the overlay, "Regenerating"): the body box sits 2.5 mm
behind the lens and `head_camera_optical_frame` is the lens. At this design pose the ZED
body reaches through the Getac placeholder into the arm profile (both are ACM rows with
`head_camera_link`): on the real robot, check how the camera and tablet actually mount.
From the lens, `top` (the policy camera) sees only the base's front deck; the wider
`head_zed` also sees the Getac face and the head light below it (its bottom rows look 5°
past straight down) and, when they swing below it, the arms. The self-filter masks
those. `tests/unit/test_openamrobot_v2_head_zed.py` checks the intrinsics, that TF, the
Isaac mount and the MuJoCo twin place the camera at one pose, that it looks down in
front of the arms, the lens's clearance from every collision box, and what each camera
sees of the robot.

## Costmap clearing

`scenes/deploy/openamrobot_v2_walls_clearing.yaml` adds two boxes that sink out of the
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
(`+inf`) clears nothing.

In Isaac the lidar sits inside the chassis collision box (`lidar_link` at x 0.33 m,
z 0.124 m; the box spans x ±0.39 m, z up to 0.197 m). PhysX reports a ray that starts
inside a shape as a hit at distance 0. The scan used to step 1 mm past each self hit and
give up after 8 steps, so almost every beam read "no return". In
`warehouse_multiple_shelves`, 66 of 360 beams returned, which is why the first map
showed only side wedges and nothing ahead. In `full_warehouse`, no beam returned and
the map stayed empty. The scan now uses PhysX's every-hit query and drops the robot's own
hits (`resolve_beam_range`).

## HALs

- **sim:** `OpenAMRobotV2MujocoHAL` drives one MJCF of the whole robot. In Isaac,
  `SimAttachedHAL` drives the imported URDF.
- **real:** `ComposedHAL` reuses the OpenArm instead of re-implementing it. Parts
  are routed by joint name:
  - **arm:** `OpenArmRealHAL`, built from `robots/openarm/robot.yaml` and unchanged.
    Its 16 joints carry the same names here. The lifecycle node wires its ros2_control
    transport (the composite delegates the seam), so 16-D OpenArm rSkills reach the
    arms as the same four-slot tick they send to the OpenArm. Upstream's real profile
    runs the same `openarm_hardware` over CAN-FD.
  - **base:** `PlatformBaseHAL`, **odometry-only**. It reads the platform's odometry
    from `/odom` and publishes no twist: the manifest sets no `cmd_vel_topic`,
    so a non-zero `BODY_TWIST` is refused (`ROSCapabilityMismatch`) before any part
    moves, and a zero twist is a no-op. See "Command ownership" below.
  - **lift:** `HeldJointsHAL` at a declared height. The real lift has no
    ros2_control interface (its firmware owns it), so a command that would move it
    is refused. When upstream exposes it, the lift becomes a stock `RosControlHAL` on
    `lift_controller` (upstream's mock and Gazebo profiles already have that
    controller, a position `JointTrajectoryController`). The swap is one `parts` entry,
    shown commented in `robot.yaml`. `ComposedHAL` merges it with the arm onto one
    ros2_control transport, and `/openral/estop` deactivates `lift_controller` with the
    arm controllers. `tests/sim/test_openamrobot_v2_lift_ros2_control.py` runs this against
    a real `controller_manager` with mock hardware. In sim the lift is already
    commanded: the MuJoCo twin and Isaac drive `lift_joint` from `JOINT_POSITION`.
    Upstream's register LIFT interlock (lift motion only with the arms stowed and the
    base stopped) is **designed, not yet implemented** in the lift firmware, and
    OpenRAL does not enforce it. Nothing provides a measured lift position to this HAL
    yet either; upstream's lift design includes position feedback, homing and validity.
    The model's continuous `lift_joint` is right for geometry; the permitted
    destinations (L1000, L1175, L1350) and the interlocks belong in a command policy,
    not in the model. The commanded lift waits on the implemented interlock, a measured
    position, that policy and Safety-WG sign-off.

Open before the first real run (agreed with the OpenAMRobot project in openral#497):

- **Command ownership.** On OpenAMRobot the final `/cmd_vel` is the
  `collision_monitor` output to the base controller, after the platform's `twist_mux`
  arbitration, its software stop and its smoother (`openamr-platform-sw` #90). A
  remapped copy of that topic is not an input: publishing there bypasses the platform
  chain, and `MobileBaseBridge`'s own Nav2 path bypasses OpenRAL's supervisor. The deploy
  command decides who owns navigation (ADR-0127 Decision 1, no setting):
  - `openral deploy sim`: OpenRAL owns SLAM, Nav2, odometry, TF and the base command, as
    the scenes in this directory do.
  - `openral deploy run`: the platform owns navigation, localization, odometry, TF and the
    final command. OpenRAL launches no SLAM or Nav2, attaches no `MobileBaseBridge`, reads
    the platform's `/odom`, and sends base motion as `navigate_to_pose` goals to the
    platform's Nav2 (the Nav2 rSkill joins the reasoner's palette once that server is
    advertised). A real scene that enables `enable_slam` or `enable_nav2` is refused.
    Direct base
    velocity, if needed, waits for a separately reviewed arbitrated input upstream
    defines, with its own priority, timeout, stale-message rejection and stop
    behaviour; only then does the base part get a `cmd_vel_topic`.
- **Stop.** The platform's software stop covers the base only. Its recovery contract is
  decided (`openamr-platform-sw` #119, closed 2026-11-07): engaging it cancels every
  navigation and docking goal, and release waits until those cancellations are confirmed
  and every direct `cmd_vel` input is quiet. It is not published yet: `openamr-platform-sw`
  #90 and its `StopState` interface (`openamrobot-interfaces` #35, `/safety/stop_state`)
  are still open. OpenRAL waits for both before building the two-way stop (ADR-0127
  Decision 4): each stop inhibits the other's base commands, a stop cancels the active
  goal, and recovery never restarts old work. OpenRAL's own half lands with openral#501: an
  OpenRAL e-stop cancels its in-flight Nav2 goal and nothing re-sends it. The
  hardwired E-stop stays separate from both.
- **TF.** On `deploy run` OpenRAL publishes no `odom → base` transform; the platform
  must publish `odom → base_footprint` (this manifest's `base_frame`, the composed URDF
  root). Upstream's legacy EKF publishes `odom → base_link` instead, which gives
  `base_link` two parents next to `robot_state_publisher`'s `base_footprint → base_link`
  and leaves `base_footprint` off the `odom` tree. Upstream is migrating its real bringup
  to the composed model; a real deploy waits for that.
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
- **Platform command timeout.** Any arbitrated twist input upstream defines for
  OpenRAL must time out on its own: if the HAL node dies, nothing republishes it.

## Known gaps

- **Pinned to an untagged release candidate.** The URDF follows v2.0.0-rc.1 candidate 2
  (upperbody `6687f65`, platform `2ac4070`, pinned on `openamrobot-manifest` branch
  `release/v2.0.0-rc.1-pins` on 2026-11-26), which upstream has not tagged yet. Against
  the earlier pin (upperbody PR #48 @ `e6ab3d1`, platform `713574f`) the geometry differs
  only in the drive wheels' loaded collision radius (0.100 → 0.098 m, platform decision
  of 15 Nov) and the ToF cliff sensors' pitch (20° → 10°, platform #93). Re-sync when
  rc.1 is tagged and at v2.0.0. Upstream publishes no expanded URDF or MJCF, so the
  expansion above stays ours. The upper body is still upstream's placeholder boxes: neither the lift
  column, the adapter plate nor the arm profile has CAD upstream yet.

- **Collision model: reused, not refitted; Safety-WG sign-off owed.** The arms
  import `robots/openarm`'s `collision_geometry` at load (`collision_geometry_imports`:
  box + hull per link, only the link names translated), so an OpenArm re-lowering
  reaches this robot; the body is one box per URDF link, bounded
  from the MuJoCo twin's geoms (+1 mm), with `base_link`'s box covering the
  whole chassis subtree. The ACM is `openral collision lower --robot robots/openamrobot_v2/robot.yaml --acm-only`
  (mjcf route: `select_lowering` now treats the partially resolving URDF as
  not usable for lowering) unioned with `openamrobot_v2.srdf`: OpenArm's six arm
  rows renamed, plus three lift-internal rows (the position flag through the stop
  sensors' forks), re-measured over the whole stroke on the 8 Nov geometry. The
  adapter plate (23 mm clear of the column) and the flag rod (1.9 mm clear of every
  sensor) lost their rows on 2026-11-11 and are checked again. No arm link is
  exempted against any body part. Owed: a
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
  - Not modelled: the wrist cameras (frames only in the URDF).
  - The head ZED body and bracket are upstream's placeholders moved to the design
    lens pose; they overlap the Getac and arm-profile placeholders (three ACM rows,
    2026-11-16). Owed with the rest: the measured camera mount.
- **Sim-only allowances; Safety-WG review and hazard-log entries owed.** This robot's
  sim twins run with allowances the real robot must not inherit, kept out of the
  manifest so the real HAL never reads them:
  - The Isaac twin's arms run position gains without gravity feed-forward (the stocking
    policy was trained on that sag), and the elbows settle 0.064-0.065 rad short of the
    stocking starting pose. The runner's `starting_pose_tolerance` param admits 0.08 rad
    for the Isaac deploys only; the manifest keeps the OpenArm cell's 0.05 rad. The
    0.08 also widens every other revolute joint's arrival check in that sim.
  - The shelf aisle scene carries the tote on the lift carriage as a rigid Isaac
    link attachment (no fixture, no slip); the real robot's tote mount is unspecified.
  - The head ZED pose is a design pose (`static_transform_calibrated: false`), moved
    2 mm off the Getac placeholder for the self-filter; the real mount is unmeasured.
  Owed: a Safety-WG review of each and a private hazard-log entry before any of them
  informs a real deployment.
- **Upstream placeholders.** Upstream marks these as placeholders: the wrist
  camera poses (replaced here by OpenArm's mount, see "Regenerating"), the 0.400 m wheel track, the lift speed and effort (DOLD
  catalogue, "provisional") and every acceleration limit. Upstream has also not
  accepted any hardware physically yet. Re-sync at v2.0.0 (2026-12-18).
- **Placeholder sensors.** Camera intrinsics here are nominal. The base speed caps
  (`safety.max_base_*`: 0.20 m/s, 0.5 rad/s) and the Nav2 footprint are upstream's
  own Nav2 values (`openamrobot_nav2/config/nav2_params.yaml` @ `2ac4070`, unchanged since `713574f`); its
  acceleration limits (1.0 / −2.5 m/s², 3.0 rad/s²) are not carried over, since the
  manifest has no field for them and OpenRAL's Nav2 profile keeps its own.
- **Isaac base is kinematic.** The Isaac base integrates any planar twist, so the
  HAL refuses `vy` before it reaches the sidecar (`SimAttachedHAL`, on the manifest's
  `base_kinematics: differential`), as the MuJoCo twin and `PlatformBaseHAL` do. The
  MuJoCo twin rolls on the wheels; the Isaac scene does not yet. It also position-holds
  every articulation DOF, so upstream's free wheel joints spin past 2π and PhysX
  refuses their drive targets (3,727 errors in one warehouse run, and an arm swung to
  its joint limit). Driving the Isaac base through its wheels is the next change.
