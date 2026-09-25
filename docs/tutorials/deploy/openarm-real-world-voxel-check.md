# OpenArm real-cell world-voxel check (attended)

This runbook turns on the safety kernel's **world-voxel check** on the real OpenArm cell and
measures what it does. The ZED Mini cloud goes through `octomap_server` and
`openral_octomap_bridge` to `/openral/world_voxels`, and from there to the kernel's
`world_voxel_enabled` phase. It is the plan the draft decision record *ADR-0109, "One kernel
world representation"* calls the *attended Thor + OpenArm evaluation*. Until now no shipped
real-hardware scene enabled this check. `scenes/deploy/openarm_zed_octomap.yaml` feeds the
real ZED into the map, but it drives a MuJoCo twin and keeps the kernel check off.

!!! danger "This drives real motors"
    Bringing the OpenArm graph up **moves the arms**: the vendor hardware's `on_activate`
    steps all 16 motors to zero, **unramped**, before any OpenRAL code runs. The kernel and
    the deadman are backstops, not a substitute for a hand on the hardware E-stop. Operator
    presence authorizes **one** bringup or dispatch. Re-confirm it before every one, and
    never leave a script queued to dispatch after an unattended wait.

Each step is marked **[offline]**, meaning any workstation with ROS 2 Jazzy and this
checkout, or **[human, rig]**, meaning a person at the cell.

## What to expect: two known gaps on current master

The evaluation exists to measure these gaps, so expect each one to show up rather than
expecting a clean pass:

1. **The arm sees itself — now filtered, with a blind shell to accept.** The sim HAL's
   depth synthesis makes the robot transparent (`exclude_body_ids`); a real camera cannot.
   `deploy run` therefore routes the depth cloud through `openral_octomap_bridge`'s
   `robot_self_filter` before `octomap_server`: it poses the kernel's own collision
   primitives at the cloud's capture stamp (joint states from the topic the runtime nodes
   read: the scene's `joint_states_topic`, else the HAL's `~/joint_states` republish;
   camera pose from tf2) and removes every return within the rig's
   `runtime.robot_self_filter_padding_m` (default 5 cm, provisional: derive it from depth
   noise, extrinsic error, capture-to-joint-state motion and half a voxel) of them, plus a held payload's primitives once an
   attachment producer exists. Without a pose at the capture stamp it drops the cloud, so
   the map goes stale and the kernel fails closed rather than seeing the arm as an obstacle.
   The cost is a padding-wide shell around the arm in which the kernel cannot see a real obstacle
   (hazard log). Before the filter, the ZED's view of the left gripper stopped tick 1
   (`safety.collision kind=world a=openarm_left_finger_pair`). Record the filter's
   `self-filter:` log line (share of points removed, ms per cloud, drops).
2. **A grasped object stops the gripper holding it.** Real hardware runs with
   attached-payload checking **off** (`_attached_collision_enabled("real")` returns `False`
   in `deploy_e2e.launch.py`). Nothing on the real graph publishes
   `/openral/attachment_state`: the HAL's vision attachment leg exists but is opt-in and not
   wired into the deploy launch. So the bridge never clears the object's cells, and the
   gripper stops against its own payload. The sim contract, where the payload leaves world
   occupancy and is checked as attached geometry, does **not** hold here.
Camera loss **fails closed** (it used to fail open; fixed with hazard-log Entry 033):
`openral_octomap_bridge` stops publishing `/openral/world_voxels` once its last octree is
older than `max_octree_age_s` (default 1.0 s, equal to the kernel's `world_voxel_deadline_ms`;
both are `DeployRuntime` fields), and
the kernel's deadline then turns the silence into `DROP_VOXEL_UNAVAILABLE`, at most ~2.0 s
after the last cloud. Verified on Thor with the ZED stopped (bridge half).

The exit criteria in step 5 cannot all be met until gap 1 is fixed. Record the
numbers anyway: they are the evidence those fixes need.

## Hosts

The cell (ZED Mini, PEAK CAN adapter `openarm_left` / `openarm_right`) is attached to
**Thor**. The ZED leg of this pipeline has been verified only on the lab **Orin**
(`qorin1`, 2026-09-07: ZED SDK 5.4.1, `/octomap_point_cloud_centers` at about 5 Hz,
`/openral/world_voxels` at about 10 Hz). On Thor, confirm each of these before step 1:
nothing here has been proven there.

- ZED SDK and `zed_wrapper` are built for Thor's JetPack, and their overlay is sourced.
- `ros-jazzy-octomap-server` is installed.
- `openral_octomap_bridge` is built into the overlay you source (`ros2 pkg prefix openral_octomap_bridge`).
- The checkout is on this branch. Thor's checkout has diverged from the pushed branches
  before; reconcile it first.
- Thor has no `just` and no `uv`. Use the venv's `python` and `openral` directly.
- Thor's root disk has run near full, and a full disk has killed `ros2_control_node`
  mid-run. Check `df -h` before recording bags.

## 0. Preconditions [human, rig]

- Two people. One holds the **hardware E-stop** for the whole session. For bringup, a second
  person stands on the stop.
- The arms are parked **near zero** before any bringup, because bringup snaps them to zero.
- Nothing is inside the arms' reach except what a step places there.
- The motion gates are exported by the person at the cell, only for step 4 and only in that
  terminal: `OPENRAL_OPENARM_ALLOW_MOTION=1` and `OPENRAL_OPENARM_ATTENDED=1`. These are the
  same gates `tests/hil/test_openarm_slot_group_motion.py` uses. `openral deploy run` does
  not check them; `tools/openarm_world_voxel_run.sh` does.

## 1. ZED intrinsics: use the SDK's [human, rig]

The ZED SDK computes depth and the point cloud from the camera's **factory calibration**.
`openral calibrate camera` wraps `camera_calibration cameracalibrator`, a monocular
checkerboard fit of intrinsics. It does not feed the SDK's cloud and cannot correct the
pose, so do not use it for this camera. Its default topic, `/<sensor>/image_raw`, does not
exist on a ZED anyway. The manifest's `head_zed` intrinsics (fx = 960, derived from the
published field of view) are not used by the voxel check.

Verify that the SDK is serving its calibration. Arms **unpowered**; this step does not
touch CAN.

```bash
source /opt/ros/jazzy/setup.bash && source <zed_ws>/install/setup.bash
ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zedm camera_name:=zed \
    publish_tf:=false publish_map_tf:=false
# second terminal
ros2 topic list | grep -E 'camera_info|point_cloud'
ros2 topic echo --once <one of the left camera_info topics>   # k[] non-zero, width/height match
ros2 topic hz /zed/zed_node/point_cloud/cloud_registered
```

Topic names differ between `zed_wrapper` 4.x and 5.x. Pick from the listing rather than
assuming. The cloud topic above is the one the scene pins, and it was verified on the Orin.

## 2. Extrinsic: measure and verify with a pass criterion

The kernel places every obstacle through `openarm_base -> zed_camera_link`. The manifest's
value is **partly measured**: roll −0.7° / pitch 67.7° come from a level-surface fit on a real
ZED cloud in the Thor cell (2026-09-24; the earlier 45° placeholder was 22.7° off), but the
position (0.20 m above `openarm_base`, x = y = 0) and yaw are still approximations, because
one level plane cannot observe them. A
wrong pose does two things. It puts obstacles where there are none, which causes false
stops. It also moves real obstacles away from where they are, which causes missed stops.

The pose is per unit. There are two physical OpenArm cells, on **Thor** and on the **Orin**
(`qorin1`), and their ZED mounts and units differ: measured on 2026-09-24, the old
placeholder's pitch error was 22.5° on Thor and 24.9° on the Orin, and the two units'
intrinsics were fx = 1497.9 (Thor) and fx = 1492.4 (Orin). Each cell has a unit overlay,
`robots/openarm/units/<unit>.yaml` (`thor`, `orin`), selected by `OPENRAL_ROBOT_UNIT` or a
scene's `robot_unit`. A unit's calibrated pose is `static_transform_xyz_rpy` under its
`head_zed` entry; a unit without one publishes the manifest's nominal pose (the Thor
measurement). The camera is bolted to the rig, so its pose is robot geometry: every OpenArm
scene publishes it as the only parent of `zed_camera_link`, and no scene may redefine the
sensor. `openral check`, `openral deploy validate`, `deploy run` and `deploy sim` all
refuse a scene sensor entry that reuses a robot sensor's name
(`openral_core.check_scene_sensor_overrides`), and a unit overlay may only set the
binding, driver topic, pose and intrinsics, never the frames (`SensorOverlay`).
`tools/depth_extrinsic_check.py --sensor head_zed --unit <unit>` measures exactly that
unit's effective pose, and `openral deploy run` itself (for any robot, whenever the
world-voxel check is on and the robot has a depth camera) refuses to launch until
`OPENRAL_ROBOT_UNIT` (or the scene's `robot_unit`) names the cell and a passing report for
exactly that unit's pose is committed at
`robots/openarm/calibration/<unit>/head_zed_extrinsic.json`.
`tools/openarm_world_voxel_run.sh` checks the same report before asking for the operator's
confirmation. Only the Thor unit has a
calibrated pose today; do not enable the kernel check on the Orin cell until `orin.yaml`
carries its own.

### 2a. Record a calibration bag [human, rig]

Arms unpowered. Only the ZED driver runs; the OpenRAL graph is not needed.

1. Leave a clear patch of table in front of the robot, in the ZED's view.
2. Measure the table-top height `TABLE_Z` in `openarm_base`. The frame is `x` forward,
   `z` up, and its origin is between the two arm mounts: `openarm_base -> openarm_left_link0`
   is `(0, +0.031, 0)` per the manifest's `fixed_attachments`. Measure from the left arm's
   base flange with a square and tape. If you can, confirm that offset first on a running
   twin pass (step 3) with `ros2 run tf2_ros tf2_echo openarm_base openarm_left_link0`.
3. Place **two** flat markers, 20–30 mm thick and about 8 cm square (a block or a thick
   board), at least 30 cm apart. Tape-measure the `(x, y)` of each centre in
   `openarm_base`. Two are required: one marker cannot tell a yaw error from a translation.
4. Start the recorder **before** the driver, so the ZED's own `/tf_static` is captured.
   Record for about 3 s: a full-resolution cloud is tens of MB per message.

```bash
ros2 bag record -s mcap -o ~/zed_extrinsic_$(date +%F-%H%M) \
    /zed/zed_node/point_cloud/cloud_registered /tf_static
# other terminal: the step-1 zed_wrapper launch; Ctrl-C the recorder after ~3 s of clouds
```

### 2b. Compute residuals and a suggested pose [offline]

```bash
source /opt/ros/jazzy/setup.bash && source install/setup.bash
uv run python tools/depth_extrinsic_check.py check \
    --robot robots/openarm/robot.yaml --sensor head_zed --unit <unit> --bag <bag_dir> \
    --cloud-topic /zed/zed_node/point_cloud/cloud_registered \
    --table-z <TABLE_Z> --table-roi <XMIN> <XMAX> <YMIN> <YMAX> \
    --marker <X1> <Y1> --marker <X2> <Y2> --out /tmp/zed_extrinsic.json
```

On a host without `uv` (Thor), use `python` from the activated venv. The table ROI must
contain only bare table and the markers, with no arm and no clutter.

The report gives:

- `residuals`: table tilt and height error, and each marker's `(x, y)` error, all for the
  **current** effective pose of that unit.
- `suggested_static_transform_xyz_rpy`: the same data with its tilt, height and planar
  (`x`, `y`, yaw) corrections composed onto the pose. It is fitted to this bag, so its own
  residuals are near zero by construction and prove nothing.

The pass criteria are derived in `openral_core.depth_extrinsic` from the real world-voxel
margin (`REAL_WORLD_VOXEL_MARGIN_M`, 20 mm, the value the launch gives the kernel), so a
margin change moves them too. They are **proposed**, not measured on a rig.

| residual | pass |
|---|---|
| table tilt | ≤ atan(10 mm / 1 m) ≈ 0.57° (half the margin at 1 m) |
| table height error | ≤ 10 mm (half the margin) |
| each marker's `(x, y)` error | ≤ 15 mm (three quarters of the margin) |
| markers | ≥ 2 |

`head_zed`'s parent is the base frame, so the bag needs only the ZED driver. For a depth
camera whose `parent_frame` is a moving link (a head on a torso, a wrist camera), also record
`/tf` from `robot_state_publisher` with the robot held still: the tool resolves
`base_frame -> parent_frame` from it and refuses if that chain moved during the recording.
RGB-only cameras cannot be checked: there is no cloud to fit.

### 2c. Adopt the suggestion, then verify on a new bag [human, rig] + [offline]

1. Copy `suggested_static_transform_xyz_rpy` into `static_transform_xyz_rpy` under the
   `head_zed` entry of `robots/openarm/units/<unit>.yaml`.
2. **Move both markers** to new measured positions and record a **second** bag, as in 2a.
3. Run `check --unit <unit>` on the second bag with
   `--out robots/openarm/calibration/<unit>/head_zed_extrinsic.json`.
   It must print `PASS`. Do not loosen the criteria: `verify` refuses a report checked at
   looser ones.
4. `uv run python tools/depth_extrinsic_check.py verify --robot robots/openarm/robot.yaml --sensor head_zed --unit <unit>` (the report defaults to `robots/openarm/calibration/<unit>/head_zed_extrinsic.json`)
   must print `extrinsic verified`.
5. Commit the unit's pose and the report together. Any later edit to the pose invalidates the
   report, and `openral deploy run` (and the launch script) refuse until the pose is re-verified.

If the camera is ever bumped, re-seated or re-mounted, go back to 2a.

## 3. Twin pass: real ZED, MuJoCo twin, motors unpowered [human, rig]

The same scene, pose and cloud are used, but the HAL is the MuJoCo twin, so no motor
command exists. The arms stay unpowered. `drivers:` is ignored on the sim path, so start the
ZED driver by hand as in step 1, before or after `deploy sim`: its launch purge
(`dds_transport_ready: … shm_purged=N shm_kept_live=M`) only removes Fast-DDS files no live
process uses, so a running driver keeps publishing (on Thor, 2026-09-24: ZED started first,
5.4 Hz cloud / 5.6 Hz octree / 4.7 Hz voxels). Run from inside a container, where a driver
in another PID namespace is invisible, the purge removes nothing; `OPENRAL_FASTDDS_SHM_CLEAN=0`
turns it off entirely. Then:

```bash
openral deploy sim --config scenes/deploy/openarm_real_world_voxels.yaml \
    --hal viewer_enabled=false --foxglove
```

`MUJOCO_GL=egl` is needed over ssh. The kernel runs with `world_voxel_enabled: true`, but on
the **sim** tuning: 0 m margin, 15 mm cells and sim octomap thresholds. The real values are
20 mm margin, 20 mm cells, occupancy threshold 0.6 and clamping max 0.97, and apply only
under `deploy run`. The reasoner is off and nothing dispatches, so the kernel evaluates no
chunk and `SafetyStatus` stays at its activation value. This pass checks the **world**, not
the verdicts.

Check and record:

- **Liveness.** `ros2 topic hz /octomap_binary` tracks the cloud, at about 5 Hz on the Orin.
  `/openral/world_voxels` republishes between octrees (up to ~10 Hz) and stops once the
  octree is older than 1.0 s, so its rate is not a cloud rate but its silence is meaningful.
- **Frame.** `ros2 topic echo --once /openral/world_voxels --field header` shows
  `frame_id: openarm_base`.
- **One parent.** `ros2 run tf2_tools view_frames` shows `zed_camera_link` with exactly one
  parent, `openarm_base`. A second parent means `publish_tf` was left on.
- **Overlay.** In Foxglove, check that the voxels sit on the table, the markers and the real
  arms. The robot model shown is the twin at zero, so park the real arms at zero to compare.
  Note any voxels on the arm links (self-occupancy, gap 1) and any free-floating speckle
  near the arm envelope, each of which is a future false stop.
- **Unplug.** Pull the ZED USB for 10 s: `/octomap_binary` stops and, about 1 s later,
  `/openral/world_voxels` stops too (the bridge logs `last octree is … old … publishing
  nothing`); it resumes when the camera returns. Continuing voxels here is a failure.

Record a bag for the write-up:

```bash
ros2 bag record -s mcap -o ~/twin_pass_$(date +%F-%H%M) \
    /openral/world_voxels /octomap_binary /openral/safety_status /tf /tf_static
```

## 4. Real arms, attended [human, rig]

Before each launch and each dispatch, confirm presence (step 0). Then, in the terminal of
the person at the cell:

```bash
source /opt/ros/jazzy/setup.bash && source install/setup.bash && source <zed_ws>/install/setup.bash
export OPENRAL_OPENARM_ALLOW_MOTION=1 OPENRAL_OPENARM_ATTENDED=1 OPENRAL_ROBOT_UNIT=thor  # this cell
tools/openarm_world_voxel_run.sh --foxglove
```

The script refuses without both gates, without `OPENRAL_ROBOT_UNIT`, without a verified extrinsic for that unit (step 2), and without an
interactive terminal. It asks you to type `ESTOP IN HAND`, then runs
`openral deploy run --config scenes/deploy/openarm_real_world_voxels.yaml`, which brings up
the vendor controllers (**arms snap to zero**), the ZED driver, octomap, the bridge and the
kernel with `world_voxel_enabled: true`, `world_voxel_margin_m: 0.02` and
`world_voxel_deadline_ms: 1000`. The cloud reaches octomap by topic
(`runtime.octomap_cloud_topic`), not through the sensor leg; `head_zed`'s manifest
`deploy_binding` only feeds its depth image to the world state.

In a second terminal, record the evidence for every test below:

```bash
ros2 bag record -s mcap -o ~/attended_$(date +%F-%H%M) \
    /openral/safety_status /openral/failure/safety /openral/estop /openral/world_voxels \
    /octomap_binary /joint_states /tf /tf_static
```

Motion comes only from a direct dispatch of an rSkill already validated on this cell. The
reasoner is off in this scene. One goal is one confirmed action:

```bash
python tools/_validation_matrix_dispatch.py --deadline-s 60 \
    --rskill-id <rskill validated on this cell> --prompt "<its task>"
```

Read a stop from `/openral/safety_status`: `latched: true`, `drop_reason: 10`
(`KIND_COLLISION`). The `FailureTrigger` on `/openral/failure/safety` names the stopping link,
the `voxel_<n>` cell and `min_distance_m` in `evidence_json`. To tell a real obstacle from
self-occupancy, locate the cell. `idx = x + size_x*(y + size_y*z)`; `orientation` is the
identity for a fixed-base arm mapped in `openarm_base`, and the snippet asserts that:

```bash
python3 - <voxel_index> <<'EOF'
import sys, rclpy
from openral_msgs.msg import OccupancyVoxels
rclpy.init(); n = rclpy.create_node("cell"); got = []
n.create_subscription(OccupancyVoxels, "/openral/world_voxels", got.append, 1)
while not got: rclpy.spin_once(n)
m, i = got[0], int(sys.argv[1]); o = m.orientation
assert abs(o.w) > 0.9999, "non-identity lattice: rotate the offset by orientation"
x, y, z = i % m.size_x, (i // m.size_x) % m.size_y, i // (m.size_x * m.size_y)
print([getattr(m.origin, a) + (k + 0.5) * m.resolution for a, k in zip("xyz", (x, y, z))])
EOF
```

A cell on or next to the stopping link itself is self-occupancy (gap 1), not an obstacle.
After each stop, clear the cell, confirm presence, then reset with
`ros2 service call /openral/estop_reset std_srvs/srv/Trigger`.

Run these tests in order:

1. **Idle.** Bring up and dispatch nothing for 60 s. Expect no stop. Note whether the arms
   at zero appear as voxels (Foxglove).
2. **Planted obstacle.** Put a soft obstacle (a foam block) on the table **in the camera's
   view**, in the path of the dispatched skill's first motion. Expect a `KIND_COLLISION`
   stop before contact, with the cell on the block and a `min_distance_m` consistent with
   where the link was (within about one 20 mm cell of the block's surface). Run it at least
   3 times. A stop whose cell is on the arm is
   gap 1, not this test passing.
3. **Grasped object.** Dispatch a grasp. Per gap 2, expect the gripper to stop against its
   own payload once it closes on the object in view. Record the link, the cell and the
   distance. This is **not** an attached-payload-vs-voxels check: that check is off on real
   hardware.
4. **Camera unplug.** During a dispatch, unplug the ZED. Expect `/openral/world_voxels` to
   stop about 1 s after `/octomap_binary`, and the kernel to drop the next chunks with
   `DROP_VOXEL_UNAVAILABLE` within ~2.0 s of the last cloud (a drop, not a latch). Motion or
   certified chunks after that window is a failure: have the E-stop ready. Record both
   topics' last-message times and the first `DROP_VOXEL_UNAVAILABLE`.

## 5. Exit criteria and what to record [offline]

The criteria for making this check **default-on** for the OpenArm are ADR-0109's:

- zero missed stops on the planted obstacle;
- a false-stop rate the TSC accepts, counted over a fixed task set, with stops on the table
  and self-occupancy stops counted separately;
- a staleness drop, not motion, when the camera is unplugged (`DROP_VOXEL_UNAVAILABLE`
  within ~2.0 s of the last cloud).

For the write-up, record:

- the extrinsic report (committed), with its `residuals` and the bag names;
- the twin-pass numbers: `/octomap_binary` rate, the `/openral/world_voxels` rate and
  frame, and self-occupancy observed yes or no;
- for every attended dispatch: the rSkill id, the outcome, and each stop's `drop_reason`,
  link, cell centre, `min_distance_m`, and whether it was an obstacle, self-occupancy, the
  payload or the table;
- the counts of `SafetyStatus` transitions by `drop_reason`, from the bag;
- the bags themselves (stored off-host, because Thor's disk runs full).

## Files

- `scenes/deploy/openarm_real_world_voxels.yaml`: the real-cell scene. The check is on, and
  it holds the ZED driver include. It declares no `head_zed` entry.
- `robots/openarm/robot.yaml`: `head_zed`'s nominal `static_transform_xyz_rpy`.
- `robots/openarm/units/<unit>.yaml`: each cell's camera bindings and calibrated ZED pose.
- `tools/depth_extrinsic_check.py`: `check` (bag to residuals and report) and `verify`
  (report against the unit's effective pose).
- `tools/openarm_world_voxel_run.sh`: the guarded launcher for step 4.
- `robots/openarm/calibration/<unit>/head_zed_extrinsic.json`: the committed passing report.
  It does not exist until step 2 is done, and until then the launcher refuses.
