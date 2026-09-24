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

## What to expect: three known gaps on current master

The evaluation exists to measure these gaps, so expect each one to show up rather than
expecting a clean pass:

1. **The arm sees itself.** No depth self-filter exists for a real camera; the only one,
   `exclude_body_ids` in the sim HAL's depth synthesis, is MuJoCo-only. Any arm link inside
   the ZED's field of view therefore becomes occupancy around that link. The kernel then
   stops that link against its own surface with `KIND_COLLISION` and a `voxel_<n>` cell. With
   the arms hanging at zero, below and behind a camera pitched 45° down, they may be out of
   view; a policy reaching into the workspace brings them into view. A ROS 2 candidate is
   `leggedrobotics/robot_self_filter` (BSD-3-Clause, see §22.4 of the
   [collision-safety alternatives survey](../../reference/collision-safety-alternatives-survey.md)),
   but that is its own decision.
2. **A grasped object stops the gripper holding it.** Real hardware runs with
   attached-payload checking **off** (`_attached_collision_enabled("real")` returns `False`
   in `deploy_e2e.launch.py`). Nothing on the real graph publishes
   `/openral/attachment_state`: the HAL's vision attachment leg exists but is opt-in and not
   wired into the deploy launch. So the bridge never clears the object's cells, and the
   gripper stops against its own payload. The sim contract, where the payload leaves world
   occupancy and is checked as attached geometry, does **not** hold here.
3. **Camera loss does not fail closed.** `openral_octomap_bridge` republishes its **last**
   octree on a 10 Hz wall timer, stamped `now()`
   (`octomap_voxel_bridge_node.cpp`, `on_timer`), and the kernel times voxel freshness from
   message **receipt** (`voxel_stamp_ = this->now()`). `octomap_server` publishes only when a
   cloud arrives, so when the ZED stops, `/openral/world_voxels` keeps arriving on time with
   a frozen map. Expect **no** `DROP_VOXEL_UNAVAILABLE`, and expect obstacles placed after the
   loss to be invisible. That fails open. The fix belongs in the bridge: refuse to publish
   once the octree is older than a bound. It changes a safety input, so it needs a
   hazard-log entry and a safety-WG reviewer.

The exit criteria in step 5 cannot all be met until gaps 1 and 3 are fixed. Record the
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
value is an **uncalibrated approximation**: 0.20 m above `openarm_base`, pitched 45° down. A
wrong pose does two things. It puts obstacles where there are none, which causes false
stops. It also moves real obstacles away from where they are, which causes missed stops.

The pose lives in exactly one place: `static_transform_xyz_rpy` of the `head_zed` entry in
`robots/openarm/robot.yaml`. The camera is bolted to the rig, so its pose is robot geometry:
every OpenArm scene publishes it as the only parent of `zed_camera_link`, and no scene may
restate it. `openral check`, `openral deploy validate`, `deploy run` and `deploy sim` all
refuse a scene sensor entry that restates a robot sensor's mount, intrinsics or frame
(`openral_core.check_scene_sensor_overrides`). `tools/zed_extrinsic_check.py` measures
exactly the manifest value, and `tools/openarm_world_voxel_run.sh` refuses to launch until a
passing report for exactly that value is committed at
`robots/openarm/calibration/head_zed_extrinsic.json`.

!!! warning "Two OpenArm cells, one manifest pose"
    There are two physical OpenArm cells, on **Thor** and on the **Orin** (`qorin1`). Their
    ZED mounts and units differ. Measured on 2026-09-24, the manifest placeholder's pitch
    error was 22.5° on Thor and 24.9° on the Orin, and the two units' intrinsics were
    fx = 1497.9 (Thor) and fx = 1492.4 (Orin). One manifest holds one pose, so the pose you
    calibrate here is the **Thor** cell's. Committing it makes it the pose every OpenArm
    scene publishes, including `openarm_zed_octomap.yaml` on the Orin, where it is wrong
    until per-unit calibration exists. Do not enable the kernel check on the Orin cell on
    the strength of a Thor calibration.

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
uv run python tools/zed_extrinsic_check.py check \
    --robot robots/openarm/robot.yaml --bag <bag_dir> \
    --cloud-topic /zed/zed_node/point_cloud/cloud_registered \
    --table-z <TABLE_Z> --table-roi <XMIN> <XMAX> <YMIN> <YMAX> \
    --marker <X1> <Y1> --marker <X2> <Y2> --out /tmp/zed_extrinsic.json
```

On a host without `uv` (Thor), use `python` from the activated venv. The table ROI must
contain only bare table and the markers, with no arm and no clutter.

The report gives:

- `residuals`: table tilt and height error, and each marker's `(x, y)` error, all for the
  **current** manifest pose.
- `suggested_static_transform_xyz_rpy`: the same data with its tilt, height and planar
  (`x`, `y`, yaw) corrections composed onto the pose. It is fitted to this bag, so its own
  residuals are near zero by construction and prove nothing.

The pass criteria are the tool's defaults. They are **proposed**, not measured on a rig:
each is at most half the real 20 mm voxel margin.

| residual | pass |
|---|---|
| table tilt | ≤ 0.75° (13 mm at 1 m) |
| table height error | ≤ 10 mm |
| each marker's `(x, y)` error | ≤ 15 mm |
| markers | ≥ 2 |

### 2c. Adopt the suggestion, then verify on a new bag [human, rig] + [offline]

1. Copy `suggested_static_transform_xyz_rpy` into the `head_zed` entry's
   `static_transform_xyz_rpy` in `robots/openarm/robot.yaml`.
2. **Move both markers** to new measured positions and record a **second** bag, as in 2a.
3. Run `check` on the second bag with `--out robots/openarm/calibration/head_zed_extrinsic.json`.
   It must print `PASS`. Do not loosen the criteria: `verify` refuses a report checked at
   looser ones.
4. `uv run python tools/zed_extrinsic_check.py verify --robot robots/openarm/robot.yaml --report robots/openarm/calibration/head_zed_extrinsic.json`
   must print `extrinsic verified`.
5. Commit the manifest pose and the report together. Any later edit to the pose invalidates the
   report, and the launch script refuses until the pose is re-verified.

If the camera is ever bumped, re-seated or re-mounted, go back to 2a.

## 3. Twin pass: real ZED, MuJoCo twin, motors unpowered [human, rig]

The same scene, pose and cloud are used, but the HAL is the MuJoCo twin, so no motor
command exists. The arms stay unpowered. `drivers:` is ignored on the sim path, so start the
ZED driver by hand as in step 1. Then:

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
  `ros2 topic hz /openral/world_voxels` reads about 10 Hz **whatever the camera is doing**
  (gap 3), so it is not a liveness signal.
- **Frame.** `ros2 topic echo --once /openral/world_voxels --field header` shows
  `frame_id: openarm_base`.
- **One parent.** `ros2 run tf2_tools view_frames` shows `zed_camera_link` with exactly one
  parent, `openarm_base`. A second parent means `publish_tf` was left on.
- **Overlay.** In Foxglove, check that the voxels sit on the table, the markers and the real
  arms. The robot model shown is the twin at zero, so park the real arms at zero to compare.
  Note any voxels on the arm links (self-occupancy, gap 1) and any free-floating speckle
  near the arm envelope, each of which is a future false stop.
- **Unplug.** Pull the ZED USB for 10 s and watch `/octomap_binary` stop while
  `/openral/world_voxels` keeps coming (gap 3). Record it; do not treat it as a pass.

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
export OPENRAL_OPENARM_ALLOW_MOTION=1 OPENRAL_OPENARM_ATTENDED=1
tools/openarm_world_voxel_run.sh --foxglove
```

The script refuses without both gates, without a verified extrinsic (step 2), and without an
interactive terminal. It asks you to type `ESTOP IN HAND`, then runs
`openral deploy run --config scenes/deploy/openarm_real_world_voxels.yaml`, which brings up
the vendor controllers (**arms snap to zero**), the ZED driver, octomap, the bridge and the
kernel with `world_voxel_enabled: true`, `world_voxel_margin_m: 0.02` and
`world_voxel_deadline_ms: 1000`. `openral deploy validate` warns that `head_zed` has no
`deploy_binding`. That is expected: the cloud reaches octomap by topic, not through the
sensor leg.

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
4. **Camera unplug.** During a dispatch, unplug the ZED. Per gap 3, expect **no**
   `DROP_VOXEL_UNAVAILABLE` and the kernel still checking a frozen map. Have the E-stop
   ready and do not rely on the kernel for anything new that enters the cell. Record the
   `/octomap_binary` gap against continuing `/openral/world_voxels`.

## 5. Exit criteria and what to record [offline]

The criteria for making this check **default-on** for the OpenArm are ADR-0109's:

- zero missed stops on the planted obstacle;
- a false-stop rate the TSC accepts, counted over a fixed task set, with stops on the table
  and self-occupancy stops counted separately;
- a staleness drop, not motion, when the camera is unplugged. This is blocked by gap 3.

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
- `robots/openarm/robot.yaml`: `head_zed`'s `static_transform_xyz_rpy`, the single-sourced
  ZED pose.
- `tools/zed_extrinsic_check.py`: `check` (bag to residuals and report) and `verify`
  (report against the manifest pose).
- `tools/openarm_world_voxel_run.sh`: the guarded launcher for step 4.
- `robots/openarm/calibration/head_zed_extrinsic.json`: the committed passing report.
  It does not exist until step 2 is done, and until then the launcher refuses.
