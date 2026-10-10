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
   `runtime.robot_self_filter_padding_m` (2 cm: measured at rest on both cells, 2026-09-25,
   where the robot's own returns end 5-7.5 mm outside the hulls; the tail with the arms
   moving is still to be measured) of them, plus a held payload's primitives once an
   attachment producer exists. Without a pose at the capture stamp it drops the cloud, so
   the map goes stale and the kernel fails closed rather than seeing the arm as an obstacle.
   The cost is a padding-wide shell around the arm in which the kernel cannot see a real obstacle
   (hazard log). Before the filter, the ZED's view of the left gripper stopped tick 1
   (`safety.collision kind=world a=openarm_left_finger_pair`). Record the filter's
   `self-filter:` log line (share of points removed, ms per cloud, drops).
2. **A grasped object stops the gripper holding it.** Real hardware runs with
   attached-payload checking **off** while the scene's `runtime.vision_attachment.enabled`
   is false (`_attached_collision_enabled("real", False)` returns `False` in
   `deploy_e2e.launch.py`), and then nothing on the real graph publishes
   `/openral/attachment_state`. So the bridge never clears the object's cells, and the
   gripper stops against its own payload. The sim contract, where the payload leaves world
   occupancy and is checked as attached geometry, does **not** hold here.

   **Vision attachment leg (off by default).** `runtime.vision_attachment` in this scene
   wires the leg — the SAM 2.1 segmenter lifecycle node (`openral_segmenter`) plus the HAL's
   attachment-evidence bridge, fed by the ZED's RGB, depth and their `camera_info` topics —
   but commits it **off**: its grasp trigger is the gripper *position* stalling short of a close
   command (the OpenArm driver reports effort as a constant 0.0). Its calibration
   (`closure_calibration`: rest offset, stall gap, settle tolerance) is a set of robot-type
   constants of the gripper mechanism in `robots/openarm/robot.yaml`, derived from teleop
   data; not per cell, so no unit overrides it and there is no calibration session. Verify it
   once by commanding an empty close on the rig and reading where the jaw rests.
   Turning it on for real is pending Safety-WG review and a hazard-log entry. Enabled,
   `deploy run` always turns the kernel's attached-payload check on with it
   (`attached_collision_enabled`, 1000 ms deadline), never the leg alone: the bridge's payload
   clearing and the self-filter act on any published attachment whatever the kernel flag, so a
   leg without the kernel check would hide the payload from every check. Do not enable it in
   this runbook.

   **Release (frozen window, the leg's default behaviour).** The trigger's DETACH fires when the
   jaws *open* past their hold, with the fingers still around the object. The leg does not drop the payload then:
   it keeps publishing it as an attached record frozen in `openarm_base` at its DETACH pose
   (`touch_links` = the hand and its finger pair, exactly what the held record exempted;
   every other link and the world are still checked against it), so the octomap bridge keeps
   clearing its cells while the fingers back out. The record goes once the hand and fingers
   are `release_clear_m` (0.04 m = the 20 mm world margin + one 20 mm voxel) clear of it, a
   new ATTACH on that gripper, or `release_timeout_s` (3.0 s) — after which a hand still
   inside the margin of the re-marked object is stopped, fail-closed. Look for
   `release window opened` / `release window closed reason=separation|timeout|attach` in the
   HAL log. Both bounds are uncalibrated (hazard log).

   **Grasp-target exemption (off by default).** `runtime.grasp_allowance_enabled` (default
   `false`) forwards `grasp_allowance_enabled:=true` to the kernel; the launch always passes
   `grasp_contact_links` = the manifest's `role: gripper` child links
   (`openarm_left_finger_pair`, `openarm_right_finger_pair`). With it on, the exemption still
   applies only inside a producer-measured `GraspDeclaration` region, which nothing on the
   real graph measures yet. Do not enable it in this runbook.

   **Place producer leg (off by default).** The HAL parameter
   `vision_attachment_place_target_enabled` (default `false`; scene
   `runtime.vision_attachment.place_target_enabled`) runs the real place producer inside the
   vision attachment bridge. Nothing is surveyed and no place target is named: while a payload
   is held it measures the surface directly under it in the live voxel map (the whole
   footprint occupied on one layer, headroom for the payload above it, within
   `vision_attachment_place_target_search_depth_m`), declares that patch as a place region
   for that payload, and a payload resting on it gets a `map_support_proximity` support
   witness — proximity to a map-measured plane, not sensed contact, not a proven support. It
   rests on drafted, unapproved ADR-0097 / ADR-0092 D6 amendments and its thresholds are
   uncalibrated. Do not enable it in this runbook.

Camera loss **fails closed** (hazard-log Entry 033):
`openral_octomap_bridge` stops publishing `/openral/world_voxels` once its last octree is
older than `max_octree_age_s` (default: equal to `world_voxel_deadline_s`, 1.0 s, which reaches
the kernel as `world_voxel_deadline_ms`; both are `DeployRuntime` fields), and
the kernel's deadline then turns the silence into `DROP_VOXEL_UNAVAILABLE`, at most ~2.0 s
after the last cloud. Verified on Thor with the ZED stopped (bridge half).

The exit criteria in step 5 cannot all be met until gap 1 is fixed. Record the
numbers anyway: they are the evidence those fixes need.

## Hosts

The cell (ZED Mini, PEAK CAN adapter `openarm_left` / `openarm_right`) is attached to
**Thor**. The ZED leg of this pipeline has been verified only on the lab **Orin**
(2026-09-07: ZED SDK 5.4.1, `/octomap_point_cloud_centers` at about 5 Hz,
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
    publish_tf:=false publish_map_tf:=false enable_ipc:=false \
    param_overrides:="depth.point_cloud_freq:=30.0;depth.point_cloud_res:=REDUCED;depth.max_depth:=2.5"
# second terminal
ros2 topic list | grep -E 'camera_info|point_cloud'
ros2 topic echo --once <one of the left camera_info topics>   # k[] non-zero, width/height match
ros2 topic hz /zed/zed_node/point_cloud/cloud_registered
```

The `param_overrides` are the same ones the scene's `drivers:` entry passes under
`deploy run`: a 30 Hz, REDUCED (224x128) cloud cut at 2.5 m. `deploy sim` ignores
`drivers:`, so a hand-launched ZED needs them too. The stock 10 Hz COMPACT cloud with 10 m
depth gave ~2 Hz and gaps of up to 3 s, which the kernel's 1 s voxel deadline turned into
stops.

`enable_ipc:=false` is not optional either. With the wrapper's default intra-process mode,
it publishes `zed_camera_center -> zed_left_camera_frame` as a **dynamic** `/tf`, stamped
per grab, so every cloud waits on its own TF in `octomap_server`'s `tf2_ros::MessageFilter`.
A cloud that arrives before its TF is inserted later, on the TF listener's own thread, and
that thread stops ingesting `/tf` while it inserts. The filter's 5-deep queue fills ("Message
Filter dropping message: frame 'zed_left_camera_frame' ... queue is full") and the map
stalls. Thor, 2026-10-04, twin with the real ZED, motors off:

| | `/octomap_binary` max gap | gaps > 1 s | "queue is full" | `/openral/world_voxels` max gap |
|---|---|---|---|---|
| IPC on (wrapper default), 15 mm | 1.75-3.0 s | 2 / 60 s, 14 / 120 s | 9 in 3 min | 0.9-1.6 s, kernel `voxel_stale` |
| IPC off, 15 mm, 2 x 300 s | 0.47 s / 0.67 s | 0 | 0 | 0.70 s / 0.43 s |
| IPC off, 20 mm (the real-path cell), 150 s | 0.40 s | 0 | 0 | 0.13 s |

The remaining sub-second gaps match gaps in the ZED cloud itself (0.70 s at most). With IPC
off the camera TFs are static, so a cloud never waits. Every consumer runs in another
process, so IPC gains nothing here.

Topic names differ between `zed_wrapper` 4.x and 5.x. Pick from the listing rather than
assuming. The cloud topic above is the one the scene pins, and it was verified on the Orin.

## 2. Extrinsic: calibrate the camera yourself, declare it per unit [human, rig]

The kernel places every obstacle through `openarm_base -> zed_camera_link`, so this one
transform has to be right to about the kernel's 2 cm world margin. A wrong pose does two
things. It puts obstacles where there are none (false stops, including the robot "hitting"
its own leftover returns), and it moves real obstacles away from where they are (missed
stops). Only the world-voxel check needs it: the policy uses images alone, and the
environment itself is never calibrated (octomap senses it live).

**OpenRAL does not measure this transform.** Calibrate it with the tool you trust and
declare the result. A hand-eye calibration against a printed ChArUco board on the ZED's
rectified left image (`/zed/zed_node/left/image_rect_color` + its `camera_info`) is the
standard route and works for any robot with a gripper. What OpenRAL fixes is the contract:

- **Frames.** The pose is `parent_frame -> frame_id` of the manifest's `head_zed` entry:
  `openarm_base` (the manifest `base_frame`, at shoulder height; the URDF root `world` is
  0.698 m below it) to `zed_camera_link`, the ZED **body** frame, midway between the
  lenses. The ZED driver publishes its cloud in `zed_left_camera_frame` and places that
  frame relative to `zed_camera_link` from the camera's factory calibration (half the
  63 mm baseline), so a hand-eye result for the left lens is converted through the
  driver's own TF, not declared directly.
- **Accuracy** (`openral_core.depth_extrinsic.MAX_*`, derived from the real world-voxel
  margin, `REAL_WORLD_VOXEL_MARGIN_M` = 20 mm): height within 10 mm,
  planar x/y within 15 mm, tilt within atan(10 mm / 1 m) ≈ 0.57°, yaw within
  atan(15 mm / 1 m) ≈ 0.86°, at the 1 m range the check covers. A hand-eye calibration
  reaches this; a tape measure does not.
- **Where it goes.** The unit overlay, `robots/openarm/units/<unit>.yaml`, selected by
  `OPENRAL_ROBOT_UNIT` or a scene's `robot_unit`: `static_transform_xyz_rpy` under its
  `head_zed` entry, `[x, y, z, roll, pitch, yaw]` in metres and radians, fixed-axis XYZ
  (`static_transform_publisher` convention). Each unit carries its own mount; a value
  measured on one cell's camera is wrong on another's. The exception is the same robot and
  camera moving between hosts: on 2026-10-07 the OpenArm and its ZED moved from Thor to
  Orin, so `orin.yaml` carries Thor's measurement (checked on Orin against a fresh capture). The
  camera is robot geometry: no scene may redefine it
  (`openral_core.check_scene_sensor_overrides`), and a unit overlay may only set the
  binding, driver topic, pose and intrinsics (`SensorOverlay`). Record how and when it was
  measured in the file's header comment.
- **The gate.** `openral deploy run` (any robot, whenever the world-voxel check is on) and
  `tools/openarm_world_voxel_run.sh` refuse to launch unless every cloud source's mount is
  declared by the selected unit's overlay
  (`openral_core.depth_extrinsic.depth_extrinsic_problems`). Every manifest ships a
  nominal mount from CAD, so a value being present proves nothing; a unit overlay
  exists only because someone set that cell up. The gate checks provenance, not
  correctness: a wrong number in the unit file runs.

If the camera is ever bumped, re-seated or re-mounted, measure again. Nothing in the stack
notices a camera that moved: on Thor the ZED's view of the floor shifted by about 1°
between two days (2026-10-01/02) with nothing deliberately touched. A cheap check that
needs no motion: with the arms at rest, the robot's own returns should end within the
self-filter's padding of the kernel's collision primitives (step 3 reads this off the
filter's log line); a mount that is centimetres off shows up there first.

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

`MUJOCO_GL=egl` is needed over ssh. Source the OpenArm vendor workspace (the one holding
`openarm_description`) in this terminal as well: the robot model in Foxglove is the manifest
URDF on `/robot_description`, and its `package://openarm_description/...` meshes are served by
the bridge, which can only resolve packages on its own `AMENT_PREFIX_PATH`. Without it the 3D
panel lists `Failed to retrieve asset` for every link and draws no robot. The kernel runs with `world_voxel_enabled: true`, but on
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
- **Overlay.** Import the layout `deploy sim` prints
  (`[deploy_e2e] foxglove: … import the scene-matched layout from /tmp/openral_layout_openarm_v2.json`)
  — re-import it if you saved an older one; layouts saved before the URDF layer have no robot.
  The hero 3D panel (and the *World voxels* tab) follows `openarm_base` and draws the robot
  model from `/robot_description` beside `/octomap_point_cloud_centers` and
  `/openral/world_voxels_cloud`. Check that the voxels sit on the table, the shelf and the real
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

The script refuses without both gates, without `OPENRAL_ROBOT_UNIT`, without a declared
head_zed mount in that unit's overlay (step 2), and without an
interactive terminal. It asks you to type `ESTOP IN HAND`, then runs
`openral deploy run --config scenes/deploy/openarm_real_world_voxels.yaml`, which brings up
the vendor controllers (**arms snap to zero**), the ZED driver, octomap, the bridge and the
kernel with `world_voxel_enabled: true`, `world_voxel_margin_m: 0.02` and
`world_voxel_deadline_ms: 1000`. The cloud reaches octomap by topic
(`runtime.octomap_cloud_topic`), not through the sensor leg; `head_zed`'s manifest
`deploy_binding` only feeds its depth image to the world state.

To run a local copy of the scene instead (test 3's grasp target), pass
`--scene scenes/deploy/local/<copy>.yaml`. The copy gets the same gates, and the script also
refuses it unless it lies under `scenes/deploy/local/` (gitignored, so operator copies never
dirty the tree or join the tracked scene registry), validates as a `DeployScene`, and, once parsed, is
identical to the committed scene except for the top-level `grasp_declaration` block. Any
other difference (octomap off, a wider `robot_self_filter_padding_m`, an allowance or extra
collision pair, a safety envelope, other drivers or HAL parameters, even a renamed
`scene.id`) is refused, naming each differing key path. Never launch a copy with a bare
`openral deploy run`: it skips every gate above.

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
After each stop, clear the cell, confirm presence, then reset with `openral estop reset`.
It calls the kernel's `/openral/estop_reset` first and, only if the kernel accepts, broadcasts
`/openral/estop_cleared`, the same sequence as the dashboard's reset button. Do not call
`ros2 service call /openral/estop_reset ...` on its own: that clears only the kernel, so the
runner and HAL stay latched and the runner rejects every later goal (it logs
`rskill_runner.goal_rejected: e-stop latched`). Exit code 1 means the kernel refused, usually
because the cooldown has not passed yet, and nothing was cleared. Wait a moment and run it again.

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

   The committed scenes name no grasp target and no coordinates: what to pick is task
   knowledge, and it arrives as language. Launch the autonomous posture,
   `scenes/deploy/openarm_real_autonomous.yaml` (this scene plus the reasoner, the
   open-vocabulary detector and spatial-memory ingest), and name a label:

   ```bash
   export OPENRAL_REASONER_MODEL=<a model from openral_core.REASONER_MODELS>
   tools/openarm_world_voxel_run.sh --autonomous
   # second terminal, presence confirmed for this goal:
   openral prompt "pick up the boxes"
   ```

   The detector lifts every box it sees; the reasoner's `WORLD_STATE` lists each instance
   (`scene_objects[...]`) and its spatial-memory id (`memory_objects[...]: box
   id=obj_box_1@(...)`), so it decomposes the goal into one subtask per box and names each
   one by `grasp_target.object_id`; perception grounds the search box
   ([design §2.2](../../reference/real-pick-place-design.md)). Grounding refuses a bare
   label that several detections carry. The only OpenArm VLA today is task-specific (the
   restock policy): a general "pick up X" needs a language-conditioned pick/place policy
   installed for the OpenArm, and until one is, expect the reasoner to hand off.

   Direct dispatch (debug only, reasoner off): copy the committed scene into the gitignored
   `scenes/deploy/local/` (create it), add a `grasp_declaration` block (the only difference
   the wrapper accepts) whose `search_box` you measured on the live voxel map of this cell,
   today, around exactly ONE item (a box covering two is refused as AMBIGUOUS), and launch it
   with `tools/openarm_world_voxel_run.sh --scene <copy>` (add `--autonomous` for a copy of
   the autonomous scene). Never reuse another cell's or another day's numbers: the box only
   seeds perception; nothing about the cell is surveyed (the place surface is measured live
   under the carried payload).
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

- the unit overlay's `head_zed` pose and how it was measured (its header comment);
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
- `scenes/deploy/openarm_real_autonomous.yaml`: the same cell with the reasoner, the
  open-vocabulary detector and spatial-memory ingest on (`--autonomous`); the vision
  attachment leg stays off.
- `robots/openarm/robot.yaml`: `head_zed`'s nominal `static_transform_xyz_rpy` (sim twins
  only; a real world-voxel deploy never runs on it).
- `robots/openarm/units/<unit>.yaml`: each cell's camera bindings and its calibrated ZED
  pose, measured by the operator (step 2). The grippers' `closure_calibration` is a
  robot-type constant in `robot.yaml`, not per unit. No furniture is surveyed: place surfaces are
  measured live from the voxel map under the carried payload
  ([real pick-and-place design](../../reference/real-pick-place-design.md) §2.3).
- `python/core/src/openral_core/depth_extrinsic.py`: the accuracy the pose needs and the
  gate `deploy run` applies.
- `tools/openarm_world_voxel_run.sh`: the guarded launcher for step 4.
