# OpenArm real-cell world-voxel check (attended)

**Goal:** turn on the safety kernel's world-voxel check on the real OpenArm cell
and measure what it does.

The ZED Mini cloud goes through `octomap_server` and `openral_octomap_bridge` to
`/openral/world_voxels`, and from there to the kernel's `world_voxel_enabled`
phase. This is the attended evaluation behind the "One kernel world
representation" decision in the private decision log. No other shipped
real-hardware scene enables the check: `scenes/deploy/openarm_zed_octomap.yaml`
feeds the real ZED into the map but drives a MuJoCo twin with the kernel check
off.

!!! danger "This drives real motors"
    Bringing the OpenArm graph up **moves the arms**: the vendor hardware's
    `on_activate` steps all 16 motors to zero, **unramped**, before any OpenRAL
    code runs. The kernel and the deadman are backstops, not a substitute for a
    hand on the hardware E-stop. Operator presence authorizes **one** bringup or
    dispatch. Re-confirm it before every one, and never leave a script queued to
    dispatch after an unattended wait.

Each step is marked **[offline]** (any workstation with ROS 2 Jazzy and this
checkout) or **[human, rig]** (a person at the cell).

## Prerequisites

- The cell (ZED Mini, PEAK CAN adapter `openarm_left` / `openarm_right`) is
  attached to one lab Jetson. Its camera bindings and calibration are in
  `robots/openarm/robot.yaml`, so whichever host drives it selects nothing.
- On the cell host, before step 1:
    - ZED SDK and `zed_wrapper` are built for the host's JetPack and their
      overlay is sourced.
    - `ros-jazzy-octomap-server` is installed.
    - `openral_octomap_bridge` is built into the overlay you source
      (`ros2 pkg prefix openral_octomap_bridge`).
    - The checkout is the commit you are testing; lab checkouts can drift from
      the pushed branches, so reconcile first.
    - The root disk has room (`df -h`); a full disk has killed
      `ros2_control_node` mid-run.
- Two people. One holds the **hardware E-stop** for the whole session; for
  bringup, a second person stands on the stop.
- The arms are parked **near zero** before any bringup, because bringup snaps
  them to zero. Nothing is inside the arms' reach except what a step places there.
- The motion gates are exported by the person at the cell, only for step 4 and
  only in that terminal: `OPENRAL_OPENARM_ALLOW_MOTION=1` and
  `OPENRAL_OPENARM_ATTENDED=1`. They are the gates
  `tests/hil/test_openarm_slot_group_motion.py` uses. `openral deploy run` does
  not check them; `tools/openarm_world_voxel_run.sh` does.

## What to expect: two known gaps

The evaluation exists to measure these gaps. Expect each to show up, not a clean
pass.

**Gap 1: the arm sees itself, so a blind shell is accepted.** The sim HAL makes
the robot transparent in synthesized depth (`exclude_body_ids`); a real camera
cannot. `deploy run` therefore routes the cloud through `openral_octomap_bridge`'s
`robot_self_filter` before `octomap_server`. The filter poses the kernel's own
collision primitives at the cloud's capture stamp (joint states from the scene's
`joint_states_topic`, else the HAL's `~/joint_states`; camera pose from tf2) and
removes every return within `runtime.robot_self_filter_padding_m` of them. The
default is 2 cm, measured at rest, where the robot's own returns end 5-7.5 mm
outside the hulls; the tail with the arms moving is not yet measured. Without a
pose at the capture stamp it drops the cloud, so the map goes stale and the
kernel fails closed. The cost is a padding-wide shell around the arm in which the
kernel cannot see a real obstacle (hazard log). Record the filter's `self-filter:`
log line (share of points removed, ms per cloud, drops).

**Gap 2: a grasped object stops the gripper holding it.** On real hardware the
kernel's attached-payload check is off while `runtime.vision_attachment.enabled`
is false, and then nothing on the real graph publishes `/openral/attachment_state`.
So the bridge never clears the object's cells and the gripper stops against its
own payload. The sim contract (the payload leaves world occupancy and is checked
as attached geometry) does not hold here. The bridge clears payload cells only for
a tracker-owned attachment (`OctomapVoxelBridge::clear_by_identity`), which needs
`runtime.object_tracking: tracking`. That mode is sim-only: `deploy run` refuses
it on a real HAL, so real runs keep it `off`. A geometry-only payload therefore
stays mapped and the gripper may stop on what it holds. That is the decided
fail-closed posture
([octomap bridge README](https://github.com/OpenRAL/openral/blob/master/packages/openral_octomap_bridge/README.md#attached-objects-and-world-occupancy)).

**Leave these off in this runbook.** All four ship off in the scene and stay off
here; each is pending safety review and a hazard-log entry.

| Scene key | What it does | Why it stays off |
| --- | --- | --- |
| `runtime.vision_attachment.enabled` | SAM 2.1 segmenter plus the HAL attachment-evidence bridge; the trigger is the gripper position stalling short of a close command. When on, `deploy run` always turns the kernel's attached-payload check on with it. | Its closure calibration (`closure_calibration` in `robots/openarm/robot.yaml`) is a robot-type constant derived from teleop data, not verified per cell. |
| the leg's release window | On DETACH the leg keeps the payload as a frozen attached record until the hand is `release_clear_m` (0.04 m) clear, a new ATTACH, or `release_timeout_s` (3.0 s). Look for `release window opened` / `release window closed reason=separation\|timeout\|attach` in the HAL log. | Both bounds are uncalibrated. |
| `runtime.grasp_allowance_enabled` | Forwards the grasp exemption to the kernel, applied only inside a producer-measured `GraspDeclaration` region. | Nothing on the real graph measures that region yet. |
| `runtime.vision_attachment.place_target_enabled` | Place producer: measures the surface under a held payload in the live voxel map and declares it as a place region. | Its thresholds are uncalibrated and it rests on unapproved decision-log amendments. |

`deploy run` also refuses `workspace_contact_enabled` and any
`object_tracking` other than `off` on a real HAL, so leave those at their
defaults.

**Camera loss fails closed.** `openral_octomap_bridge` stops publishing
`/openral/world_voxels` once its last octree is older than `max_octree_age_s`
(default: `world_voxel_deadline_s`, 1.0 s, which reaches the kernel as
`world_voxel_deadline_ms`). The kernel's deadline then turns the silence into
`DROP_VOXEL_UNAVAILABLE`, at most about 2.0 s after the last cloud.

The step 5 exit criteria cannot all be met until gap 1 is closed. Record the
numbers anyway: they are the evidence the fix needs.

## 1. ZED intrinsics: use the SDK's [human, rig]

The ZED SDK computes depth and the cloud from the camera's **factory
calibration**. Do not use `openral calibrate camera` for this camera: it wraps a
monocular checkerboard fit that does not feed the SDK's cloud and cannot correct
the pose, and its default topic `/<sensor>/image_raw` does not exist on a ZED.
The manifest's `head_zed` intrinsics are not used by the voxel check.

Verify the SDK is serving its calibration. Arms **unpowered**; this step does not
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

You should see a non-zero `k[]` and a cloud rate near 30 Hz.

The `param_overrides` are the ones the scene's `drivers:` entry passes under
`deploy run`: a 30 Hz, REDUCED (224x128) cloud cut at 2.5 m. `deploy sim` ignores
`drivers:`, so a hand-launched ZED needs them too. The stock 10 Hz COMPACT cloud
with 10 m depth gave gaps of up to 3 s, which the kernel's 1 s voxel deadline
turns into stops.

`enable_ipc:=false` is required. With the wrapper's default intra-process mode
the camera's frames go on `/tf` as **dynamic** transforms, so every cloud waits on
its own TF in `octomap_server`'s `tf2_ros::MessageFilter`, the filter's queue
fills and the map stalls. Measured on the real ZED with a twin, motors off:

| | `/octomap_binary` max gap | gaps > 1 s | "queue is full" | `/openral/world_voxels` max gap |
|---|---|---|---|---|
| IPC on (wrapper default), 15 mm | 1.75-3.0 s | 2 / 60 s, 14 / 120 s | 9 in 3 min | 0.9-1.6 s, kernel `voxel_stale` |
| IPC off, 15 mm | 0.47-0.67 s | 0 | 0 | 0.43-0.70 s |
| IPC off, 20 mm (the real-path cell) | 0.40 s | 0 | 0 | 0.13 s |

Topic names differ between `zed_wrapper` 4.x and 5.x; pick from the listing. The
cloud topic above is the one the scene pins.

## 2. Extrinsic: calibrate the camera yourself, declare it per unit [human, rig]

The kernel places every obstacle through `openarm_base -> zed_camera_link`, so
this one transform must be right to about the kernel's 2 cm world margin. A wrong
pose puts obstacles where there are none (false stops, including the robot
"hitting" its own leftover returns) and moves real obstacles away from where they
are (missed stops). Only the world-voxel check needs it; the policy uses images
alone, and the environment itself is never calibrated (octomap senses it live).

**OpenRAL does not measure this transform.** Calibrate it with the tool you trust
and declare the result. A hand-eye calibration against a printed ChArUco board on
the ZED's rectified left image (`/zed/zed_node/left/image_rect_color` plus its
`camera_info`) is the standard route. OpenRAL fixes the contract:

- **Frames.** The pose is `parent_frame -> frame_id` of the manifest's `head_zed`
  entry: `openarm_base` (the manifest `base_frame`, at shoulder height; the URDF
  root `world` is 0.698 m below it, and the torso there is in the kernel's own
  collision model, self-filtered out of the cloud like the arms) to
  `zed_camera_link`, the ZED **body** frame midway between the lenses. The driver
  publishes its cloud in `zed_left_camera_frame` and places it relative to
  `zed_camera_link` from factory calibration (half the 63 mm baseline), so convert
  a left-lens hand-eye result through the driver's TF rather than declaring it.
- **Accuracy** (`openral_core.depth_extrinsic.MAX_*`, derived from
  `REAL_WORLD_VOXEL_MARGIN_M` = 20 mm): height within 10 mm, planar x/y within
  15 mm, tilt within atan(10 mm / 1 m) ≈ 0.57°, yaw within atan(15 mm / 1 m) ≈
  0.86°, at the 1 m range the check covers. A hand-eye calibration reaches this;
  a tape measure does not.
- **Where it goes.** `robots/openarm/robot.yaml`: `static_transform_xyz_rpy` under
  the `head_zed` entry, `[x, y, z, roll, pitch, yaw]` in metres and radians
  (fixed-axis XYZ, the `static_transform_publisher` convention), with
  `static_transform_calibrated: true`. The camera is robot geometry: no scene may
  redefine it (`openral_core.check_scene_sensor_overrides`). Record how and when
  it was measured in the comment above the entry. A robot model run on several
  cells with different mounts puts each cell's in a unit overlay
  (`robots/<id>/units/<unit>.yaml`); OpenArm has one.
- **The gate.** `openral deploy run` (any robot, whenever the world-voxel check is
  on) and `tools/openarm_world_voxel_run.sh` refuse to launch unless every cloud
  source's mount is declared as measured
  (`openral_core.depth_extrinsic.depth_extrinsic_problems`). Every manifest ships
  a nominal CAD mount, so a value being present proves nothing; the flag is the
  operator's statement that it was measured. The gate checks provenance, not
  correctness: a wrong number marked calibrated runs.

If the camera is bumped, re-seated or re-mounted, measure again. Nothing in the
stack notices a camera that moved. A cheap check that needs no motion: with the
arms at rest, the robot's own returns should end within the self-filter's padding
of the kernel's collision primitives (step 3 reads this off the filter's log
line); a mount that is centimetres off shows up there first.

## 3. Twin pass: real ZED, MuJoCo twin, motors unpowered [human, rig]

Same scene, pose and cloud, but the HAL is the MuJoCo twin, so no motor command
exists. The arms stay unpowered. The twin advances exactly one control period per
action (derived from `action_spec.control_freq_hz` at connect and logged once as
`sim.time_base`), so a speed or tracking verdict here is the verdict the real
pass would give.

`drivers:` is ignored on the sim path, so start the ZED driver by hand as in
step 1, before or after `deploy sim`. The launch's `dds_transport_ready:` purge
(`shm_purged=N shm_kept_live=M`) removes only Fast-DDS files no live process
uses, so a running driver keeps publishing. Inside a container, where a driver in
another PID namespace is invisible, the purge removes nothing;
`OPENRAL_FASTDDS_SHM_CLEAN=0` turns it off entirely.

```bash
uv run openral deploy sim --config scenes/deploy/openarm_real_world_voxels.yaml \
    --hal viewer_enabled=false --foxglove
```

`MUJOCO_GL=egl` is needed over ssh. Source the OpenArm vendor workspace (the one
holding `openarm_description`) in this terminal too: the Foxglove robot model is
the manifest URDF on `/robot_description`, and the bridge resolves its
`package://openarm_description/...` meshes only from its own `AMENT_PREFIX_PATH`.
Without it the 3D panel reports `Failed to retrieve asset` for every link.

The kernel runs with `world_voxel_enabled: true` but on the **sim** tuning: 0 m
margin, 15 mm cells and sim octomap thresholds. The real values (20 mm margin,
20 mm cells, occupancy threshold 0.6, clamping max 0.97) apply only under
`deploy run`. The reasoner is off and nothing dispatches, so the kernel evaluates
no chunk and `SafetyStatus` stays at its activation value. This pass checks the
**world**, not the verdicts.

Check and record:

- **Liveness.** `ros2 topic hz /octomap_binary` tracks the cloud (about 5 Hz on
  the Orin). `/openral/world_voxels` republishes between octrees (up to about
  10 Hz) and stops once the octree is older than 1.0 s, so its silence is
  meaningful, not its rate.
- **Frame.** `ros2 topic echo --once /openral/world_voxels --field header` shows
  `frame_id: openarm_base`.
- **One parent.** `ros2 run tf2_tools view_frames` shows `zed_camera_link` with
  exactly one parent, `openarm_base`. A second parent means `publish_tf` was left on.
- **Overlay.** Import the layout `deploy sim` prints (`[deploy_e2e] foxglove: ...
  import the scene-matched layout from /tmp/openral_layout_openarm_v2.json`);
  re-import it if you saved an older one, since layouts saved before the URDF
  layer draw no robot. The 3D panel follows `openarm_base` and draws the robot
  beside `/octomap_point_cloud_centers` and `/openral/world_voxels_cloud`. Check
  that voxels sit on the table, the shelf and the real arms. The model shown is
  the twin at zero, so park the real arms at zero to compare. Note voxels on arm
  links (self-occupancy, gap 1) and free-floating speckle near the arm envelope;
  each is a future false stop.
- **Unplug.** Pull the ZED USB for 10 s. `/octomap_binary` stops and about 1 s
  later `/openral/world_voxels` stops too (the bridge logs `last octree is ... old
  ... publishing nothing`); it resumes when the camera returns. Continuing voxels
  here is a failure.

Record a bag for the write-up:

```bash
ros2 bag record -s mcap -o ~/twin_pass_$(date +%F-%H%M) \
    /openral/world_voxels /octomap_binary /openral/safety_status /tf /tf_static
```

## 4. Real arms, attended [human, rig]

Before each launch and each dispatch, confirm presence (prerequisites). Then, in
the terminal of the person at the cell:

```bash
source /opt/ros/jazzy/setup.bash && source install/setup.bash && source <zed_ws>/install/setup.bash
export OPENRAL_OPENARM_ALLOW_MOTION=1 OPENRAL_OPENARM_ATTENDED=1
tools/openarm_world_voxel_run.sh --foxglove
```

The script refuses without both gates, without a measured `head_zed` mount in
`robots/openarm/robot.yaml` (step 2), and without an interactive terminal. It
asks you to type `ESTOP IN HAND`, then runs
`openral deploy run --config scenes/deploy/openarm_real_world_voxels.yaml`, which
brings up the vendor controllers (**arms snap to zero**), the ZED driver, octomap,
the bridge and the kernel with `world_voxel_enabled: true`,
`world_voxel_margin_m: 0.02` and `world_voxel_deadline_ms: 1000`. The cloud
reaches octomap by topic (`runtime.octomap_cloud_topic`), not through the sensor
leg.

To run a local copy of the scene (test 3's grasp target), pass
`--scene scenes/deploy/local/<copy>.yaml`. The copy gets the same gates, and the
script refuses it unless it lies under `scenes/deploy/local/` (gitignored),
validates as a `DeployScene`, and is identical to the committed scene except for
the top-level `grasp_declaration` block. Any other difference (octomap off, a
wider `robot_self_filter_padding_m`, an allowance, an extra collision pair, a
safety envelope, other drivers or HAL parameters, even a renamed `scene.id`) is
refused, naming each differing key path. Never launch a copy with a bare
`openral deploy run`: it skips every gate above.

In a second terminal, record the evidence for every test:

```bash
ros2 bag record -s mcap -o ~/attended_$(date +%F-%H%M) \
    /openral/safety_status /openral/failure/safety /openral/estop /openral/world_voxels \
    /octomap_binary /joint_states /tf /tf_static
```

Motion comes only from a direct dispatch of an rSkill already validated on this
cell. The reasoner is off in this scene. One goal is one confirmed action:

```bash
uv run python tools/_validation_matrix_dispatch.py --deadline-s 60 \
    --rskill-id <rskill validated on this cell> --prompt "<its task>"
```

Read a stop from `/openral/safety_status`: `latched: true`, `drop_reason: 10`
(`KIND_COLLISION`). The `FailureTrigger` on `/openral/failure/safety` names the
stopping link, the `voxel_<n>` cell and `min_distance_m` in `evidence_json`. To
tell a real obstacle from self-occupancy, locate the cell with
`idx = x + size_x*(y + size_y*z)` (`orientation` is the identity for a fixed-base
arm mapped in `openarm_base`, and the snippet asserts that):

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

A cell on or next to the stopping link itself is self-occupancy (gap 1), not an
obstacle.

**After each stop**, clear the cell, confirm presence, then reset:

```bash
uv run openral estop reset --robot openarm
```

`--robot` puts the command on the deploy's per-robot ROS domain; without it a
terminal that has not exported the deploy's `ros scope:` reaches nothing and the
robot stays latched. The command calls the kernel's `/openral/estop_reset` first
and, only if the kernel accepts, broadcasts `/openral/estop_cleared`, the same
sequence as the dashboard's reset button. Do not call `ros2 service call
/openral/estop_reset ...` alone: it clears only the kernel, the runner and HAL stay
latched and the runner rejects every later goal (`rskill_runner.goal_rejected:
e-stop latched`). Exit code 1 means the kernel refused, usually because the
cooldown has not passed; wait and run it again.

Run these tests in order:

1. **Idle.** Bring up and dispatch nothing for 60 s. Expect no stop. Note whether
   the arms at zero appear as voxels. With nothing dispatched the kernel judges
   no chunk, so a quiet minute does not show the map would let the robot move;
   the first dispatch is the first world-voxel verdict.
2. **Planted obstacle.** Put a soft obstacle (a foam block) on the table **in the
   camera's view**, in the path of the dispatched skill's first motion. Expect a
   `KIND_COLLISION` stop before contact, with the cell on the block and a
   `min_distance_m` consistent with where the link was (within about one 20 mm
   cell of the block's surface). Run it at least 3 times. A stop whose cell is on
   the arm is gap 1, not this test passing.
3. **Grasped object.** Dispatch a grasp. Per gap 2, expect the gripper to stop
   against its own payload once it closes on the object in view. Record the link,
   cell and distance. This is **not** an attached-payload-vs-voxels check: that
   check is off on real hardware.

   The committed scenes name no grasp target and no coordinates; what to pick
   arrives as language. Launch the autonomous posture,
   `scenes/deploy/openarm_real_autonomous.yaml` (this scene plus the reasoner,
   the open-vocabulary detector and spatial-memory ingest), and name a label:

   ```bash
   export OPENRAL_REASONER_MODEL=<a curated model id>
   tools/openarm_world_voxel_run.sh --autonomous
   # second terminal, presence confirmed for this goal:
   uv run openral prompt --robot openarm "pick up the boxes"
   ```

   The detector lifts every box it sees; the reasoner's `WORLD_STATE` lists each
   instance and its spatial-memory id, so it decomposes the goal into one subtask
   per box and names each by `grasp_target.object_id`; perception grounds the
   search box ([design §2.2](../../reference/real-pick-place-design.md)).
   Grounding refuses a bare label that several detections carry. The only OpenArm
   VLA today is task-specific, so a general "pick up X" needs a
   language-conditioned pick/place policy installed for the OpenArm; until then
   expect the reasoner to hand off.

   Direct dispatch (debug only, reasoner off): copy the committed scene into the
   gitignored `scenes/deploy/local/`, add a `grasp_declaration` block (the only
   difference the wrapper accepts) whose `search_box` you measured on this cell's
   live voxel map today, around exactly ONE item (a box covering two is refused as
   AMBIGUOUS), and launch it with `tools/openarm_world_voxel_run.sh --scene
   <copy>` (add `--autonomous` for a copy of the autonomous scene). Never reuse
   another cell's or another day's numbers.
4. **Camera unplug.** During a dispatch, unplug the ZED. Expect
   `/openral/world_voxels` to stop about 1 s after `/octomap_binary` and the
   kernel to drop the next chunks with `DROP_VOXEL_UNAVAILABLE` within about
   2.0 s of the last cloud (a drop, not a latch). Motion or certified chunks after
   that window is a failure: have the E-stop ready. Record both topics'
   last-message times and the first `DROP_VOXEL_UNAVAILABLE`.

## 5. Exit criteria and what to record [offline]

The criteria for making the check **default-on** for the OpenArm:

- zero missed stops on the planted obstacle;
- a false-stop rate the TSC accepts, counted over a fixed task set, with stops on
  the table and self-occupancy stops counted separately;
- a staleness drop, not motion, when the camera is unplugged
  (`DROP_VOXEL_UNAVAILABLE` within about 2.0 s of the last cloud).

For the write-up, record:

- `head_zed`'s pose in `robot.yaml` and how it was measured (its comment);
- the twin-pass numbers: `/octomap_binary` rate, the `/openral/world_voxels` rate
  and frame, and whether self-occupancy was observed;
- for every attended dispatch: the rSkill id, the outcome, and each stop's
  `drop_reason`, link, cell centre, `min_distance_m`, and whether it was an
  obstacle, self-occupancy, the payload or the table;
- the counts of `SafetyStatus` transitions by `drop_reason`, from the bag;
- the bags themselves, stored off-host.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| The script refuses to start | Check both motion gates, a measured `head_zed` mount (`static_transform_calibrated: true`), and an interactive terminal. |
| Map stalls, "queue is full" in `octomap_server` | The ZED driver is running with IPC on; relaunch with `enable_ipc:=false`. |
| `zed_camera_link` has two parents | `publish_tf` or `publish_map_tf` was left on; set both to `false`. |
| Foxglove shows `Failed to retrieve asset` | Source the vendor workspace holding `openarm_description` before starting the bridge. |
| Stop on a cell next to the stopping link | Self-occupancy (gap 1), not an obstacle. Re-check the extrinsic (step 2). |
| Robot stays latched after `estop reset` | Run it with `--robot openarm`; exit code 1 means the kernel's cooldown has not passed. |

## Files

- `scenes/deploy/openarm_real_world_voxels.yaml`: the real-cell scene. The check
  is on and it holds the ZED driver include. It declares no `head_zed` entry.
- `scenes/deploy/openarm_real_autonomous.yaml`: the same cell with the reasoner,
  the open-vocabulary detector and spatial-memory ingest on (`--autonomous`); the
  vision attachment leg stays off.
- `robots/openarm/robot.yaml`: the cameras' bindings, `head_zed`'s calibrated
  pose (step 2) and the grippers' `closure_calibration`. No furniture is
  surveyed; place surfaces are measured live from the voxel map under the carried
  payload ([real pick-and-place design](../../reference/real-pick-place-design.md) §2.3).
- `python/core/src/openral_core/depth_extrinsic.py`: the accuracy the pose needs
  and the gate `deploy run` applies.
- `tools/openarm_world_voxel_run.sh`: the guarded launcher for step 4.
