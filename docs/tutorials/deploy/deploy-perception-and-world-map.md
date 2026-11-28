# Cameras, world map and the collision view

**Goal:** bind a camera whose stream only exists as a ROS topic (a ZED, for
example), turn its depth into the world map the safety kernel checks, and read
that map in Foxglove.

This is the perception half of a deployment. The core flow (scene, dry run,
run, dashboard) is in [Run a deployment and open the dashboard](deploy-run-and-dashboard.md).

## Prerequisites

- A working `DeployScene` for your robot (see the core tutorial, step 1).
- For the ZED recipe: the **ZED SDK** and `zed_wrapper` installed on the deploy
  host.
- `uv run openral calibrate camera` done for any camera whose depth you project
  into a point cloud. A VLA conditions on pixels, so uncalibrated `fx/fy/cx/cy`
  are inert for it; they matter once depth becomes a cloud for octomap or
  collision.

## 1. Bind a camera that only exists as a ROS topic (`ros2_image`)

`opencv_thread` opens a `/dev/video*` node, which covers most USB cameras. It
cannot reach a stream a **vendor SDK computes**, most importantly depth from a
passive stereo camera. A StereoLabs ZED over USB presents one UVC node with both
eyes side by side and no depth; the ZED SDK rectifies and stereo-matches on the
host GPU and `zed_wrapper` publishes the result.

Bind those streams with the `ros2_image` backend and let the scene start the
driver. This recipe is used on the OpenArm bench
(`scenes/deploy/openarm_bench.yaml`) and applies to any cell with a ZED-M under
the wrapper's default `zed` camera name and `zed_node` node name. Another ZED
model changes `camera_model` and needs its own override file; a different camera
or node name changes the `/zed/zed_node/...` topic root on every binding.

```yaml
# The scene launches zed_wrapper itself (real path only; `deploy sim` renders).
drivers:
  - package: zed_wrapper
    launch_file: zed_camera.launch.py
    args:
      camera_model: zedm            # zed / zedm / zed2 / zed2i / zedx ...
      # The camera is bolted to the robot, so its pose belongs to the robot's TF
      # tree (the manifest's mount), not to the wrapper's visual odometry. With
      # these on, zed_camera_link gets a SECOND parent and octomap drops clouds.
      publish_tf: "false"
      publish_map_tf: "false"
      # With IPC on (the wrapper default) the camera frames go on /tf as DYNAMIC
      # transforms and octomap_server stalls waiting on them.
      enable_ipc: "false"
      # Scene-relative. Pins VGA, turns the raw image topics off (`top` reads
      # rgb/color/rect/image), positional tracking off and depth stabilization to 0.
      ros_params_override_path: drivers/zedm_openarm_override.yaml

runtime:
  enable_octomap: true
  octomap_cloud_topic: /zed/zed_node/point_cloud/cloud_registered
```

A driver started from the scene's `drivers:` block inherits the deploy's Fast DDS
large-data profile (`FASTRTPS_DEFAULT_PROFILES_FILE`, named on the
`dds_transport_ready:` line). A driver you start in another shell does not;
export the same variable there, or its samples above about 512 KB are mostly
lost ([DDS transport for large messages](../../reference/dds-large-messages.md)).

**Where the bindings live.** The scene has no `sensors:` block for the ZED. It is
bolted to the robot, so both streams are robot cameras, and a deploy scene never
touches a camera the robot manifest defines (a scene entry that reuses a
manifest sensor name is refused). The manifest owns each camera's identity,
calibrated mount and `deploy_binding`; per-cell mounts can go in
`robots/<robot>/units/<unit>.yaml`. `robots/openarm/robot.yaml` declares, in
abridged form:

```yaml
sensors:
  - name: head_zed                     # depth: SDK-computed, so it exists only as a topic
    static_transform_xyz_rpy: [0.06597, -0.004548, 0.094539, -0.055916, 1.145014, -0.036625]
    static_transform_calibrated: true  # measured, so a real world-voxel deploy accepts it
    deploy_binding:
      backend: ros2_image
      backend_params:
        topic: /zed/zed_node/depth/depth_registered
        reliability: best_effort       # also matches a RELIABLE publisher; the reverse gets nothing
        qos_depth: 5
      max_age_ms: 500
  - name: top                          # RGB: the wrapper's rectified left image (aligned with the depth)
    deploy_binding:
      backend: ros2_image
      backend_params:
        topic: /zed/zed_node/rgb/color/rect/image
        reliability: best_effort
        qos_depth: 5
      max_age_ms: 200
```

A scene's own `sensors:` block is only for a camera that belongs to the *cell*
(an overhead camera on a stand). It takes a name the manifest does not use and
carries its own geometry and binding:

```yaml
sensors:
  - name: overhead
    modality: rgb
    frame_id: overhead_optical_frame
    parent_frame: world
    static_transform_xyz_rpy: [0.4, 0.0, 1.2, 3.1416, 0.0, 0.0]
    rate_hz: 30.0
    encoding: rgb8
    deploy_binding:
      backend: opencv_thread
      backend_params: {device: /dev/v4l/by-id/<your-camera>-video-index0, fps: 30}
```

**What the policy receives.** Every frame in the world state is decoded by its
encoding. Depth rides along in the observation as a `uint16` `(H, W, 1)` array
under its sensor name (an RGB-only policy never reads it), and a JPEG or PNG
camera is decoded to RGB. The reader converts `32FC1` metre depth to `DEPTH16`
(uint16 millimetres); non-finite samples or samples outside `[0, 65.535] m`
become `0`, the "no reading" value. A frame the decoder cannot handle is logged
as `runner.frame_skipped`; if that sensor feeds a required camera slot, the goal
fails with `ROSPerceptionStale` rather than run the policy without the view.

**What the bus carries.** The bus is not the policy's path. Cameras without a
GStreamer tee reach `/openral/cameras/<name>/image` through a Python republisher
capped at 3 Hz and 320x240; the policy reads the native frame in-process. Per
camera, `backend_params` takes `topic_rate_hz: <hz>` to lift the rate and
`topic_native: true` to publish at capture size, for a bus reader that needs
exact pixels such as a calibration pass. To record what the policy saw, pass
`--dataset-out <file.mcap>`.

The same backend covers RealSense aligned depth, Orbbec, GMSL/Isaac drivers or
any rectified stream a calibration node publishes.

## 2. Build a world map from real depth (`octomap_cloud_topic`)

`enable_octomap: true` alone is not enough on real hardware. `octomap_server`
subscribes to whatever `octomap_cloud_topic` names. Under `deploy sim`, left
unset, it is `/openral/cameras/<name>/points` of the manifest's one depth sensor
with intrinsics (a manifest with several must pin one); the **sim** sensor bridge
publishes that cloud. Nothing in-tree publishes a cloud under `hal_mode:=real`.

So `deploy run` refuses with `ROSConfigError` before launch when octomap is on
and no topic is pinned. This includes the auto-enable: any depth or 3D-lidar
sensor in the manifest or scene turns octomap on. Without the check every node
comes up healthy while the octree, `/openral/world_voxels` and the dashboard's
POINTCLOUD card stay empty, and the kernel drops every chunk as
`DROP_VOXEL_UNAVAILABLE`. Pin the driver's topic, or set `enable_octomap: false`.

Point it at the cloud your depth driver already publishes. Reusing it is
deliberate: `zed_wrapper`, RealSense and Orbbec already project on the GPU.

```yaml
runtime:
  enable_octomap: true
  # zed_wrapper's registered cloud. RealSense: /camera/depth/color/points.
  octomap_cloud_topic: /zed/zed_node/point_cloud/cloud_registered
  # Optional, per rig (defaults in parentheses):
  # world_voxel_deadline_s: 2.0          # how long the kernel trusts the last grid (1.0 s; hard cap 2.0 s)
  # max_octree_age_s: 2.0                # how long the bridge republishes the last octree (= the deadline, never above)
  # world_voxel_data_age_budget_s: 1.5   # max camera-data age behind a grid at chunk check (1.5 s; hard cap 3.0 s)
  # robot_self_filter_padding_m: 0.02    # margin past the collision model where depth is removed as "the robot"
```

Raise the first two for a source slower than about 1 Hz. Measure your rig's
latency tail before lowering `world_voxel_data_age_budget_s`: below the tail the
robot stops. `robot_self_filter_padding_m` is also a blind shell around the arm;
derive it from depth noise, extrinsic error, capture-to-joint-state motion and
half a voxel, and do not raise it without that derivation (hard cap 0.10 m).

On a `deploy sim` twin, a pinned topic outside `/openral/cameras/` is treated as
a real driver stamping wall-clock, so the graph runs on host wall time without a
`clock_origin` pin (pinning `simulation` with it is refused).

Two worked scenes, both `deploy sim` but with fields that apply unchanged under
`hal_mode:=real`:
[`openarm_zed_octomap.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/deploy/openarm_zed_octomap.yaml)
sets `enable_octomap` and the cloud topic together;
[`openarm_tabletop.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/deploy/openarm_tabletop.yaml)
pins `enable_octomap: false` to override the auto-enable. Copy one; never leave
the pair half-set.

Enabling this leg also attaches the `WorldCloudBridge`, which feeds the
dashboard's `world.pointcloud` span. The card stays dark unless
`enable_octomap` or `slam_mono_camera` is on.

**Before enabling it on a robot that moves:**

- `octomap_server` needs the cloud's `frame_id` resolvable against `odom`
  through TF. A driver publishing in its own optical frame needs that frame
  connected to the robot's tree, which the sensor's `frame_id` / `parent_frame`
  provides.
- With `enable_octomap_kernel_check` at its default, these voxels are a
  **safety input**: the C++ kernel rasterises the arm's capsules against them and
  E-stops on overlap, so a noisy or mis-framed cloud shows up as an E-stop. Bring
  the leg up with the arm unpowered and check the card first.

To let the map **stop** a real arm rather than just draw it, use
[`openarm_real_world_voxels.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/deploy/openarm_real_world_voxels.yaml)
and its [attended runbook](openarm-real-world-voxel-check.md). It adds
`enable_octomap_kernel_check: true` and a guarded launcher that refuses until the
camera pose is declared as measured in `robots/openarm/robot.yaml`. OpenRAL does
not calibrate that pose; the runbook states the frames and accuracy the check
needs. The gate is robot-agnostic. Its autonomous sibling,
[`openarm_real_autonomous.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/deploy/openarm_real_autonomous.yaml),
adds the reasoner, the open-vocabulary detector and spatial-memory ingest so a
goal is language (`uv run openral prompt "pick up the boxes"`), behind the same
launcher's `--autonomous`.

## 3. Read the collision view in Foxglove

`openral deploy sim` starts a read-only Foxglove bridge next to the graph (with
the dashboard, or `--foxglove`). Its 3D panel shows the world the safety kernel
checks: the voxel map, the declared fixtures and the boxes the grasp leg measures.
The frames below come from a bimanual OpenArm pick-and-place in Isaac Sim with the
[manipulation-workspace contact](../../reference/manipulation-workspace-contact.md)
rule on; the camera tiles, sim-time header and event text were added by the
recording tool (in a live session the event text is kernel log lines).

Each element is tagged **detected** (the graph measures it live, as on a real
robot) or **from simulator** (the sim supplied it; on a real cell it must come
from the cell's scene graph, CAD or a survey).

**Grasp phase.** The grasp leg has already measured the target carton from head
depth, before any finger touches it.

![Foxglove collision view during the grasp phase: camera tiles, height-coloured voxels, colour
depth points, black-outlined static geometry and the green grasp region, each numbered and
tagged detected or from simulator](images/collision-view-grasp.webp)

1. **Camera tiles** (from simulator): the frames the policy acts on.
2. **Voxels, coloured by height** (detected): the kernel's occupancy map, built
   from head depth by octomap at 15 mm. Cells inside a declared box fixture are
   occupied in the kernel but left out of the drawing (`hide_static_voxels`).
3. **Colour points** (detected): the raw head-camera cloud with the robot
   filtered out. The kernel does not check it directly.
4. **Black-outlined boxes** (from simulator): the declared static geometry
   (`runtime.static_geometry_file`: tote walls, rack, decks, shelf stock).
   `tools/derive_scene_inputs.py` exports them from the stage's static colliders.
   They are always occupied, seen or not.
5. **Green box, grasp region** (detected): the target carton, measured from head
   depth inside the grasp declaration's `search_box`. The fingers may touch it.

**Carry phase.** The carton is attached to the left hand and rides with it.

![Foxglove collision view during the carry phase: the orange payload box on the left hand, a
tolerated-contact line, a brake line with its creep fraction, and the static geometry over the
shelf](images/collision-view-carry.webp)

1. **Orange box, payload** (detected): the held carton's collision box, from the
   measured grasp region (`evidence_kind: grasp_target_region`). The kernel
   checks it against the world and robot as part of the hand. Its cells are
   **not** removed from the voxel map unless the object tracker owns the
   attachment (`runtime.object_tracking: tracking`, sim-only). With tracking
   `off` the carton stays a mapped obstacle and the kernel may stop the arm on
   what it holds, which is the fail-closed posture
   ([octomap bridge README](https://github.com/OpenRAL/openral/blob/master/packages/openral_octomap_bridge/README.md#attached-objects-and-world-occupancy)).
2. **Contact tolerated** (`safety.workspace_contact`): a hand link or its payload
   touches an occupied cell inside the declared `workspace_box`, slower than
   `workspace_contact_max_speed_mps` and within `workspace_contact_max_lead_m`.
   The line reports the link, depth into the cell and hand speed. Arm links never
   get this tolerance.
3. **BRAKE ... creep** (`safety.workspace_brake`): the chunk failed **only**
   because the hand was too fast or led its measured pose by too much, so the
   kernel scales the chunk toward the measured pose (`creep=0.099` is about 10 %
   of the commanded step). Any other reason to refuse still stops the robot.
4. **Black-outlined boxes**: the same declared static geometry (shelf stock).

**A payload from wrist segmentation.** The right hand holds a carton whose box
was measured by wrist-camera segmentation (`evidence_kind: vision_segmentation`).

![Foxglove collision view with a cyan payload box measured by wrist-camera segmentation, two
tolerated-contact lines and the height-coloured voxels](images/collision-view-segmentation.webp)

Other colours: blue for an active place region, amber for the map-confirmed
workspace box, yellow for the jaw-span fallback payload, and a red sphere where a
kernel stop happened. They are defined in
[`bucket2_markers.py`](https://github.com/OpenRAL/openral/blob/master/packages/openral_foxglove_bringup/openral_foxglove_bringup/bucket2_markers.py);
the topics are listed in the
[Foxglove bringup README](https://github.com/OpenRAL/openral/blob/master/packages/openral_foxglove_bringup/README.md).

Workspace contact is off by default (`runtime.workspace_contact_enabled`) and
sim-only: `openral deploy run` on a real HAL refuses it.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `ROSPerceptionStale` naming a topic | The driver is not running or the QoS mismatches (the two look identical to the subscriber). Start `zed_wrapper`; keep `reliability: best_effort`. |
| `ROSConfigError` before launch about the octomap cloud | Octomap is on with no `octomap_cloud_topic`. Pin the driver's cloud, or set `enable_octomap: false`. |
| Octree and POINTCLOUD card stay empty, kernel drops chunks as `DROP_VOXEL_UNAVAILABLE` | The pinned topic publishes nothing, or its frame does not resolve against `odom`. Check the topic and the sensor's `frame_id` / `parent_frame`. |
| octomap drops every cloud | A second parent for `zed_camera_link`: set `publish_tf` and `publish_map_tf` to `"false"`. |
| octomap_server stalls for seconds | Set `enable_ipc: "false"` on the ZED driver. |
| Large driver samples mostly lost | Export `FASTRTPS_DEFAULT_PROFILES_FILE` in the shell that starts the driver. |

## Next

- [Run a deployment and open the dashboard](deploy-run-and-dashboard.md)
- [Attended real-world voxel check (OpenArm)](openarm-real-world-voxel-check.md)
- [ROS graph security](../../reference/ros-graph-security.md)
