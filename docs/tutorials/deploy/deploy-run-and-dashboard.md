# Run a deployment on a robot and open the dashboard

`openral deploy run` is the real-hardware sibling of `openral sim run`. It boots
the full production ROS graph — the HAL lifecycle node, the C++ safety kernel,
the reasoner, world state (plus SLAM/Nav2 when the robot declares a lidar) — and
ticks an rSkill against your **real** robot, driven by a `DeployScene`
YAML. This tutorial writes a deploy scene, dry-runs it
against a digital twin, then runs it on hardware with the live dashboard.

## Prerequisites

```bash
just bootstrap && just sync   # always `just sync`, never bare `uv sync` —
                              # see docs/contributing/toolchain.md
openral install ros           # the ROS 2 graph deploy run launches
uv run openral doctor         # confirm ROS 2, GPU, USB and the reasoner
```

You need a `RobotDescription` for your robot under
[`robots/<robot_id>/robot.yaml`](https://github.com/OpenRAL/openral/blob/master/robots/),
and an installed rSkill (see
[Write an rSkill](../rskill/write-and-publish-an-rskill.md)).

The in-tree manifests are:

| Class | `robot_id` |
| --- | --- |
| Single arm | `so100_follower`, `so101_follower`, `franka_panda`, `ur5e`, `ur10e`, `rizon4`, `sawyer`, `widowx`, `galaxea_a1` |
| Bimanual | `aloha_bimanual`, `aloha_agilex`, `openarm`, `anvil_openarm_v2` |
| Mobile manipulator | `panda_mobile`, `panda_mobile_vslam`, `google_robot`, `r1pro` |
| Humanoid | `g1`, `h1`, `gr1` |

Class is each manifest's `embodiment_kind`. `pusht_2d` is omitted: it is a
sim-only scene-pseudo-robot with no hardware path.

Being in that table is not itself a hardware path. `deploy run` resolves
`hal.real` from the manifest, and only nine declare one — `so100_follower`,
`so101_follower`, `franka_panda`, `ur5e`, `ur10e`, `sawyer`, `galaxea_a1`,
`aloha_bimanual` and `openarm`. The rest carry `hal.real: null` (or no `hal:`
block at all) and `build_hal` raises `ROSCapabilityMismatch` under
`hal_mode:=real`. Watch the two OpenArms in particular: `openarm` (Enactic) has
`OpenArmRealHAL`, while `anvil_openarm_v2` (Anvil) is sim-only.

### Set a reasoner model — the graph will not plan without one

`deploy run` boots the S2 reasoner, and the reasoner has **no hidden default
model** by design. Set a curated one before launching:

```bash
export OPENRAL_REASONER_MODEL=claude-opus-4-8   # or gpt-5.5 / gpt-5.6 / cosmos3-edge
export OPENRAL_REASONER_API_KEY=sk-ant-...      # only where the endpoint needs it
```

Full model/endpoint matrix, the uncurated escape hatch, and what `openral
doctor`'s `Reasoner LLM` row checks: [reasoner
reference](../../reference/reasoner.md#reasoner-model-selection).

`cosmos3-edge` is the on-device option — no key, no cloud. Budget the whole
GPU for it: the checkpoint is 8.6 GB on disk (a 6.3 GB reasoner tower plus a
934 MB vision encoder; the VAE is not served) and loads at 4.66 GiB, so on an
8 GB card it fits only with the card to itself — a co-resident job holding even
2 GB makes the boot fail. The sidecar's defaults are also too small for a real
palette: the reasoner's prompt measures 10K tokens for a 4-skill embodiment and
20K for a 14-skill one, against a `--max-model-len` default of 8192. What works
on 8 GB, verified end to end:

```bash
python tools/cosmos3_reasoner_sidecar.py --port 8901 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.97 --max-model-len 12288
```

That serves the small palette with room for the reply, and not the large one. `openral doctor` cannot
see any of this — the `Reasoner LLM` row reports `ok` because the *model
resolves*; only a live tick exercises the sidecar. Sizing, platform status and
the live-validation record:
[`docs/reference/cosmos3-edge-reasoner.md`](../../reference/cosmos3-edge-reasoner.md).

## 1. Write a `DeployScene` config

Deploy configs live in
[`scenes/deploy/`](https://github.com/OpenRAL/openral/blob/master/scenes/deploy/).
A `DeployScene` pins the workcell: scene id, `robot_id`, optional sim
composition, safety tightening, and additive allowed collision pairs. Robot
facts such as serial ports, IPs, sensors, poses, rates, and limits live in
`robots/<robot_id>/robot.yaml`.

### Option A — create/update the robot manifest with `openral detect`

If the robot is plugged in, let detection write or refresh the robot manifest:

```bash
# A bare detect resolves a plugged-in Feetech arm to so101_follower by default.
openral detect \
    --output robots/so101_follower/robot.yaml \
    --deployment scenes/deploy/so101_bench.yaml
```

Detection records robot-owned facts in `robot.yaml`; it does not create a
deploy scene unless `--deployment` is passed. The wizard always runs (there is
no `--interactive` flag) and opens the camera binding wizard: a robot camera's
binding lands in `robot.yaml`, a workcell camera (its own name) in that deploy
scene — a deploy scene never touches a robot camera. A robot type with more than
one unit or host keeps each host's camera bindings (and per-unit calibration) in
`robots/<id>/units/<unit>.yaml` instead; the scene's `robot_unit` or
`OPENRAL_ROBOT_UNIT` selects it, and a real deploy of such a robot refuses until
one is selected (SO-101 bench: `units/bench_laptop.yaml`; OpenArm: `thor`, `orin`).
Use `--include usb,gpu,cameras_v4l2,cameras_realsense` to limit probes, and
`--report detect.json --no-write` when you only want the raw detection report.
The rSkill that drives the robot is **not** set in deploy config — the reasoner
selects it at runtime from the installed `rskills/` registry.

> The SO-101 is electrically identical to the SO-100 over USB (same Feetech
> controller), so the bus alone can't distinguish them — the current SO-101 is
> the default. To target the older SO-100 instead, add `--robot so100` (the flag
> accepts a short slug like `so100` or a manifest directory name like
> `so100_follower`).

### Option B — write the workcell by hand

Create `scenes/deploy/so100_pick_cube.yaml`:

```yaml
scene:
  id: so100_pick_cube
  backend: mujoco
robot_id: so100_follower          # matches robots/so100_follower/robot.yaml
safety:
  workspace_box_min_xyz: [-0.3, -0.3, 0.0]
  workspace_box_max_xyz: [0.3, 0.3, 0.5]
```

The full schema is `openral_core.schemas.DeployScene`. `safety` is optional and
must tighten the robot manifest's `SafetyEnvelope`.

List what's available:

```bash
openral deploy list      # walks scenes/deploy/*.yaml
```

### Cameras whose stream only exists as a ROS topic (`ros2_image`)

`opencv_thread` opens a `/dev/video*` node, which covers most USB cameras. It
cannot reach a stream that a **vendor SDK computes** rather than the device
emitting it — most importantly **depth from a passive stereo camera**.

A StereoLabs ZED is the clear case. Over USB it presents *one* UVC node carrying
both eyes side by side (`1344×376` on a ZED Mini at WVGA); the device produces no
depth at all. The ZED SDK rectifies and stereo-matches on the **host GPU**, and
`zed_wrapper` publishes the result. So a host without the SDK sees one wide RGB
camera and nothing else — which is exactly what `openral detect` reports.

Bind those streams with the `ros2_image` backend — and let the scene start the
driver, so there is no second terminal to forget. This is the recipe verified on
the OpenArm bench (`scenes/deploy/openarm_bench.yaml`, ZED-M, SDK 5.4.1) and it
is not OpenArm-specific: a cell with a ZED-M bolted to the robot, launched under
the wrapper's default `zed` camera name and `zed_node` node name, uses it as is.
Another ZED model changes `camera_model` and needs its own override file (the
ZED-M one pins settings for its USB hub and IMU); a different camera or node
name changes the `/zed/zed_node/...` topic root on every binding below.

```yaml
# The scene launches zed_wrapper itself (real path only; `deploy sim` renders).
drivers:
  - package: zed_wrapper
    launch_file: zed_camera.launch.py
    args:
      camera_model: zedm            # zed / zedm / zed2 / zed2i / zedx …
      # The camera is bolted to the robot, so its pose belongs to the robot's
      # TF tree (the manifest's mount), not to the wrapper's visual odometry.
      # With these on, zed_camera_link gets a SECOND parent, tf2 lookups go
      # order-dependent and octomap silently drops every cloud.
      publish_tf: "false"
      publish_map_tf: "false"
      # Scene-relative. Pins HD720 (the ZED-M shares a USB hub with other
      # cameras and reboots in a loop at HD1080), turns positional tracking
      # off and depth stabilization to 0 (the SDK force-enables tracking
      # otherwise, defeating the setting above).
      ros_params_override_path: drivers/zedm_openarm_override.yaml

runtime:
  # A depth SensorSpec with intrinsics auto-enables the octomap leg, but the
  # cloud topic keeps a sim-only launch default unless the scene pins it —
  # leaving the octree empty behind a healthy-looking graph.
  enable_octomap: true
  octomap_cloud_topic: /zed/zed_node/point_cloud/cloud_registered
```

The scene has **no `sensors:` block**. The ZED is bolted to the robot, so both of
its streams are robot cameras, and a deploy scene never touches a camera the
robot manifest defines (`check_scene_sensor_overrides` refuses a scene entry
that reuses a manifest sensor's name). The bindings go on the manifest's own
entries, in `robots/<robot>/robot.yaml`, next to the geometry they belong to —
this is what `robots/openarm/robot.yaml` commits:

```yaml
sensors:
  # Depth: SDK-computed, so it exists only as a topic. `32FC1` metres on the
  # wire (or `16UC1` millimetres with `openni_depth_mode: true`) — either way
  # it reaches the world state as DEPTH16, uint16 millimetres.
  - name: head_zed
    modality: depth
    frame_id: zed_camera_link
    parent_frame: openarm_base      # the mount: robot geometry, calibrated here
    # … static_transform_xyz_rpy, intrinsics …
    deploy_binding:
      backend: ros2_image
      backend_params:
        # Verified against a running zed_wrapper: the topic root is
        # /<camera_name>/<node_name>/, so the node name is part of the path.
        topic: /zed/zed_node/depth/depth_registered
        # best_effort (the default) also matches a RELIABLE publisher; a
        # `reliable` subscriber gets NOTHING from a best-effort one.
        reliability: best_effort
        qos_depth: 5
      max_age_ms: 500

  # RGB: the wrapper's rectified left image, the policy's `top` view. It is
  # published as `bgra8`; the reader drops the constant alpha plane and
  # delivers `bgr8`.
  - name: top
    modality: rgb
    # … frame_id, intrinsics, vla_feature_key …
    deploy_binding:
      backend: ros2_image
      backend_params:
        topic: /zed/zed_node/rgb/color/rect/image
        reliability: best_effort
        qos_depth: 5
      max_age_ms: 200
```

A camera that is part of the *cell* rather than the robot — an overhead or
front camera on a stand — is the one thing a scene's `sensors:` block is for.
It takes a name the manifest does not use, and carries its own geometry and
binding:

```yaml
sensors:
  - name: overhead               # not a manifest sensor name
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

Every frame in the world state, RGB and depth alike, is decoded by its
encoding (`decode_inline_frame`): the depth frame rides along in the policy's
observation as a `uint16` `(H, W, 1)` array under its own sensor name, and an
RGB-only policy simply never reads it. Until 2026-09-22 the runner read every
frame as `uint8`, and a ZED depth frame next to the RGB slots aborted the first
real OpenArm dispatch with `cannot reshape array of size 1843200`. A camera
that delivers JPEG or PNG (an MJPEG USB camera) is decoded to RGB the same way.
A frame the decoder cannot handle is logged as `runner.frame_skipped` with the
sensor and encoding; if that sensor feeds one of the policy's required camera
slots, the goal fails with `ROSPerceptionStale` rather than running the policy
without that view.

Prerequisites on the host: the **ZED SDK** installed, and `zed_wrapper` running
and publishing. Without them the reader opens fine and then every `read_latest`
raises `ROSPerceptionStale` naming the topic — a driver that isn't running and a
QoS mismatch look identical from the subscriber side, so the error message calls
both out.

The reader converts `32FC1` metre depth into OpenRAL's `DEPTH16` layout (uint16
millimetres). Samples that are non-finite (`NaN` is a stereo matcher's "no
match") or outside `[0, 65.535] m` become `0`, the ROS "no reading" value —
never a wrapped `uint16`, which would read as a confident *near* distance for
something far away.

The same backend covers RealSense aligned depth, Orbbec, GMSL/Isaac drivers, or
any rectified stream published by a calibration node.

> **Calibrate before you project.** A VLA conditions on pixels, so uncalibrated
> `fx/fy/cx/cy` are inert for it. They stop being inert the moment depth is
> projected into a point cloud for octomap / nvblox / collision. Run
> `openral calibrate camera` first.
### Building a world map from a real depth camera (`octomap_cloud_topic`)

Turning `enable_octomap: true` on is not enough on real hardware. `octomap_server`
subscribes to whatever `octomap_cloud_topic` names. Under `deploy sim`, left unset,
it is `/openral/cameras/<name>/points` of the manifest's one depth sensor with
intrinsics (e.g. `head_zed` on `openarm`, `front_depth` on `panda_mobile`) — a
topic published by the **sim** sensor bridge, which back-projects the digital
twin's depth raster; a manifest with several depth sensors must pin the one to map.
Nothing in-tree publishes a cloud under `hal_mode:=real` (the sensor leg publishes a
bound depth sensor's *image* on `/openral/cameras/<name>/depth/image`).

So `deploy run` refuses with `ROSConfigError` before launch when octomap is on and
nothing is pinned. That includes the auto-enable: any depth or point-cloud
(3D lidar) sensor in the manifest or the scene turns octomap on. Before this check
the failure was silent in the worst way: every node came up healthy, the octree,
`/openral/world_voxels` and the dashboard's POINTCLOUD card stayed empty, and the
kernel dropped every chunk as `DROP_VOXEL_UNAVAILABLE`. Pin the driver's topic, or
set `enable_octomap: false` to run without the world map.

Point it at the cloud your depth driver already publishes:

```yaml
runtime:
  enable_octomap: true
  # zed_wrapper's own registered cloud. RealSense: /camera/depth/color/points.
  octomap_cloud_topic: /zed/zed_node/point_cloud/cloud_registered
  # Optional, per rig: how long the kernel trusts the last voxel grid (default 1.0 s)
  # and how long the octomap bridge republishes the last octree (default = the
  # deadline, never above it). Raise both for a source slower than ~1 Hz, up to the
  # schema's hard cap of 2.0 s on the deadline (above it the scene is refused).
  # world_voxel_deadline_s: 2.0
  # max_octree_age_s: 2.0
  # Optional, per rig: how old the camera data behind a voxel grid may be when the
  # kernel checks a chunk, from capture (default 1.5 s, measured on the Thor ZED-M:
  # p99 ~1.0 s; hard cap 3.0 s). Measure yours; below the rig's latency tail the robot stops.
  # world_voxel_data_age_budget_s: 1.5
  # Optional, per rig (real camera only): how far past the robot's collision model a
  # depth return is removed as the robot before octomap. Provisional 0.02 m; derive
  # it from depth noise, extrinsic error, capture-to-joint-state motion and half a voxel.
  # It is also a blind shell around the arm, so do not raise it without that derivation
  # (hard cap 0.10 m).
  # robot_self_filter_padding_m: 0.02
```

On a `deploy sim` twin, a pinned topic outside `/openral/cameras/` is a real
driver stamping on wall-clock, so the graph runs on host wall time without a
`clock_origin` pin (pinning `simulation` with it is refused: octomap would drop
every cloud as from the future).

Two worked examples make the choice explicitly:
[`scenes/deploy/openarm_zed_octomap.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/deploy/openarm_zed_octomap.yaml)
sets `enable_octomap` and `octomap_cloud_topic` together, and
[`scenes/deploy/openarm_tabletop.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/deploy/openarm_tabletop.yaml)
pins `enable_octomap: false` to override the auto-enable that the manifest's
`head_zed` depth `SensorSpec` would otherwise trigger (`deploy run` resolves
that auto-enable through the same code path as `deploy sim`). Both are
`deploy sim` scenes, but the same fields apply unchanged under
`hal_mode:=real`. Copy one of them; never leave it half-set.

To let that map **stop** a real arm, not just draw it, see
[`scenes/deploy/openarm_real_world_voxels.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/deploy/openarm_real_world_voxels.yaml)
and its [attended runbook](openarm-real-world-voxel-check.md): it adds
`enable_octomap_kernel_check: true` and a guarded launcher that refuses until the camera
pose in `robots/openarm/robot.yaml` is calibrated and verified.

That is deliberate reuse rather than a new node: `zed_wrapper` (and the RealSense
and Orbbec drivers) already stereo-match and project on the GPU, so composing a
depth-to-cloud converter would redo work the driver has done.

Enabling this leg also attaches the `WorldCloudBridge`, which is what feeds the
dashboard's `world.pointcloud` span — the card is gated on
`enable_octomap or slam_mono_camera`, so it stays dark until one of them is on.

Two things worth knowing before flipping it on a robot that moves:

- `octomap_server` needs the cloud's `frame_id` resolvable against `odom` through
  TF. A driver publishing in its own optical frame needs that frame connected to
  the robot's tree — the sensor's `frame_id` / `parent_frame` in the deploy scene
  is what does that.
- With `enable_octomap_kernel_check` left at its default, these voxels become a
  **safety input**: the C++ kernel rasterises the arm's capsules against them and
  E-stops on overlap. That conservative posture means a noisy or mis-framed
  cloud surfaces as an E-stop rather than as a bad picture. Bring the leg up
  with the arm unpowered and check the card first.

## 2. Dry-run against a digital twin first

Before touching hardware, validate the whole graph against a simulated HAL
with `openral deploy sim`. It boots the **same** graph (dashboard + safety
kernel + reasoner + prompt router + runtime + HAL) but against a digital-twin
HAL driven by the same `DeployScene` YAML — no robot required:

```bash
openral deploy sim \
  --config scenes/deploy/so100_pick_cube.yaml
```

`deploy sim` takes no `--rskill` — the reasoner picks the active rSkill from the
in-tree `rskills/` palette at `on_configure`, embodiment-filtered.

This is the safe place to shake out manifest, sensor, and rSkill-compatibility
errors.

### Two cheaper checks before that

`deploy sim --dry-run` resolves the scene, the HAL params and the launch argv
and prints them without shelling out to `ros2 launch`:

```bash
openral deploy sim --config scenes/deploy/so100_pick_cube.yaml --dry-run
```

`openral deploy validate` is the readiness check for **real** hardware — no
ROS launch, no robot required. It catches exactly the gaps that otherwise fail
late, at HAL configure or on the first sensor read:

```bash
openral deploy validate --config scenes/deploy/so100_pick_cube.yaml
```

It checks three things:

- **HAL transport** — a serial `port` is declared, and that device exists now.
- **Calibration** — a serial HAL with `calibrate_on_connect: false` has an `id`
  and `calibration_dir`, and `<calibration_dir>/<id>.json` is actually there.
  Missing, and every `send_action` fails with "has no calibration registered".
- **Camera bindings** — each deploy sensor (the robot manifest's cameras, bound
  in the host's unit overlay or `robot.yaml`, plus the scene's workcell cameras) has a `deploy_binding`
  (without one it is never published, and a camera VLA silently gets an empty
  observation), and any `/dev/*` path exists now.

It separates **ERROR** (committed data is missing — exits non-zero) from
**WARN** (the device just is not plugged in right now), and resolves HAL
params with the same precedence `deploy run` uses: `--hal` > scene `hal` >
`robot.yaml`.

### RoboCasa scenes — let the HAL provision the backend

RoboCasa kitchen scenes (e.g. `scenes/deploy/robocasa_navigate.yaml`) need the
RoboCasa fork, which is **not** installed by `just sync --group robocasa` — that
group only supplies robosuite + supporting deps. The fork is git-cloned and
installed editable **at runtime** by the deploy-sim HAL's `on_configure` via
`openral_sim._deps.ensure_backend_deps('robocasa_kitchen')`. Auto-install is on
by default; run it like so:

```bash
just sync --group robocasa    # robosuite + deps (swaps out the libero/sim group)
OPENRAL_AUTO_INSTALL_DEPS=1 openral deploy sim \
  --config scenes/deploy/robocasa_navigate.yaml
```

Do **not** hand-install `robocasa` / `robosuite` — that pulls the wrong
robosuite and wrecks the managed env. To avoid the first-run build stalling the
lifecycle transition, pre-build the clone once beforehand:

```bash
OPENRAL_AUTO_INSTALL_DEPS=1 python -c \
  "from openral_sim._deps import ensure_backend_deps; ensure_backend_deps('robocasa_kitchen')"
```

LIBERO and RoboCasa pin conflicting robosuite versions and cannot coexist, so
swap groups per task: `just sync --group robocasa` for kitchens, `just sync
--group sim` (or `--group libero`) to go back. Full details in
[Managing the Python environment & dependency
groups](../../contributing/toolchain.md#managing-the-python-environment-dependency-groups).

## 3. Run on hardware

With the robot powered, connected, and within a clear workspace:

```bash
openral deploy run --config scenes/deploy/so100_pick_cube.yaml
```

What happens:

- The robot is resolved from `--config`; `build_hal(mode="real")` constructs
  the real HAL. If no hardware is attached, `connect()` **fails loudly**; a
  simulation-only robot raises `ROSCapabilityMismatch` (use `deploy sim`).
- Robot HAL defaults (`port` / `robot_ip` / `fci_ip` / adapter params) come from
  `robots/<robot_id>/robot.yaml`. Override at the CLI with repeatable `--hal key=value`:

  ```bash
  openral deploy run --config scenes/deploy/so100_pick_cube.yaml --hal port=/dev/ttyUSB1
  ```

- The C++ safety kernel sits between the policy and the motors: Python
  proposes, C++ disposes, and `ROSSafetyViolation` is never silently caught.
  Keep your E-stop within reach.

### SO-101 SmolVLA TensorRT fast path (OpenRAL Pro)

For the public SO-101 SmolVLA skill, the real deploy scene and rSkill are:

```bash
openral rskill install OpenRAL/rskill-smolvla-so101-eraser_place-bf16
openral deploy run \
  --config scenes/deploy/so101_bench.yaml
```

The `OPENRAL_SMOLVLA_TRT=1` split ONNX/TensorRT fast path (and the GStreamer
NVMM zero-copy camera leg it pairs with) is an **OpenRAL Pro plugin** — it ships in the
private `openral-pro-trt` package, not this repo. With `openral-pro-trt`
installed, `OPENRAL_SMOLVLA_TRT=1` before `openral deploy run` attaches the
same way it always did (the env var is read by the pro-side hook, looked up
by name via `openral_rskill.backend_registry.maybe_attach_pro_hooks`); see
`openral-pro`'s own docs for the engine pre-build recipe.

Without `openral-pro-trt` installed, the policy runs in eager PyTorch —
logged, not a silent skip. For the SO-101 NVMM camera path, either install
the OpenRAL Pro plugin or disable NVMM for the relevant cameras in the
deploy scene.

### Give the robot a task

The reasoner selects *which* rSkill to run, but it needs a goal. Without one it
comes up idle and waits. Three ways to give it one:

**At launch**, with `--initial-task`:

```bash
openral deploy run --config scenes/deploy/so101_bench.yaml \
  --initial-task "pick the bowl and place it on the plate, then push the mug back"
```

A multi-step goal is decomposed into an ordered `MissionState` queue via the
reasoner's `decompose_mission` tool, and the queue only advances when the
active task passes the reward gate.

**While it runs**, from another terminal:

```bash
openral prompt "put the pen back in the cup"
```

This publishes one `PromptStamped` on `/openral/prompt_in/cli`; the
prompt-router fans it out to `/openral/prompt` for the reasoner. It needs a
sourced ROS 2 install.

By default a prompt arriving mid-mission is treated as **conversational
context** — an answer to a reasoner question, or a hint — and does *not*
rebuild the task queue. To replace the current mission outright:

```bash
openral prompt --new-goal "stop that, clear the table instead"
```

On a cold-booted graph the router may not have discovered the publisher yet;
`--discovery-wait-s 15` covers the stale-shared-memory case the 5 s default
does not.

**From the dashboard**, using the prompt box on the live pane, which posts to
the same path as the CLI.

### Optional reward monitor

`deploy run` can bring up the same reward/progress monitor as `deploy sim`:

```bash
openral deploy run \
  --config scenes/deploy/so101_bench.yaml \
  --enable-reward-monitor
```

The monitor is advisory only; it serves `/openral/perception/query_task_progress`
for the reasoner and never gates motors.

## 4. Open the dashboard

`deploy run` spawns the live dashboard by default (`--dashboard/--no-dashboard`,
port `--dashboard-port`, default **4318**). It's a read-only pane over the OTel
stream — the most recent `rskill.execute`, `skill.chunk_inference`, and
`safety.check` spans, rolling metric histograms, per-camera thumbnails, and an
event log. Operator discovery / write endpoints still exist for explicit tooling
flows, but they are kept off the main dashboard surface.

```
http://localhost:4318
```

The connection indicator in the top right turns green within a few hundred
milliseconds. To run without it (e.g. headless CI), pass `--no-dashboard`; to
move the port, `--dashboard-port 4400`.

You can also launch the dashboard standalone and point other workloads at it:

```bash
openral dashboard            # binds 127.0.0.1:4318
```

then export `OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318` and
`OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf` for the workload (see the
[dashboard quickstart](../../quickstart/dashboard.md) for the in-process demo
mode).

## See also

- [`scenes/README.md`](https://github.com/OpenRAL/openral/blob/master/scenes/README.md) — DeployScene / SimScene / BenchmarkScene tiers.
- [`openral dashboard` quickstart](../../quickstart/dashboard.md).
- [Reasoner (S2) reference](../../reference/reasoner.md) — tool palette, bounded
  replanning, missions and memory.
- [Your first sim rollout](../sim/first-rollout.md) — the no-hardware path.
- `openral detect` — auto-generate `robot.yaml` by probing USB devices and sensors.
