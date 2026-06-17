# openral_foxglove_bringup

A read-only live [Foxglove](https://foxglove.dev/) visualisation surface for
OpenRAL's live ROS scene — camera images, the `/map` occupancy grid, the octomap
point cloud (voxels), joint states, TF, the robot model (**Bucket-1**, native),
plus the custom OpenRAL world types re-published as standard markers/clouds
(**Bucket-2**, via a converter node).

Governed by [**ADR-0059**](../../docs/adr/0059-foxglove-live-scene-visualization.md):
this is the live-scene half of a **hybrid** — Foxglove owns the live 3D/2D
scene; the `openral dashboard` OTel receiver (ADR-0017) keeps traces, metrics,
system health, and the reasoner/safety cards. Foxglove is a visualization tool,
not an observability backend, so the OTel plane does **not** port here (see the
[feasibility study](../../docs/investigations/foxglove-dashboard-port-feasibility.md)).

The surface is **read-only and cannot actuate the robot**. Any path that
re-enables a write capability (E-stop reset, Publish/Teleop, prompt input) is out
of scope and requires safety-WG sign-off (CLAUDE.md §3, ADR-0059 §Safety).

## What it does

Wraps upstream `foxglove_bridge` with an OpenRAL-specific, safety-conscious
default:

| Choice | This package | Upstream default | Why |
|---|---|---|---|
| Bind address | `127.0.0.1` (loopback) | `0.0.0.0` | Matches dashboard posture (issue #44); no auth on the bridge |
| Capabilities | `[connectionGraph, assets]` | adds `clientPublish, services, parameters…` | **Read-only** — a viewer cannot publish topics or call services (no remote actuation / E-stop poke) |
| Topics | explicit Bucket-1 allowlist | `['.*']` (everything) | Safety/e-stop/action topics are never exposed |

## Install the bridge (one-time)

```bash
sudo apt install -y ros-jazzy-foxglove-bridge
```

## Run inside deploy-sim (recommended)

`openral deploy sim` can spawn the read-only bridge as part of the runtime
graph (ADR-0059 Phase 1), ordered **after** the topic producers to dodge the
stale-bridge gotcha (see `VERIFICATION.md`):

```bash
openral deploy sim --config scenes/deploy/<scene>.yaml --foxglove
# custom port:
openral deploy sim --config scenes/deploy/<scene>.yaml --foxglove --foxglove-port 8770
```

Default is `--no-foxglove`. The flag is view-only — it cannot actuate the robot.
When a manifest robot carries an `assets.urdf` (ADR-0058), deploy-sim already
runs a `robot_state_publisher`, so `/tf` + `/robot_description` are on the bus
and the 3D panel draws the robot with no extra wiring.

## Run stand-alone

```bash
# 1. Start whatever publishes the topics (a deploy-sim session, or a
#    robot_state_publisher for just the robot model + joints + TF).
# 2. Launch the bridge:
ros2 launch openral_foxglove_bringup foxglove.launch.py
#    against a sim that publishes /clock:
ros2 launch openral_foxglove_bringup foxglove.launch.py use_sim_time:=true
```

Then open **https://app.foxglove.dev** (or the desktop app) →
**Open connection** → **Foxglove WebSocket** → `ws://localhost:8765` →
import the layout from `config/openral_layout.json`.

## Layout panels

| Panel | Topic | Message | Shows |
|---|---|---|---|
| Image | `/openral/cameras/0/image` | `sensor_msgs/Image` | Camera feed |
| 3D (top-down) | `/map`, `/odom`, `/scan` | `OccupancyGrid` + `Odometry` + `LaserScan` | 2D nav map |
| 3D (perspective) | `/octomap_point_cloud_centers`, `/map` | `PointCloud2` | Voxels / world cloud in real 3D |
| 3D (Bucket-2) | `/openral/world_collisions_markers`, `/openral/world_voxels_cloud` | `MarkerArray` + `PointCloud2` | Collision capsules + voxel grid (converter node) |
| Plot | `/joint_states` | `sensor_msgs/JointState` | Joint position traces |
| Raw Messages | `/joint_states` | — | Live message inspection |
| Topic Graph | — | — | Connection graph (via `connectionGraph`) |

> **Note:** Foxglove's *Map* panel is geographic (GPS/`NavSatFix`), **not** for
> occupancy grids. `nav_msgs/OccupancyGrid` renders in the **3D panel** as a
> ground layer — hence the top-down 3D panel instead of a Map panel.

## Render `/tf` + the robot model

deploy-sim publishes `/joint_states` but not dynamic `/tf`, so the 3D panel
can't draw the robot. Opt in to a `robot_state_publisher` (turns
`/joint_states` + URDF → `/tf` + `/robot_description`):

```bash
ros2 launch openral_foxglove_bringup foxglove.launch.py \
  with_robot_state_publisher:=true \
  robot_description_urdf:=/path/to/robot.urdf
# Standalone (no sim) — also synthesise /joint_states (zeros):
ros2 launch openral_foxglove_bringup foxglove.launch.py \
  with_robot_state_publisher:=true with_joint_state_publisher:=true \
  robot_description_urdf:=/path/to/robot.urdf
```

Under a real deploy-sim, set **only** `with_robot_state_publisher:=true` — the
sim is the real `/joint_states` source; a second publisher would fight it.
Resolve a manifest robot's URDF via `robot_descriptions` (e.g.
`panda_description` for `franka_panda` / `panda_mobile`). `openarm` has no local
URDF (ADR-0027). Meshes render only when the URDF's `package://` paths resolve
to an ament package on the ROS path. See `VERIFICATION.md`.

## Compress camera images (ADR-0059 Phase 2)

Raw `sensor_msgs/Image` is ~9 MB/s per camera and saturates a laptop link and
Foxglove's send buffer. Opt in to `image_transport` republishers that emit
`sensor_msgs/CompressedImage` siblings (~10× smaller; Foxglove renders them
natively in the Image panel). The `/compressed` topics are on the Bucket-1
allowlist:

```bash
ros2 launch openral_foxglove_bringup foxglove.launch.py \
  republish_compressed:=true \
  compressed_camera_topics:="/openral/cameras/base/image /openral/cameras/left_wrist/image"
```

Default is off, so the raw path stays available for fidelity-sensitive use.

## Bucket-2 markers (ADR-0059 Phase 3)

The custom OpenRAL world types don't render richly in Foxglove on their own. A
small read-only converter node re-publishes them as **standard** viz types so
Foxglove draws them natively — no TypeScript extension:

| In (`openral_msgs`) | Out (standard) | Topic |
|---|---|---|
| `WorldCollision` (capsules) | `visualization_msgs/MarkerArray` (cylinders) | `/openral/world_collisions_markers` |
| `OccupancyVoxels` | `sensor_msgs/PointCloud2` (voxel centres) | `/openral/world_voxels_cloud` |

```bash
ros2 launch openral_foxglove_bringup bucket2.launch.py
```

Capsules are approximated as cylinders (the hemispherical end-caps aren't a
single standard Marker type); a sphere obstacle renders as a zero-length
cylinder. The conversion math lives in pure, unit-tested functions.

## Record an MCAP (ADR-0059 Phase 4)

Record the Bucket-1 topics to Foxglove's native MCAP format for offline replay,
scoped by the same allowlist (the safety/e-stop/action topics are **not**
recorded):

```bash
ros2 launch openral_foxglove_bringup record.launch.py output_dir:=my_session
# under a sim that publishes /clock:
ros2 launch openral_foxglove_bringup record.launch.py use_sim_time:=true
```

## Escape hatch (debug only)

```bash
ros2 launch openral_foxglove_bringup foxglove.launch.py expose_all_topics:=true
```

Drops the Bucket-1 allowlist and exposes every topic **read-only**. It does
**not** re-enable `clientPublish`/`services`. Never use on a shared or
robot-connected run.

## Not covered (by design)

Traces, OTLP metrics, system-health gauges, the reasoner/safety cards — these
live on the OpenTelemetry plane, which Foxglove cannot ingest. Keep
Jaeger/OTLP for those (see the feasibility report).
