# openral_foxglove_bringup (PROTOTYPE)

A read-only live [Foxglove](https://foxglove.dev/) visualisation surface for
OpenRAL's **Bucket-1** topics — the data Foxglove renders natively with **zero
custom extensions**: camera images, the `/map` occupancy grid, the octomap
point cloud (voxels), joint states, TF, and the robot model.

This is the proof-of-concept from
[`docs/investigations/foxglove-dashboard-port-feasibility.md`](../../docs/investigations/foxglove-dashboard-port-feasibility.md).
It is a **spike**, not a supported component — graduating it (and anything that
re-enables actuation, e.g. an E-stop button) needs the ADR named in that report
plus safety-WG review (CLAUDE.md §3).

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

## Run

```bash
# 1. Start whatever publishes the topics (e.g. a deploy-sim session, or a
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
| Plot | `/joint_states` | `sensor_msgs/JointState` | Joint position traces |
| Raw Messages | `/joint_states` | — | Live message inspection |
| Topic Graph | — | — | Connection graph (via `connectionGraph`) |

> **Note:** Foxglove's *Map* panel is geographic (GPS/`NavSatFix`), **not** for
> occupancy grids. `nav_msgs/OccupancyGrid` renders in the **3D panel** as a
> ground layer — hence the top-down 3D panel instead of a Map panel.

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
