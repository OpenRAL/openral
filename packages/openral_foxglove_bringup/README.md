# openral_foxglove_bringup

A read-only live [Foxglove](https://foxglove.dev/) visualisation surface for
OpenRAL's live ROS scene — camera images, the `/map` occupancy grid, the octomap
point cloud (voxels), joint states, TF, the robot model (**Bucket-1**, native),
plus the custom OpenRAL world types re-published as standard markers/clouds
(**Bucket-2**, via a converter node), plus the ROS-side telemetry plane —
world state, diagnostics, node logs, episode/mission transitions, reward
scores, detected objects.

This is the live-scene half of a **hybrid** — Foxglove owns the live 3D/2D
scene; the `openral dashboard` OTel receiver keeps traces, metrics,
system health, and the reasoner/safety cards. Foxglove is a visualization tool,
not an observability backend, so the OTel plane does **not** port here.

**Nothing has been removed from the dashboard.** The telemetry mirrored into
Foxglove is read from ROS topics; the dashboard keeps rendering its own
OTel-derived copy. Where the two disagree, the dashboard is the one wired to
the latched `/openral/safety_status` subscriber and to the trace index.

The surface is **read-only and cannot actuate the robot**. Any path that
re-enables a write capability (E-stop reset, Publish/Teleop, prompt input) is out
of scope and requires safety-WG sign-off (CLAUDE.md §3).

## What it does

Wraps upstream `foxglove_bridge` with an OpenRAL-specific, safety-conscious
default:

| Choice | This package | Upstream default | Why |
|---|---|---|---|
| Bind address | `127.0.0.1` (loopback) | `0.0.0.0` | Matches dashboard posture (issue #44); no auth on the bridge |
| Capabilities | `[connectionGraph, assets]` | adds `clientPublish, services, parameters…` | **Read-only** — a viewer cannot publish topics or call services (no remote actuation / E-stop poke) |
| Topics | explicit Bucket-1 allowlist | `['.*']` (everything) | Safety/e-stop/action topics are never exposed |

The allowlist lives in `openral_foxglove_bringup/topics.py` as four named
groups, so it is reviewable a group at a time:

| Group | What it carries |
|---|---|
| `SCENE_TOPICS` | Camera images (+ `/compressed` siblings, `camera_info`), `/map`, octomap cloud, `/scan`, `/odom`, `/joint_states`, `/robot_description`, TF |
| `DEPTH_TOPICS` | Per-camera depth + `points`, the DA3 metric-depth sidecar, nvblox filtered depth + ESDF slice, `/openral/imu`, cuVSLAM odometry |
| `BUCKET2_TOPICS` | The converter's `MarkerArray` + `PointCloud2` outputs |
| `TELEMETRY_TOPICS` | `world_state_fast`/`_slow`, `policy_state`, `episode`, `critic/score`, `reward/active_task`, `perception/objects`, `attachment_state`(`_applied`), `skill_registry_changed`, `/diagnostics`, `/rosout` |

Every entry is an observation topic a node publishes *about itself*. The
command plane (`/openral/estop`, `estop_reset`, `safe_action`,
`candidate_action`, `execute_rskill`, `prompt`, …) is on none of them, and
`test/test_foxglove_launch.py` proves each one stays unreachable.

`/openral/safety_status` is **deliberately withheld** even though it is
read-only status: it sits in the safety plane this package promises never to
advertise, so exposing it needs safety-WG sign-off (CLAUDE.md §3). Read the
latch state on the dashboard's *Safety · current state* card.

Vendor driver namespaces are **not** on any list either. `zed_wrapper`'s
`/zed/zed_node/point_cloud/cloud_registered` is the raw stereo cloud the
OpenArm cell feeds to `octomap_server`; what reaches Foxglove is the world
model built from it — `/octomap_point_cloud_centers` and
`/openral/world_voxels_cloud` — not the driver's own output. That is the
allowlist working as designed (the bridge speaks OpenRAL's topic contract, not
a per-vendor one), but it does mean "I can see no cloud from my camera" is the
expected result and not a fault. Add the vendor topic to `SCENE_TOPICS` only
if you want the dense cloud on the wire, and mind the bandwidth.

> **Bag size:** `record.launch.py` records exactly what the bridge exposes
> (one source of truth), so a scene publishing depth images and point clouds
> now produces a substantially larger MCAP than before the depth group
> existed. Narrow the `--regex` by hand for a long capture.

## Install the bridge (one-time)

```bash
sudo apt install -y ros-jazzy-foxglove-bridge
```

## Run inside deploy-sim (recommended)

`openral deploy sim` can spawn the read-only bridge as part of the runtime
graph, ordered **after** the topic producers to dodge the
stale-bridge gotcha (see `VERIFICATION.md`):

```bash
openral deploy sim --config scenes/deploy/<scene>.yaml --foxglove
# custom port:
openral deploy sim --config scenes/deploy/<scene>.yaml --foxglove --foxglove-port 8770
```

Default is `--no-foxglove`. The flag is view-only — it cannot actuate the robot.
When a manifest robot carries an `assets.urdf`, deploy-sim already
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
import the layout — the scene-matched one the deploy logs a path to, or
`config/openral_layout.json` when you brought the graph up by hand (see
[Layout](#layout)).

## Layout

```
┌───────────────────────────┬──────────────┐
│ 3D · robot + environment  │ camera 0     │
│ URDF · TF · map · voxels  │ camera 1     │
│ · collisions · odom       │ camera 2     │
├───────────────────────────┼──────────────┤
│ tabs: nav map · joints ·  │ tabs: log ·  │
│ collisions · policy state │ health · …   │
└───────────────────────────┴──────────────┘
```

One hero 3D panel draws the robot inside its environment; the scene's cameras
stack beside it; everything else is tabbed, one click away.

**Main area**

| Panel | Topics | Shows |
|---|---|---|
| 3D · scene (hero) | `/robot_description`, `/tf`, `/map`, `/octomap_point_cloud_centers`, `/openral/world_voxels_cloud`, `/openral/world_collisions_markers`, `/odom`, `/scan` | The robot in its world — URDF posed by TF, occupancy grid, voxels, collision capsules |
| Image ×N | `/openral/cameras/<slot>/image` | One panel per camera slot in the scene's `cameras:` list |

**Scene tabs**

| Tab | Topics | Shows |
|---|---|---|
| Nav · 2D map | `/map`, `/odom`, `/scan` | Top-down 2D nav view |
| Joints | `/joint_states.position[:]` | Every joint's position trace (`[:]` slices any DOF count) |
| Collisions · voxels | `/openral/world_collisions_markers`, `/openral/world_voxels_cloud` | Bucket-2 geometry close-up |
| Policy state | `/openral/policy_state`, `world_state_fast.staleness_ms[:]`, `.battery_pct` | Step-locked policy vector; staleness/battery off by default |

**Telemetry tabs**

| Tab | Topic | Mirrors (dashboard card) |
|---|---|---|
| Log | `/rosout` | Event log (the ROS lane; the dashboard's is OTel-fed) |
| Health | `/diagnostics` | System health |
| World state | `/openral/world_state_fast` | World state — one `WorldStateStamped` carries joint state, base pose/twist, EE poses, per-component diagnostics + staleness, battery, detected objects |
| Mission | `/openral/episode` (`phase`, `success`, `task_string`) | Reasoner · mission, as a State Transitions timeline |
| Reward | `/openral/critic/score` (`score`, `threshold`) | rSkill reward bars |
| Objects | `/openral/perception/objects` | Spatial memory · scene objects |
| Attachments | `/openral/attachment_state` | — (grasped-object set) |
| Topics | — | Connection graph (via `connectionGraph`) |

> **Note:** Foxglove's *Map* panel is geographic (GPS/`NavSatFix`), **not** for
> occupancy grids. `nav_msgs/OccupancyGrid` renders in the **3D panel** as a
> ground layer — hence the top-down 3D panel instead of a Map panel.

### Regenerate the layout for your scene

A Foxglove layout is imported client-side, so a launch argument cannot reach
it — the camera panels in a shipped layout are baked in. Scenes disagree about
camera names (`behavior_r1pro` → `head` / `left_wrist` / `right_wrist`;
`isaac_franka` → `top`), so generate one for yours:

```bash
python -m openral_foxglove_bringup.layout \
  --cameras head left_wrist right_wrist -o /tmp/openral_layout.json
# point the Image panels at the /compressed siblings instead:
python -m openral_foxglove_bringup.layout --cameras head left_wrist --compressed
# keep the hero view pinned to the world origin rather than the robot:
python -m openral_foxglove_bringup.layout --follow-frame map
# ...or at the robot's own root, when it does not use the base_link default:
python -m openral_foxglove_bringup.layout --follow-frame openarm_base
```

`--follow-frame` defaults to the ROS-conventional `base_link`, and a robot
that names its root otherwise (OpenArm broadcasts `openarm_base`) needs it
passed. Getting it wrong is not a partial failure: Foxglove renders **nothing**
in a 3D panel whose follow frame is absent from TF — no robot model, no point
clouds, no collision markers — while every topic underneath keeps publishing.
The generated layout takes this from `RobotDescription.base_frame`, so it is
only the hand-run generator above that needs the flag.

The shipped `config/openral_layout.json` is this generator's output for
`DEFAULT_CAMERAS` (`top` / `wrist_left` / `wrist_right`); regenerate it in place
with `--write-default` rather than hand-editing, and a test asserts the two
match.

**Prefer the layout the deploy generates.** Camera slots are sensor names from
the robot manifest ∪ the deploy scene, and only a sensor carrying a
`deploy_binding` gets a reader — so a panel aimed at a declared-but-unbound slot
reads "Image topic does not exist", which looks exactly like a dead camera.
`openral deploy sim/run --foxglove` writes a layout for the slots *that deploy
actually publishes* and logs the path:

```
[deploy_e2e] foxglove: ws://127.0.0.1:8765 — import the scene-matched layout
  from /tmp/openral_layout_<robot>.json (cameras: top, wrist_left, wrist_right)
```

Import that one. `DEFAULT_CAMERAS` cannot know your scene; it is only the
fallback for a viewer opened against a graph nobody launched from the CLI.

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
URDF. Meshes render only when the URDF's `package://` paths resolve
to an ament package on the ROS path. See `VERIFICATION.md`.

## Compress camera images

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

## Bucket-2 markers

The custom OpenRAL world types don't render richly in Foxglove on their own. A
small read-only converter node re-publishes them as **standard** viz types so
Foxglove draws them natively — no TypeScript extension:

| In (`openral_msgs`) | Out (standard) | Topic |
|---|---|---|
| `WorldCollision` (capsules) | `visualization_msgs/MarkerArray` (cylinders) | `/openral/world_collisions_markers` |
| `OccupancyVoxels` | `sensor_msgs/PointCloud2` (voxel centres) | `/openral/world_voxels_cloud` |

`openral deploy sim/run --foxglove` spawns this converter as part of the graph,
so these two topics are live on any deploy that has Foxglove on. Standalone
(pairing with `foxglove.launch.py`, or against a graph you brought up yourself):

```bash
ros2 launch openral_foxglove_bringup bucket2.launch.py
```

Capsules are approximated as cylinders (the hemispherical end-caps aren't a
single standard Marker type); a sphere obstacle renders as a zero-length
cylinder. The conversion math lives in pure, unit-tested functions.

## Record an MCAP

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

The dashboard stays the surface for everything below; none of it is being
retired.

| Dashboard capability | Why it does not port |
|---|---|
| Trace chip, `/api/traces`, `/api/spans/{trace_id}`, Jaeger deep link | Foxglove has no span/waterfall panel, and its data model is one message per timestamp — it cannot hold nested spans with durations |
| OTLP metric histograms (p50/p95, threshold bands, freeze/zoom) | Same: the histogram + threshold semantics are OTel-side |
| `openral sim run` / `benchmark run` telemetry | Those run HAL-only with no ROS graph at all; there is nothing for `foxglove_bridge` to bridge |
| E-stop, prompt input (text + voice), skill execute, param set | Write paths. The bridge advertises no `clientPublish`/`services`, and the dashboard's e-stop uses a launch-time, pre-matched publisher precisely because a freshly-created one raced DDS discovery and dropped the message (see `dashboard/estop_publisher.py`) |
| Safety · current state (latched kernel state) | Read-only, but `/openral/safety_status` is withheld pending safety-WG sign-off (above) |

Keep `openral dashboard` (and Jaeger/OTLP) running alongside the bridge; the
two are complementary, not alternatives.
