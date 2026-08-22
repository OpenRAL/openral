# openral_nav2_bringup

Bringup wrapper for upstream **Nav2** as a Reasoner-managed
background service (the second such service, after
[`openral_slam_bringup`](../openral_slam_bringup/)). We do **not** reimplement
Nav2 — this package only ships the per-deployment launch + parameter glue so the
OpenRAL Reasoner can start/stop navigation on demand and route a
`NavigateToPose` goal.

```
lidar profile:  /scan ─▶ payload scan filter ─▶ /openral/nav2/scan ─▶ Nav2 (obstacle_layer)
visual profile: /map ───────────────────────────────────────────────▶ Nav2 (static_layer)
                                                          └──/navigate_to_pose──▶ /cmd_vel
                /openral/world_state_fast ─▶ payload footprint ─▶ */footprint
```

## Costmap profiles (backend-agnostic)

Nav2 is selected to match the SLAM backend via the `slam_backend` launch arg, so
navigation works **regardless of how the 2D map is built**:

| `slam_backend` | Config | Costmap obstacle source |
|---|---|---|
| `lidar` (default) | `nav2_panda_mobile.yaml` | `/scan` via `obstacle_layer`/`voxel_layer` |
| `visual` | `nav2_visual.yaml` | **`/map`** `OccupancyGrid` via `static_layer` |

The **visual** profile lets a lidar-less robot (cuVSLAM + nvblox)
navigate: the global+local costmaps consume the backend-agnostic `/map` (which
nvblox publishes, remapped from its `static_occupancy_grid`) via `static_layer`
with `map_subscribe_transient_local: False` (nvblox's `/map` is RELIABLE+VOLATILE,
not latched), and the collision_monitor's `/scan` source is disabled. Everything
else mirrors the lidar base — `nav2_visual.yaml` is **generated** from it by
`tools/gen_nav2_visual.py` (re-run after editing the base). Verified live: the
visual profile activated and `ComputePathToPose` returned a path consuming only
`/map` (no `/scan`).

> 3D-lifted detected objects are already backend-agnostic — they
> use the `map` **TF frame** (which cuVSLAM publishes like slam_toolbox), not the
> `/map` topic, so they map into the world identically on both backends.

## Run

Normally started by the Reasoner as a background service when the active goal
needs navigation; the `openral deploy sim` / `deploy run` graph wires
the leg when the robot declares a lidar (`has_lidar`) **or** vision SLAM
(`has_vision_slam`) and forwards the resolved `slam_backend`.
Standalone:

```bash
ros2 launch openral_nav2_bringup nav2.launch.py slam_backend:=lidar    # /scan
ros2 launch openral_nav2_bringup nav2.launch.py slam_backend:=visual   # /map
```

Requires upstream `ros-${ROS_DISTRO}-navigation2` / `nav2_bringup` (in the deploy
images) and a map source — `openral_slam_bringup` (slam_toolbox `/map` for lidar,
or cuVSLAM+nvblox `/map` for visual).

## The Nav2 ↔ safety-kernel boundary

Two independent world models run side by side here, and conflating them is the
mistake this section exists to prevent. Neither one covers for the other.

| | **Nav2** | **C++ safety kernel** |
|---|---|---|
| Checks | the **base footprint** — one 2-D polygon | the **arm's link capsules + attached payload primitives** — 3-D |
| Against | the 2-D costmap | `/openral/world_voxels`, a 3-D occupancy grid |
| Fed by | `/openral/nav2/scan` (lidar) or `/map` (visual) | `openral_octomap_bridge`, from `octomap_server`'s octree |
| Acts by | refusing a path / trajectory, then `/cmd_vel` | vetoing `/openral/candidate_action` → `/openral/safe_action`, or E-stopping |
| Blind to | anything not in the 2-D grid: overhangs, table tops, the arm's own reach | anything outside the local voxel box, and the base's path |

**`/cmd_vel` does not pass through the kernel, and that is a recorded decision**
(ADR-0040: base collision avoidance relies entirely on Nav2's costmap). Nav2
publishes `geometry_msgs/Twist`; `openral_hal.mobile_base_bridge.MobileBaseBridge`
maps each message to a `BODY_TWIST` `Action` and applies it through the HAL
node's `_send_action_traced` — it never reaches `/openral/candidate_action`, so
the kernel never sees it, never bounds it, and never vetoes it. Velocity caps on
this path are Nav2's own `velocity_smoother`. This is stated in the bridge's
module docstring too; it is a boundary, not an oversight, and it is written
down in both places on purpose.

**The one thing that does cross.** `_on_cmd_vel` returns early while the HAL
node is latched E-stopped, so an E-stop — the kernel's included — does stop the
Nav2 path, at the HAL rather than in the kernel. Stopping is the only kernel
authority over base motion. There is no per-command bound.

**What this costs, and why the dynamic footprint below matters.** A carried
object is checked in 3-D by the kernel and is *not* checked at all by Nav2
unless Nav2's footprint says it is there. Without the footprint publisher, the
robot's arm and payload are protected while the chassis drives them into a
counter. So the payload has to appear in **exactly one** place on each side of
the boundary, and the two sides are deliberately opposite:

* **Kernel side** — the payload is *robot*: collision-active attached geometry
  the kernel keeps checking, and `openral_octomap_bridge` clears its cells out
  of the voxel grid so the robot is not stopped by itself.
* **Nav2 side** — the payload is *robot* here too: it joins the footprint
  polygon, and `payload_scan_filter_node` drops its returns from the scan the
  costmaps and the collision monitor read, for the same reason.

Both sides read the same attachment set on `/openral/world_state_fast`, so they
cannot disagree about what is attached.

## Dynamic footprint while carrying

Two nodes ride with Nav2 (launch arg `payload_footprint`, default on):

| Node | In | Out |
|---|---|---|
| `openral_nav2_payload_footprint` | `/openral/world_state_fast` + `base_frame ← attach_link` TF | `geometry_msgs/Polygon` on `/local_costmap/footprint` + `/global_costmap/footprint` |
| `openral_nav2_payload_scan_filter` | `/scan` + the same attachment set | `sensor_msgs/LaserScan` on `/openral/nav2/scan` |

The published polygon is the convex hull of the manifest's
`footprint_polygon` and the ground projection of every attached collision
primitive, each placed by `TF(base_frame ← attach_link) · pose_in_link ·
pose_in_object` — the kernel's own composition. Boxes project exactly; spheres
and capsules project onto *circumscribed* N-gons, so the polygon contains the
true shape rather than inscribing it.

**Verified against the Jazzy binaries, not from memory.** `nav2_costmap_2d`'s
`Costmap2DROS` subscribes an **unstamped `geometry_msgs/Polygon`** on its own
*relative* `footprint` topic (`/local_costmap/footprint` — not `~/footprint` on
the parent server), calls `setRobotFootprintPolygon` on every message, and
republishes the oriented result as a `PolygonStamped` on `published_footprint`,
padded by `footprint_padding` (default 0.01 m). `nav2_collision_monitor`,
`nav2_behaviors`, `opennav_docking` and `IsPathValid` all read that republished
polygon, so writing the two costmaps reaches every consumer.

Config consequences, both in `config/nav2_panda_mobile.yaml`:

* The costmaps declare a **`footprint` polygon**, not just `robot_radius`. A
  circle cannot express an asymmetric payload. Nav2 reads `""` and `"[]"` as
  "use `robot_radius`", so `RobotDescription.nav2_param_overrides()` emits
  `"[]"` for a robot whose manifest has no `footprint_polygon` rather than
  letting it inherit panda's outline.
* `CostCritic.consider_footprint` is **`true`**. With `false`, MPPI scores the
  centre cell only and the payload-inclusive polygon changes nothing for the
  controller — a forward-reaching payload grows the circumscribed radius, but
  the inflation layer keys its cost cache off the *inscribed* one, which the
  payload does not move. This is upstream's expensive option and its cost has
  **not** been measured under a live scene; see "What is still open" below.

**Failure directions are opposite on purpose.** The footprint publisher refuses
to narrow: a missing attach-link TF, a primitive the kernel would itself reject,
or a stale world state makes it hold the last polygon, because shrinking the
footprint is the only direction that can drive a payload into something. The
scan filter fails the other way — it republishes the scan **unfiltered**, which
leaves the payload in the costmap and makes Nav2 more cautious, and keeps the
topic alive so the costmap never goes blind.

Nav2 also clears its own footprint (`obstacle_layer`'s
`footprint_clearing_enabled`, default `True`, verified live). That is a second
line, not a substitute: it reaches only cells inside the current polygon, and
the collision monitor reads the scan with no costmap in between.

### What is still open (issue #108)

* **No scene drives the base while carrying.** Every RoboCasa *atomic*
  PickPlace task pins `init_robot_base_ref` to one fixture, so nothing here has
  been exercised against a base that actually translates mid-carry. Closing
  #108's last acceptance box needs a `composite/*` task or a custom
  deterministic scene — a maintainer decision, not made in this package.
* **`consider_footprint: true` is unmeasured on the controller loop.** The
  20 Hz MPPI budget with full SE2 footprint costing has no reference-host
  number yet; `trajectory_point_step` is the knob if it misses.
* **The robot's own geometry** is filtered upstream, not here — the sim
  ray-cast skips the robot's kinematic tree and a real lidar masks its own
  structure in the driver. A real robot whose lidar sees its own arm would need
  that half added to `payload_scan_filter_node`.
