# openral_nav2_bringup

ADR-0025 — bringup wrapper for upstream **Nav2** as a Reasoner-managed
background service (the second such service, after
[`openral_slam_bringup`](../openral_slam_bringup/)). We do **not** reimplement
Nav2 — this package only ships the per-deployment launch + parameter glue so the
OpenRAL Reasoner can start/stop navigation on demand and route a
`NavigateToPose` goal.

```
lidar profile:  /scan ─────▶ Nav2 (obstacle_layer) ──/navigate_to_pose──▶ /cmd_vel
visual profile: /map ──────▶ Nav2 (static_layer)   ──/navigate_to_pose──▶ /cmd_vel
```

## Costmap profiles (ADR-0064 — backend-agnostic)

Nav2 is selected to match the SLAM backend via the `slam_backend` launch arg, so
navigation works **regardless of how the 2D map is built**:

| `slam_backend` | Config | Costmap obstacle source |
|---|---|---|
| `lidar` (default) | `nav2_panda_mobile.yaml` | `/scan` via `obstacle_layer`/`voxel_layer` |
| `visual` | `nav2_visual.yaml` | **`/map`** `OccupancyGrid` via `static_layer` |

The **visual** profile lets a lidar-less robot (cuVSLAM + nvblox, ADR-0064)
navigate: the global+local costmaps consume the backend-agnostic `/map` (which
nvblox publishes, remapped from its `static_occupancy_grid`) via `static_layer`
with `map_subscribe_transient_local: False` (nvblox's `/map` is RELIABLE+VOLATILE,
not latched), and the collision_monitor's `/scan` source is disabled. Everything
else mirrors the lidar base — `nav2_visual.yaml` is **generated** from it by
`tools/gen_nav2_visual.py` (re-run after editing the base). Verified live: the
visual profile activated and `ComputePathToPose` returned a path consuming only
`/map` (no `/scan`).

> 3D-lifted detected objects (ADR-0035/0052) are already backend-agnostic — they
> use the `map` **TF frame** (which cuVSLAM publishes like slam_toolbox), not the
> `/map` topic, so they map into the world identically on both backends.

## Run

Normally started by the Reasoner as a background service when the active goal
needs navigation (ADR-0025); the `openral deploy sim` / `deploy run` graph wires
the leg when the robot declares a lidar (`has_lidar`) **or** vision SLAM
(`has_vision_slam`, ADR-0064) and forwards the resolved `slam_backend`.
Standalone:

```bash
ros2 launch openral_nav2_bringup nav2.launch.py slam_backend:=lidar    # /scan
ros2 launch openral_nav2_bringup nav2.launch.py slam_backend:=visual   # /map
```

Requires upstream `ros-${ROS_DISTRO}-navigation2` / `nav2_bringup` (in the deploy
images) and a map source — `openral_slam_bringup` (slam_toolbox `/map` for lidar,
or cuVSLAM+nvblox `/map` for visual).
