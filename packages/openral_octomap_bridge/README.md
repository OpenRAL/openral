# openral_octomap_bridge

ADR-0030 — lowers a **3-D OctoMap** into the safety kernel's dense, base-frame
`openral_msgs/OccupancyVoxels` grid for the kernel's allocation-free
capsule-vs-voxel world-collision check.

## Why a bridge (and not the kernel)

The C++ safety kernel must stay small, auditable, and allocation-free on its
hot path, so it does **not** parse a raw OctoMap octree (octree deserialization
allocates, querying isn't time-bounded, and `octomap` is a heavy dependency).
Instead — "perception proposes, the kernel disposes" — this Layer-2 node does
the octree work off the real-time path and publishes a bounded dense grid the
kernel rasterizes capsules against.

```
octomap_msgs/Octomap (map frame)
   │  octomap_msgs::msgToMap → octomap::OcTree
   │  tf2 lookup: octomap_frame ← base_frame
   │  crop a bounded local box around the robot, query the octree per voxel
   ▼
openral_msgs/OccupancyVoxels (base frame, /openral/world_voxels)
   ▼
C++ safety kernel  ──  check_voxel_collision (allocation-free)
```

## Run

### Integrated (recommended) — via `openral deploy sim`

`openral deploy sim --enable-octomap` brings up the whole world-collision leg in
one graph: octomap_server (from the HAL's depth `PointCloud2`), this bridge,
and the kernel's capsule-vs-voxel check (`world_voxel_enabled:=true`). It
**auto-enables** when the robot manifest declares a depth `SensorSpec`:

```bash
openral deploy sim --config scenes/sim/robocasa_panda_mobile_kitchen.yaml   # panda_mobile → auto-on
# or force it / point at a different depth topic:
openral deploy sim --config <cfg> --enable-octomap
```

Requires `ros-${ROS_DISTRO}-octomap-server` apt-installed and this package
colcon-built (both are in the deploy Docker images).

### Standalone

```bash
ros2 launch openral_octomap_bridge octomap_voxel_bridge.launch.py \
    base_frame:=base_link octomap_topic:=/octomap_binary
```

and launch the kernel with `world_voxel_enabled:=true` (plus
`world_voxel_max_cells` ≥ the grid's `size_x*size_y*size_z`).

Requires TF from `base_frame` into the OctoMap's `header.frame_id` (usually
`map`) — published by your SLAM / localization stack.

## Parameters

| Param | Default | Meaning |
|---|---|---|
| `base_frame` | `base_link` | Output frame; the kernel expects obstacles here. |
| `octomap_topic` | `/octomap_binary` | Input `octomap_msgs/Octomap`. |
| `output_topic` | `/openral/world_voxels` | Output `OccupancyVoxels`. |
| `resolution` | `0.05` | Output voxel edge length (m). |
| `box_size_{x,y,z}` | `2.0` | Local volume extent around the robot (m). |
| `box_center_{x,y,z}` | `0,0,0.5` | Local volume centre in `base_frame`. |
| `publish_rate_hz` | `10.0` | Republish rate (the grid follows the robot via TF). |

`size_{x,y,z} = ceil(box_size / resolution)`. Keep
`size_x*size_y*size_z ≤ world_voxel_max_cells` (kernel default 262144), or the
kernel fails closed.

## Producing the upstream OctoMap

The bridge consumes an `octomap_msgs/Octomap`; the canonical producer is
`octomap_server` (`ros-${ROS_DISTRO}-octomap-server`, bundled in both the dev and
inference images alongside `octomap` / `octomap-msgs` / `tf2-geometry-msgs`). It
builds the octree from a **3-D depth point cloud** (`sensor_msgs/PointCloud2`):

```bash
ros2 run octomap_server octomap_server_node \
    --ros-args -p resolution:=0.05 -p frame_id:=map -r cloud_in:=/camera/points
```

## Testing

`test_octree_to_grid` unit-tests the rasterization core (`rasterize_octree_to_grid`)
against a real `octomap::OcTree` — no ROS graph needed. A full
octomap → bridge → kernel chain needs a live OctoMap producer and is a
HIL / on-robot test (the `octomap` Python bindings aren't in CI, so a synthetic
OctoMap publisher would itself be C++).

**Sim status.** The sim target is `scenes/sim/robocasa_panda_mobile_kitchen.yaml`
— a mobile manipulator in a cluttered RoboCasa kitchen, the only scene with real
3-D obstacles and an obstacle-avoidance task. The deploy-sim HAL now publishes a
**depth `PointCloud2`** for it: the panda_mobile node ray-casts each depth
`SensorSpec` (`robots/panda_mobile/robot.yaml` → `front_depth`) from MuJoCo via
`openral_sim.backends.depth_camera.synthesize_depth_pointcloud` and publishes
`/openral/cameras/front_depth/points` (camera optical frame) + a live
`base_link → front_depth_optical_frame` TF. This is robot-agnostic — declare a
depth `SensorSpec` (with `metadata.mjcf_camera`) on any robot and its HAL node
publishes the same — so the full chain is:

```
panda_mobile HAL  ──/openral/cameras/front_depth/points──▶  octomap_server
   (depth ray-cast)                                              │ /octomap_binary
                                                                 ▼
   openral_octomap_bridge  ──/openral/world_voxels──▶  C++ safety kernel
```

Run `octomap_server` against the published cloud (see *Producing the upstream
OctoMap* above) and this bridge against its output. Booting the full RoboCasa
kitchen + octomap_server is a HIL / on-host integration step (RoboCasa assets +
GPU render); the synth, packing, and TF are unit-tested
(`tests/unit/test_depth_camera_synth.py`, `tests/unit/test_depth_cloud_helpers.py`)
and the kernel-side voxel check by
`tests/sim/safety/test_kernel_voxel_collision_synthetic.py` (feeds
`OccupancyVoxels` directly).
