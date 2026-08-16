# openral_octomap_bridge

Lowers a **3-D OctoMap** into the safety kernel's dense, base-frame
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

## The frame contract (ADR-0095)

`OccupancyVoxels` carries **no TF of its own**. `header.frame_id` names the
frame `origin` is measured in, and the kernel rasterizes its FK'd link capsules
against the grid *directly* — no transform is applied on the way in. So the
grid's **content** and the kernel's **collision FK root** must be expressed in
one and the same frame, and that frame must be the one `base_frame` denotes on
`/tf`. Three parties have to agree:

| Party | What it must use |
|---|---|
| whatever feeds `octomap_server` (the sim HAL's depth cloud, a real RGB-D driver) | an extrinsic measured against the body `base_frame` denotes |
| this bridge | `base_frame` as `header.frame_id`, with `origin` in that frame |
| the kernel | per-link `origin_xyz` measured from that same `base_frame` |

On robosuite/RoboCasa mobile manipulators that is not the MJCF chassis root.
The OmronMobileBase stacks a ground-level root (`mobilebase0_base`, world z 0)
under a 0.70 m pedestal whose top plate (`mobilebase0_support`) carries the arm
and the robot-mounted cameras — and `base_link` on `/tf` is the **pedestal
top**, because `MobileBaseBridge` publishes `odom -> base_link` from the HAL's
`base_pose_6dof()` (robosuite's `robot0_base_pos`). Measuring the sim camera
extrinsic against the chassis root instead made the grid's *content*
world-referenced while its *label* said `base_link`, so every TF consumer (Nav2,
SLAM, the dashboard) saw obstacles a pedestal too high. The kernel agreed with
the grid only because PR #103 had pushed the same 0.70 m into its FK root, which
cancelled the error for the kernel alone.

ADR-0095 fixes it at the source: `openral_hal.depth_cloud.resolve_base_frame_body_name`
resolves the arm-mount body for every `base_frame -> …` extrinsic, and the
`panda_mobile` manifests' `panda_joint1` origin is the plain Franka 0.333 m
again. **Both halves must land together** — a half-applied change leaves the
kernel a full pedestal away from the obstacles it is checked against (hazard
HZ-0095-1). `VoxelCollision.BaseFrameAlignmentPreservesTheProtectiveEnvelope`
and `VoxelCollision.HalfAppliedFrameAlignmentMovesTheKernelByThePedestal`
(`cpp/openral_safety_kernel/test/test_collision.cpp`) pin both facts, and
`tests/unit/test_voxel_grid_frame_alignment.py` pins the producer side against a
real `MjModel`.

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
`openral_sim.backends.depth_camera.synthesize_depth_image` (one cast per frame,
back-projected to a cloud by `openral_hal.depth_cloud.points_from_depth_grid`) and publishes
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
`OccupancyVoxels` directly). The frame this chain is expressed in is pinned by
`tests/unit/test_voxel_grid_frame_alignment.py` and, against the live kitchen,
by `tests/sim/safety/test_panda_mobile_robocasa_collision_fk.py` — see *The
frame contract* above.
