# openral_slam_bringup

Bringup wrapper for SLAM as a deployment service. We do **not** reimplement SLAM —
this package ships only the per-deployment launch + parameter glue so the SLAM leg
provides a `/map` (+ `map → odom` TF) to
[`openral_nav2_bringup`](../openral_nav2_bringup/). Two backends, selected by
robot capability:

| Backend | `RobotCapabilities` flag | Sensor | Lifecycle |
|---|---|---|---|
| **lidar** (`slam_toolbox`) | `has_lidar` | `sensor_msgs/LaserScan` on `/scan` | Reasoner-managed lifecycle node |
| **visual** (cuVSLAM) | `has_vision_slam` | stereo / mono+IMU / RGB-D cameras | plain composable node (live once composed) |

```
lidar:   /scan ─────────────▶ slam_toolbox ──▶ /map  (+ map → odom TF)
visual:  cameras (+ IMU) ───▶ cuVSLAM ───────▶ map → odom TF
                                  └─▶ (+ nvblox, Phase 2) ──▶ /map costmap
```

`deploy_sim.py` resolves the backend (`lidar` wins when both flags are set — it
needs no AI depth model) and forwards `slam_backend:=lidar|visual|none`;
`sim_e2e.launch.py` composes the matching nodes when `enable_slam` is true.

## Run

Normally started by the deploy graph when the robot declares a lidar
(`has_lidar`) or vision SLAM (`has_vision_slam`). Standalone:

```bash
# lidar backend — needs ros-${ROS_DISTRO}-slam-toolbox + a /scan
ros2 launch openral_slam_bringup slam_toolbox.launch.py

# visual backend — needs the operator's NVIDIA Isaac ROS install
# (isaac_ros_visual_slam) + rectified camera streams
ros2 launch openral_slam_bringup cuvslam.launch.py

# occupancy for Nav2 (visual) — needs nvblox_ros + a depth stream.
# Pass robot_yaml so the prefilter derives the floor-excluded body-height
# band from the robot's footprint/collision/link measurements + live TF.
ros2 launch openral_slam_bringup nvblox.launch.py robot_yaml:=/abs/path/to/robots/<id>/robot.yaml
```

Every collision volume the manifest declares must be placeable relative to
`base_frame` — through the `joints` chain, or through the
`assets.urdf.root_frame` + `base_to_root_xyz_rpy` bridge that
`sim_e2e.launch.py` publishes as a static TF (this is how UR manifests reach
their upstream `base_link` root, which no movable joint has as a child).
A manifest that declares a volume on a link neither of those reaches makes
the node **refuse at startup** with `ROSConfigError` naming the links, rather
than measure the subset it can reach: a band covering an arbitrary part of the
robot still reports `collision_geometry` as its source, so nothing downstream
could tell it apart from a real measurement, and `/map` would claim free space
at the heights the omitted links occupy.

A manifest that declares **no** collision geometry is not that case — it falls
back to `min_body_height_m` (0.30 m) as documented. If a robot's placement gap
cannot be fixed in the manifest, set the explicit `min_height_m` /
`max_height_m` override pair instead.

### Installing the NVIDIA Isaac ROS stack (visual backend only)

cuVSLAM + nvblox are **not bundled** (closed NVIDIA binaries, license
guard). Install them once on the GPU host (Ubuntu 24.04 x86_64 / supported
Jetson, CUDA 13.0+, driver 580+):

```bash
just install-isaac-ros        # adds the two NVIDIA apt repos + installs cuVSLAM + nvblox (sudo)
```

Run it in a **real terminal** (it needs a tty for the sudo password — not the `!`
session prefix). It runs the [official Isaac ROS apt steps](https://nvidia-isaac-ros.github.io/getting_started/)
**plus** the NVIDIA Jetson x86_64 repo, which provides the VPI + `nvsci` libraries
Isaac ROS NITROS depends on (`libnvvpi4` / `vpi4-dev` / `nvsci`) — without it the
install fails with `libnvvpi4 ... not installable` / `held broken packages` on x86:

```bash
# Isaac ROS repo (cuVSLAM + nvblox)
k="/usr/share/keyrings/nvidia-isaac-ros.gpg"
curl -fsSL https://isaac.download.nvidia.com/isaac-ros/repos.key | sudo gpg --dearmor | sudo tee "$k" > /dev/null
f="/etc/apt/sources.list.d/nvidia-isaac-ros.list"
s="deb [signed-by=$k] https://isaac.download.nvidia.com/isaac-ros/release-4 noble main"
sudo touch "$f"; grep -qxF "$s" "$f" || echo "$s" | sudo tee -a "$f"

# Jetson x86_64 repo (VPI + nvsci — NITROS deps; also used on x86 dGPU)
jk="/usr/share/keyrings/nvidia-jetson.gpg"
curl -fsSL https://repo.download.nvidia.com/jetson/jetson-ota-public.asc | sudo gpg --dearmor | sudo tee "$jk" > /dev/null
jf="/etc/apt/sources.list.d/nvidia-jetson-x86.list"
js="deb [signed-by=$jk] https://repo.download.nvidia.com/jetson/x86_64/noble r38.2 main"
sudo touch "$jf"; grep -qxF "$js" "$jf" || echo "$js" | sudo tee -a "$jf"

sudo apt-get update
sudo apt-get install -y ros-jazzy-isaac-ros-visual-slam ros-jazzy-isaac-ros-nvblox
# verify: ros2 pkg prefix isaac_ros_visual_slam && ros2 pkg prefix nvblox_ros
```

### PyCuVSLAM: the same engine without the Isaac ROS apt stack

NVIDIA also ships the cuVSLAM engine as a pip wheel with a Python API —
[PyCuVSLAM](https://github.com/nvidia-isaac/cuVSLAM) (NVIDIA Community
License: commercial OK, NVIDIA hardware only; still **not bundled**).
`pycuvslam_node.py` runs it in-process under the workspace's Python 3.12:
it synchronizes a **rectified stereo pair** from the camera bus, tracks with
`cuvslam.Tracker`, and fills the same `map → odom` TF edge (rig ≡ left camera
optical frame; the right `CameraInfo.p` must carry `-fx·baseline`, the
standard rectified-stereo convention). Use it when the Isaac ROS apt install
is unavailable or too heavy (e.g. sim hosts); it has no NITROS zero-copy
path and no raw-image undistortion — for those, use `cuvslam.launch.py`.

```bash
# operator installs the wheel matching the host (Py3.12 / CUDA 12 or 13):
#   https://github.com/nvidia-isaac/cuVSLAM/releases  (v16.0.0+)
uv pip install ./cuvslam-16.0.0+cu13-cp312-abi3-manylinux_2_39_x86_64.whl

ros2 launch openral_slam_bringup pycuvslam.launch.py
```

> If the host also has the apt Isaac ROS stack installed, make sure the
> wheel's bundled `libcuvslam.so` wins over `/opt/ros/<distro>/lib`'s older
> copy on `LD_LIBRARY_PATH`, or imports fail with an `undefined symbol`.
> The wheel resolves CUDA runtime libs from the host (`/usr/local/cuda`);
> pick the `cu12`/`cu13` wheel matching the installed toolkit.

**Selecting the impl from a deploy scene.** `openral deploy sim/run` composes
the visual backend when the robot declares `has_vision_slam`; which engine
(and which stereo cameras) is a committed **workcell** choice on
`DeployScene.runtime`:

```yaml
runtime:
  slam_visual_impl: pycuvslam        # default: isaac_ros (the composable node)
  slam_stereo_cameras: [left, right] # camera names → /openral/cameras/<name>/…
```

`deploy_sim.py` forwards these as `slam_visual_impl:=` / `slam_stereo_cameras:=`,
and `sim_e2e.launch.py` composes the matching launch file with the rig's topics
remapped onto that impl's camera args (Isaac ROS `image_0/1_topic`, PyCuVSLAM
`left/right_image_topic`). Omitting `slam_stereo_cameras` keeps the impl's
default `left`/`right` topics.

**Multi-camera mode (sim rigs).** For pycuvslam, `sim_e2e.launch.py` also passes
`robot_yaml`, so `pycuvslam_node.py` derives the rig frame from the manifest
`base_frame` and reads each camera's `rig_from_camera` extrinsic from TF — cuVSLAM's
default multi-camera mode. This handles arbitrary base-mounted rigs (e.g. a
toed-in pair) without a rectified stereo image, so a sim robot's existing cameras
work as-is. The first shipped example is the lidar-less **`panda_mobile_vslam`**
robot (`has_lidar: false`, `has_vision_slam: true`) whose two `shoulder_left` /
`shoulder_right` cameras (RoboCasa `agentview_left`/`right`, a 0.70 m base rig)
localise the mobile base — run `scenes/deploy/robocasa_vslam.yaml` and watch
`/openral/visual_slam/odometry` as it drives. A standalone rectified RealSense
pair (no `robot_yaml`/`rig_frame`) keeps the baseline path.

For **mono-only** robots, also start the metric-depth sidecar that feeds nvblox:

```bash
python tools/da3_depth_sidecar.py --port 5771   # DA3-Small, ~0.27 GB / ~27 Hz on an 8 GB Ada
```

> **cuVSLAM is the camera-based backend for lidar-less robots.** It produces
> `map → odom` localization, **not** an occupancy grid — Nav2's costmap needs the
> companion **nvblox** stage (depth + cuVSLAM pose → `/map`). The cuVSLAM/nvblox
> engines are precompiled NVIDIA binaries under an
> NVIDIA EULA — **not bundled** by OpenRAL; install them on the target GPU host
> behind the NVIDIA license guard. Live bring-up is operator-run (needs a GPU +
> the Isaac ROS stack); the in-tree tests are hermetic launch-contract checks.
