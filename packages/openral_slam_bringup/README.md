# openral_slam_bringup

Bringup wrapper for SLAM as a deployment service. We do **not** reimplement SLAM —
this package ships only the per-deployment launch + parameter glue so the SLAM leg
provides a `/map` (+ `map → odom` TF) to
[`openral_nav2_bringup`](../openral_nav2_bringup/). Two backends, selected by
robot capability:

| Backend | `RobotCapabilities` flag | Sensor | ADR | Lifecycle |
|---|---|---|---|---|
| **lidar** (`slam_toolbox`) | `has_lidar` | `sensor_msgs/LaserScan` on `/scan` | ADR-0025 | Reasoner-managed lifecycle node |
| **visual** (cuVSLAM) | `has_vision_slam` | stereo / mono+IMU / RGB-D cameras | ADR-0064 | plain composable node (live once composed) |

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
(`has_lidar`, ADR-0025) or vision SLAM (`has_vision_slam`, ADR-0064). Standalone:

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

### Installing the NVIDIA Isaac ROS stack (visual backend only)

cuVSLAM + nvblox are **not bundled** (closed NVIDIA binaries, ADR-0064 license
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

For **mono-only** robots, also start the metric-depth sidecar that feeds nvblox:

```bash
python tools/da3_depth_sidecar.py --port 5771   # DA3-Small, ~0.27 GB / ~27 Hz on an 8 GB Ada
```

> **cuVSLAM is the camera-based backend for lidar-less robots.** It produces
> `map → odom` localization, **not** an occupancy grid — Nav2's costmap needs the
> companion **nvblox** stage (depth + cuVSLAM pose → `/map`), which is Phase 2 of
> ADR-0064. The cuVSLAM/nvblox engines are precompiled NVIDIA binaries under an
> NVIDIA EULA — **not bundled** by OpenRAL; install them on the target GPU host
> behind the ADR-0064 license guard. Live bring-up is operator-run (needs a GPU +
> the Isaac ROS stack); the in-tree tests are hermetic launch-contract checks.
