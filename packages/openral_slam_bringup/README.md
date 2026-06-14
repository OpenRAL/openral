# openral_slam_bringup

ADR-0025 — bringup wrapper for upstream **slam_toolbox** as a Reasoner-managed
background service. We do **not** reimplement SLAM — this package ships only the
per-deployment launch + parameter glue so the OpenRAL Reasoner can start/stop
mapping/localization on demand and provide a `/map` to
[`openral_nav2_bringup`](../openral_nav2_bringup/).

```
/scan ──▶ slam_toolbox ──▶ /map  (+ map → odom TF)
```

## Run

Normally started by the Reasoner as a background service when the robot declares
a lidar (ADR-0025); the `openral deploy sim` / `deploy run` graph brings up the
SLAM + Nav2 leg only then. Standalone:

```bash
ros2 launch openral_slam_bringup slam_toolbox.launch.py
```

Requires upstream `ros-${ROS_DISTRO}-slam-toolbox` (in the deploy images) and a
`sensor_msgs/LaserScan` on `/scan`.
