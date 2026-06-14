# openral_nav2_bringup

ADR-0025 — bringup wrapper for upstream **Nav2** as a Reasoner-managed
background service (the second such service, after
[`openral_slam_bringup`](../openral_slam_bringup/)). We do **not** reimplement
Nav2 — this package only ships the per-deployment launch + parameter glue so the
OpenRAL Reasoner can start/stop navigation on demand and route a
`NavigateToPose` goal.

```
/scan + /map ──▶ Nav2 (planner + controller + bt_navigator) ──/navigate_to_pose──▶ /cmd_vel
```

## Run

Normally started by the Reasoner as a background service when the active goal
needs navigation (ADR-0025); the robot must declare a lidar for the leg to be
wired (the `openral deploy sim` / `deploy run` graph brings up SLAM + Nav2 only
then). Standalone:

```bash
ros2 launch openral_nav2_bringup nav2.launch.py
```

Requires upstream `ros-${ROS_DISTRO}-navigation2` / `nav2_bringup` (in the deploy
images) and a map source — typically `openral_slam_bringup` (slam_toolbox).
