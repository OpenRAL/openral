# openral_hal_panda_mobile

ROS 2 lifecycle-node wrapper around the `panda_mobile` HAL (a Franka 7-DoF arm
on a holonomic 3-DoF planar base) so the RoboCasa mobile-manipulator embodiment
can participate in the `openral deploy sim` graph (`sim_e2e.launch.py` → C++
safety kernel → HAL).

Spawned by `openral deploy sim --robot panda_mobile` via
`_ROBOT_HAL_REGISTRY["panda_mobile"]` (see
`python/cli/src/openral_cli/deploy_sim.py`). Subscribes `/openral/safe_action`
+ `/openral/estop`, publishes `/joint_states`, `/openral/candidate_action`,
`/cmd_vel` (base), and — under ADR-0034 sim scene-attach — `/openral/cameras/*`
(incl. a depth `PointCloud2` + `/scan` for the ADR-0030 octomap/Nav2 leg).

The HAL drives the RoboCasa kitchen via robosuite; ADR-0028c/0028d add the
Cartesian-gripper and joint-velocity-torso action handlers, and ADR-0030 wires
the depth/lidar streams for collision checking. Eval scene:
[`scenes/sim/robocasa_panda_mobile_kitchen.yaml`](../../scenes/sim/robocasa_panda_mobile_kitchen.yaml).
