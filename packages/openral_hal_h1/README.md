# openral_hal_h1

ROS 2 lifecycle-node wrapper around `openral_hal.H1MujocoHAL` so the
Unitree H1 humanoid (19-DoF) can participate in the `openral deploy sim` graph
(`sim_e2e.launch.py` → C++ safety kernel → HAL).

Spawned by `openral deploy sim --robot h1` via
`_ROBOT_HAL_REGISTRY["h1"]` (see
`python/cli/src/openral_cli/deploy_sim.py`).

The HAL is MuJoCo-backed; `HAL.connect()` pulls the MJCF from
`robot_descriptions` on first use.
