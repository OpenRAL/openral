# openral_hal_rizon4

ROS 2 lifecycle-node wrapper around `openral_hal.Rizon4MujocoHAL` so the
Flexiv Rizon 4 7-DoF arm can participate in the `openral deploy sim` graph
(`sim_e2e.launch.py` → C++ safety kernel → HAL).

Spawned by `openral deploy sim --robot rizon4` via
`_ROBOT_HAL_REGISTRY["rizon4"]` (see
`python/cli/src/openral_cli/deploy_sim.py`).

The HAL is MuJoCo-backed; `HAL.connect()` pulls the MJCF from
`robot_descriptions` on first use.
