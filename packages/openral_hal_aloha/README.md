# openral_hal_aloha

ROS 2 lifecycle-node wrapper around `openral_hal.AlohaMujocoHAL` so the
bimanual ALOHA (14-DoF, leader+follower) can participate in the `openral deploy sim` graph
(`sim_e2e.launch.py` → C++ safety kernel → HAL).

Spawned by `openral deploy sim --robot aloha` via
`_ROBOT_HAL_REGISTRY["aloha"]` (see
`python/cli/src/openral_cli/deploy_sim.py`).

The HAL is MuJoCo-backed; `HAL.connect()` pulls the MJCF from
`robot_descriptions` on first use.
