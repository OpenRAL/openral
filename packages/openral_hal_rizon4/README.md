# openral_hal_rizon4

ROS 2 lifecycle-node host for the manifest-driven HAL (`hal.sim: null` → `MujocoArmHAL.from_description`) so the
Flexiv Rizon 4 7-DoF arm can participate in the `openral deploy sim` graph
(`deploy_e2e.launch.py` → C++ safety kernel → HAL).

Spawned by `openral deploy sim --robot rizon4` via
`_derive_hal_spec` (see `python/cli/src/openral_cli/deploy_sim.py`): a
robot's own `openral_hal_<robot_id>` package hosts it when one ships, else
the generic `openral_hal_scene_attached` node — both run the same
manifest-driven node, so `robots/<id>/robot.yaml` is the only per-robot input.

The HAL is MuJoCo-backed; `HAL.connect()` pulls the MJCF from
`robot_descriptions` on first use.
