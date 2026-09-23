# openral_hal_aloha

ROS 2 lifecycle-node host for the manifest-driven HAL (`hal.sim: null` → `MujocoArmHAL.from_description`) so the
bimanual ALOHA (14-DoF, leader+follower) can participate in the `openral deploy sim` graph
(`deploy_e2e.launch.py` → C++ safety kernel → HAL).

**Not spawned by `openral deploy sim` any more.** The robot id is
`aloha_bimanual` and `_derive_hal_spec` (see
`python/cli/src/openral_cli/deploy_sim.py`) only picks a robot's own
`openral_hal_<robot_id>` package, so ALOHA runs on the generic
`openral_hal_scene_attached` node — the same manifest-driven node this package
ships. Retiring this package needs a recorded decision.

The HAL is MuJoCo-backed; `HAL.connect()` pulls the MJCF from
`robot_descriptions` on first use.
