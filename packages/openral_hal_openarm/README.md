# openral_hal_openarm

ROS 2 lifecycle-node wrapper around `openral_hal.OpenArmMujocoHAL` so the
Enactic **OpenArm v2** 16-DoF bimanual arm can participate in the
`openral deploy sim` graph (`sim_e2e.launch.py` → C++ safety kernel → HAL).

Spawned by `openral deploy sim --robot openarm` via
`_ROBOT_HAL_REGISTRY["openarm"]` (see
`python/cli/src/openral_cli/deploy_sim.py`). Subscribes `/openral/safe_action`
+ `/openral/estop`, publishes `/joint_states`, and — under sim
scene-attach — `/openral/cameras/*` + the MuJoCo viewer.

This node wraps the MuJoCo twin; `HAL.connect()` resolves the MJCF on first
use. Lifecycle coverage in `tests/integration/test_openarm_hal_lifecycle.py`.

The real arm is reached a different way — `openral_hal.openarm_real:OpenArmRealHAL`
(the manifest's `hal.real`) commands `openarm_bringup`'s own `ros2_control`
stack, so on hardware the `controller_manager` and the C++ `openarm_hardware`
SystemInterface own the 400 Hz loop and this node is not in the path.
