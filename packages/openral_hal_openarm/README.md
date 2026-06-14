# openral_hal_openarm

ROS 2 lifecycle-node wrapper around `openral_hal.OpenArmMujocoHAL` so the
Enactic **OpenArm v2** 16-DoF bimanual arm can participate in the
`openral deploy sim` graph (`sim_e2e.launch.py` → C++ safety kernel → HAL).

Spawned by `openral deploy sim --robot openarm` via
`_ROBOT_HAL_REGISTRY["openarm"]` (see
`python/cli/src/openral_cli/deploy_sim.py`). Subscribes `/openral/safe_action`
+ `/openral/estop`, publishes `/joint_states`, and — under ADR-0034 sim
scene-attach — `/openral/cameras/*` + the MuJoCo viewer.

The HAL is MuJoCo-backed (sim path only — a real-HW OpenArm driver via lerobot
upstream is a planned follow-up); `HAL.connect()` resolves the MJCF on first
use. Lifecycle coverage in `tests/integration/test_openarm_hal_lifecycle.py`.
