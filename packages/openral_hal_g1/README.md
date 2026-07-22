# openral_hal_g1

ROS 2 lifecycle-node wrapper around `openral_hal.G1MujocoHAL` so the
Unitree G1 humanoid (29-DoF) can participate in the `openral deploy sim` graph
(`sim_e2e.launch.py` → C++ safety kernel → HAL).

Spawned by `openral deploy sim --robot g1` via
`_ROBOT_HAL_REGISTRY["g1"]` (see
`python/cli/src/openral_cli/deploy_sim.py`).

The HAL is MuJoCo-backed. Its default contract/glide mode pulls the MJCF from
`robot_descriptions`; `scenes/deploy/g1_vln.yaml` explicitly enables ADR-0089's
hash-pinned MuJoCo Playground ONNX walking controller and matching MJCF.
