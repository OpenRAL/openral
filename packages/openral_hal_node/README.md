# `openral_hal_node`

The one manifest-driven HAL lifecycle node. Every robot runs it; robot
identity comes from the `robot_yaml` parameter (`robots/<id>/robot.yaml`),
never from the package.

`lifecycle_node.py` calls `openral_hal.lifecycle.make_lifecycle_main_from_manifest`,
which spins `ManifestHALLifecycleNode`. On `configure` it loads the manifest and
builds the HAL through `openral_hal.build_hal`:

| `hal_mode` | HAL |
| --- | --- |
| `sim` + `sim_env_yaml` | `SimAttachedHAL` around the scene's rollout (scene-attach) |
| `sim` | the manifest's `hal.sim` entrypoint, else `MujocoArmHAL.from_description` (bare twin) |
| `real` | the manifest's `hal.real` entrypoint |

`openral deploy sim|run` spawns it from `deploy_e2e.launch.py` with
`hal_package:=openral_hal_node` and `hal_node_name:=openral_hal_<robot_id>`.

## Package name vs node name

- **Package** `openral_hal_node` — the one install that hosts the executable.
- **Node name** `openral_hal_<robot_id>` (e.g. `openral_hal_ur5e`) — set by the
  launch (`__node:=` remap); it namespaces `~/joint_states` and the services and
  becomes the dashboard `service.name` (`openral.hal.<robot_id>`).

Real-hardware vendor bringups that must run beside the node are not part of this
package: a manifest points at them through `hal.real_bringup`
(`"<ros_package>:<launch_file>"`, e.g. `openral_hal_openarm:real_bringup.launch.py`).

## Run by hand

```bash
source /opt/ros/jazzy/setup.bash && just ros2-build && source install/setup.bash
ros2 run openral_hal_node lifecycle_node.py --ros-args \
    -r __node:=openral_hal_ur5e \
    -p robot_yaml:=robots/ur5e/robot.yaml -p hal_mode:=sim
```

## Parameters, topics, lifecycle

Declared in `ManifestHALLifecycleNode` / `HALLifecycleNodeBase`
(`python/hal/src/openral_hal/lifecycle.py`): `robot_yaml`, `hal_mode`,
`sim_env_yaml`, real-transport overrides (`port`, `robot_ip`, `fci_ip`, `id`,
`calibration_dir`, `hal_transport_json`), `scene_composition_json`,
`viewer_enabled`, `walking_enabled`, the camera / scan / depth / odom rates and
`publish_rate_hz`. Publishes `/joint_states` + `~/joint_states`; subscribes
`/openral/safe_action` and `/openral/estop`; mobile bases also publish `/odom`,
the `odom -> base_link` TF and consume `/cmd_vel`.

## Tests

`test/` holds the package tests, parametrised by robot manifest:

| File | Covers |
| --- | --- |
| `test_lifecycle_node.py` | full lifecycle cycle for every in-tree sim-twin manifest; `hal.read_state` / `hal.send_action` spans |
| `test_lifecycle_estop_safety.py` | e-stop forwarding, recovery policy and diagnostics against the Galaxea A1 real HAL |
| `test_real_hal_spans.py` | spans from a real-HAL adapter (`SO100FollowerHAL` on its digital twin) |
| `test_mobile_base_helpers.py` | mobile-base odom / scan helpers (panda_mobile manifest frames) |
| `test_panda_mobile_sensor_bridge_regression.py` | `/scan`, depth cloud and `/odom` on the RoboCasa kitchen (not in `colcon test`; run directly) |

```bash
just ros2-test
```
