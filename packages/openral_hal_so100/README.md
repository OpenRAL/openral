# `openral_hal_so100`

ROS 2 lifecycle-node wrapper for the SO-100 follower arm.

This package wraps `openral_hal.so100_follower.SO100FollowerHAL` as a managed
ROS 2 lifecycle node. It is the **shipped** path: see
`openral_hal_so100/lifecycle_node.py` for the full implementation
and `tests/hil/test_so100.py` for the corresponding HIL coverage.

## Synopsis

```bash
# After: just bootstrap && uv sync --all-packages
source /opt/ros/jazzy/setup.bash
just ros2-build                 # colcon build --merge-install for selected packages
source install/setup.bash

ros2 run openral_hal_so100 lifecycle_node \
    --ros-args -p port:=/dev/ttyUSB0 -p publish_rate_hz:=30.0
```

## Lifecycle contract

| Transition | Action |
| --- | --- |
| `configure` | Open `SO100FollowerHAL.connect()`. |
| `activate` | Start joint-state publish timer + command subscriber. |
| `deactivate` | Stop timer; destroy publisher / subscriber. |
| `cleanup` | `SO100FollowerHAL.disconnect()`. |
| `shutdown` | Force-disconnect. |

## Parameters

| Name | Type | Default | Notes |
| --- | --- | --- | --- |
| `port` | string | `/dev/ttyUSB0` | USB serial port for the SO-100 controller. |
| `publish_rate_hz` | double | `30.0` | Rate of `JointState` publication. |
| `calibrate_on_connect` | bool | `false` | Run the lerobot calibration wizard at configure time. |

## Topics

| Direction | Topic | QoS | Message |
| --- | --- | --- | --- |
| Pub | `/joint_states` | RELIABLE / VOLATILE / KEEP_LAST=10 | `sensor_msgs/JointState` |
| Pub | `~/joint_states` | RELIABLE / VOLATILE / KEEP_LAST=10 | `sensor_msgs/JointState` |
| Sub | `/openral/safe_action` | RELIABLE / VOLATILE / KEEP_LAST=1 | `openral_msgs/ActionChunk` |
| Sub | `/openral/estop` | RELIABLE / VOLATILE / KEEP_LAST=10 | `std_msgs/Empty` |

The QoS profile follows CLAUDE.md §5.3 for control-class data. Actuator
commands flow through the ADR-0018 F1/F5 `/openral/safe_action` path
(produced by `rskill_runner_node` and clamped by `safety_node`).

## Embodiment

| Field | Value |
| --- | --- |
| Embodiment tag | `so100_follower` (`robots/so100_follower/robot.yaml`) |
| HAL backend | `openral_hal.so100_follower.SO100FollowerHAL` |
| Description | `openral_hal.SO100_DESCRIPTION` |
| `sdk_kind` | `open` (LeRobot SDK, Apache-2.0) |

## Build & test

```bash
just ros2-build                                  # builds msgs + hal_so100 + world_state
just ros2-test                                   # colcon test + colcon test-result --verbose
just hil so100                                   # tests/hil/test_so100.py — requires a connected SO-100
```

The HIL job runs on the `[self-hosted, lab-so100]` runner; CI parity is
in `.github/workflows/hil-so100.yml`.

## See also

- `python/hal/README.md` — the Python adapter documentation.
- `robots/so100_follower/README.md` — canonical SO-100 description.
- `skills/smolvla-libero/` — gold-standard rSkill that targets this embodiment.
- CLAUDE.md §5.3 (QoS profiles) and §6.1 (layer discipline).
