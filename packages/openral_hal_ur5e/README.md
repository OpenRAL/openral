# `openral_hal_ur5e`

ROS 2 lifecycle-node skeleton for the Universal Robots UR5e 6-DoF arm.

This package will wrap `openral_hal.ur.UR5eHAL` as a managed ROS 2
lifecycle node. The Python HAL adapter ships today (sim-only, MuJoCo via
the public `robot_descriptions` URDF); the ROS package is a **skeleton**
— `lifecycle_node.py` is ~25 lines and its transition handlers land with
the M3 hardware bring-up.

## Status

| Component | Status |
| --- | --- |
| `UR5eHAL` Python adapter | ✓ shipped (sim-only via MuJoCo) |
| `UR5e_DESCRIPTION` (Pydantic) | ✓ shipped |
| `ur5e_with_sensors` factory | ✓ shipped |
| ROS 2 lifecycle node | skeleton only — handlers are TODOs |
| HIL test on real UR5e | M3 (planned) |

## Intended interface

Once filled in, the contract follows the SO-100 package (see
`packages/openral_hal_so100/README.md`):

| Element | Value |
| --- | --- |
| Lifecycle states | `configure → activate → deactivate → cleanup` |
| Pub topics | `/joint_states`, `~/joint_states` (`sensor_msgs/JointState`) |
| Sub topics | `/openral/safe_action` (`openral_msgs/ActionChunk`), `/openral/estop` (`std_msgs/Empty`) per ADR-0018 F1/F5 |
| QoS | RELIABLE / VOLATILE / KEEP_LAST=10 (control-class) |
| HAL backend | `openral_hal.ur.UR5eHAL` |

## Embodiment

| Field | Value |
| --- | --- |
| Embodiment tag | `ur5e` |
| Datasheet | 5 kg payload, 0.85 m reach, ≤ π rad/s joint velocity |
| `sdk_kind` | `closed` for real hardware (UR RTDE / URScript) |

## Build

```bash
source /opt/ros/jazzy/setup.bash
colcon build --merge-install --packages-select openral_hal_ur5e
```

Not yet included in `just ros2-build`; add via `--packages-select` until
the lifecycle node is complete.

## See also

- `python/hal/README.md` — `UR5eHAL`, `MujocoArmHAL`.
- `tests/sim/test_ur5e_hal_mujoco.py` — MuJoCo closed-loop test (gravity-off
  convergence, safety bounds).
- `openral_hal_ur10e` — the larger sibling (12.5 kg / 1.30 m reach).
