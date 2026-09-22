# `openral_hal_ur10e`

ROS 2 lifecycle node for the Universal Robots UR10e 6-DoF arm.

Mirror of `openral_hal_ur5e` for the larger 12.5 kg / 1.30 m reach
robot. The Python HAL adapter (`openral_hal.ur.UR10eHAL`) is shipped
and sim-tested; the ROS package is shipped via the shared
`make_lifecycle_main_from_manifest` (same pattern as `openral_hal_so100`).

## Status

| Component | Status |
| --- | --- |
| `UR10eHAL` Python adapter | ✓ shipped (sim-only via MuJoCo) |
| `UR10e_DESCRIPTION` (Pydantic) | ✓ shipped |
| `ur10e_with_sensors` factory | ✓ shipped |
| ROS 2 lifecycle node | shipped — manifest-driven (`make_lifecycle_main_from_manifest`) |
| HIL test on real UR10e | M3 (planned) |

## Intended interface

Once filled in, the contract follows the SO-100 package (see
`packages/openral_hal_so100/README.md`):

| Element | Value |
| --- | --- |
| Lifecycle states | `configure → activate → deactivate → cleanup` |
| Pub topics | `/joint_states`, `~/joint_states` (`sensor_msgs/JointState`) |
| Sub topics | `/openral/safe_action` (`openral_msgs/ActionChunk`), `/openral/estop` (`std_msgs/Empty`) — under `hal_mode:=real` the stop deactivates `scaled_joint_trajectory_controller` via `controller_manager` and calls `/dashboard_client/stop`; recovery is `RESTART_REQUIRED` (see [`robots/ur10e/README.md`](../../robots/ur10e/README.md#e-stop-and-recovery)) |
| QoS | RELIABLE / VOLATILE / KEEP_LAST=10 (control-class) |
| HAL backend | `openral_hal.ur.UR10eHAL` |

## Embodiment

| Field | Value |
| --- | --- |
| Embodiment tag | `ur10e` |
| Datasheet | 12.5 kg payload, 1.30 m reach, ≤ 3.142 rad/s joint velocity |
| `sdk_kind` | `closed` for real hardware (UR RTDE / URScript) |

## Build

```bash
source /opt/ros/jazzy/setup.bash
colcon build --merge-install --packages-select openral_hal_ur10e
```

The package builds via `just ros2-build` (see `Justfile`'s `ros2-build`
recipe, which lists it in `--packages-select`).

## See also

- `python/hal/README.md` — `UR10eHAL`, `MujocoArmHAL`.
- `tests/sim/test_ur10e_hal_mujoco.py` — MuJoCo closed-loop test pinning
  the UR10e datasheet limits.
- `openral_hal_ur5e` — the smaller sibling (5 kg / 0.85 m reach).
