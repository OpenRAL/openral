# `openral_hal_franka`

ROS 2 lifecycle node for the Franka Emika Panda 7-DoF arm.

This package wraps `openral_hal.franka_panda.FrankaPandaHAL` (sim) /
`openral_hal.franka_panda_real.FrankaPandaRealHAL` (real) as a managed
ROS 2 lifecycle node, shipped via the shared
`make_lifecycle_main_from_manifest` (same pattern as `openral_hal_so100`).
The Python HAL adapter is shipped and sim-tested via MuJoCo
(`tests/sim/test_franka_panda_hal_mujoco.py`).

## Status

| Component | Status |
| --- | --- |
| `FrankaPandaHAL` Python adapter | ✓ shipped (sim-only via MuJoCo) |
| `FRANKA_PANDA_DESCRIPTION` (Pydantic) | ✓ shipped |
| `franka_panda_with_sensors` factory | ✓ shipped |
| ROS 2 lifecycle node | shipped — manifest-driven (`make_lifecycle_main_from_manifest`), same pattern as `openral_hal_so100` |
| HIL test on real Panda hardware | M3 (planned) |

## Intended interface

Once the lifecycle node is filled in, the contract follows the SO-100
package (see `packages/openral_hal_so100/README.md`):

| Element | Value |
| --- | --- |
| Lifecycle states | `configure → activate → deactivate → cleanup` |
| Pub topics | `/joint_states`, `~/joint_states` (`sensor_msgs/JointState`) |
| Sub topics | `/openral/safe_action` (`openral_msgs/ActionChunk`), `/openral/estop` (`std_msgs/Empty`) — under `hal_mode:=real` the stop deactivates `franka_arm_controller` via `controller_manager` (`franka_hardware` then calls `stopRobot()`); recovery is `RESTART_REQUIRED` (see [`robots/franka_panda/README.md`](../../robots/franka_panda/README.md#e-stop-and-recovery)) |
| QoS | RELIABLE / VOLATILE / KEEP_LAST=10 (control-class) |
| HAL backend | `openral_hal.franka_panda.FrankaPandaHAL` |

## Embodiment

| Field | Value |
| --- | --- |
| Embodiment tag | `franka_panda` (and `libero` for the LIBERO sim variant) |
| Robot description | `robots/franka_panda/robot.yaml` |
| `sdk_kind` | `closed` for real Panda (FCI); `open` for the LIBERO sim variant |

## Build

```bash
source /opt/ros/jazzy/setup.bash
colcon build --merge-install --packages-select openral_hal_franka
```

The package builds via `just ros2-build` (see `Justfile`'s `ros2-build`
recipe, which lists it in `--packages-select`).

## See also

- `python/hal/README.md` — `FrankaPandaHAL`, `MujocoArmHAL`.
- `tests/sim/test_franka_panda_hal_mujoco.py` — MuJoCo closed-loop test.
- `tests/unit/test_franka_panda.py` — description / kinematic / safety tests.
- CLAUDE.md §7.4 — VLA license matrix; the LIBERO Franka sim is open,
  but real Panda hardware uses Franka's closed FCI SDK.
