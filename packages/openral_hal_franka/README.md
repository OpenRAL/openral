# `openral_hal_franka`

ROS 2 lifecycle-node skeleton for the Franka Emika Panda 7-DoF arm.

This package will wrap `openral_hal.franka_panda.FrankaPandaHAL` as
a managed ROS 2 lifecycle node. The Python HAL adapter is shipped and
sim-tested via MuJoCo (`tests/sim/test_franka_panda_hal_mujoco.py`); this ROS
package itself is a **skeleton** — `lifecycle_node.py` is ~25 lines and
its full transition handlers land alongside the M3 hardware bring-up.

## Status

| Component | Status |
| --- | --- |
| `FrankaPandaHAL` Python adapter | ✓ shipped (sim-only via MuJoCo) |
| `FRANKA_PANDA_DESCRIPTION` (Pydantic) | ✓ shipped |
| `franka_panda_with_sensors` factory | ✓ shipped |
| ROS 2 lifecycle node | skeleton only — handlers are TODOs |
| HIL test on real Panda hardware | M3 (planned) |

## Intended interface

Once the lifecycle node is filled in, the contract follows the SO-100
package (see `packages/openral_hal_so100/README.md`):

| Element | Value |
| --- | --- |
| Lifecycle states | `configure → activate → deactivate → cleanup` |
| Pub topics | `/joint_states`, `~/joint_states` (`sensor_msgs/JointState`) |
| Sub topics | `/openral/safe_action` (`openral_msgs/ActionChunk`), `/openral/estop` (`std_msgs/Empty`) per ADR-0018 F1/F5 |
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

The package is **not** included in `just ros2-build` today — that recipe
only builds `msgs`, `hal_so100`, and `world_state`. Add it explicitly
with `--packages-select` while the lifecycle node is iterated on.

## See also

- `python/hal/README.md` — `FrankaPandaHAL`, `MujocoArmHAL`.
- `tests/sim/test_franka_panda_hal_mujoco.py` — MuJoCo closed-loop test.
- `tests/unit/test_franka_panda.py` — description / kinematic / safety tests.
- CLAUDE.md §7.4 — VLA license matrix; the LIBERO Franka sim is open,
  but real Panda hardware uses Franka's closed FCI SDK.
