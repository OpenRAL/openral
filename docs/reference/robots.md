# Robot Descriptions

Every embodiment is a typed `RobotDescription` manifest under `robots/<robot_id>/robot.yaml`, validated by `openral_core` and consumed by the HAL adapter, the rSkill loader (embodiment-tag check), and the `openral sim run` runner.

## Supported robots

| Robot | Manifest | HAL | Status |
|---|---|---|---|
| SO-100 (LeRobot follower arm) | [`robots/so100_follower/`](../../robots/so100_follower/) | `SO100FollowerHAL` (real) + `SO100MujocoHAL` (sim) + `openral_hal_so100` lifecycle node | ✓ HW + sim |
| SO-101 (LeRobot follower arm) | [`robots/so101_follower/`](../../robots/so101_follower/) | shares SO-100 family — `SO100FollowerHAL` (real) + `SO100MujocoHAL` (sim) + `openral_hal_so100` lifecycle node | ✓ HW + sim |
| Franka Panda | [`robots/franka_panda/`](../../robots/franka_panda/) | `FrankaPandaHAL` (`MujocoArmHAL`, ADR-0023) + `openral_hal_franka` | ✓ sim · HW bring-up M3 (#56) |
| UR5e | [`robots/ur5e/`](../../robots/ur5e/) | `UR5eHAL` (`MujocoArmHAL`) + `openral_hal_ur5e` | ✓ sim · HW bring-up M3 |
| UR10e | [`robots/ur10e/`](../../robots/ur10e/) | `UR10eHAL` (`MujocoArmHAL`) + `openral_hal_ur10e` | ✓ sim · HW bring-up M3 |
| Flexiv Rizon 4 | [`robots/rizon4/`](../../robots/rizon4/) | `Rizon4MujocoHAL` (`MujocoArmHAL`) | ✓ sim |
| ALOHA bimanual (gym-aloha) | [`robots/aloha_bimanual/`](../../robots/aloha_bimanual/) | `AlohaMujocoHAL` (`MujocoArmHAL`, bimanual) + real-HW `AlohaHAL` over Interbotix XS | ✓ sim · ✓ HW |
| Enactic OpenArm v2 bimanual | [`robots/openarm/`](../../robots/openarm/) | `OpenArmMujocoHAL` (`MujocoArmHAL`, 16-DoF bimanual) | ✓ sim |
| Unitree H1 humanoid | [`robots/h1/`](../../robots/h1/) | `H1MujocoHAL` (software PD loop, no S0 cerebellum) | ✓ sim |
| Unitree G1 humanoid | [`robots/g1/`](../../robots/g1/) | `G1MujocoHAL` (Menagerie MJCF) | ✓ sim |
| Rethink Sawyer | [`robots/sawyer/`](../../robots/sawyer/) | eval-only · real-HW HAL planned (#57) | ✓ sim |
| Fourier GR1 | [`robots/gr1/`](../../robots/gr1/) | eval-only (RoboCasa GR1 fork + RLDX-1) | ✓ sim |
| Panda mobile (RoboCasa kitchen) | [`robots/panda_mobile/`](../../robots/panda_mobile/) | eval-only (drives RoboCasa kitchen via robosuite) | ✓ sim |
| Google Robot (SimplerEnv) | [`robots/google_robot/`](../../robots/google_robot/) | eval-only (SimplerEnv `fractal20220817_data` bridge env) | ✓ sim |
| WidowX (SimplerEnv) | [`robots/widowx/`](../../robots/widowx/) | eval-only (SimplerEnv `bridge_orig` env) | ✓ sim |
| PushT 2-D (gym-pusht) | [`robots/pusht_2d/`](../../robots/pusht_2d/) | eval-only (`pymunk` 2-D rigid-body) | ✓ sim |

## Manifest format

```yaml
# robots/so100_follower/robot.yaml (excerpt)
name: "so100_follower"
embodiment_kind: "manipulator"
base_frame: "base"
joints:
  - name: "shoulder_pan"
    joint_type: "revolute"
    axis_xyz: [0.0, 0.0, 1.0]
    position_limits: [-2.0944, 2.0944]   # radians (URDF-native)
    velocity_limit: 4.5
    actuator_kind: "servo"
  # … 5-DoF arm + 1-DoF gripper
```

## VLA compatibility

See [docs/reference/vla_compatibility.md](vla_compatibility.md) for the full matrix of which VLA checkpoints run on each robot under which simulators, including verified observation/action dimensions and normalisation notes.
