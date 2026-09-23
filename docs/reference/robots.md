# Robot Descriptions

Every embodiment is a typed `RobotDescription` manifest under `robots/<robot_id>/robot.yaml`, validated by `openral_core` and consumed by the HAL adapter, the rSkill loader (embodiment-tag check), and the `openral sim run` runner.

## Supported robots

Every robot launched by `openral deploy sim|run` runs the one generic ROS 2 HAL lifecycle node,
[`packages/openral_hal_node`](https://github.com/OpenRAL/openral/tree/master/packages/openral_hal_node/),
under the node name `openral_hal_<robot_id>`; the HAL column names the Python
adapter it builds from the manifest. Eval-only entries (`sim run`, benchmarks) build
their environment directly and start no lifecycle node.

| Robot | Manifest | HAL | Status |
|---|---|---|---|
| SO-100 (LeRobot follower arm) | [`robots/so100_follower/`](https://github.com/OpenRAL/openral/tree/master/robots/so100_follower/) | `SO100FollowerHAL` (real) + `SO100MujocoHAL` (sim) | ✓ HW + sim |
| SO-101 (LeRobot follower arm) | [`robots/so101_follower/`](https://github.com/OpenRAL/openral/tree/master/robots/so101_follower/) | shares SO-100 family — `SO100FollowerHAL` (real) + `SO100MujocoHAL` (sim) | ✓ HW + sim |
| Galaxea A1 | [`robots/galaxea_a1/`](https://github.com/OpenRAL/openral/tree/master/robots/galaxea_a1/) | `GalaxeaA1HAL` (real-only) + isolated operator-provided ROS 1 SDK sidecar | ✓ unit + offline integration + real observation/hold/joint/gripper + C++ kernel HIL |
| Franka Panda | [`robots/franka_panda/`](https://github.com/OpenRAL/openral/tree/master/robots/franka_panda/) | `FrankaPandaHAL` (`MujocoArmHAL`) + real-HW `FrankaPandaRealHAL` (`franka_ros2` / `libfranka` over the FCI) | ✓ sim · HW adapter landed, never validated — no physical Franka; `tests/hil/test_franka_panda.py` is gated on a `lab-franka` runner that does not exist (#56) |
| UR5e | [`robots/ur5e/`](https://github.com/OpenRAL/openral/tree/master/robots/ur5e/) | `UR5eHAL` (`MujocoArmHAL`) + real-HW `UR5eRealHAL` (`ur_robot_driver`, URCap/RTDE) | ✓ sim · HW adapter landed, no recorded run — `tests/hil/test_ur5e.py` needs `UR5E_HOST` plus a `lab-ur5e` runner, and neither has ever been set |
| UR10e | [`robots/ur10e/`](https://github.com/OpenRAL/openral/tree/master/robots/ur10e/) | `UR10eHAL` (`MujocoArmHAL`) + real-HW `UR10eRealHAL` (`ur_robot_driver`, URCap/RTDE) | ✓ sim · HW adapter landed, no recorded run — `tests/hil/test_ur10e.py` needs `UR10E_HOST` plus a `lab-ur10e` runner, and neither has ever been set |
| Flexiv Rizon 4 | [`robots/rizon4/`](https://github.com/OpenRAL/openral/tree/master/robots/rizon4/) | `Rizon4MujocoHAL` (`MujocoArmHAL`) | ✓ sim |
| ALOHA bimanual (gym-aloha) | [`robots/aloha_bimanual/`](https://github.com/OpenRAL/openral/tree/master/robots/aloha_bimanual/) | `AlohaMujocoHAL` (`MujocoArmHAL`, bimanual) + real-HW `AlohaHAL` over Interbotix XS (`HALBase`, not `RosControlHAL` — the vendor driver owns the bus) | ✓ sim · HW adapter landed, never validated — no physical ALOHA rig (`tests/hil/test_aloha.py` is gated on a `lab-aloha` runner that does not exist), **and its wire contract is wrong** ([#250](https://github.com/OpenRAL/openral/issues/250)): bring-up starts `interbotix_xs_sdk`'s `xs_sdk`, so the `arm_controller` / `gripper_controller` topics `AlohaHAL` publishes to do not exist on a real rig and `send_action` would succeed without moving the arm |
| ALOHA-AgileX (RoboTwin 2.0) | [`robots/aloha_agilex/`](https://github.com/OpenRAL/openral/tree/master/robots/aloha_agilex/) | eval-only (RoboTwin 2.0 dual-arm, 14-DoF; py3.10 SAPIEN sidecar) — targeted by `smolvla-robotwin` | ✓ sim |
| Enactic OpenArm v2 bimanual | [`robots/openarm/`](https://github.com/OpenRAL/openral/tree/master/robots/openarm/) | `OpenArmMujocoHAL` (`MujocoArmHAL`, 16-DoF bimanual) + real-HW `OpenArmRealHAL` (four `openarm_bringup` ros2_control controllers over Damiao CAN FD) | ✓ sim · ✓ HW bus — CAN transport + motor round-trip pass on the wired rig (`tests/hil/test_openarm_can_live.py`, read-only: it queries motor state and never energises). Run by hand on a lab host: no CI runner, no in-tree real-HW deploy scene, and the command→motion gate (`tests/hil/test_openarm_slot_group_motion.py`) has no recorded run, so nothing has yet driven the arm through `send_action` |
| Anvil OpenARM 2.0 bimanual | [`robots/anvil_openarm_v2/`](https://github.com/OpenRAL/openral/tree/master/robots/anvil_openarm_v2/) | `AnvilOpenArmV2MujocoHAL` (`MujocoArmHAL`, 16-DoF bimanual; v2 + Anvil J1/J6 range deltas + wrist support bracket) | ✓ sim |
| Unitree H1 humanoid | [`robots/h1/`](https://github.com/OpenRAL/openral/tree/master/robots/h1/) | `H1MujocoHAL` (software PD loop, no S0 cerebellum) | ✓ sim |
| Unitree G1 humanoid | [`robots/g1/`](https://github.com/OpenRAL/openral/tree/master/robots/g1/) | `G1MujocoHAL` (ADR-0087 glide fallback; ADR-0089 pinned ONNX walking controller + head camera for VLN) | ✓ sim |
| Rethink Sawyer | [`robots/sawyer/`](https://github.com/OpenRAL/openral/tree/master/robots/sawyer/) | eval-only sim (MetaWorld backend; `hal.sim` is null) + real-HW `SawyerRealHAL` (`sawyer_robot` fork, `intera_sdk` lineage) | ✓ sim · HW adapter landed, never validated — no physical Sawyer (#57) |
| Fourier GR1 | [`robots/gr1/`](https://github.com/OpenRAL/openral/tree/master/robots/gr1/) | eval-only (RoboCasa GR1 fork + RLDX-1) | ✓ sim |
| Panda mobile (RoboCasa kitchen) | [`robots/panda_mobile/`](https://github.com/OpenRAL/openral/tree/master/robots/panda_mobile/) | eval-only (drives RoboCasa kitchen via robosuite) | ✓ sim |
| Galaxea R1 Pro | [`robots/r1pro/`](https://github.com/OpenRAL/openral/tree/master/robots/r1pro/) | generic scene-attached HAL + official BEHAVIOR/OmniGibson sidecar | ✓ sim |
| Google Robot (SimplerEnv) | [`robots/google_robot/`](https://github.com/OpenRAL/openral/tree/master/robots/google_robot/) | eval-only (SimplerEnv `fractal20220817_data` bridge env) | ✓ sim |
| WidowX (SimplerEnv) | [`robots/widowx/`](https://github.com/OpenRAL/openral/tree/master/robots/widowx/) | eval-only (SimplerEnv `bridge_orig` env) | ✓ sim |
| PushT 2-D (gym-pusht) | [`robots/pusht_2d/`](https://github.com/OpenRAL/openral/tree/master/robots/pusht_2d/) | eval-only (`pymunk` 2-D rigid-body) | ✓ sim |

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
