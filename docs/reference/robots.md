# Robot Descriptions

Every embodiment is a typed `RobotDescription` manifest under `robots/<robot_id>/robot.yaml`. `openral_core` validates it; the HAL adapter, the rSkill loader (embodiment-tag check) and `openral sim run` consume it. This page is the one inventory of all 27 manifests. Directory conventions and how to add a robot are in [`robots/README.md`](https://github.com/OpenRAL/openral/blob/master/robots/README.md).

## How to read the table

`openral deploy sim|run` starts one generic ROS 2 HAL lifecycle node,
[`openral_hal_node`](https://github.com/OpenRAL/openral/tree/master/packages/openral_hal_node/),
named `openral_hal_<robot_id>`.

- **Real HAL**: the manifest's `hal.real`. `none` means the robot has no hardware path.
- **Sim HAL**: `hal.sim` when set; otherwise "derived `MujocoArmHAL`" means the manifest has a `sim:` block and `openral_hal.build_hal` builds `MujocoArmHAL.from_description` from it (no per-robot class is named). `none` means the robot runs only through an eval environment, which starts no lifecycle node.
- **Deploy scenes**: `scenes/deploy/*.yaml` for `openral deploy sim|run`. A `*_twin` scene is a bare MuJoCo twin of the robot.

| Robot dir | Robot | Real HAL | Sim HAL | Deploy scenes | Status |
|---|---|---|---|---|---|
| [`aloha_agilex`](https://github.com/OpenRAL/openral/tree/master/robots/aloha_agilex/) | ALOHA-AgileX (RoboTwin 2.0) | none | derived `MujocoArmHAL` (bare twin, `robotwin:aloha_agilex`); RoboTwin SAPIEN sidecar for eval | `aloha_agilex_twin` | sim |
| [`aloha_bimanual`](https://github.com/OpenRAL/openral/tree/master/robots/aloha_bimanual/) | ALOHA bimanual (gym-aloha) | `AlohaHAL` (Interbotix XS; `HALBase`, not ros2_control) | derived `MujocoArmHAL` (gym-aloha MJCF) | - | sim; real HAL never validated (no rig, #250): speaks the `xs_sdk` contract (`JointGroupCommand` per arm, `JointSingleCommand` per gripper) through the production `InterbotixXSTransport`, with stock-Interbotix defaults not yet confirmed on a cell |
| [`anvil_openarm_v2`](https://github.com/OpenRAL/openral/tree/master/robots/anvil_openarm_v2/) | Anvil OpenARM 2.0 bimanual | none | derived `MujocoArmHAL` | - | sim |
| [`droid_franka_robotiq`](https://github.com/OpenRAL/openral/tree/master/robots/droid_franka_robotiq/) | DROID Franka + Robotiq 2F-85 | none | none (Isaac Sim only, composed URDF) | `isaac_arena_droid_banana_in_bowl` | Isaac Sim; target of `pi05-droid-jointpos-arena` |
| [`franka_panda`](https://github.com/OpenRAL/openral/tree/master/robots/franka_panda/) | Franka Panda | `FrankaPandaRealHAL` (`franka_ros2` / libfranka FCI) | derived `MujocoArmHAL` | `isaac_franka_urdf`; LIBERO scenes | sim; real HAL never validated (no rig) |
| [`g1`](https://github.com/OpenRAL/openral/tree/master/robots/g1/) | Unitree G1 humanoid | `UnitreeHgRealHAL` (NVIDIA ros2_control stack, C++ S0 WBC) | `G1MujocoHAL` (glide fallback; sim S0 walker / GR00T WBC) | `g1_vln` | sim; real HAL untested on hardware |
| [`galaxea_a1`](https://github.com/OpenRAL/openral/tree/master/robots/galaxea_a1/) | Galaxea A1 | `GalaxeaA1HAL` (operator-provided ROS 1 SDK sidecar) | derived `MujocoArmHAL` (twin from the vendor URDF) | `galaxea_a1_twin`, `galaxea_a1_bench` | sim twin; real: observation / hold / joint / gripper checked, C++ kernel HIL |
| [`google_robot`](https://github.com/OpenRAL/openral/tree/master/robots/google_robot/) | Google Robot (SimplerEnv) | none | none (eval-only SimplerEnv bridge env) | - | eval |
| [`gr1`](https://github.com/OpenRAL/openral/tree/master/robots/gr1/) | Fourier GR1 | none | none (eval-only, RoboCasa GR1 fork) | - | eval |
| [`h1`](https://github.com/OpenRAL/openral/tree/master/robots/h1/) | Unitree H1 humanoid | none | `H1MujocoHAL` (software PD loop, no S0) | - | sim |
| [`h2`](https://github.com/OpenRAL/openral/tree/master/robots/h2/) | Unitree H2 humanoid | `UnitreeHgRealHAL` (same stack as G1) | `H2MujocoHAL` (pinned-base glide + PD; `unitree:h2` MJCF) | `h2_walk` | sim; real HAL untested on hardware |
| [`makermods_metal`](https://github.com/OpenRAL/openral/tree/master/robots/makermods_metal/) | MakerMods Metal arm | none (Damiao CAN arm with no brakes; a real HAL needs a reviewed stop behaviour) | derived `MujocoArmHAL` (`makermods:metal`, derived at runtime from a pinned upstream clone) | - | sim |
| [`makermods_metal_station`](https://github.com/OpenRAL/openral/tree/master/robots/makermods_metal_station/) | MakerMods Metal Arm Station | none | derived `MujocoArmHAL` (`makermods:metal_station`; arm spacing and camera mount provisional) | - | sim |
| [`openamrobot_v2`](https://github.com/OpenRAL/openral/tree/master/robots/openamrobot_v2/) | OpenAMRobot 2.0 mobile bimanual | `ComposedHAL` (`OpenArmRealHAL` arms + `PlatformBaseHAL` odometry + held lift) | `OpenAMRobotV2MujocoHAL` (planar base, 20-DoF) | `openamrobot_v2_*`, `isaac_openamrobot_v2_warehouse` | sim (MuJoCo + Isaac Sim); real never run |
| [`openarm`](https://github.com/OpenRAL/openral/tree/master/robots/openarm/) | Enactic OpenArm v2 bimanual | `OpenArmRealHAL` (ros2_control over Damiao CAN FD) | derived `MujocoArmHAL` | `openarm_tabletop`, `openarm_bench`, `openarm_real_*`, `openarm_zed_octomap`, `isaac_openarm_warehouse` | sim; CAN bus + motor round trip pass on the wired rig; `send_action` motion gate has no recorded run |
| [`panda_mobile`](https://github.com/OpenRAL/openral/tree/master/robots/panda_mobile/) | Panda mobile (RoboCasa kitchen) | none | `PandaMobileHAL` (RoboCasa / robosuite) | `robocasa_*`, `isaac_panda_mobile_*` | sim |
| [`panda_mobile_vslam`](https://github.com/OpenRAL/openral/tree/master/robots/panda_mobile_vslam/) | Panda mobile, visual-SLAM twin | none | `PandaMobileHAL`; cuVSLAM from two base RGB cameras instead of a LaserScan | `robocasa_vslam*` | sim |
| [`pusht_2d`](https://github.com/OpenRAL/openral/tree/master/robots/pusht_2d/) | PushT 2-D (gym-pusht) | none | none (eval-only `pymunk`) | - | eval |
| [`r1pro`](https://github.com/OpenRAL/openral/tree/master/robots/r1pro/) | Galaxea R1 Pro | none | derived `MujocoArmHAL` (twin on the `behavior:r1pro` MJCF); BEHAVIOR/OmniGibson sidecar for eval | `r1pro_twin`, `behavior_r1pro` | sim |
| [`rizon4`](https://github.com/OpenRAL/openral/tree/master/robots/rizon4/) | Flexiv Rizon 4 | none | derived `MujocoArmHAL` | - | sim |
| [`rizon_dual`](https://github.com/OpenRAL/openral/tree/master/robots/rizon_dual/) | Flexiv Rizon 4s dual (q5090 rig) | `GroupedRosControlHAL` (config-only; one `JointTrajectoryController` per arm, `flexiv_hardware`) | none (real only, 14-DoF arms) | `rizon_dual_bench` | real; safety thresholds provisional |
| [`sawyer`](https://github.com/OpenRAL/openral/tree/master/robots/sawyer/) | Rethink Sawyer | `SawyerRealHAL` (`intera_sdk` lineage) | none (eval-only, MetaWorld) | - | sim; real HAL never validated (no rig) |
| [`so100_follower`](https://github.com/OpenRAL/openral/tree/master/robots/so100_follower/) | SO-100 follower arm | `SO100FollowerHAL` (serial) | derived `MujocoArmHAL` (`rd:so_arm100_mj_description`) | - | sim + real |
| [`so101_follower`](https://github.com/OpenRAL/openral/tree/master/robots/so101_follower/) | SO-101 follower arm | `SO100FollowerHAL` (serial) | derived `MujocoArmHAL` | `so101_box`, `so101_bench`, `isaac_leisaac_pick_orange` | sim + real |
| [`ur10e`](https://github.com/OpenRAL/openral/tree/master/robots/ur10e/) | UR10e | `UR10eRealHAL` (`ur_robot_driver`, URCap/RTDE) | derived `MujocoArmHAL` | - | sim; real HAL never run (no `lab-ur10e` runner) |
| [`ur5e`](https://github.com/OpenRAL/openral/tree/master/robots/ur5e/) | UR5e | `UR5eRealHAL` (`ur_robot_driver`, URCap/RTDE) | derived `MujocoArmHAL` | - | sim; real HAL never run (no `lab-ur5e` runner) |
| [`widowx`](https://github.com/OpenRAL/openral/tree/master/robots/widowx/) | WidowX-250s (SimplerEnv) | none | derived `MujocoArmHAL` (Menagerie MJCF twin); SimplerEnv `bridge_orig` for eval | `widowx_twin` | sim |

A real HAL marked "never validated" or "never run" exists in code and unit tests but has not driven a physical robot; HIL tests for those need a `lab-<robot>` runner that does not exist.

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
  # ... 5-DoF arm + 1-DoF gripper
```

## VLA compatibility

[vla_compatibility.md](vla_compatibility.md) lists which VLA checkpoints run on which robot and simulator, with observation/action dimensions and normalisation notes.
