# Robot manifests (`robots/`)

This directory holds one `RobotDescription` manifest (`robot.yaml`) per
embodiment — the normative joint / sensor / capability / safety / HAL
contract every other layer (rSkills, scenes, the eval registry, the HAL
itself) validates against. `python/sim/src/openral_sim`'s robots adapter
scans `robots/*/robot.yaml` at import time and auto-registers each one (no
hand-maintained registry tuple); override the search path with
`$OPENRAL_ROBOTS_DIR` for out-of-tree manifests.

21 robot directories exist today. Each carries a `robot.yaml` and, for most,
its own `README.md` with the manifest's specific provenance notes
(kinematic source, MJCF wiring, sim-vs-real caveats). `hal.real` on the
manifest is the source of truth for real-hardware support — `null`/absent
means the embodiment is sim-only (no HAL, no HIL tests, no physical rig).
`hal.sim` being `null` does **not** mean no sim support: several embodiments
(`aloha_agilex`, `gr1`, `pusht_2d`, `so101_follower`, `r1pro`) drive a sim
rollout through a scene-adapter's `fixed_robot=` binding or an out-of-process
sidecar instead of the generic `MujocoArmHAL.from_description` path. `widowx`
is different again: it drives a free-axis (multi-robot) SimplerEnv/ManiSkill3
scene with no `fixed_robot=` binding — the robot is selected by the scene
YAML's `robot_id:` field instead.

| Robot dir | Description | Embodiment tags | Sim support | Real-HW support | README |
| --- | --- | --- | --- | --- | --- |
| `aloha_agilex` | Bimanual 6-DoF AgileX PiPER arms + parallel grippers (RoboTwin 2.0's default embodiment; 14-DoF action space) | `aloha_agilex`, `aloha`, `lerobot` | yes — `fixed_robot="aloha_agilex"` scene adapter (SAPIEN, via the RoboTwin sidecar); no on-disk URDF/MJCF | no (`hal.real` null) | [README.md](aloha_agilex/README.md) |
| `aloha_bimanual` | Bimanual Trossen ALOHA (ViperX arms) | `aloha`, `lerobot` | yes — `AlohaMujocoHAL` (gym-aloha MJCF twin) | yes — `AlohaHAL` (Trossen `interbotix_xs_sdk`, not `ros2_control`) | [README.md](aloha_bimanual/README.md) |
| `anvil_openarm_v2` | Bimanual Anvil OpenARM 2.0 arms | `openarm`, `anvil_openarm_v2`, `anvil`, `bimanual` | yes — `AnvilOpenArmV2MujocoHAL` | no (`hal.real` null) | [README.md](anvil_openarm_v2/README.md) |
| `franka_panda` | 7-DoF collaborative arm (Franka Emika Panda); LIBERO's canonical embodiment | `franka_panda`, `franka`, `panda`, `libero` | yes — `FrankaPandaHAL` (mujoco_menagerie MJCF) | yes — `FrankaPandaRealHAL` (FCI) | [README.md](franka_panda/README.md) |
| `g1` | Unitree G1 humanoid, floating base | `g1`, `unitree_g1`, `humanoid`, `mobile_base` | yes — `G1MujocoHAL` (mujoco_menagerie MJCF) | no (`hal.real` null) | [README.md](g1/README.md) |
| `galaxea_a1` | 6-DoF manipulator (Galaxea A1), joint/gripper-space real-hardware manifest | `galaxea_a1` | no (`hal.sim` null) | yes — `GalaxeaA1HAL` | no README |
| `google_robot` | Google Everyday Robot mobile manipulator (discontinued 2023; coarse kinematic approximation) | `google_robot`, `everyday_robot`, `rt1`, `rt2`, `oxe` | no today — no scene currently binds it; the `simpler_env/google_robot_*` benchmark suite was removed because the upstream ManiSkill3 GoogleRobot envs are unregistered (`NameNotFound`) | no (`hal.real` null) | [README.md](google_robot/README.md) |
| `gr1` | Fourier GR-1 humanoid — arms + waist + Fourier dex hands (legs/head disabled) | `gr1`, `fourier_gr1`, `humanoid`, `robocasa` | yes — `fixed_robot="gr1"` scene adapter (RoboCasa GR1 Tabletop Tasks backend) | no (`hal.real` null) | no README |
| `h1` | Unitree H1 humanoid, bipedal floating base | `h1`, `unitree_h1`, `humanoid` | yes — `H1MujocoHAL` (mujoco_menagerie MJCF) | no (`hal.real` null) | [README.md](h1/README.md) |
| `openarm` | Enactic OpenArm v2 bimanual arms (manifest's own `name:` is `openarm_v2`, directory is `openarm`) | `openarm`, `openarm_v2`, `enactic`, `bimanual` | yes — `OpenArmMujocoHAL` | yes — `OpenArmRealHAL` | [README.md](openarm/README.md) |
| `panda_mobile` | Franka Panda 7-DoF arm on a 3-DoF holonomic base (RoboCasa's default robot composition; 2-D LiDAR SLAM) | `panda_mobile`, `robocasa`, `franka`, `panda`, `mobile_base` | yes — `PandaMobileHAL` (RoboCasa/robosuite backend) | no (`hal.real` null) | no README |
| `panda_mobile_vslam` | Lidar-less visual-SLAM twin of `panda_mobile` — localises/maps from two base-mounted RGB cameras via cuVSLAM instead of a `LaserScan` | `panda_mobile`, `robocasa`, `franka`, `panda`, `mobile_base` | yes — `PandaMobileHAL` (RoboCasa/robosuite backend) | no (`hal.real` null) | no README |
| `pusht_2d` | PushT 2-D pseudo-robot (lerobot PushT benchmark) | `pusht`, `lerobot` | yes — `fixed_robot="pusht_2d"` scene adapter (gym-pusht) | no (`hal.real` null) | [README.md](pusht_2d/README.md) |
| `r1pro` | Galaxea R1 Pro mobile bimanual manipulator — the official 2026 BEHAVIOR-1K OmniGibson evaluator embodiment | `r1pro`, `mobile_base` | yes — out-of-process OmniGibson/Isaac Sim sidecar (`tools/behavior_scene_sidecar.py`), 61-D state / 23-D action | no (`hal.real` null) | [README.md](r1pro/README.md) |
| `rizon4` | 7-DoF collaborative arm (Flexiv Rizon 4), no gripper | `rizon4`, `flexiv` | yes — `Rizon4MujocoHAL` (mujoco_menagerie MJCF) | no (`hal.real` null) | [README.md](rizon4/README.md) |
| `sawyer` | 7-DoF collaborative arm (Rethink Robotics Sawyer) | `sawyer`, `rethink` | no (`hal.sim` null) | yes — `SawyerRealHAL` (`intera_sdk`/`sawyer_robot` fork, BSD-3) | [README.md](sawyer/README.md) |
| `so100_follower` | 6-DoF follower arm (LeRobot SO-100), serial — not `ros2_control` | `so100_follower`, `lerobot` | no (`hal.sim` null) | yes — `SO100FollowerHAL` (serial, `/dev/ttyUSB0`) | [README.md](so100_follower/README.md) |
| `so101_follower` | 6-DoF follower arm (LeRobot SO-101), serial — shares `SO100FollowerHAL` | `so101_follower`, `lerobot` | yes — `fixed_robot="so101_follower"` scene adapter (`so101_box`, MuJoCo tabletop arena) | yes — `SO100FollowerHAL` (serial, `/dev/ttyUSB0`) | [README.md](so101_follower/README.md) |
| `ur10e` | 6-DoF collaborative arm (Universal Robots UR10e), no gripper | `ur10e`, `ur` | yes — `UR10eHAL` (mujoco_menagerie MJCF) | yes — `UR10eRealHAL` | [README.md](ur10e/README.md) |
| `ur5e` | 6-DoF collaborative arm (Universal Robots UR5e), no gripper | `ur5e`, `ur` | yes — `UR5eHAL` (mujoco_menagerie MJCF) | yes — `UR5eRealHAL` | [README.md](ur5e/README.md) |
| `widowx` | Trossen WidowX-250s 6-DoF arm (BridgeData V2 / SimplerEnv embodiment) | `widowx`, `widowx_250s`, `bridge`, `bridgedata_v2`, `oxe` | yes — SimplerEnv/ManiSkill3 scenes (`scenes/benchmark/widowx_carrot_on_plate.yaml`) | no (`hal.real` null) | [README.md](widowx/README.md) |

## Add a new robot

```bash
openral detect         # probe the host and write robots/<name>/robot.yaml directly
```

or author `robots/<name>/robot.yaml` by hand against the `RobotDescription`
schema (`python/core/src/openral_core/schemas.py`) — see any manifest above
for the shape (joints, `capabilities.embodiment_tags`, `safety`,
`hal.{sim,real}`, optional `sim:` MuJoCo wiring). A new manifest is
auto-registered on next import; no code change is needed to make it visible
to `openral sim list` / `openral rskill check` / the eval registry.

A manifest with `hal.real` set must declare every value the real path reads —
the schema refuses it otherwise, naming each gap: `action_spec.control_freq_hz`,
the kernel's `safety` thresholds, `safety.joint_state_staleness_limit_s` (the
HAL's and the runner's joint-state window, measured on the raw stream; measure
it with `tools/joint_state_staleness_probe.py --robot`. A runner that reads the
HAL's rate-limited `~/joint_states` republish gets two republish periods on top,
capped at 0.5 s), and the starting-pose approach
per joint type — `safety.starting_pose_max_joint_speed_rad_s` /
`starting_pose_tolerance_rad` for revolute joints, `..._m_s` / `..._m` when the
robot has a prismatic joint (the runner also caps each joint at its
`velocity_limit`). A value copied from another robot rather than measured on
this one is marked `provisional` in the manifest comment.

Robot manifests are at `schema_version: "0.2"`, and only `"0.2"` loads. There is
no migration path: a `"0.1"` manifest is refused with a `ROSConfigError`. To update
one, move `hal.parameters.defaults.staleness_limit_s` to
`safety.joint_state_staleness_limit_s`, measure and declare every real-hardware
safety field above on the rig, and set `schema_version: "0.2"`.
