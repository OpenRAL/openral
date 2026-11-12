# openral_hal_unitree

Real-hardware bringup (only) for the **Unitree hg humanoid family** — the G1 today; the
H2 uses the same NVIDIA hardware interface (`robot_variant: h2`) and joins by adding a
`config/h2/` and a row in `openral_hal.unitree_real._MODELS`.

The HAL lifecycle node is the generic [`openral_hal_node`](../openral_hal_node/README.md)
running `openral_hal.unitree_real:UnitreeHgRealHAL` (`hal.real` in
[`robots/g1/robot.yaml`](../../robots/g1/robot.yaml)). This package ships what that node
cannot: `launch/real_bringup.launch.py` (named by `hal.real_bringup`), the controller
configs under `config/<model>/`, and `deps.repos`.

> **Status: yellow.** Configs, launch and HAL are unit-tested against the pinned upstream
> sources and the G1 manifest; nothing here has run on a G1 yet. Treat the first
> bring-up as a validation run, on a gantry.

## Architecture: NVIDIA owns S0, OpenRAL proposes

On real hardware the legs are **not** commanded from Python. NVIDIA's whole-body ONNX
policy runs as a `ros2_control` controller in the C++ loop; OpenRAL's HAL sends it a base
velocity and drives the upper body (CLAUDE.md §3 "Python proposes; C++ disposes").

```text
Skill → Action → OpenRAL safety kernel (cpp/openral_safety_kernel) → UnitreeHgRealHAL
   JOINT_POSITION, upper body only ─► /upper_body_forward_joint_command_controller/joint_commands_trajectory
                                       isaac_ros_deploy_interfaces/msg/JointCommandTrajectory
   BODY_TWIST (vx, vy, wz)        ─► /inference_controller/cmd_vel     geometry_msgs/msg/Twist
                                          │
controller_manager @ 200 Hz ──────────────┼───────────────────────────────────────────────
   upper_body_forward_joint_command_controller ─┐   (writes <joint>/<if>_raw)
   inference_controller  (ONNX via Triton, 50 Hz)┤
                                                 ▼
   safety_controller: blend_ratio · velocity clamp · out-of-domain → freeze_controller
                                                 ▼
   unitree_hg_ros2_control/UnitreeHgSystemInterface: CRC32 · 500 ms state freshness ·
   non-finite refusal · unitree_sdk2 LowCmd over DDS → G1 motors
```

| What | Topic / interface | Type |
|---|---|---|
| Upper-body command (OpenRAL → S0 graph) | `/upper_body_forward_joint_command_controller/joint_commands_trajectory` | `isaac_ros_deploy_interfaces/msg/JointCommandTrajectory` |
| Base velocity (OpenRAL → policy) | `/inference_controller/cmd_vel` | `geometry_msgs/msg/Twist` (`linear.x`, `linear.y`, `angular.z` used) |
| Joint state | `/joint_states` (`joint_state_broadcaster`) | `sensor_msgs/msg/JointState` |
| IMU | `/imu_sensor_broadcaster/imu` | `sensor_msgs/msg/Imu` |
| Policy blend | `/safety_controller` param `blend_ratio` | double, 0.0–1.0 |

**Base height is not commandable.** OpenRAL's `Action` has no base-height field, and
upstream's only height-carrying converter (`command/body/velocity_height`, isaac_ros_deploy
f3aa6b89) hardcodes 0.7 m; the GR00T bundle fixes GR00T's height input at its runner's 0.74 m.
Both policies take the planar twist (vx, vy, wz) only.

**Joint ownership** is the selected policy's, read from its policy file by the HAL
(`openral_hal.unitree_s0_policy`) and by the launch alike. GR00T Decoupled WBC (default)
owns 12 leg joints + all three waist axes, leaving OpenRAL both 7-DoF arms; AGILE owns
12 legs + `waist_roll_joint` + `waist_pitch_joint`, leaving OpenRAL `waist_yaw_joint` too.
A `JOINT_POSITION` action that names (or, with `Action.joint_names=None`, implies) a
policy-owned joint is refused with `ROSCapabilityMismatch` before anything is published.
The HAL's `s0_policy` (manifest `hal.parameters.defaults.s0_policy`) and the launch's
`policy:=` default must name the same file (test-enforced for the defaults); override
both together. Consequence: a skill whose
whole-vector `starting_pose` ramp covers the legs cannot run on the real G1 until it is
expressed over the upper-body joints.

## Layout

```text
config/g1/controller_manager.yaml        policy-independent graph (29 DoF, no Dex3 hands)
config/g1/policies/gr00t_decoupled_wbc.yaml   GR00T WBC (DEFAULT): inference params + split + arm gains
config/g1/policies/agile_velocity.yaml   WBC-AGILE: inference_controller params + joint split
launch/real_bringup.launch.py            robot_model:=g1 policy:=gr00t_decoupled_wbc
../../tools/build_gr00t_wbc_leapp.py     builds the GR00T LEAPP bundle on the robot host
deps.repos                               isaac_ros_robots + isaac_ros_deploy at full SHAs
```

### S0 policy (swappable)

Each `config/<model>/policies/<policy>.yaml` holds exactly what differs between whole-body
policies: the `inference_controller` parameters (LEAPP config, decimation, command
sources and their message types, input timeout), the joint split (`s0_joints`, and the
upper-body controller's `joints`), and policy-specific arm gains. Selecting one is
`policy:=<stem>` (launch) + `s0_policy=<stem>` (HAL); the graph in
`controller_manager.yaml` does not change. A `status: placeholder` file is refused.

**GR00T Decoupled WBC is the default** — it won the G1 S0 bake-off on robustness (0/90
falls under aggressive arm motion + payload vs 3/90 for AGILE; 100 % recovery from 400 N
standing / 300 N walking pushes vs 12–68 %); AGILE tracks velocity better and stays
selectable. GR00T ships two plain ONNX MLPs (Walk, Balance; NVlabs/GR00T-WholeBodyControl
@ `1e1ec4f5`), not a LEAPP export, so NVIDIA's `InferenceController` cannot load them
directly. `tools/build_gr00t_wbc_leapp.py` composes them, with GR00T's runner logic, into
one LEAPP bundle the controller loads unchanged — no new controller, no Python in the
loop:

* 86-D observation per tick (`cmd·(2,2,0.5)`, height 0.74, torso rpy 0, pelvis angular
  velocity·0.5, projected gravity from the IMU quaternion, `q − q_default`, `dq·0.05`,
  last action), 6-frame history (previous 5 frames + last action are LEAPP
  `feedback_flow` tensors, zero-initialised as in GR00T's runner);
* Walk if `‖(vx, vy, wz)‖ > 0.05` else Balance (both evaluated, `Where`-selected);
* `q* = 0.25·a + default` for the 15 S0 joints, GR00T's `kps`/`kds` as `kp`/`kd` outputs.

`tests/unit/test_gr00t_wbc_leapp_bundle.py` runs the bundle — with its feedback looped the
way the controller loops it — against an independent port of GR00T's runner on the
original networks for 120 seeded ticks crossing the walk threshold both ways: max
|Δq*| < 1e-5. Build it on the robot host (downloads the pinned weights, SHA-256-checked;
they are NVIDIA Open Model License and never vendored here):

```bash
uv run python tools/build_gr00t_wbc_leapp.py   # → ~/.cache/openral/unitree/g1/gr00t_decoupled_wbc/
```

What the bundle cannot reproduce from the runner, stated plainly: the observation's
joint state comes from `UnitreeHgSystemInterface` (motor encoders) and the angular
velocity / orientation from the pelvis IMU through isaac_ros_deploy's state converters,
not from MuJoCo `qpos`/`qvel`; that mapping (frames, the IMU's `imu_in_pelvis` mount) is
the first thing to validate on hardware.

WBC-AGILE (`policy:=agile_velocity`): its LEAPP export (`agile_velocity.yaml` + ONNX)
ships in the `isaac_ros_agile_unitree_g1` data package, which is **not** in
`isaac_ros_robots` or `isaac_ros_deploy` — install it per NVIDIA's
`isaac_ros_deploy/tutorials/deploy_agile.html`, or pass
`inference_controller_config_path:=/abs/path/agile_velocity.yaml`. Its 14-joint split
follows upstream's own comment ("12 legs + waist roll + waist pitch. Matches the AGILE
action output set."); the Velocity-Height AGILE export in WBC-AGILE's own repository
commands only the 12 legs, so check the installed export's `target/joint/position`
element names against `s0_joints` before use. Its licence is not stated in the pinned
repos — check it before redistributing a deployment image.

## Install

Host: ROS 2 **Jazzy** (upstream also targets Lyrical; the hardware-interface CMake and
package manifests carry both), a CUDA GPU (Jetson Thor/Orin or x86 + NVIDIA dGPU), and the
**Isaac ROS apt repository** (release 5.0, per isaac_ros_robots' 2026-09-21 note).

```bash
vcs import ros2_ws/src < packages/openral_hal_unitree/deps.repos
rosdep install --from-paths ros2_ws/src --ignore-src -y   # resolves Isaac ROS apt keys
colcon build --packages-up-to unitree_hg_ros2_control unitree_g1_description \
    isaac_ros_deploy_ros2_control
just ros2-build                                            # this package
```

Build-time network fetches you should know about:

* `unitree_hg_ros2_control` FetchContent-downloads `unitree_sdk2` @ `21d0a3b2` and Cyclone
  DDS 0.10.2 @ `9995905b` (patched for malformed TypeLookup replies from the G1) — pinned.
* `unitree_g1_description` downloads `unitree_ros` **master** (zip) and `unitree_mujoco`
  **main** files — **unpinned upstream**; a rebuild can pick up different meshes/URDF.
  Mirror those artifacts if you need a reproducible image.

## Bring-up

0. **S0 bundle** (GR00T, the default): `uv run python tools/build_gr00t_wbc_leapp.py` once
   on the robot host.
1. **Network.** Wire the host NIC to the robot and give it `192.168.123.99/24` (robot:
   `192.168.123.161`; values from isaac_ros_robots' `setup_network.py`, which also adds the
   `239.0.0.0/8` multicast route DDS discovery needs). Export the NIC name:
   `export OPENRAL_UNITREE_NETWORK_INTERFACE=enp3s0` (default `eth0`). `deploy run`
   includes the bringup without arguments, so this env var is how the NIC reaches it.
2. **Support the robot.** Hang it on a gantry. Activating the hardware interface releases
   Unitree's built-in locomotion service (`MotionSwitcherClient::ReleaseMode`); from then
   on only this graph holds it up.
3. **Start.** Either the whole deploy (`openral deploy run --config <scene>` with
   `robot_id: g1` — it includes this launch through `hal.real_bringup`), or the graph
   alone to check wiring, under the OpenRAL venv interpreter (the launch imports
   `openral_hal`, which `/opt/ros`'s system Python cannot see):
   `.venv/bin/python "$(command -v ros2)" launch openral_hal_unitree real_bringup.launch.py`.
   Never both: two graphs
   double-publish `/joint_states` and `deploy run` refuses to start.
4. **Verify.** `ros2 control list_controllers` → broadcasters, `safety_controller`,
   `inference_controller`, `upper_body_forward_joint_command_controller` active,
   `freeze_controller` inactive. `ros2 topic hz /joint_states` ≈ 200 Hz.
5. **Blend the policy in.** It starts blended out (`blend_ratio: 0.0`) and nothing in
   OpenRAL raises it. With a velocity command flowing (the HAL publishes zero twist every
   tick), the operator runs `ros2 param set /safety_controller blend_ratio 1.0` (ramped at
   `max_blend_ratio_speed: 1.0`/s), then lowers the gantry.
   Under SROS2 ([ADR-0130](../../docs/reference/ros-graph-security.md#sros2-enclaves-adr-0130))
   only the operator enclave may set `/safety_controller` parameters: prefix the command
   with `ROS_SECURITY_ENCLAVE_OVERRIDE=/openral/operator`.

## E-stop, freeze and stale commands

| Event | What happens |
|---|---|
| `/openral/estop` | `UnitreeHgRealHAL.estop()` swaps `safety_controller`, `inference_controller`, `upper_body_forward_joint_command_controller` **out** and `freeze_controller` **in**, in **one** `switch_controller` request (BEST_EFFORT, confirmed via `list_controllers`; outcome in `/diagnostics` `downstream_stop`). Recovery: `RESTART_REQUIRED`. |
| Joint velocity out of domain | NVIDIA `SafetyController` (mean > 10 rad/s or max > 35 rad/s) performs the same swap itself. |
| OpenRAL stops publishing (crash, stall) | After 500 ms (`topic_input_timeout_ms`) the policy's output is invalidated and `SafetyController` holds the measured pose. Arms hold their last trajectory end. |
| Robot state stale > 500 ms | `UnitreeHgSystemInterface` read/write return ERROR → the hardware component leaves `active`. |

Why a swap and not OpenRAL's usual "deactivate the controllers": the Unitree hg hardware
interface sends **motor mode 0 (zero torque)** for every joint no controller claims
(`compose_motor_command`, isaac_ros_robots 520aa94f). Deactivating a standing humanoid's
controllers drops it. The swap leaves no control cycle with the joints unclaimed.

**`freeze_controller` is not a balance controller.** It holds the measured pose at kp 10 /
kd 1 (upstream's SDK-matched gains). A free-standing G1 can still fall after an e-stop or
an out-of-domain trip; a stale-input hold (kp 100 / kd 2 through `safety_controller`) is
stiffer but equally not balance. Run supported until this has been characterised on
hardware.

### Layering with OpenRAL's safety (nothing weakened)

1. OpenRAL's C++ safety kernel validates every `ActionChunk` against `robots/g1/robot.yaml`
   (joint limits, speed factor, `max_base_linear_speed_m_s: 0.5`,
   `max_base_angular_speed_rad_s: 1.0`, collision) **before** the HAL sees it — unchanged.
2. The HAL refuses leg targets, non-planar or non-finite twists, wrong dimensions.
3. NVIDIA `SafetyController` then blends, rate-limits (arms ≤ 1.2 rad/s target motion) and
   watches for out-of-domain velocities; the hardware interface checks finiteness, CRC and
   state freshness. These are **additional** layers below OpenRAL's — none replaces or
   relaxes the kernel, and no OpenRAL flag disables any of them.

Neither policy's trained velocity-command range is stated in its pinned repo; the
kernel's BODY_TWIST bound is still the walking twin's (0.5 m/s, 1.0 rad/s). Tighten it, not
loosen it, if a policy's range turns out smaller.

## Rates

| Loop | Rate | Source |
|---|---|---|
| `controller_manager` (hardware read/write, safety, forward controller) | 200 Hz | `config/g1/controller_manager.yaml` (upstream's validated value) |
| S0 policy (`inference_controller`, decimation 4) | 50 Hz | policy file (GR00T's own rate: dt 0.005 × 4) |
| OpenRAL runner / HAL (`action_spec.control_freq_hz`) | 50 Hz | `robots/g1/robot.yaml` |

CLAUDE.md §3 sketches S0 at 500–1000 Hz. **This S0 runs at 200 Hz** (policy at 50 Hz),
because that is the rate NVIDIA validated for this hardware interface and policy; raising
it is unvalidated and not done here. The 200 Hz loop is still C++ and bounded (§1.5).

## Licences

| Component | Licence | Notes |
|---|---|---|
| This package, `openral_hal.unitree_real` | Apache-2.0 | |
| `isaac_ros_robots` @ 520aa94f | Apache-2.0 | hardware interface, G1 description |
| `isaac_ros_deploy` @ f3aa6b89 | Apache-2.0 | controllers, interfaces |
| Isaac ROS apt deps (`isaac_ros_triton`, `isaac_ros_tensor_msgs`, `cuda_buffer`, …) | Apache-2.0 | from the Isaac ROS apt repo |
| `unitree_sdk2` @ 21d0a3b2 (fetched by CMake) | BSD-3-Clause wrapper, **prebuilt closed core** (`libunitree_sdk2.a`) | why the manifest is `sdk_kind: closed_with_api` |
| Cyclone DDS 0.10.2 (fetched by CMake) | EPL-2.0 OR EDL-1.0 | dynamically linked runtime |
| Triton Inference Server (`triton-server` rosdep key) | BSD-3-Clause | ONNX runtime for the policy |
| libtorch (`python3-torch-pip-shim`) | BSD-3-Clause (PyTorch); pip wheels bundle NVIDIA CUDA libraries under NVIDIA's EULA | |
| CUDA / TensorRT runtime | NVIDIA proprietary EULA | **GPU mandatory** for the Isaac ROS stack |
| GR00T-WholeBodyControl @ 1e1ec4f5 Walk/Balance ONNX (default S0) | **NVIDIA Open Model License** (weights; code Apache-2.0) | commercial use permitted under its terms; fetched + SHA-256-checked by `tools/build_gr00t_wbc_leapp.py`, never vendored |
| `isaac_ros_agile_unitree_g1` (AGILE LEAPP export) | **not stated in the pinned repos** | verify before redistribution |

No GPL. Nothing here is vendored: all third-party code is fetched at the pinned SHAs by
the operator, as for `openral_hal_openarm`.
