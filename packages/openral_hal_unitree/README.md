# openral_hal_unitree

Real-hardware bringup for the **Unitree hg humanoid family** — the **G1** (29 DoF)
and the **H2** (31 DoF), both on NVIDIA's `unitree_hg_ros2_control` hardware interface
(`robot_variant: g1` / `h2`).

The HAL lifecycle node is the generic [`openral_hal_node`](../openral_hal_node/README.md)
running `openral_hal.unitree_real:UnitreeHgRealHAL` (`hal.real` in
[`robots/g1/robot.yaml`](../../robots/g1/robot.yaml) and
[`robots/h2/robot.yaml`](../../robots/h2/robot.yaml)). This package ships what that node
cannot: `launch/real_bringup.launch.py` (named by `hal.real_bringup`), the controller
configs under `config/<model>/`, the H2's real-hardware xacro (`urdf/`; NVIDIA ships none),
and `deps.repos`.

Simulation is separate and needs none of this package: `G1MujocoHAL` and `H2MujocoHAL`
(`openral_hal.g1` / `openral_hal.h2`) are pure Python MuJoCo twins. With
`s0_controller` set (G1 `gr00t_wbc`; H2 `agile_velocity`) the twin runs the same S0
policy on a gravity-on free base and a `BODY_TWIST` is its velocity command; the default
`none` is the kinematic glide. For example `uv run openral deploy sim --config
scenes/deploy/h2_walk.yaml` (downloads the H2 policy first; see its header).

> **Status: yellow.** Configs, launch and HAL are unit-tested against the pinned upstream
> sources and the G1 / H2 manifests; nothing here has run on a G1 or an H2 yet. Treat the
> first bring-up of each as a validation run, on a gantry. The H2 has further open
> hardware gates — see [H2](#h2).

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
   safety_controller: blend_ratio · velocity clamp · joint-limit clamp · out-of-domain → freeze_controller
      (NVIDIA's SafetyController through OpenRAL's GuardedSafetyController, see below)
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
| Policy blend | `/safety_controller` param `blend_ratio` | double, 0.0–1.0 (`max_blend_ratio_speed` is fixed at configure) |

**Base height is not commandable.** OpenRAL's `Action` has no base-height field, and
upstream's only height-carrying converter (`command/body/velocity_height`, isaac_ros_deploy
v4.6-0 3e84f85a) hardcodes 0.7 m; the GR00T bundle fixes GR00T's height input at its runner's 0.74 m.
Both policies take the planar twist (vx, vy, wz) only.

**Joint ownership** is the selected policy's, read from its policy file by the HAL
(`openral_hal.unitree_s0_policy`) and by the launch alike. GR00T Decoupled WBC (default)
owns 12 leg joints + all three waist axes, leaving OpenRAL both 7-DoF arms; AGILE owns
12 legs + `waist_roll_joint` + `waist_pitch_joint`, leaving OpenRAL `waist_yaw_joint` too.
A `JOINT_POSITION` action that names (or, with `Action.joint_names=None`, implies) a
policy-owned joint is refused with `ROSCapabilityMismatch` before anything is published.
The HAL's `s0_policy` (manifest `hal.parameters.defaults.s0_policy`) and the launch's
`policy:=` default must name the same file (test-enforced for the defaults); override
both together. On the H2, WBC-AGILE owns the 12 legs + waist roll/pitch, leaving OpenRAL
waist yaw, both 7-DoF arms and the head (17 joints). Consequence: a skill whose
whole-vector `starting_pose` ramp covers the legs cannot run on the real robot until it is
expressed over the upper-body joints.

## Layout

```text
config/g1/controller_manager.yaml        policy-independent graph (29 DoF, no Dex3 hands)
config/g1/policies/gr00t_decoupled_wbc.yaml   GR00T WBC (DEFAULT): inference params + split + arm gains
config/g1/policies/agile_velocity.yaml   G1 WBC-AGILE: inference_controller params + joint split
config/h2/controller_manager.yaml        the same graph for the H2 (31 DoF, SDK motor order)
config/h2/policies/agile_velocity.yaml   H2 WBC-AGILE (DEFAULT): inference params + split + gains + Hub pins
urdf/h2_real_hardware.urdf.xacro         H2 robot_description: pinned unitree_ros H2.urdf + ros2_control block
launch/real_bringup.launch.py            robot_model:=g1|h2 [policy:=<stem>] [network_interface:=<nic>]
                                         [inference_controller_config_path:=<abs yaml>]
../../tools/build_gr00t_wbc_leapp.py     builds the GR00T LEAPP bundle on the robot host
include/, src/guarded_{safety,inference}_controller.cpp   the guarded controller plugins
test/test_guarded_*_controller.cpp       their gtests against the real upstream controllers
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

**H2: WBC-AGILE** (`config/h2/policies/agile_velocity.yaml`, the only H2 policy). NVIDIA
publishes the export on the Hub — `nvidia/agile_wbc_unitree_h2` @ `222da1df`, LEAPP 0.5.2
(`Velocity-H2-History-v0`), NVIDIA Open Model License, trained on the **H2 Plus**. 50 Hz;
the action is 14 absolute joint targets (12 legs + waist roll/pitch) plus per-joint kp/kd;
inputs are base angular velocity and orientation, the velocity command and all 31 joints
by name. The policy file pins the revision and every file's SHA-256 (`hf_repo`,
`hf_revision`, `sha256`), and the launch re-checks them each time it resolves the bundle:

```bash
hf download nvidia/agile_wbc_unitree_h2 --revision 222da1dfecdb708fb77a64096451ce162feec2da \
    --local-dir ~/.cache/openral/unitree/h2/agile_velocity
```

The upper body runs at the gains the policy was trained with (the export's default
stiffness/damping — not measured H2 motor gains). `tests/unit/test_h2_agile_velocity_rollout.py`
drives the ONNX as the controller does (feedback looped, initial values from the
safetensors) with the state held fixed: finite targets on exactly the 14 S0 joints, a live
velocity input. In that open loop it asks for a left ankle roll of −0.371 rad against
`H2.urdf`'s −0.349 rad limit — see [H2](#h2).

## Install

Host: ROS 2 **Jazzy**, a CUDA GPU (Jetson Thor/Orin or x86 + NVIDIA dGPU), and the
**Isaac ROS apt repository, release 4** (its `ros-jazzy-*` builds; release 5.0 ships only
`ros-lyrical-*`).

`deps.repos` pins `isaac_ros_deploy` to tag **v4.6-0** (`3e84f85a`, the source of the
release-4 `ros-jazzy-isaac-ros-deploy-*` 4.6.0 debs), not its HEAD: v5.0-0 (`f3aa6b89`)
needs ROS 2 Lyrical (`cuda_buffer` from `rosidl_buffer_backends`, `urdf/model.hpp`) and does
not build on Jazzy. v4.6 lacks v5.0's stale-input watchdog; OpenRAL's guarded
controllers (below) restore it. `isaac_ros_robots` stays at `520aa94f` — release-4 has no
`unitree_hg_ros2_control`.

```bash
vcs import ros2_ws/src < packages/openral_hal_unitree/deps.repos
rosdep install --from-paths ros2_ws/src --ignore-src -y   # resolves Isaac ROS apt keys
colcon build --packages-up-to unitree_hg_ros2_control unitree_g1_description \
    isaac_ros_deploy_ros2_control
just ros2-build                                            # this package
```

The H2 needs no NVIDIA description package: the launch renders `urdf/h2_real_hardware.urdf.xacro`
around the pinned `unitree_ros` `H2.urdf`, which `openral_hal._unitree_h2_assets.ensure_h2_urdf()`
fetches (SHA-256-checked) into `~/.cache/openral/unitree_h2/` on first use.

Build this package with the NVIDIA workspace sourced: its `GuardedSafetyController` and
`GuardedInferenceController` plugins are compiled only when CMake finds
`isaac_ros_deploy_ros2_control`. Built without it, the bringup fails to load
`safety_controller` / `inference_controller` — it never falls back to NVIDIA's unguarded
classes. On a robot host make that a build error instead:

```bash
colcon build --packages-select openral_hal_unitree \
    --cmake-args -DOPENRAL_REQUIRE_GUARDED_SAFETY_CONTROLLER=ON
```

`isaac_deploy_core`'s exported CMake config fetches a Triton tarball whenever a dependent
package configures. On aarch64 that is the public `tritonserver2.60.0-agx.tar`; on x86_64
the default URL is NVIDIA-internal, so pass the tarball you built `isaac_deploy_core` with:
`-DTRITON_SERVER_TARBALL_URL=file:///path/to/tritonserver.tar.gz`.

The guards' gtests (`colcon test --packages-select openral_hal_unitree`) run NVIDIA's real
controllers; the policy tests also need the GR00T bundle
(`OPENRAL_G1_LEAPP_CONFIG=<bundle_dir>/gr00t_wbc_g1.yaml`) and Triton's ONNX Runtime backend
(`TRITON_BACKEND_DIRECTORY`, from the `triton-server` deb's tarball), and skip without the
bundle.

Build-time network fetches you should know about:

* `unitree_hg_ros2_control` FetchContent-downloads `unitree_sdk2` @ `21d0a3b2` and Cyclone
  DDS 0.10.2 @ `9995905b` (patched for malformed TypeLookup replies from the G1) — pinned.
* `unitree_g1_description` downloads `unitree_ros` **master** (zip) and `unitree_mujoco`
  **main** files — **unpinned upstream**; a rebuild can pick up different meshes/URDF.
  Mirror those artifacts if you need a reproducible image.

## Bring-up

0. **S0 bundle**, once on the robot host — G1 (GR00T, the default):
   `uv run python tools/build_gr00t_wbc_leapp.py`; H2: the `hf download` above.
1. **Network.** Wire the host NIC to the robot and give it `192.168.123.99/24` (robot:
   `192.168.123.161`; values from isaac_ros_robots' `setup_network.py`, which also adds the
   `239.0.0.0/8` multicast route DDS discovery needs). Export the NIC name:
   `export OPENRAL_UNITREE_NETWORK_INTERFACE=enp3s0` (default `eth0`). `deploy run`
   includes the bringup without arguments, so this env var is how the NIC reaches it.
2. **Support the robot.** Hang it on a gantry. Activating the hardware interface releases
   Unitree's built-in locomotion service (`MotionSwitcherClient::ReleaseMode`); from then
   on only this graph holds it up.
3. **Start.** Either the whole deploy (`openral deploy run --config <scene>` with
   `robot_id: g1` or `h2` — it includes this launch through `hal.real_bringup`, and the
   launch takes `robot_model` and the default `policy` from that manifest's
   `hal.parameters.defaults`, refusing a contradicting `robot_model:=`), or the graph
   alone to check wiring, under the OpenRAL venv interpreter (the launch imports
   `openral_hal`, which `/opt/ros`'s system Python cannot see). A hand run must name the
   model — the G1 and H2 joint tables are not interchangeable, so there is no default:
   `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST .venv/bin/python "$(command -v ros2)" launch openral_hal_unitree real_bringup.launch.py robot_model:=g1`
   (it refuses an unset or `SUBNET` range — what `setup.bash` exports — unless
   `OPENRAL_ALLOW_NETWORK_ROS=1` under SROS2 Enforce, ADR-0130; see `docs/reference/ros-graph-security.md`).
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

### Blend guard (`GuardedSafetyController`)

NVIDIA's `SafetyController` (isaac_ros_deploy v4.6-0 `3e84f85a`) declares `blend_ratio` and
`max_blend_ratio_speed` writable and applies a runtime set on the next tick, and its
`on_activate` snaps the applied blend to its target with no ramp. `safety_controller`
therefore runs as `openral_hal_unitree/GuardedSafetyController`, a subclass that touches
nothing on the hot path and:

| Step | Guard |
|---|---|
| configure | refuses `max_blend_ratio_speed` outside (0, 1.0] /s — faster blends the policy in in under a second, 0 freezes the blend so it can no longer be blended **out** |
| runtime parameter set | refuses every change to `max_blend_ratio_speed` (edit the YAML and relaunch) |
| activate | **resets** `blend_ratio` to 0.0 first (WARN in the log), so re-arming the chain after an e-stop with the policy still blended in starts it blended out instead of applying it at once. It never refuses: a strict `switch_controller` whose activation fails keeps its deactivations, so re-arming from `freeze_controller` would leave every joint unclaimed — zero torque. |

`blend_ratio` stays writable: it is the operator's blend-in, and a process that can
reach `/safety_controller`'s parameter services can still blend the policy in, at the
same ≤ 1 /s ramp. Who can reach them is a graph-access question — host confinement today,
SROS2 enclaves next (`docs/reference/ros-graph-security.md`).

### Stale-input watchdog (`GuardedInferenceController` + hold)

isaac_ros_deploy grew a stale-input watchdog in v5.0 (`topic_input_timeout_ms`,
`topic_input_sources`); the pinned Jazzy release, v4.6, has none, and accepts those keys
in a policy file without a word — a crashed OpenRAL side would leave the G1 walking on its
last `cmd_vel`. On a NaN position reference v4.6's `SafetyController` also re-reads the
measured pose every tick, so a blended-in joint has no restoring force. OpenRAL restores
v5.0's behaviour:

| Piece | What it does |
|---|---|
| `inference_controller` runs as `openral_hal_unitree/GuardedInferenceController` | declares both keys. Refuses to configure on a negative, non-finite or non-double timeout, or a `topic_input_sources` entry v4.6 does not read from a topic. With a timeout, a topic input with no message since activation, or none within the timeout (measured on the controller manager's clock), sets **every** policy command interface to NaN and skips the policy step. |
| `safety_controller`'s hold (`GuardedSafetyController`) | while **any** position reference is NaN, commands every joint to the measured pose captured at the first invalid tick, fully weighted (blend input 1.0) — v5.0's `stale_position_hold`. Released as soon as every reference is finite again. |
| launch | `require_watchdog_enforced` refuses a policy file without a positive float `topic_input_timeout_ms`, and any `type` for `inference_controller` other than the guarded one, wherever a params file sets it (the manager's block, the controller's own block, namespaced or `/**` keys) — nothing is silently ignored. |

Measured against NVIDIA's real v4.6 controllers and the GR00T bundle (gtests): upstream
v4.6 keeps commanding the policy 1000 ms after the last velocity command (timeout 200 ms in
the test); the guarded controller invalidates it 201 ms after.

Same as v5.0, and worth knowing before a free-standing run: the hold covers **every**
joint when any reference is NaN (a bad arm reference holds the legs too), leaving it steps
straight back to the blended policy target (legs have no `interpolate_max_velocity`), and
the policy resumes on the history it had at stale entry. **The heartbeat is per
`send_action`:** the HAL publishes `cmd_vel` only when an action arrives, so with no skill
running for 500 ms the policy is invalidated and the robot is held stiff, not balanced.
Whether an idle-but-alive OpenRAL should keep a zero-twist heartbeat is an open
Safety-WG decision (hazard log Entry 090).

## E-stop, freeze and stale commands

| Event | What happens |
|---|---|
| `/openral/estop` | `UnitreeHgRealHAL.estop()` swaps `safety_controller`, `inference_controller`, `upper_body_forward_joint_command_controller` **out** and `freeze_controller` **in**, in **one** `switch_controller` request (**STRICT**, confirmed via `list_controllers`; outcome in `/diagnostics` `downstream_stop`). `freeze_controller` cannot come in *before* the chain goes out — they claim the same command interfaces and `controller_manager` refuses it. A BEST_EFFORT swap would drop an activation that cannot happen (freeze not `inactive`) yet apply the deactivations, leaving every joint at zero torque; STRICT refuses the whole request instead and the chain keeps the joints. So the HAL first configures an `unconfigured` freeze controller (advisory pre-check, ≤ 0.2 s; a fault in it is logged and the swap goes ahead), swaps out only the chain members still active if part of the chain is already out, and, if `on_activate` fails inside the switch (no rollback in the manager), activates freeze alone at once (CRITICAL `hal.estop.freeze_retry`). The whole sequence is bounded by one `stop_timeout_s` (5 s). Recovery: `RESTART_REQUIRED`. |
| Joint velocity out of domain | NVIDIA `SafetyController` (mean > 10 rad/s or max > 35 rad/s) performs the same swap itself. |
| OpenRAL stops publishing (crash, stall) | After 500 ms (`topic_input_timeout_ms`, enforced by `GuardedInferenceController`) the policy's output is invalidated and `safety_controller` holds every joint at the measured pose captured at that moment (also before the first velocity command after activation). |
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
3. NVIDIA `SafetyController` (through `GuardedSafetyController`) then blends, rate-limits (arms ≤ 1.2 rad/s target motion) and
   watches for out-of-domain velocities; the guard then **clamps every joint's position
   command to the manifest's limits** (`joint_position_limits` in `config/<model>/
   controller_manager.yaml`, pinned to the manifest by the bring-up test; hazard-log
   Entry 105 C1). Nothing else bounds an S0 target: the kernel never sees one, upstream's
   `SafetyController` does not range-check positions, and the hardware interface checks only
   finiteness, CRC and state freshness. These are **additional** layers below OpenRAL's — none replaces or
   relaxes the kernel, and no OpenRAL flag disables any of them.

Neither policy's trained velocity-command range is stated in its pinned repo; the
kernel's BODY_TWIST bound is still the walking twin's (0.5 m/s, 1.0 rad/s). Tighten it, not
loosen it, if a policy's range turns out smaller.

## H2

Same graph, HAL and e-stop as the G1, with these differences:

* **Description.** NVIDIA's isaac_ros_robots @ `520aa94f` supports the H2 only in the
  hardware interface (`h2_body_joint_index_table`, a transcription of unitree_sdk2's
  `H2JointIndex`). `urdf/h2_real_hardware.urdf.xacro` adds upstream's ros2_control block
  (same plugin, same five command / three state interfaces per joint, `robot_variant: h2`,
  the `imu` sensor) to the pinned `H2.urdf`. Joint order = that table = SDK motor order =
  `robots/h2/robot.yaml`: ankle **roll before pitch** (the G1 is pitch-then-roll), head
  pitch/yaw last (motors 29–30). `test_unitree_real_bringup_config.py` pins the xacro, the
  three whole-body controllers and the manifest to one list, renders the xacro with `xacro`,
  and — with `$OPENRAL_ISAAC_ROS_ROBOTS_SRC` pointing at an isaac_ros_robots checkout —
  checks names and motor indices against upstream's table.
* **Gains.** `freeze_controller` (10 / 1) and `safety_controller` (100 / 2) are the G1's
  values, unchanged — no H2 value has been measured. On a ~70 kg robot with 360 N·m hips
  the freeze hold is further still from balance.
* **Hardware gates still open** (hazard-log Entry 074 gates 1–4 and Entry 105): the robot stays on a gantry until freeze is
  characterised; the IMU frame (`imu_in_pelvis`) is checked on hardware; the S0 loop runs at
  200 Hz (policy 50 Hz); H2 gains and safety fields are measured on the gantry (C2); wrist
  pitch stays at ±0.576 rad or tighter (C3); the unit's variant and mass are confirmed, since
  the checkpoint was trained on the H2 Plus (C4). C1 (joint-limit clamping of S0 targets) is
  met by `GuardedSafetyController` above, for the G1 too.

## Rates

| Loop | Rate | Source |
|---|---|---|
| `controller_manager` (hardware read/write, safety, forward controller) | 200 Hz | `config/<model>/controller_manager.yaml` (upstream's validated G1 value; the H2 reuses it, unvalidated) |
| S0 policy (`inference_controller`, decimation 4) | 50 Hz | policy file (GR00T's own rate: dt 0.005 × 4; H2 AGILE: the export's `frequency`) |
| OpenRAL runner / HAL (`action_spec.control_freq_hz`) | 50 Hz | `robots/g1/robot.yaml`, `robots/h2/robot.yaml` |

CLAUDE.md §3 sketches S0 at 500–1000 Hz. **This S0 runs at 200 Hz** (policy at 50 Hz),
because that is the rate NVIDIA validated for this hardware interface and policy; raising
it is unvalidated and not done here. The 200 Hz loop is still C++ and bounded (§1.5).

## Licences

| Component | Licence | Notes |
|---|---|---|
| This package, `openral_hal.unitree_real` | Apache-2.0 | |
| `isaac_ros_robots` @ 520aa94f | Apache-2.0 | hardware interface, G1 description |
| `isaac_ros_deploy` @ v4.6-0 (3e84f85a) | Apache-2.0 | controllers, interfaces |
| Isaac ROS apt deps (`isaac_ros_tensor_list_interfaces`, …) | Apache-2.0 | from the Isaac ROS apt repo, release 4 |
| `unitree_sdk2` @ 21d0a3b2 (fetched by CMake) | BSD-3-Clause wrapper, **prebuilt closed core** (`libunitree_sdk2.a`) | why the manifest is `sdk_kind: closed_with_api` |
| Cyclone DDS 0.10.2 (fetched by CMake) | EPL-2.0 OR EDL-1.0 | dynamically linked runtime |
| Triton Inference Server (`triton-server` rosdep key) | BSD-3-Clause | ONNX runtime for the policy |
| libtorch (`python3-torch-pip-shim`) | BSD-3-Clause (PyTorch); pip wheels bundle NVIDIA CUDA libraries under NVIDIA's EULA | |
| CUDA / TensorRT runtime | NVIDIA proprietary EULA | **GPU mandatory** for the Isaac ROS stack |
| GR00T-WholeBodyControl @ 1e1ec4f5 Walk/Balance ONNX (default S0) | **NVIDIA Open Model License** (weights; code Apache-2.0) | commercial use permitted under its terms; fetched + SHA-256-checked by `tools/build_gr00t_wbc_leapp.py`, never vendored |
| `isaac_ros_agile_unitree_g1` (AGILE LEAPP export) | **not stated in the pinned repos** | verify before redistribution |
| `nvidia/agile_wbc_unitree_h2` @ 222da1df (H2 S0) | **NVIDIA Open Model License** | commercial use permitted under its terms; downloaded + SHA-256-checked, never vendored |
| `unitree_ros` @ 41dec58c `H2.urdf` + meshes | BSD-3-Clause | fetched + SHA-256-checked by `openral_hal._unitree_h2_assets` |

No GPL. Nothing here is vendored: all third-party code is fetched at the pinned SHAs by
the operator, as for `openral_hal_openarm`.
