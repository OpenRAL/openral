# `openral_hal`

Hardware Abstraction Layer, Layer 0 of the seven-layer architecture
(CLAUDE.md §3): the `HAL` Protocol every robot adapter satisfies, plus the
concrete adapters for the robots OpenRAL targets today.

## Synopsis

```python
from openral_hal import HAL, SO100FollowerHAL, SO100DigitalTwin, SO100DigitalTwinConfig
from openral_core import Action, ControlMode

hal: HAL = SO100FollowerHAL(port="/dev/ttyUSB0")        # real SO-100 follower arm
# or, for offline / CI, the same adapter over the in-process simulator:
hal = SO100FollowerHAL(robot=SO100DigitalTwin(SO100DigitalTwinConfig()))

hal.connect()
state = hal.read_state()                          # → openral_core.JointState
hal.send_action(
    Action(control_mode=ControlMode.JOINT_POSITION, joint_targets=[[0.0] * 6])
)
hal.disconnect()
```

The `HAL` Protocol is **structural**: any class with `connect / disconnect /
read_state / send_action / estop` and a `description: RobotDescription`
attribute satisfies it, without inheriting from anything.

## What's in here

One line per module; per-robot detail lives in `robots/<id>/README.md` and
[`docs/reference/`](../../docs/reference/).

**Protocol and shared plumbing**

| Module | Role |
| --- | --- |
| `protocol.py` | The `HAL` Protocol (canonical interface), `EStopRecovery`, `HALHealthReport`. |
| `_base.py` | `HALBase` mixin: connected flag, action validation, default `disconnect`. |
| `lifecycle.py` | `ManifestHALLifecycleNode` and lifecycle helpers shared by the ROS 2 adapters. |
| `resolver.py` | `build_hal(description, *, mode, transport=None, sim_env_yaml=None)`: the one seam from manifest + mode to a constructed HAL. |
| `_real_description.py` | `make_real_description`: derive a real-hardware `RobotDescription` from a sim baseline. |
| `_sensor_wiring.py`, `_camera_rig.py` | Catalog-backed sensor wiring; splice manifest-declared cameras into a bare-arm MJCF. |
| `_slot_group.py` | Compose one whole-robot command from a tick's slot-dispatched actions (ADR-0102). |
| `_tracking_watch.py` | Flags a position-commanded joint that does not follow its command. |

**`ros2_control` and real hardware**

| Module | Role |
| --- | --- |
| `ros_control.py` | `RosControlHAL` and `GroupedRosControlHAL` (several controllers, one robot; config-only via `groups` in the manifest). |
| `ros_control_transport.py` | `RosControlTransport`, the production `rclpy` transport, and the `ControllerStopSeam` it stops through. |
| `sim_transport.py` | `SimTransport` (in-memory transport and stop seam) and `SimTorqueSeam` for unit tests. |
| `interbotix_transport.py` | `InterbotixXSTransport`: the production `xs_sdk` transport for the Interbotix XS arms (commands, merged per-namespace joint states, torque-off stop); attached by the lifecycle node under `hal_mode:=real`, never run on a rig (#250). |
| `ur_real.py`, `franka_panda_real.py`, `sawyer_real.py` | `UR5eRealHAL` / `UR10eRealHAL`, `FrankaPandaRealHAL`, `SawyerRealHAL`. |
| `openarm_real.py` | `OpenArmRealHAL`: OpenArm v2 over `openarm_bringup`'s `ros2_control` stack (CAN stays in C++). |
| `unitree_real.py`, `unitree_s0_policy.py` | `UnitreeHgRealHAL` (G1, H2 over NVIDIA's Isaac ROS stack; S0 stays NVIDIA's C++ controller) and the typed reader of its S0 policy files. |
| `galaxea_a1.py` | `GalaxeaA1HAL` over a loopback JSON-lines transport to an isolated sidecar (no vendor code bundled). |
| `aloha.py` | `AlohaHAL` (Interbotix XS, not `ros2_control`; driven through `InterbotixXSTransport`; `connect()` checks both arms' operating modes) and `AlohaMujocoHAL`. |
| `so100_follower.py`, `feetech_torque_watcher.py` | `SO100FollowerHAL` over the `lerobot` SDK, and the separate process that cuts servo torque if the HAL dies. |
| `composed.py`, `platform_base.py` | `ComposedHAL` (one robot from part HALs that own disjoint joints) and `PlatformBaseHAL` (a base driven by its own stack via odometry and a twist topic); used by `openamrobot_v2`. |

**Simulation twins**

| Module | Role |
| --- | --- |
| `sim_attached.py` | `SimAttachedHAL`: the one sim HAL shell wrapping any `openral_sim.SimRollout` (send path, idle stepper, clock, e-stop latch, attachment API, `reset_episode`). |
| `sim_bringup.py`, `sim_scene.py` | Build a `SimRollout` from a scene YAML; export the console's sim-scene mesh. |
| `mujoco_twin.py` | `MujocoTwinRollout`: the one MuJoCo driver behind every `MujocoArmHAL` twin. Per-robot twins subclass it. Needs only `mujoco`. |
| `_mujoco_arm.py` | `MujocoArmHAL.from_description(desc)`: the manifest-driven twin. No Python file is needed to add a MuJoCo robot (below). |
| `so100_sim.py`, `so100_mujoco.py` | `SO100DigitalTwin` (pure Python, no GPU) and `SO100MujocoHAL`. |
| `franka_panda.py`, `ur.py`, `flexiv_rizon4.py`, `openarm.py`, `anvil_openarm_v2.py` | Thin per-robot MuJoCo wrappers and in-code `*_DESCRIPTION` mirrors. |
| `g1.py`, `h1.py`, `h2.py` | Unitree humanoid twins. G1: glide by default, optional `playground_walk` or `gr00t_wbc` S0 (`_g1_walking.py`, `_g1_gr00t_wbc.py`). H1: software PD over torque actuators. H2: glide, or the sim-only WBC-AGILE S0 (`_h2_agile_velocity.py`). |
| `panda_mobile.py`, `openamrobot_v2.py` | `PandaMobileHAL` (RoboCasa's Panda on a holonomic base) and the OpenAMRobot 2.0 twin (base + lift + two OpenArms). |
| `_floating_base.py` | Humanoid base-pin helpers (ADR-0087 glide). |
| `sim_sensor_bridge.py`, `mobile_base_bridge.py`, `depth_cloud.py`, `proprio_snapshot.py` | ROS wiring for sim HALs: camera/viewer republishing, `/odom` and base streams, depth to `PointCloud2`, thread-safe proprio snapshot. |
| `convex_distance.py` | Certified signed distance between two MuJoCo convex geoms (use it, never `mj_geomDistance`). |

**Attachment evidence (grasp and place)**

| Module | Role |
| --- | --- |
| `_sim_attachment_evidence.py` | MuJoCo ground-truth attachment evidence for deploy sim. |
| `_vision_attachment_evidence.py`, `vision_attachment_bridge.py` | Real-hardware counterpart: masked wrist depth to payload geometry, and the ROS client calling `SegmentInView`. |
| `_grasp_trigger.py`, `_grasp_target_leg.py`, `_place_target_leg.py` | When to ask perception what is in the jaws; pre-grasp and place target producers. |

**Asset fetchers** (pinned revisions, fail closed, nothing vendored when upstream has no licence)

`_pinned_files.py`, `_pinned_clone.py`, `ros_package_overlay.py`, `_openarm_v2_assets.py`, `_openarm_description_assets.py`, `_anvil_openarm_v2_assets.py`, `_unitree_h2_assets.py`, `_galaxea_a1_assets.py`, `_makermods_metal_assets.py`, `_openamrobot_v2_assets.py`, `_openamrobot_platform_assets.py`, `_behavior_r1pro_assets.py`, `_robotwin_aloha_agilex_assets.py`, `_robosuite_assets.py`. Each backs an `assets:` scheme in `robot.yaml` (`rd:`, `gym_aloha:`, `openarm:`, `makermods:`, `behavior:`, `robotwin:`, `unitree:`, `galaxea:`, `file:`).

## Supported robots

`robots/<id>/robot.yaml` is the runtime source of truth: `build_hal` passes the
loaded manifest to every HAL as `description=`. A manifest with `hal.sim: null`
and a `sim:` block builds `MujocoArmHAL.from_description(manifest)`; the
per-robot classes (`FrankaPandaHAL`, `UR5eHAL`, ...) are thin wrappers over the
in-code `*_DESCRIPTION` mirrors, kept for direct construction in tests.

| Robot (`robots/<id>`) | Sim | Real |
| --- | --- | --- |
| `so100_follower`, `so101_follower` | `SO100DigitalTwin`, `SO100MujocoHAL` | `SO100FollowerHAL` |
| `franka_panda` | MuJoCo twin | `FrankaPandaRealHAL` (`franka_ros2` / FCI) |
| `ur5e`, `ur10e` | MuJoCo twin | `UR5eRealHAL`, `UR10eRealHAL` (`ur_robot_driver`) |
| `sawyer` | MetaWorld scene adapter, no twin | `SawyerRealHAL` |
| `aloha_bimanual` | `AlohaMujocoHAL` (gym-aloha MJCF) | `AlohaHAL` (never run on a rig) |
| `aloha_agilex` | RoboTwin SAPIEN sidecar; manifest-driven twin | none |
| `galaxea_a1` | manifest-driven twin | `GalaxeaA1HAL` (sidecar) |
| `openarm`, `anvil_openarm_v2` | manifest-driven twin | `OpenArmRealHAL` (`openarm`; Anvil has none) |
| `rizon4` | `Rizon4MujocoHAL` / manifest-driven twin | none |
| `rizon_dual` | none | `GroupedRosControlHAL` (two arms, upstream `flexiv_hardware`; arms only) |
| `g1`, `h2` | `G1MujocoHAL`, `H2MujocoHAL` | `UnitreeHgRealHAL` (`robot_model` g1 / h2) |
| `h1` | `H1MujocoHAL` | none |
| `panda_mobile`, `panda_mobile_vslam` | `PandaMobileHAL` (RoboCasa) | none |
| `openamrobot_v2` | `OpenAMRobotV2MujocoHAL` | `ComposedHAL` (OpenArm + `PlatformBaseHAL`) |
| `r1pro` | manifest-driven twin (`behavior:r1pro`) | none |
| `makermods_metal`, `makermods_metal_station` | manifest-driven twin | none |
| `widowx` | SimplerEnv; manifest-driven twin | none |
| `droid_franka_robotiq`, `gr1`, `google_robot`, `pusht_2d` | scene-adapter embodiments (Isaac, RoboCasa GR1, ManiSkill3 / SimplerEnv, gym-pusht) | none |

Read each manifest's `hal:` block before relying on a cell; real adapters marked
unit-tested have not necessarily run on hardware. The Unitree real HAL has never
run on a G1 or H2; the robot READMEs list the open gates. The Python
descriptions are mirrors of the manifests, and the manifest wins:
`tests/unit/test_robot_manifests_match_hal_constants.py` fails on drift.

## Adding a new MuJoCo-backed robot

For any robot — single-arm, floating-base humanoid, **or** bimanual —
whose MJCF lives in `robot_descriptions`, `gym_aloha`, or on disk, **no
Python file is required**:

1. Add `robots/<id>/robot.yaml` with the usual `joints`, `capabilities`,
   `safety` blocks **plus** a `sim:` block:

   ```yaml
   assets:
     # mjcf ref schemes: rd:<robot_descriptions module> | gym_aloha:<scene>
     #                   | openarm:<variant> | makermods:<variant> | file:<relpath>
     mjcf: "rd:<module>"
   sim:
     floating_base: false                       # true for humanoids (G1, H1, H2)
     # Optional joint-index overrides — only needed when the MJCF's qpos
     # order doesn't match description.joints (e.g. OpenArm skips qpos 8
     # and qpos 17 for passive follower fingers).
     # joint_qpos_addr: { joint_a: 0, joint_b: 1 }
     # actuator_index:  { joint_a: 0, joint_b: 1 }
     grippers:                                  # zero, one, or two entries
       - joint: "<gripper_joint_name>"
         ctrl_range: [low, high]
         qpos_addrs: [<qpos_index>, ...]
         qpos_scale: <float>
         read_mode: "sum_over_scale"            # | "affine_low_high" | "passthrough"
         write_mode: "normalised"               # | "passthrough"
         # actuator_index: 6                    # override; defaults to actuator_index map
         # mirror_actuator_index: 7             # Aloha: writes -ctrl to the negative finger
     # keyframe_index: 0                        # mj_resetDataKeyframe at connect (Aloha)
     # seed_ctrl_from_qpos: true                # ctrl = qpos at connect (OpenArm v2)
   ```

2. Load the manifest and instantiate the HAL — no per-robot class:

   ```python
   from openral_core import RobotDescription
   from openral_hal import MujocoArmHAL
   desc = RobotDescription.from_yaml("robots/<id>/robot.yaml")
   hal = MujocoArmHAL.from_description(desc, gravity_enabled=False)
   hal.connect()
   ```

The default 1:1 joint→qpos/actuator mapping (offset by 7/6 if
`floating_base: true`) is correct for every robot in tree without
passive-follower qpos quirks; `sim.joint_qpos_addr` /
`sim.actuator_index` are only needed when the MJCF declares joints in a
different order than `description.joints` (Aloha) or interleaves
passive follower-finger slots (OpenArm).

### Worked patterns in tree

| Pattern | Example robot |
|---|---|
| single-arm + revolute jaw gripper (`affine_low_high`) | `so100_follower` |
| single-arm + parallel jaw (`sum_over_scale`) | `franka_panda` |
| single-arm, no gripper | `ur5e`, `ur10e`, `rizon4` |
| floating-base humanoid (`floating_base: true`) | `g1`, `h1` |
| floating-base humanoid + `joint_qpos_addr` / `joint_qvel_addr` override (MJCF tree ≠ SDK order) | `h2` |
| bimanual + two `PASSTHROUGH` grippers + `mirror_actuator_index` + `keyframe_index` | `aloha_bimanual` |
| bimanual + two `PASSTHROUGH` grippers + `joint_qpos_addr` override + `seed_ctrl_from_qpos` | `openarm` |

## Lifecycle e-stop: what `/openral/estop` does to the robot

The generic lifecycle node latches `/openral/estop` and forwards it to any HAL
implementing `LifecycleEStopHAL` (`tests/unit/test_real_hal_estop_fleet_conformance.py` derives every
`hal.real` adapter from `robots/*/robot.yaml` and fails on a new one that omits it).
Each adapter's downstream stop and declared `EStopRecovery`:

| Adapter | Downstream stop (acknowledged by) | Vendor step | Recovery |
| --- | --- | --- | --- |
| `RosControlHAL` (generic) | `controller_manager` deactivates `controller_names()` (STRICT) → `list_controllers` reads `inactive` | — | `RESTART_REQUIRED` |
| `UR5eRealHAL` / `UR10eRealHAL` | same | `/dashboard_client/stop` (`std_srvs/Trigger`) stops the pendant program | `RESTART_REQUIRED` |
| `FrankaPandaRealHAL` | same (`franka_hardware.on_deactivate` → `stopRobot()`) | — (`/error_recovery` is a recovery, never called on stop) | `RESTART_REQUIRED` |
| `SawyerRealHAL` | same | `std_msgs/Empty` on `/robot/set_super_stop` | `RESTART_REQUIRED` |
| `OpenArmRealHAL` | same, all four controllers | — | `RESETTABLE` — `reset_estop` re-activates and reconnects only once confirmed `active` |
| `GroupedRosControlHAL` (`rizon_dual`) | same, both arm controllers (each arm then holds its last NRT target) | — | `RESTART_REQUIRED` (until an in-place re-arm is exercised on the rig) |
| `AlohaHAL` | `/<arm>/torque_enable(enable=false)` per arm (`interbotix_xs_msgs`) | — | `RESTART_REQUIRED` |
| `SO100FollowerHAL` | `Torque_Enable=0` written to every servo (status packet) and read back — an IEC 60204-1 category-0 stop (arm goes limp); proven, the bus stays open read-only (`EStopObservableHAL`), else it is disconnected | — | `RESETTABLE` — `reset_estop` re-arms the torque watcher, sets `Goal := Present` and enables torque, both read back; an unproven stop refuses (restart) |
| `UnitreeHgRealHAL` (G1, H2) | one `switch_controller` request swaps NVIDIA's S0 controller chain for `freeze_controller` (`ControllerSwapSeam`); the freeze holds the measured pose with low gains and does not balance | — | `RESTART_REQUIRED` |
| `GalaxeaA1HAL` | ROS 1 sidecar stopped (acked after driver + roscore exit) | — | `RESTART_REQUIRED` |

A HAL records a `DownstreamStopReport`; the lifecycle node logs FATAL and
publishes `downstream_stop=unacknowledged` on `/diagnostics` when the stop was
not proven, and keeps its latch regardless. `RESETTABLE` adapters clear only
after `reset_estop()` succeeds; `RESTART_REQUIRED` adapters reject
`/openral/estop_cleared` and need a fresh lifecycle start and alignment — the
procedure is in each `robots/<robot>/README.md` under "E-stop and recovery".

### SO-100/SO-101: torque off when the HAL process dies

`estop()` and `disconnect()` cut torque from inside the HAL process, so a
SIGKILL or an OOM kill skips both and the STS3215 servos would keep holding.
`SO100FollowerHAL.connect()` therefore spawns a `FeetechTorqueWatcher` **before**
lerobot's `connect()` enables torque, and refuses to connect if the watcher
does not report armed. The watcher is a separate process in its own session
(a terminal Ctrl-C or a process-group kill aimed at the HAL does not reach
it). It waits on the HAL's `pidfd` and on a stdin pipe from the HAL:

| Event | Watcher action |
| --- | --- |
| HAL process dies (any signal, OOM) | open the port (the kernel closed the HAL's fd), `Torque_Enable=0` per servo with 5 tries each, read back, log `hal.torque_watcher.torque_off` (or `..._unproven`), exit 0 only if every servo reads 0 |
| `disconnect()` / `estop()` proved torque off | HAL sends `q`: the watcher exits without touching the bus (`hal.torque_watcher.clean_exit`) |
| `disconnect()` raised, or `estop()` was unacknowledged | HAL closes the pipe without `q`: the watcher runs its own torque-off pass (fail closed) |
| The watcher itself exits while the HAL runs | the HAL checks it before every bus write (`send_action`, each `reset_to_pose` waypoint) and e-stops (`hal.torque_watcher.lost`, torque off, `ROSEStopRequested`) |

The watcher writes nothing but `Torque_Enable=0`. The HIL bound from SIGKILL to
every servo reading back `Torque_Enable=0` is 1 s
(`tests/hil/test_so101_torque_watcher_live.py`; the bench arm measures far
inside it). It does not survive a whole-cgroup kill (`systemd stop` with
`KillMode=control-group`, a container stop); that case stays with the hardware
E-stop. The STS3215 has no communication-loss timeout: a servo with torque on
holds its goal indefinitely with the host silent, so the host-side watcher and
the hardware E-stop are the only cuts.

### SO-100/SO-101: torque-on holds the current pose

An STS3215 enabling torque drives to its
`Goal_Position` register, which reads 0 after power-up. `connect()` wraps the bus's `enable_torque` so each servo first gets
`Goal_Position := Present_Position`, read back, and refuses the torque-on on a
mismatch.

## Pairing with the ROS 2 lifecycle node

Every HAL adapter runs inside the one manifest-driven ROS 2 lifecycle node,
`packages/openral_hal_node/` (`ManifestHALLifecycleNode` from
`openral_hal.lifecycle`). The robot comes from the `robot_yaml` parameter;
`build_hal` picks the manifest's `hal.sim` / `hal.real` entrypoint (or derives
`MujocoArmHAL`). `openral deploy sim|run` runs it under the ROS node name
`openral_hal_<robot_id>` — a node name, not a package. A vendor real-hardware
bringup the node needs beside it is declared by the manifest's
`hal.real_bringup` (e.g. `packages/openral_hal_openarm`).

See `packages/openral_hal_node/README.md` for the lifecycle contract,
parameters and topic names.

## Tests

- Unit: `tests/unit/test_hal.py` (`RosControlHAL` over `SimTransport`) and one
  `test_<adapter>.py` per adapter.
- Protocol conformance: `tests/unit/test_hal_protocol_conformance.py`
  parametrizes the `HAL` invariants across every adapter that runs without
  hardware; `tests/unit/test_real_hal_estop_fleet_conformance.py` does the same
  for the real-adapter e-stop contract.
- Sim, per HAL: `tests/sim/test_<robot>_hal_mujoco.py`.
- Sim, cross-HAL: `tests/sim/test_all_hals_via_runner.py` drives every HAL twin
  through the production runner with a real `WorldStateAggregator` and an
  echo-current-pose skill, catching joint-indexing, units, lifecycle and
  aggregator breaks that per-robot tests miss. `just hal-twin-sweep` runs both
  sim suites with the ROS pytest-plugin workaround set.
- HIL: `tests/hil/test_<robot>.py`, run by hand on the cell and skipped off-rig
  ([development](../../docs/contributing/development.md#hardware-in-the-loop)).
  The bridge `tests/hil/_ros_control_transport.py` raises at import without
  `rclpy`, so HIL tests guard with `importlib.util.find_spec("rclpy")`. The
  ALOHA HIL test needs no bridge: it drives the production
  `InterbotixXSTransport`.
- Doctests: `just test-doctest` runs `--doctest-modules` over a curated package
  list that includes `python/hal/src/openral_hal` (the list is `DOCTEST_TARGETS`
  in `tests/unit/test_doctest_runner.py`).

## See also

- `openral_core.RobotDescription`, `JointSpec`, `Action`, `JointState`,
  `SafetyEnvelope`: the typed contract this package consumes and emits.
- CLAUDE.md §3 (architecture) and §5 (the `ROSError` hierarchy HAL adapters
  raise from).
- Isaac RGB-D images and clouds keep their capture stamps; their dynamic
  optical TF uses the measured backend state's simulation time and is published
  before the raster. Consumers require transforms at the exact capture time, and
  current camera poses are never restamped onto older images.
