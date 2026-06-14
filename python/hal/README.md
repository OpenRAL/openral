# `openral_hal`

Hardware Abstraction Layer for **OpenRAL** — Layer 0 in the eight-layer
architecture (CLAUDE.md §2, §6.1). This package owns the `HAL` Protocol
that every robot adapter must satisfy and ships concrete adapters for the
robots OpenRAL targets today.

## Synopsis

```python
from openral_hal import HAL, SO100FollowerHAL, SO100DigitalTwin
from openral_core import Action, ControlMode

hal: HAL = SO100FollowerHAL(port="/dev/ttyUSB0")        # real SO-100 follower arm
# or, for offline / CI:
hal = SO100DigitalTwin()                          # in-process simulator

hal.connect()
state = hal.read_state()                          # → openral_core.JointState
hal.send_action(
    Action(control_mode=ControlMode.JOINT_POSITION, joint_positions=[0.0]*6)
)
hal.disconnect()
```

The `HAL` Protocol is **structural**: any class with `connect / disconnect /
read_state / send_action / e_stop` and an `embodiment_tag: str` field
satisfies it. There is no inheritance requirement.

## What's in here

| Module | Role |
| --- | --- |
| `protocol.py` | The `HAL` Protocol — the canonical interface (RFC §8.2). |
| `lifecycle.py` | Lifecycle state machine helpers shared by ROS 2 lifecycle adapters. |
| `ros_control.py` | `RosControlHAL` — adapter on top of `ros2_control` (no real ROS 2 import; works against `SimTransport` for unit tests and the live transport at runtime). |
| `sim_transport.py` | `SimTransport` — typed in-memory `ros2_control` transport for unit tests. |
| `so100_follower.py` | `SO100FollowerHAL` + `SO100_DESCRIPTION` + `so100_with_sensors` — real SO-100 follower arm via the `lerobot` SDK. |
| `so100_sim.py` | `SO100DigitalTwin` + `SO100DigitalTwinConfig` — pure-Python in-process simulator (used by smoketests, sim tests, and as a stand-in when no hardware is connected). |
| `so100_mujoco.py` | `SO100MujocoHAL` — MuJoCo digital twin for the SO-100 follower, driving the `mujoco_menagerie` MJCF with the same 6-DoF action layout as `SO100FollowerHAL`. |
| `_mujoco_arm.py` | `MujocoArmHAL` + `MujocoArmHAL.from_description` + `resolve_mjcf_uri` — shared MuJoCo backend for the arms below. Post-ADR-0023 it self-constructs from `RobotDescription.sim` (`SimDescription`), so per-robot subclasses are 5-line wrappers and **no Python file is required to add a new MuJoCo HAL** — declare a `sim:` block in `robots/<id>/robot.yaml` and call `MujocoArmHAL.from_description(desc)`. |
| `franka_panda.py` | `FrankaPandaHAL` + `FRANKA_PANDA_DESCRIPTION` (sim baseline) + `franka_panda_with_sensors` (MuJoCo sim). |
| `franka_panda_real.py` | `FrankaPandaRealHAL` + `FRANKA_PANDA_REAL_DESCRIPTION` — real-hardware adapter over `franka_ros2` / FCI (issue #56). |
| `sawyer_real.py` | `SawyerRealHAL` + `SAWYER_DESCRIPTION` (sim baseline) + `SAWYER_REAL_DESCRIPTION` — real-hardware adapter over `sawyer_robot` / `intera_sdk` (issue #57). |
| `aloha.py` | `AlohaHAL` + `ALOHA_DESCRIPTION` (sim baseline) + `ALOHA_REAL_DESCRIPTION` (real HW; Trossen ALOHA via the Interbotix XS SDK + `ros2_control`, issue #58) **and** `AlohaMujocoHAL` (MuJoCo digital twin via the gym-aloha bimanual MJCF, same 14-DoF action layout as `AlohaHAL`). |
| `g1.py` | `G1MujocoHAL` + `G1_DESCRIPTION` — Unitree G1 humanoid (29-DoF) MuJoCo digital twin on the `mujoco_menagerie/unitree_g1` MJCF. Contract validator only — the floating-base humanoid falls without an S0 cerebellum, so closed-loop sim tests run with `gravity_enabled=False` (CLAUDE.md §6.2). |
| `h1.py` | `H1MujocoHAL` + `H1_DESCRIPTION` — Unitree H1 humanoid (19-DoF, predecessor to the G1) MuJoCo digital twin on the `mujoco_menagerie/unitree_h1` MJCF. Same contract-validator scope as the G1; falls without gravity disabled. First HAL whose underlying MJCF uses torque actuators rather than position actuators — runs a software PD position loop every `mj_step` via the `MujocoArmHAL._per_step_update` hook so the public action contract stays "position targets in radians". |
| `flexiv_rizon4.py` | `Rizon4MujocoHAL` + `RIZON4_DESCRIPTION` — Flexiv Rizon 4 cobot (7-DoF, whole-body force sensitivity) MuJoCo digital twin on the `mujoco_menagerie/flexiv_rizon4` MJCF. Drop-in `MujocoArmHAL` subclass — position actuators, no gripper, no floating base. Real-HW path wrapping `flexivrobotics/flexiv_rdk` is a planned follow-up. |
| `openarm.py` | `OpenArmMujocoHAL` + `OPENARM_DESCRIPTION` — Enactic OpenArm v2 bimanual humanoid arm (16-DoF: 7 arm + 1 gripper per side) on the `enactic/openarm_mujoco` **v2** bimanual MJCF. Fresh `HALBase` subclass — v2 ships native `<position>` actuators with per-class PD gains baked into the MJCF, so the HAL just writes target → ctrl and steps. v2 MJCF fetched lazily via `openral_hal._openarm_v2_assets` (the `robot_descriptions` package still pins to a pre-v2 commit; helper drops out when upstream bumps). |
| `_real_description.py` | `make_real_description(base, *, sdk_kind)` — internal helper deriving a real-HW `RobotDescription` from a sim baseline (flips `sdk_kind`; the `hal` entrypoints are shared, ADR-0031); used by all real-HW adapters. |
| `resolver.py` | `build_hal(description, *, mode, transport=None)` — the single seam mapping a manifest + mode to a constructed sim/real HAL (ADR-0031). |
| `ur.py` | `UR5eHAL` / `UR5e_DESCRIPTION` + `UR10eHAL` / `UR10e_DESCRIPTION` (and the `*_with_sensors` factories) — MuJoCo-backed sim. |
| `ur_real.py` | `UR5eRealHAL` / `UR5e_REAL_DESCRIPTION` + `UR10eRealHAL` / `UR10e_REAL_DESCRIPTION` — real UR hardware via `ros2_control` + `ur_robot_driver` (URCap / RTDE; BSD-3-Clause driver). |
| `_sensor_wiring.py` | Catalog-backed sensor wiring shared by the `*_with_sensors` factories. |

## Supported robots

| Robot | Adapter | Status | Notes |
| --- | --- | --- | --- |
| LeRobot SO-100 follower arm | `SO100FollowerHAL` | ✓ HW + sim (HIL via `tests/hil/test_so100.py`) | Embodiment tag: `so100_follower`. |
| SO-100 (pure-Python sim) | `SO100DigitalTwin` | ✓ sim | In-process; no MuJoCo, no GPU. Used by the smoketest and CI. |
| SO-100 (MuJoCo digital twin) | `SO100MujocoHAL` | ✓ sim (`tests/sim/test_so100_follower_hal_mujoco.py`) | Real MuJoCo physics on the `mujoco_menagerie` MJCF; same 6-DoF action layout as `SO100FollowerHAL`. |
| Franka Panda (sim) | `FrankaPandaHAL` | ✓ sim | MuJoCo-backed. |
| Franka Panda (real-HW) | `FrankaPandaRealHAL` | ✓ unit + HIL fixture | `franka_ros2` / `libfranka` / FCI. Lab-gated HIL via `tests/hil/test_franka_panda.py`. Issue #56. |
| Rethink Sawyer (real-HW) | `SawyerRealHAL` | ✓ unit + HIL fixture | `sawyer_robot` (community fork of `intera_sdk`). Lab-gated HIL via `tests/hil/test_sawyer.py`. Issue #57. No MuJoCo twin by design (vendor defunct; no real hardware will ever connect). |
| Trossen ALOHA bimanual (real-HW) | `AlohaHAL` | ✓ unit + HIL fixture | Interbotix XS SDK + `ros2_control` (4 controllers, 14-DoF). Lab-gated HIL via `tests/hil/test_aloha.py`. Issue #58. |
| Trossen ALOHA bimanual (MuJoCo digital twin) | `AlohaMujocoHAL` | ✓ sim (`tests/sim/test_aloha_bimanual_hal_mujoco.py`) | Real MuJoCo physics on gym-aloha's `bimanual_viperx_transfer_cube.xml`; same 14-DoF action layout as `AlohaHAL`. |
| Unitree G1 (MuJoCo digital twin) | `G1MujocoHAL` | ✓ sim (`tests/sim/test_g1_hal_mujoco.py`) | 29-DoF humanoid on the `mujoco_menagerie/unitree_g1` MJCF. Contract validator only — falls without an S0 cerebellum so closed-loop tests run with `gravity_enabled=False`. Real-HW G1 is planned under M2 (CLAUDE.md §6.2). |
| Unitree H1 (MuJoCo digital twin) | `H1MujocoHAL` | ✓ sim (`tests/sim/test_h1_hal_mujoco.py`) | 19-DoF humanoid (G1's predecessor) on the `mujoco_menagerie/unitree_h1` MJCF. Same contract-validator scope as G1; falls without gravity disabled. Software PD position loop wraps the MJCF's torque actuators (mirrors `unitree_sdk2` on hardware). Real-HW H1 also planned under M2. |
| Flexiv Rizon 4 (MuJoCo digital twin) | `Rizon4MujocoHAL` | ✓ sim (`tests/sim/test_rizon4_hal_mujoco.py`) | 7-DoF cobot with whole-body force sensitivity (0.1 N) on the `mujoco_menagerie/flexiv_rizon4` MJCF. Drop-in `MujocoArmHAL` subclass — position actuators, no gripper. Real-HW Rizon 4 path wraps `flexivrobotics/flexiv_rdk` (Python + C++ SDK; closed-with-api per CLAUDE.md §7.4). |
| Enactic OpenArm v2 (MuJoCo digital twin) | `OpenArmMujocoHAL` | ✓ sim (`tests/sim/test_openarm_hal_mujoco.py`) | 16-DoF bimanual (7 arm + 1 gripper per side) on the `enactic/openarm_mujoco` **v2** bimanual MJCF (PR #19 on master). v2's native `<position>` actuators with per-class PD baked in let the HAL just write target → ctrl. LeRobot upstream has an OpenArm driver — real-HW path planned. |
| Universal Robots UR5e (sim) | `UR5eHAL` | ✓ sim | MuJoCo-backed. |
| Universal Robots UR5e (real HW) | `UR5eRealHAL` | ✓ unit + HIL fixture | `ros2_control` + `ur_robot_driver` (URCap / RTDE; BSD-3-Clause driver). Lab-gated HIL via `tests/hil/test_ur5e.py`. Issue #54. |
| Universal Robots UR10e (sim) | `UR10eHAL` | ✓ sim | MuJoCo-backed. |
| Universal Robots UR10e (real HW) | `UR10eRealHAL` | ✓ unit + HIL fixture | Same driver as the UR5e real-HW adapter; only the URDF / payload envelope differ. Lab-gated HIL via `tests/hil/test_ur10e.py`. Issue #55. |
| `ros2_control`-driven arms | `RosControlHAL` | ✓ generic | Any `ros2_control` controller manager that exposes the standard topics/services. |

The Pydantic descriptions above (e.g. `SO100_DESCRIPTION`,
`FRANKA_PANDA_DESCRIPTION`) are paired with the canonical YAML manifests
under `robots/<robot_id>/robot.yaml` — the YAML is the source of truth
for sensors and capabilities, while the Python description pins the
kinematic / safety values that don't change at runtime.

## Adding a new MuJoCo-backed robot (ADR-0023)

For any robot — single-arm, floating-base humanoid, **or** bimanual —
whose MJCF lives in `robot_descriptions`, `gym_aloha`, or on disk, **no
Python file is required**:

1. Add `robots/<id>/robot.yaml` with the usual `joints`, `capabilities`,
   `safety` blocks **plus** a `sim:` block:

   ```yaml
   sim:
     # mjcf_uri schemes: robot_descriptions:<module> | gym_aloha:<scene>
     #                    | openarm_v2:bimanual | file:<abs-path>
     mjcf_uri: "robot_descriptions:<module>"
     floating_base: false                       # true for humanoids (G1, H1)
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
| bimanual + two `PASSTHROUGH` grippers + `mirror_actuator_index` + `keyframe_index` | `aloha_bimanual` |
| bimanual + two `PASSTHROUGH` grippers + `joint_qpos_addr` override + `seed_ctrl_from_qpos` | `openarm` |

## Pairing with ROS 2 lifecycle nodes

Each HAL adapter has (or will have) a thin ROS 2 lifecycle node under
`packages/openral_hal_<robot>/`:

| ROS package | Wraps | Status |
| --- | --- | --- |
| `openral_hal_so100` | `SO100FollowerHAL` | ✓ working (`tests/hil/test_so100.py`) |
| `openral_hal_franka` | `FrankaPandaHAL` (planned) | skeleton |
| `openral_hal_ur5e` | `UR5eHAL` (sim) / `UR5eRealHAL` (real HW) | skeleton |
| `openral_hal_ur10e` | `UR10eHAL` (sim) / `UR10eRealHAL` (real HW) | skeleton |

See each ROS package's `README.md` for the lifecycle contract, parameters,
and topic names.

## Tests

- Unit: `tests/unit/test_hal.py` (`RosControlHAL` × `SimTransport`),
  `tests/unit/test_so100_follower_hal.py`, `tests/unit/test_franka_panda.py`,
  `tests/unit/test_franka_panda_real.py`, `tests/unit/test_sawyer_real.py`,
  `tests/unit/test_aloha.py`, `tests/unit/test_mujoco_arm.py`,
  `tests/unit/test_sim_transport.py`.
- Protocol conformance: `tests/unit/test_hal_protocol_conformance.py`
  parametrizes the `HAL` invariants across **10 implementations**
  (7 invariants × 10 HALs = 70 cases) — `RosControlHAL`,
  `SO100FollowerHAL+SO100DigitalTwin`, `UR5eHAL`, `UR10eHAL`,
  `FrankaPandaHAL`, `FrankaPandaRealHAL`, `SawyerRealHAL`, `AlohaHAL`,
  `UR5eRealHAL+SimTransport`, `UR10eRealHAL+SimTransport`.
- Sim (per-HAL contract): `tests/sim/test_ur5e_hal_mujoco.py`,
  `tests/sim/test_ur10e_hal_mujoco.py`,
  `tests/sim/test_franka_panda_hal_mujoco.py`,
  `tests/sim/test_so100_follower_hal_mujoco.py`,
  `tests/sim/test_aloha_bimanual_hal_mujoco.py`,
  `tests/sim/test_g1_hal_mujoco.py`,
  `tests/sim/test_h1_hal_mujoco.py`,
  `tests/sim/test_smolvla_so100.py`.
- Sim (cross-HAL integration): `tests/sim/test_all_hals_via_runner.py`
  drives **every** HAL twin through the production `HardwareRunner`
  with a real `WorldStateAggregator` + a trivial echo-current-pose
  skill. Catches any joint-indexing / units / lifecycle / aggregator
  / safety wiring breakage at the integration boundary — every
  per-robot sim test could pass while this one still fails.
- One-line full sweep: `just hal-twin-sweep` runs both the per-HAL
  contract suite and the cross-HAL `HardwareRunner` integration
  suite (~200 tests covering 7 robot families) with the ROS 2
  plugin workaround baked in.
- HIL: `tests/hil/test_so100.py`, `tests/hil/test_franka_panda.py`,
  `tests/hil/test_sawyer.py`, `tests/hil/test_aloha.py`,
  `tests/hil/test_ur5e.py`, `tests/hil/test_ur10e.py` — lab runners only
  (`UR5E_HOST` / `UR10E_HOST` env vars + `[self-hosted, lab-ur*]` runner labels for the UR pair).
- HIL transport bridges (lab-only): `tests/hil/_ros_control_transport.py`
  exposes `RosControlHILTransport` + `make_hil_transport` for the
  single-controller real-HW HALs (UR5e / UR10e / Franka Panda / Sawyer);
  `tests/hil/_aloha_ros_transport.py` exposes `AlohaHILTransport` +
  `make_aloha_hil_transport` for the bimanual ALOHA fan-out (4
  controllers). Both raise at import time if `rclpy` is not installed,
  so HIL tests guard with `importlib.util.find_spec("rclpy")` before
  importing them.

Doctest coverage: every adapter file is in `Justfile::test-doctest`'s
curated `DOCTEST_TARGETS`; run `just test-doctest` to exercise the
docstring examples.

## See also

- `openral_core.RobotDescription`, `JointSpec`, `Action`,
  `JointState`, `SafetyEnvelope` — the typed contract this package
  consumes and emits.
- `robots/<robot_id>/robot.yaml` — canonical description manifests.
- `packages/openral_hal_*/README.md` — ROS 2 lifecycle node docs.
- CLAUDE.md §6 (architecture discipline) and §10 (exception hierarchy:
  `ROSConfigError` / `ROSRuntimeError` / `ROSSafetyViolation` are the
  only exceptions a HAL adapter raises).
