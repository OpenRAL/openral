# `so100_follower` — Robot description

Canonical `RobotDescription` manifest for the **LeRobot SO-100 follower
arm** — a 5-DoF + 1-DoF gripper teleop / imitation-learning arm driven
over a USB serial bus by Feetech servos. The same manifest covers
three execution paths: real hardware (`SO100FollowerHAL`), the
in-process kinematic twin (`SO100DigitalTwin`), and the real-physics
MuJoCo digital twin (`SO100MujocoHAL` on the `mujoco_menagerie` MJCF).

## At a glance

| Field | Value |
| --- | --- |
| `name` | `so100_follower` |
| `embodiment_kind` | `manipulator` |
| Joints | 6 (5 revolute arm + 1 revolute gripper) |
| End-effector | parallel gripper (1 DoF, 5 N max grip force, 0.5 kg payload, 0.32 m workspace radius) |
| Sensors | 2× RGB USB cameras (`front`, `wrist` — 256×256 @ 30 Hz; can be zeroed for sim) |
| Embodiment tags | `so100_follower`, `lerobot` |
| Supported control modes | `joint_position`, `gripper_position` |
| `sdk_kind` | `open` (LeRobot SDK, Apache-2.0) |
| `hal.sim` | _null_ — derives `MujocoArmHAL.from_description` from the `sim:` block; for `deploy sim` the generic HAL camera rig (ADR-0065) splices the `front` + `wrist` cameras from their `sensors[].sim_placement` (wrist parented to the SO-100 `Fixed_Jaw` body) into the bare MJCF so the twin renders them (issue #88) |
| `hal.real` | `openral_hal.so100_follower:SO100FollowerHAL` (`deploy run`) |
| Action / observation spec | 6-D joint positions @ 30 Hz / `(6,)` joint state |

Workspace box: ±0.4 m × ±0.4 m × 0.0–0.6 m. EE speed ≤ 0.5 m/s, EE
acceleration ≤ 2.0 m/s². Deadman required (`deadman_required: true`).

## Pair with

| Component | Path |
| --- | --- |
| Python HAL adapters | `openral_hal.so100_follower.SO100FollowerHAL` (real), `openral_hal.so100_sim.SO100DigitalTwin` (kinematic twin), `openral_hal.so100_mujoco.SO100MujocoHAL` (MuJoCo digital twin) |
| Python description | `openral_hal.SO100_DESCRIPTION` |
| Sensor factory | `openral_hal.so100_with_sensors` |
| ROS 2 lifecycle node | [`packages/openral_hal_so100/`](../../packages/openral_hal_so100/README.md) |
| Compatible rSkills | [`smolvla-libero`](../../skills/smolvla-libero/README.md) (digital-twin verified) |

## Joints

| Name | Type | Limits (rad) | Velocity (rad/s) | Effort (N·m) |
| --- | --- | --- | ---: | ---: |
| `shoulder_pan` | revolute | ±2.0944 | 4.5 | 5.0 |
| `shoulder_lift` | revolute | ±1.7453 | 4.5 | 5.0 |
| `elbow_flex` | revolute | ±1.7453 | 4.5 | 5.0 |
| `wrist_flex` | revolute | ±1.7453 | 4.5 | 3.0 |
| `wrist_roll` | revolute | ±3.1416 | 4.5 | 3.0 |
| `gripper` | revolute | 0 — 1.5708 | 4.5 | 1.5 |

URDF reference: `robot_descriptions` / `mujoco_menagerie` SO-100. Units
are radians (the lerobot SDK exposes degrees but OpenRAL keeps the
URDF-native units).

## Sensors

| Name | Modality | Frame | Resolution | Notes |
| --- | --- | --- | --- | --- |
| `front` | RGB | `world` | 256 × 256 @ 30 Hz | Mapped to `observation.images.camera1`. |
| `wrist` | RGB | `wrist` (parent: `wrist_roll`) | 256 × 256 @ 30 Hz | Mapped to `observation.images.camera2`. |

For the smoketest digital twin, images are synthesised as zero tensors;
state uses 6-DoF twin positions padded to 7-DoF where needed by the VLA.

## Tests

- Unit: `tests/unit/test_so100_follower_hal.py`,
  `tests/unit/test_hal_protocol_conformance.py` (parametrized over the
  HAL × `SimTransport` and `SO100DigitalTwin` combinations).
- Sim: `tests/sim/test_so100_follower_hal_mujoco.py` exercises
  `SO100MujocoHAL` end-to-end against real MuJoCo physics on the
  Menagerie MJCF (lifecycle, schema-drift guard, closed-loop convergence
  for arm + normalised gripper).
- HIL: `tests/hil/test_so100.py` (lab runner only).
- Closed-loop sim against a SmolVLA / π0.5 / xVLA finetune is driven via
  `openral sim run` against `scenes/benchmark/libero_spatial.yaml` (with `--rskill rskills/<smolvla|pi05|xvla>-libero`) etc.;
  there is no longer a dedicated `tests/sim/test_smolvla_so100.py`.

## See also

- [`python/hal/README.md`](../../python/hal/README.md) — `SO100FollowerHAL`, `SO100DigitalTwin`, sensor wiring.
- [`packages/openral_hal_so100/README.md`](../../packages/openral_hal_so100/README.md) — ROS lifecycle node.
- [`skills/smolvla-libero/README.md`](../../skills/smolvla-libero/README.md) — example rSkill for this embodiment.
- [`docs/quickstart/so100.md`](../../docs/quickstart/so100.md) — sim and hardware quickstart.
