# `so101_follower` — Robot description

Canonical `RobotDescription` manifest for the **LeRobot SO-101 follower
arm** — the hardware revision of the SO-100. Same 6-DoF kinematic
chain (5-DoF arm + 1-DoF parallel jaw) driven over a USB serial bus
by Feetech STS3215 servos; updated Onshape-derived mechanical design
and calibration. The lerobot SDK driver is unchanged (the SO-100 +
SO-101 share `SO100FollowerHAL` / `SO100DigitalTwin` on the serial
side); only the MJCF mesh set differs.

## At a glance

| Field | Value |
| --- | --- |
| `name` | `so101_follower` |
| `embodiment_kind` | `manipulator` |
| Joints | 6 (5 revolute arm + 1 revolute gripper) |
| End-effector | parallel gripper (1 DoF, 5 N max grip force, 0.5 kg payload, 0.32 m workspace radius) |
| Sensors | OAK-D Pro overhead RGB (`front` → `observation.images.camera1`) + 256×256 wrist RGB on the terminal gripper body (`wrist` → `observation.images.camera2`) |
| Embodiment tags | `so101_follower`, `lerobot` |
| Supported control modes | `joint_position`, `gripper_position` |
| `sdk_kind` | `open` (LeRobot SDK, Apache-2.0) |
| `hal.sim` | _null_ — derives `MujocoArmHAL.from_description` from the `sim:` block; for `deploy sim` the generic HAL camera rig (ADR-0065) splices the `front` + `wrist` cameras from their `sensors[].sim_placement` into the bare MJCF so the twin renders them (issue #88) |
| `hal.real` | `openral_hal.so100_follower:SO100FollowerHAL` (shared SO-100 Feetech follower; `deploy run`) |
| Action / observation spec | 6-D joint positions @ 30 Hz / `(6,)` joint state |

Workspace box: ±0.4 m × ±0.4 m × 0.0–0.6 m. EE speed ≤ 0.5 m/s, EE
acceleration ≤ 2.0 m/s². Deadman required.

## Joints

| Name | Type | Limits (rad) | Velocity (rad/s) | Effort (N·m) |
| --- | --- | --- | ---: | ---: |
| `shoulder_pan`  | revolute | ±1.9199 | 4.5 | 3.35 |
| `shoulder_lift` | revolute | ±1.7453 | 4.5 | 3.35 |
| `elbow_flex`    | revolute | −1.7453 – 1.5708 | 4.5 | 3.35 |
| `wrist_flex`    | revolute | ±1.6581 | 4.5 | 3.35 |
| `wrist_roll`    | revolute | ±2.7925 | 4.5 | 3.35 |
| `gripper`       | revolute | −0.1745 – 1.7453 | 4.5 | 3.35 |

MJCF reference: `robot_descriptions:so_arm101_mj_description` —
`TheRobotStudio/SO-ARM100/Simulation/SO101/so101_new_calib.xml`
(Apache-2.0). The MJCF declares its hinge joints as numeric names
(`"1"`…`"6"`); `MujocoArmHAL._sim_kwargs_for` resolves them
by index against `description.joints`, so the manifest's logical
names (`shoulder_pan`, …, `gripper`) drive the user-facing contract
while the MJCF stays as upstream ships it.

## Detect & deploy

The SO-101 is electrically identical to the SO-100 over USB — the same Feetech
controller and USB VID/PID — so the bus alone cannot tell them apart. The SO-101
is the current revision, so a bare `openral detect` resolves to **this** manifest
by default (no `--robot` flag needed; an SO-100 is selected with `--robot
so100`). Scaffold the matching `RobotEnvironment` in one shot:

```bash
openral connect --robot so101                    # smoke-test the serial link
openral detect \
    --output robots/so101_follower/robot.yaml \
    --deployment deployments/so101.yaml          # robot_id + port + sensors pre-filled
```

The scaffold fills everything detection knows (identity, serial `port`,
cameras) and leaves `task` + `vla` as `TODO` placeholders to edit before
`openral deploy run` (`safety` is `null`, so the limits above apply). See the
[deploy tutorial](../../docs/tutorials/deploy/deploy-run-and-dashboard.md).

## Pair with

| Component | Path |
| --- | --- |
| Sim scene | [`so101_box`](../../python/sim/src/openral_sim/backends/so101_box/) — 100 × 61.5 × 75 cm tabletop arena with wrist + overhead OAK-D Pro RGB-D and an optional tube-insertion task |
| Example config | [`scenes/sim/so101_tube_insertion.yaml`](../../scenes/sim/so101_tube_insertion.yaml) |
| Python HAL adapter | `openral_hal.so100_sim:SO100DigitalTwin` (kinematics identical to SO-100; the SO-101 hardware is drop-in on the serial-Feetech path) |

## See also

- [SO-100 manifest](../so100_follower/) — same 6-DoF contract, different MJCF.
- [`docs/reference/vla_compatibility.md`](../../docs/reference/vla_compatibility.md#so101_follower) for the rSkill lineage that targets this embodiment.
