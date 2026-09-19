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
| `hal.sim` | _null_ — derives `MujocoArmHAL.from_description` from the `sim:` block; for `deploy sim` the generic HAL camera rig splices the `front` + `wrist` cameras from their `sensors[].sim_placement` into the bare MJCF so the twin renders them (issue #88) |
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
| `gripper`       | revolute | 0 — 1 (normalised, **not** rad) | 4.5 | 3.35 |

The gripper row is the one exception to the "Limits (rad)" heading: that
channel carries a normalised `[0 = closed, 1 = open]` jaw fraction, and
`position_limits` declares the unit the safety envelope and the runner's
pre-clamp compare against. The jaw's mechanical `[−0.1745, 1.7453] rad`
range lives in `sim.grippers[].ctrl_range` and the URDF (issue #62).

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
so100`). Use detection to validate/write robot-owned facts:

```bash
openral connect --robot so101                    # smoke-test the serial link
openral detect \
    --output robots/so101_follower/robot.yaml
```

Workcell/deploy context lives in `scenes/deploy/*.yaml`; robot facts such as the
serial port, cameras, and limits stay in `robot.yaml`. The rSkill is not pinned —
the reasoner selects it at runtime from the installed `rskills/` registry. See the
[deploy tutorial](../../docs/tutorials/deploy/deploy-run-and-dashboard.md).

## Hardware-in-the-loop

The SO-101 is the only robot in this repo with a committed, self-contained
real-hardware deploy scene: [`scenes/deploy/so101_bench.yaml`](../../scenes/deploy/so101_bench.yaml)
carries the serial port, the lerobot calibration identity and both camera
bindings, and the Feetech calibration itself is committed next to it at
`scenes/deploy/calibration/so_follower.json`.

Two non-motion commands check the rig before anything is dispatched:

```bash
openral deploy validate --config scenes/deploy/so101_bench.yaml
just hil-so101       # tests/hil/test_so101_serial_live.py
```

`deploy validate` touches no hardware beyond `stat`-ing the device nodes: it
resolves the scene, then checks the inputs a real run needs and otherwise
discovers late — a declared serial port, a calibration file that actually
exists (the "has no calibration registered" failure), and a `deploy_binding`
per scene sensor. Camera paths are host-specific; a stale one is reported as a
warning, and retuning the scene against `ls -l /dev/v4l/by-id /dev/v4l/by-path`
is the fix. Never bind a raw `/dev/videoN` — USB enumeration order renumbers
them on replug.

`just hil-so101` is the HIL gate. It opens the real serial bus, runs the
pre-flight servo ping, and reads state back, asserting the joint names, shape,
units and envelope against this manifest and that the committed calibration is
the one loaded into the motors. It commands nothing — no `send_action`, no
`reset_to_pose` — and skips with a reason when the arm is unplugged or its
12 V supply is off (the USB-serial adapter enumerates on 5 V alone, so the
port looks healthy while every servo is dark). Running it from CI is manual
dispatch only (Actions tab → "Run workflow") and needs a
`[self-hosted, lab-so101]` runner listening; see
[`docs/contributing/development.md`](../../docs/contributing/development.md#registering-a-lab-so101-hil-runner).

Motion on this arm — `openral deploy run` — is an attended operation. The
deploy graph does not currently launch the deadman watchdog or human E-stop
nodes, so the physical power switch is the E-stop.

## Pair with

| Component | Path |
| --- | --- |
| Sim scene | [`so101_box`](../../python/sim/src/openral_sim/backends/so101_box/) — 100 × 61.5 × 75 cm tabletop arena with wrist + overhead OAK-D Pro RGB-D and an optional tube-insertion task |
| Example config | [`scenes/sim/so101_tube_insertion.yaml`](../../scenes/sim/so101_tube_insertion.yaml) |
| Python HAL adapter | `openral_hal.so100_sim:SO100DigitalTwin` (kinematics identical to SO-100; the SO-101 hardware is drop-in on the serial-Feetech path) |

## See also

- [SO-100 manifest](../so100_follower/) — same 6-DoF contract, different MJCF.
- [`docs/reference/vla_compatibility.md`](../../docs/reference/vla_compatibility.md#35-so-100--so-101-real-robot-or-sim) for the rSkill lineage that targets this embodiment.
