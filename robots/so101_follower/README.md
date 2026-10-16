# `so101_follower` — Robot description

Canonical `RobotDescription` manifest for the **LeRobot SO-101 follower
arm**, the hardware revision of the SO-100: the same 6-DoF chain (5-DoF arm +
1-DoF parallel jaw) on Feetech STS3215 servos over a USB serial bus, with an
updated Onshape-derived mechanical design and calibration. The lerobot SDK
driver is unchanged (both share `SO100FollowerHAL` / `SO100DigitalTwin` on the
serial side); only the MJCF mesh set differs.

## At a glance

| Field | Value |
| --- | --- |
| `name` | `so101_follower` |
| `embodiment_kind` | `manipulator` |
| Joints | 6 (5 revolute arm + 1 revolute gripper) |
| End-effector | parallel gripper (1 DoF, 5 N max grip force, 0.5 kg payload, 0.32 m workspace radius) |
| Sensors | OAK-D Pro overhead RGB (`top` → `observation.images.camera1`) + 256×256 wrist RGB on the terminal gripper body (`wrist` → `observation.images.camera2`) |
| Embodiment tags | `so101_follower`, `lerobot` |
| Supported control modes | `joint_position`, `gripper_position` |
| `sdk_kind` | `open` (LeRobot SDK, Apache-2.0) |
| `hal.sim` | _null_ — derives `MujocoArmHAL.from_description` from the `sim:` block; for `deploy sim` the generic HAL camera rig splices the `top` + `wrist` cameras from their `sensors[].sim_placement` into the bare MJCF so the twin renders them (issue #88) |
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

The gripper channel is not in rad: it carries a normalised
`[0 = closed, 1 = open]` jaw fraction, and
`position_limits` declares the unit the safety envelope and the runner's
pre-clamp compare against. The jaw's mechanical `[−0.1745, 1.7453] rad`
range lives in `sim.grippers[].ctrl_range` and the URDF (issue #62).

MJCF reference: `robot_descriptions:so_arm101_mj_description` —
`TheRobotStudio/SO-ARM100/Simulation/SO101/so101_new_calib.xml`
(Apache-2.0). The MJCF declares its hinge joints as numeric names
(`"1"`…`"6"`); `MujocoArmHAL._sim_kwargs_for` resolves them by index against
`description.joints`, so the manifest's names (`shoulder_pan`, …, `gripper`)
form the user-facing contract while the MJCF stays as upstream ships it.

## Grasp-trigger calibration

`joints[gripper].closure_calibration` holds the position-stall grasp-trigger
thresholds, in fraction units. They follow the method of
[`real-pick-place-design.md`](../../docs/reference/real-pick-place-design.md)
§1.6 / §5 as far as simulation allows:

| Value | Setting | Where it comes from |
| --- | --- | --- |
| `closed_position` | 0.0 | the channel's closed end |
| `closed_rest_offset` | 0.0023 | p99 of the LeIsaac PickOrange teleop dataset's settled commanded-closed samples; the Isaac 6.1 twin's empty close rests at 0.00012 |
| `close_band` | 0.25 | SmolVLA grasps with the jaw commanded at 0.19–0.23 (deploy trial g5 replayed in the Isaac twin); the teleop dataset at 0.19 (p95 0.25); free motion lags only at commands ≥ 0.25 |
| `stall_gap` | 0.02 | settled holds sit 0.027–0.103 short of the command in Isaac (teleop median 0.024); free and near-object motion stays ≤ 0.013 |
| `settle_tolerance` | 0.02 | the closed-loop hold in deploy trial g6, replayed on the sim clock, chatters with a 0.133 s span of p50 0.0156 (p90 0.021): at 0.01 it never settles, at 0.02 it attaches once; 0.002 caught only 3 of 53 open-loop hold ticks |

On the Isaac replay these values attach on both holds and never on free motion or an
empty close. On the dataset they attach in 53 of 60 episodes, never outside the
orange width. The smaller `stall_gap` halves the slip/release hysteresis to 0.01;
objects thinner than about 0.022 of jaw travel never read as held. A decision-record
entry is owed for `close_band`.

**These values are sim-measured** (Isaac 6.1 twin plus LeIsaac sim-teleop data) and
**unverified on a real SO-101**. Re-measure them on hardware before any real
`grasp_allowance_enabled` deploy. This is a Safety-WG item, and a hazard-log row is
owed.

## Collision model

`collision_geometry` is hand-authored (the drift guard checks its ACM only):
PCA-fit boxes for `base`, `upper_arm`, `lower_arm` and `wrist`, run with the
negative `self_collision_margin_m` documented in the manifest. `shoulder` is a
box + exact hull, and so are `gripper` and `jaw`.

`shoulder` is the `openral collision lower --tight-link shoulder` fit of its URDF
meshes. The mesh sits inside the hull to 1 nm. It replaces the hand PCA box, whose
bottom hung 1 mm below the surface the arm is mounted on: the box reached
z = 0.029 m in the base frame, while the mesh only reaches 0.0487 m. Two LeIsaac
PickOrange deploy-sim trials stopped on `shoulder` against counter voxels beside
the base, with PhysX reporting no penetration. The hull is less conservative than
the box. **It needs its own Safety-WG review and hazard-log row.**

`gripper` is fitted to the URDF `gripper` link's meshes by
`fit_link_with_tight_geometry`, and `jaw` to the URDF `jaw` meshes in the jaw
frame.

The jaw is a moving link of the collision model. A `mimic_joints` row named `jaw`
uses the URDF `gripper` joint's origin and axis, and poses it at
`q_jaw = 1.7453 · f` rad from the gripper fraction `f`. The manifest's `gripper`
joint keeps the `[0, 1]` fraction channel. Its child, `moving_jaw`, is a logical
link with no primitive. `grasp_contact_links()` is `(moving_jaw, jaw)`, so a
grasp declaration exempts the jaw too. The fraction-to-angle map is per backend:

| Backend | Jaw angle at fraction `f` | How the kernel's jaw compares |
| --- | --- | --- |
| Isaac (`_gripper_spec`: closed = 0 rad, open = the URDF upper limit) | `1.7453 · f` | exact |
| MuJoCo (`sim.grippers[].ctrl_range`) | `−0.1745 + 1.9199 · f` | up to about 10° more open than the MuJoCo jaw at `f = 0`; the gap closes to 0 at `f = 1` |
| Real arm (`SO100FollowerHAL`: `f` maps to lerobot's 0–100) | set by the servo calibration | unverified |

The kernel checks the jaw at the commanded fraction, as it does for every link.

The URDF names the wrist-roll child `gripper`, and so do the manifest and `/tf`.
The `wrist_roll` joint carries the URDF origin. Before this change it was left at
zero, which put the kernel's gripper frame at the wrist origin. The only ACM row
added are the adjacent `[gripper, wrist]` and `[gripper, jaw]`.

Without the gripper entries, a LeIsaac PickOrange deploy-sim stopped on `wrist`
against a voxel on the gripper body (PhysX: no contact). The self-filter had left
the gripper's own depth returns in the map.

**This model needs Safety-WG review and a hazard-log row** (private
`OpenRAL/management`) before anyone relies on it (CLAUDE.md §3).

## Detect & deploy

The SO-101 is electrically identical to the SO-100 over USB (same Feetech
controller and VID/PID), so the bus cannot tell them apart. As the current
revision, it is what a bare `openral detect` resolves to; select an SO-100 with
`--robot so100`. Use detection to validate/write robot-owned facts:

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

The SO-101's committed real-hardware deploy scene,
[`scenes/deploy/so101_bench.yaml`](../../scenes/deploy/so101_bench.yaml),
carries the serial port and the lerobot calibration identity and selects the
bench host's camera bindings (`robot_unit`), and the Feetech calibration itself is committed next to it at
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
per camera. The `top` / `wrist` bindings are per host, in
`units/<unit>.yaml` (the bench laptop's is `units/bench_laptop.yaml`, which
`so101_bench.yaml` selects with `robot_unit`; `OPENRAL_ROBOT_UNIT` overrides
it). A second SO-101 host adds its own unit file instead of editing
`robot.yaml`. A stale path is reported as a warning; fix it by retuning the unit
file against `ls -l /dev/v4l/by-id /dev/v4l/by-path`. Never bind a raw
`/dev/videoN`: USB enumeration renumbers them on replug.

`just hil-so101` is the HIL gate. It opens the real serial bus, runs the
pre-flight servo ping, and reads state back, asserting the joint names, shape,
units and envelope against this manifest and that the committed calibration is
the one loaded into the motors. It sends no position command (no
`send_action`, no `reset_to_pose`), then validates that the terminal E-stop
disconnects the motor bus; run it attended with the physical power switch
within reach. It skips with a reason when the arm is unplugged or its 12 V
supply is off (the USB-serial adapter enumerates on 5 V alone, so the port
looks healthy while every servo is dark). It runs by hand on the bench host,
not in CI; see
[`docs/contributing/development.md`](../../docs/contributing/development.md#running-the-so-101-hil-gate).

Motion on this arm — `openral deploy run` — is an attended operation. The
deploy graph launches the action-stream deadman watchdog and the dashboard /
human E-stop forwarder. The hardware-pendant node stays unconfigured unless a
supported device is supplied, so keep the physical power switch within reach.

## Pair with

| Component | Path |
| --- | --- |
| Sim scene | [`so101_box`](../../python/sim/src/openral_sim/backends/so101_box/) — 100 × 61.5 × 75 cm tabletop arena with wrist + overhead OAK-D Pro RGB-D and an optional tube-insertion task |
| Example config | [`scenes/sim/so101_tube_insertion.yaml`](../../scenes/sim/so101_tube_insertion.yaml) |
| Python HAL adapters | `openral_hal.so100_follower:SO100FollowerHAL` (real, shared with the SO-100 on the serial-Feetech path); sim: derived `MujocoArmHAL` (`hal.sim: null`) or the `so101_box` scene |

## See also

- [SO-100 manifest](../so100_follower/) — same 6-DoF contract, different MJCF.
- [`docs/reference/vla_compatibility.md`](../../docs/reference/vla_compatibility.md#35-so-100--so-101-real-robot-or-sim) for the rSkill lineage that targets this embodiment.
