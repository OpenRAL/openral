# `makermods_metal` — Robot description

Canonical `RobotDescription` manifest for one **MakerMods Metal arm**
([makermods.ai/metal-arm](https://www.makermods.ai/metal-arm)), an aluminium arm
marketed as "7-axis": **6 revolute joints** (`joint1`..`joint6`) plus a
**parallel gripper** whose two prismatic fingers (upstream `joint7` / `joint8`)
are coupled (`<mimic joint="joint7" multiplier="-1">` on `joint8`), so the
gripper is one actuated DoF. The joints are Damiao motors on a CAN bus,
commanded in MIT mode. **The arm has no brakes.**

The two-arm Metal Arm Station is a separate manifest:
[`robots/makermods_metal_station/`](../makermods_metal_station/README.md).

## At a glance

| Field | Value |
| --- | --- |
| `name` | `makermods_metal` |
| `embodiment_kind` | `manipulator` |
| `base_frame` | `base_link` |
| Joints | 7 actuated: 6 revolute arm joints + 1 gripper channel (driven finger `joint8`; `joint7` follows it) |
| End effector | parallel gripper, normalised `[0 = closed, 1 = open]` mapped onto `joint8`'s `[0, 0.05]` m finger travel |
| Reach | 612.5 mm (vendor figure) → `workspace_radius_m: 0.6125` |
| Payload | vendor rates the arm at 3 kg; the manifest declares 0.3 kg, capped by what a 3 N two-finger friction grasp holds (the URDF finger effort limit is the only upstream gripper figure) |
| Sensors | `top` RGB (static overhead camera, `observation.images.top`) + `wrist` RGB on `gripper_base` (`observation.images.wrist`); both poses / intrinsics are a fit on MakerMods' `HappyEthan/metal_pick_place` videos, not measured mounts |
| Embodiment tags | `makermods_metal`, `makermods` |
| Supported control modes | `joint_position`, `gripper_position` |
| Action / observation spec | 7-D joint positions @ 30 Hz / `(7,)` joint state |
| `sdk_kind` | `open` |
| `hal.sim` | `null` — `build_hal` derives `MujocoArmHAL.from_description` from the `sim:` block |
| `hal.real` | `null` — no real-hardware HAL yet (see [Real hardware](#real-hardware)) |

## Assets and licensing

Kinematics, joint limits and meshes come from the `metal_description` ROS 2
package in
[`makermods-robotics/metal-python-ros`](https://github.com/makermods-robotics/metal-python-ros)
(`metal_ros2/src/metal_description`, `urdf/metal_with_gripper.urdf`). That
repository has **no LICENSE file** at the pinned commit, although the MakerMods
site describes the project as MIT. Without a licence in the repository OpenRAL
does not redistribute it: **nothing is vendored here.**

Instead the manifest's assets use the `makermods:metal` ref
(`assets.urdf.ref` and `assets.mjcf`). `openral_core.assets.resolve_asset`
hands it to `openral_hal._makermods_metal_assets`, which:

1. clones the upstream repository at `METAL_PINNED_SHA` into
   `$OPENRAL_CACHE_DIR/makermods_metal/<sha>/` (default
   `~/.cache/openral`; needs `git` and network on first use);
2. derives the single-arm URDF, rooted at `base_link` (the upstream `world`
   link and its fixed joint are dropped so the root matches `base_frame`;
   mesh refs stay `package://metal_description/…`);
3. compiles that URDF to MJCF with MuJoCo and adds OpenRAL's `<position>`
   actuators (`joint1`..`joint6` + `joint8`), joint damping/armature, two
   contact excludes, and an `<equality>` holding `joint7 = -joint8`.

`metal_description` is also registered in
`openral_hal.ros_package_overlay.PUBLIC_ROS_PACKAGES`, so `package://` mesh
refs resolve on a host without a sourced MakerMods workspace (Isaac import,
Foxglove 3D panel). Bump the pin deliberately; the derived files are cached per
SHA and a hash of the derive module's source, so a derive change never reuses
a stale cache.

**The actuator gains are a sim-only choice.** The upstream URDF has no actuator
model, so the MJCF's `kp` / `kv` say nothing about the real Damiao control loop.

## MJCF layout

| qpos | Joint | Actuator |
| ---: | --- | ---: |
| 0–5 | `joint1`..`joint6` | 0–5 |
| 6 | `joint7` (mimic finger, equality-coupled) | — |
| 7 | `joint8` (driven finger → manifest joint `gripper`) | 6 |

`sim.joint_qpos_addr` pins `gripper` to qpos 7. `joint2` and `joint3` rest at a
range limit at qpos 0, so `sim.seed_ctrl_from_qpos: true` keeps the position
actuators holding the initial pose instead of driving to 0.

## Gripper kinematics

The manifest's `gripper` joint carries the normalised `[0, 1]` fraction, not
metres, so its `child_link` is the LOGICAL `finger_pair` (no primitive), as
`so101_follower`'s `moving_jaw`. The two real fingers are `mimic_joints` rows
at the upstream URDF joint origins / axes: `link8` at `joint8 = 0.05 f` m (the
MuJoCo twin's driven finger) and `link7` at `joint7 = -0.05 f` m (the URDF's
`<mimic multiplier="-1">` master). The safety kernel and
`openral_core.kinematics` pose both fingers from the commanded fraction, and
`openral_hal.resolver.urdf_joint_map` publishes the gripper's state to
`robot_state_publisher` as `joint7` in metres (`-0.05 f`), from which it
derives `joint8`. The real gripper's stroke-to-metres map is unverified (no
real HAL yet).

## Cameras

The camera names and feature keys match MakerMods' pick-and-place dataset
(`HappyEthan/metal_pick_place`): `top` (`observation.images.top`) and `wrist`
(`observation.images.wrist`). Both poses were fit on that dataset's videos
(top: f 373 px, principal point (287.8, 195), reprojection ~3 px; wrist:
an offline fit on 120 frames with known states against appearance-free
features — held-out desk edges 1–3 px, marker 4.4 px median; f 285.6 px,
principal point (325.8, 249.9), no measurable radial distortion), so they describe *that* rig, not a
measured mount — `static_transform_calibrated` stays `false`. The generic HAL
camera rig has no principal-point offset, so the `top` `sim_placement` is aimed
along the ray through the image centre (fovy 65.5°); the `metal_pick_place`
scene renders the exact intrinsics. `robots/makermods_metal_station` keeps its
wrist cameras identical to this one.

## Sim usage

### `metal_pick_place` scene

[`scenes/sim/makermods_metal_pick_place.yaml`](../../scenes/sim/makermods_metal_pick_place.yaml)
is a sim twin of the dataset's rig: the calibrated cameras, a dark wood desk
against a grey wall, the white cart, the desk grommet, a 3 cm orange cube and the
real marker sticker in the dataset's layouts A–D, and a visual-only leader arm.
The cube is 25 mm: at 30 mm it renders 1.23x too large in the wrist view. The
upstream finger STL is thicker than the fingers on the dataset's arm: with the
camera pinned by the desk edges and markers, the real finger silhouettes give a
mesh scale of (0.676, 1, 0.93) in the STL frame (`finger_mesh_scale`). The
gripper mapping is joint8 = stroke / 2000 - 1.25 mm (the closed real gripper
reads 2.5) with a stiff finger servo (`finger_kp`), so the simulated grip on the
cube reads stroke 22-27 where the real arm reads 24-30. The
static top view is MuJoCo's arm / cube / marker composited over a real background
plate taken from the dataset (so a policy trained on the real top images sees
the real room). The wrist view is rendered with real textures: a seamless desk-wood tile
rectified from near-nadir wrist frames (de-shaded, grain frequency matched to the
real frames), the white cart as a mosaic of rectified wrist frames, and the
marker as the real sticker. Desk, wall, cube, fingers and marker match the real
wrist frames' mean colour to within 7 % per region; a grid of soft shadow lights
keeps the gripper's shadow on the desk, which the pi0.5 LoRA uses as a cue.

```bash
openral sim run --config scenes/sim/makermods_metal_pick_place.yaml \
    --rskill rskills/<metal-pick-place-skill>
```

Replaying eight recorded episodes open-loop (two per layout, the cube and marker
measured from each episode's own top frames, its own start state) grasps, lifts
and places the cube in all eight, 0.3-0.95 cm from the marker centre, with
joint tracking errors of 0.04-0.25 deg; one episode per layout runs in
`tests/sim/test_makermods_metal_hal_metal_pick_place.py`. The scene adds
gravity compensation to the arm links (`arm_gravcomp`): the derived MJCF's
sim-only gains otherwise sag the fingertips into the desk at the dataset's
grasp poses.

### Bare HAL

Without a scene, drive the bare MuJoCo twin through the HAL:

```python
from openral_core import RobotDescription
from openral_hal import build_hal

desc = RobotDescription.from_yaml("robots/makermods_metal/robot.yaml")
hal = build_hal(desc, mode="sim")   # derived MujocoArmHAL; fetches the pinned clone on first use
hal.connect()
print(hal.read_state().name)        # joint1..joint6, gripper
hal.disconnect()
```

`openral check` validates the manifest offline; `openral check
--resolve-remote-assets` also resolves the `makermods:` refs (clone + MuJoCo
compile).

## Real hardware

No HAL exists yet (`hal.real: null`). `openral deploy run` refuses the robot
with `ROSCapabilityMismatch`.

The upstream driver (`metal-python-ros`, ROS 2 Humble) uses its own topics and
messages (`metal_controller`, `ArmJointPositionControl`), not `ros2_control`.
It runs one CAN bus per arm, and the kit can be bought with a Star Arm 102
leader arm for teleoperation. Two integration options are open:

- **(a)** a thin HAL that publishes to the upstream `metal_controller` topics;
- **(b)** a C++ `ros2_control` `SystemInterface` over Damiao MIT mode, which
  fits CLAUDE.md §1.5 (the hot path is C++ and bounded).

> **Safety: the arm has no brakes. A de-energised arm falls**
> ([livekit-examples/portal-makermods](https://github.com/livekit-examples/portal-makermods)).
> A torque cut is therefore not a safe stop for this arm. A real HAL needs a
> stop behaviour (hold, or a damped descent) reviewed by the safety WG, with a
> hazard-log entry, before it moves hardware.

The manifest's `safety` block holds sim values. None of them was measured on a
real arm, and the camera mounts are uncalibrated
(`static_transform_calibrated: false` on `top` and `wrist`). Re-measure all of them before
a real deploy.

## References

- Product page: <https://www.makermods.ai/metal-arm>
- Upstream description + driver: <https://github.com/makermods-robotics/metal-python-ros>
- No-brakes note: <https://github.com/livekit-examples/portal-makermods>
- Fetch / derive code: [`python/hal/src/openral_hal/_makermods_metal_assets.py`](../../python/hal/src/openral_hal/_makermods_metal_assets.py)
- Asset grammar: `openral_core.assets.resolve_asset` (`makermods:<metal|metal_station>`)
- Sibling manifests: [`robots/makermods_metal_station/`](../makermods_metal_station/README.md), [`robots/README.md`](../README.md)
