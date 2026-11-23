# `aloha_agilex` — AgileX dual-arm (RoboTwin 2.0 embodiment)

The bimanual **AgileX "aloha-agilex"** platform RoboTwin 2.0 evaluates by default: two
6-DoF PiPER arms with parallel grippers (**14-DoF**, 7 per arm) and three RGB cameras
(`top`, `wrist_left`, `wrist_right`, 256×256). It is the embodiment behind the
[`lerobot/robotwin_unified`](https://huggingface.co/datasets/lerobot/robotwin_unified)
dataset and the public RoboTwin checkpoints (e.g.
[`lerobot/smolvla_robotwin`](https://huggingface.co/lerobot/smolvla_robotwin)).

## Sim-only manifest

No real HAL is wired. Two sims run the robot:

- the **SAPIEN** environment owned by the RoboTwin sidecar (the benchmark scenes),
  where OpenRAL never instantiates the robot and the SAPIEN model is authoritative;
- a bare **MuJoCo twin** (`MujocoArmHAL.from_description`, the manifest's `sim:`
  block) for `openral deploy sim` and the dashboard — see below.

Joint limits are **approximate** (PiPER figures, sufficient for capability matching
and `SafetyEnvelope` authoring). `joint3` carries the ARX5 elbow's sign, `[0, 2.967]`:
RoboTwin's URDF unfolds the elbow toward +q (`lerobot/robotwin_unified` state mean
+0.81 rad), so PiPER's `[-2.967, 0]` pinned it folded.

## URDF (`assets.urdf: robotwin:aloha_agilex`)

For geometry outside the sim (renders, viewers, collision lowering), the resolver
fetches the URDF RoboTwin itself loads,
`embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf`, and the 34 meshes it
references out of `embodiments.zip` in the MIT
[`TianxingChen/RoboTwin2.0`](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0)
dataset, at a pinned revision with per-file SHA-256 (range reads, ~98 MB download,
~430 MB on disk; `openral_hal._robotwin_aloha_agilex_assets`). Nothing is vendored.

- Its arm joints are `fl_joint1..6` / `fr_joint1..6` and its grippers `fl_joint7` /
  `fr_joint7`; the manifest's `left_*` / `right_*` joints name them in
  `sim_joint_name`.
- Its joint limits are ±10 rad placeholders, so the manifest's stay authoritative.
- Most visual meshes are `.dae`, which MuJoCo cannot decode: a MuJoCo load renders
  the STL collision meshes.

## MuJoCo twin (`assets.mjcf: robotwin:aloha_agilex`)

```bash
openral deploy sim --config scenes/deploy/aloha_agilex_twin.yaml
```

`openral_hal._robotwin_aloha_agilex_assets.ensure_aloha_agilex_mjcf` compiles the
same URDF with `mujoco.MjSpec` into `derived-<hash>/aloha_agilex.xml` beside it:

- The two **front arms** are actuated: one `<position>` actuator per manifest joint,
  in manifest order (actuators 0–13), at the manifest's ranges; forces clamp at the
  manifest's 30 N·m. qpos 0–7 are `fl_joint1..8`, 8–15 `fr_joint1..8`.
- Each gripper actuator drives `joint7` (0 closed .. 0.04765 m open); an MJCF
  equality mirrors it onto `joint8`, as RoboTwin's `config.yml` couples them. The
  twin takes the normalised `[0, 1]` gripper (`normalized_open_unit`, 1 = open),
  RoboTwin's own encoding.
- The base, wheels, castors and the two **rear arms** (`lr_*` / `rr_*`) are welded
  at zero: RoboTwin drives only the front arms.
- COLLADA visuals are dropped; STL visuals (the base tray, the camera mast) are
  kept. The frame columns (`box2_Link`) exist only as `.dae`, so the arm pedestals
  render floating above the base.
- The gains are OpenRAL's sim-only choice (kv sized so the 30 N·m clamp allows the
  3 rad/s velocity limit), not a model of the real servos.
- It starts at qpos 0, RoboTwin's home pose (config.yml), inside every range.
- Cameras come from each sensor's `sim_placement`: `top` at RoboTwin's
  `head_camera` pose (its `config.yml`, moved into the robot frame), the wrist
  cameras on the URDF's `left_camera` / `right_camera` links.

Automatic sim grasp evidence (`SimAttachmentEvidenceTracker`) arms one hand per arm
(attach links `fl_link6` / `fr_link6`); deploy logs `automatic sim attachment evidence
armed`.

**Kernel self-collision model.** The manifest's links carry the URDF's names
(`fl_*` / `fr_*`) and every joint's origin and axis is the URDF's, so the kernel, `/tf`
and the camera frames use one set of frames (the old `left_link*` names existed nowhere
in the URDF, and the joint 4/5/gripper axes were wrong). `fixed_attachments` mount the
arm bases on `world` (the URDF root `footprint`, at the MJCF world origin);
`mimic_joints` pose each gripper's second finger (`*_joint8`, MJCF equality
coefficient 1), and each gripper's `urdf_curve` maps the normalised channel onto the
URDF slide (0-0.04765 m). `collision_geometry` / `allowed_collision_pairs` are
`openral collision lower` output, every link a box plus its exact hull
(`--tight-link`); the ACM is adjacent pairs plus pairs the tool proves always collide.
Safety-WG review owed.

Tests: `tests/sim/test_aloha_agilex_hal_mujoco.py` (gravity-on tracking, both
grippers, three cameras, each hand and jaw resolving through `sim_joint_name`) and `test_data_driven_mujoco_hal.py`.

## Driving it

Used by `scenes/benchmark/robotwin_*.yaml` and `benchmarks/robotwin.yaml` through the
`robotwin` scene backend. See the [benchmarks README](../../benchmarks/README.md)
for the sidecar provisioning recipe.
