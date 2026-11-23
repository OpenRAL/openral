# `aloha_agilex` — AgileX dual-arm (RoboTwin 2.0 embodiment)

The bimanual **AgileX "aloha-agilex"** platform RoboTwin 2.0 evaluates by default: two
6-DoF PiPER arms with parallel grippers (**14-DoF**, 7 per arm) and three RGB cameras
(`top`, `wrist_left`, `wrist_right`, 256×256). It is the embodiment behind the
[`lerobot/robotwin_unified`](https://huggingface.co/datasets/lerobot/robotwin_unified)
dataset and the public RoboTwin checkpoints (e.g.
[`lerobot/smolvla_robotwin`](https://huggingface.co/lerobot/smolvla_robotwin)).

## Sim-only manifest

The robot lives inside the **SAPIEN** environment owned by the RoboTwin sidecar,
and OpenRAL never instantiates it. The manifest exists so the eval layer can:

- resolve the **14-D joint-position action/state contract**, and
- run the embodiment / sensor compatibility gate against an rSkill.

Joint limits are **approximate** (sufficient for capability matching and
`SafetyEnvelope` authoring); the SAPIEN model is authoritative.

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

## Driving it

Used by `scenes/benchmark/robotwin_*.yaml` and `benchmarks/robotwin.yaml` through the
`robotwin` scene backend. See the [benchmarks README](../../benchmarks/README.md)
for the sidecar provisioning recipe.
