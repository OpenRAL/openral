# `aloha_agilex` — AgileX dual-arm (RoboTwin 2.0 embodiment)

The bimanual **AgileX "aloha-agilex"** platform RoboTwin 2.0 evaluates by default: two
6-DoF PiPER arms with parallel grippers (**14-DoF**, 7 per arm) and three RGB cameras
(`head_camera`, `left_camera`, `right_camera`, 240×320). It is the embodiment behind the
[`lerobot/robotwin_unified`](https://huggingface.co/datasets/lerobot/robotwin_unified)
dataset and the public RoboTwin checkpoints (e.g.
[`lerobot/smolvla_robotwin`](https://huggingface.co/lerobot/smolvla_robotwin)).

## Sim-only manifest

Unlike most robots in `robots/`, this manifest ships **no on-disk URDF/MJCF**. The actual
robot lives inside the **SAPIEN** environment owned by the RoboTwin sidecar
([ADR-0061](../../docs/adr/0061-robotwin-dual-arm-benchmark-backend.md)), so the openral side
never instantiates it. The manifest exists only so the eval layer can:

- resolve the **14-D joint-position action/state contract**, and
- run the embodiment / sensor compatibility gate against an rSkill.

Joint kinematics are therefore **approximate** (sufficient for capability matching and
`SafetyEnvelope` authoring) — the SAPIEN model is authoritative. Mirrors the same posture as
`robots/aloha_bimanual/robot.yaml` (gym-aloha ViperX), which also carries no shipped URDF.

## Driving it

Used by `scenes/benchmark/robotwin_*.yaml` and `benchmarks/robotwin.yaml` through the
`robotwin` scene backend. See the [benchmarks README](../../benchmarks/README.md) and ADR-0061
for the sidecar provisioning recipe.
