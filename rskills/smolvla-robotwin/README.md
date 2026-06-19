---
tags:
  - OpenRAL
  - rskill
  - smolvla
  - lerobot
  - vla
  - aloha_agilex
  - robotwin
  - bimanual
  - manipulation
license: apache-2.0
language:
  - en
---

# rskill-smolvla-robotwin

> **OpenRAL rSkill** — SmolVLA (0.45 B) finetuned on the **RoboTwin 2.0** unified
> dual-arm dataset (50 bimanual SAPIEN tasks, aloha-agilex embodiment), packaged for
> use with the [OpenRAL](https://github.com/OpenRAL/openral) robot agent framework.

This package wraps
[`lerobot/smolvla_robotwin`](https://huggingface.co/lerobot/smolvla_robotwin) with a
`rskill.yaml` manifest that adds capability checking, license surfacing, latency
budgets, and local registry integration. It does **not** copy model weights.

## What this skill does

A multi-task dual-arm policy for the RoboTwin 2.0 benchmark
([Chen et al., arXiv 2506.18088](https://arxiv.org/abs/2506.18088)). Action chunks of
length 50 across three RGB views (head + per-wrist) driving a 14-DoF dual-arm joint
command on the AgileX "aloha-agilex" embodiment.

| Field | Value |
| --- | --- |
| Actions | `generalist`, `pick`, `place`, `transfer` |
| Objects | `block`, `pot`, `cup`, `hammer` |
| Scenes  | `tabletop` |
| Embodiment | `aloha_agilex` |
| Action space | 14-D joint position |
| Cameras | `camera1` (head), `camera2` (left wrist), `camera3` (right wrist), 256×256 |

## How it works

OpenRAL loads the upstream LeRobot SmolVLA policy from `hf://lerobot/smolvla_robotwin`
and uses the in-tree `smolvla` adapter to run chunked inference. RoboTwin itself runs in a
separate SAPIEN/CuRobo sidecar; the sidecar returns three RGB views plus the 14-D
aloha-agilex joint state, and the adapter replays 50-action chunks as absolute 14-D joint
position commands.

## Sensors / observation contract

| Direction | Key | Shape | Notes |
| --- | --- | --- | --- |
| in | `observation.images.camera1` | `(3, 256, 256)` RGB | Head / overhead view, re-keyed from RoboTwin `head_camera`. |
| in | `observation.images.camera2` | `(3, 256, 256)` RGB | Left wrist view, re-keyed from RoboTwin `left_camera`. |
| in | `observation.images.camera3` | `(3, 256, 256)` RGB | Right wrist view, re-keyed from RoboTwin `right_camera`. |
| in | `observation.state` | `(14,) float32` | aloha-agilex dual-arm joint state. |
| out | action chunk | `(50, 14) float32` | Absolute dual-arm joint position commands. |

## Manifest summary

| Field | Value |
| --- | --- |
| `name` | `OpenRAL/rskill-smolvla-robotwin` |
| `version` | `0.1.0` |
| `license` | `apache-2.0` |
| `role` | `s1` |
| `model_family` | `smolvla` |
| `embodiment_tags` | `aloha_agilex` |
| `runtime` / `quantization.dtype` | `pytorch` / `bf16` |
| `weights_uri` | `hf://lerobot/smolvla_robotwin` |
| `state_contract.dim` / `action_contract.dim` | `14` / `14` |
| `chunk_size` / `n_action_steps` | `50` / `50` |
| `latency_budget.per_chunk_ms` | `250.0` |
| `evaluated_tasks` | `robotwin` |

## How to run it

RoboTwin runs on **SAPIEN** out-of-process via a Python 3.10 sidecar
([ADR-0061](../../docs/adr/0061-robotwin-dual-arm-benchmark-backend.md)) — its stack is
incompatible with the openral 3.12 venv. Provision the sidecar venv, then:

```bash
# openral-side wire (pyzmq + msgpack)
just sync --all-packages --group robotwin --inexact

# single task
openral benchmark scene \
  --config scenes/benchmark/robotwin_lift_pot.yaml \
  --rskill rskills/smolvla-robotwin

# the 5-task suite
openral benchmark run --suite robotwin --vla smolvla:rskills/smolvla-robotwin
```

See ADR-0061 for the SAPIEN+RoboTwin sidecar provisioning recipe
(`OPENRAL_ROBOTWIN_AUTO_PROVISION=1` or the manual conda recipe).

## Provenance

- **Weights:** [`lerobot/smolvla_robotwin`](https://huggingface.co/lerobot/smolvla_robotwin)
  (Apache-2.0), base [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base).
- **Dataset:** [`lerobot/robotwin_unified`](https://huggingface.co/datasets/lerobot/robotwin_unified)
  (Apache-2.0; `pepijn223/robotwin_unified_v3` renamed).
- **Eval protocol:** RoboTwin official — 100 episodes/task, sim built-in success,
  `episode_length=300`. No locally-reproduced official numbers shipped yet (`eval/` is empty).
  The current website artifact is a 150-step GPU `openral benchmark scene` smoke clip
  (`robotwin_smolvla-robotwin_fail.mp4`, `success=False`) for visual validation only;
  populate `eval/` with `openral benchmark run --suite robotwin` on the eval host.

> **STATE NOTE:** the live RoboTwin sidecar returns a 14-D aloha-agilex state, and the
> official `policy_preprocessor.json` normalization stats expect `observation.state`
> shape `(14,)`; `rskill.yaml` pins `state_contract.dim: 14` accordingly
> (ADR-0061 §Live verification).

## License

This rSkill wrapper, the upstream `lerobot/smolvla_robotwin` checkpoint, and the
`lerobot/robotwin_unified` dataset are Apache-2.0. The package does not copy weights into
this repository; runtime loading still emits OpenRAL's unverified-provenance warning until
the planned signing control exists.
