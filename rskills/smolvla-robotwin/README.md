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
  `episode_length=300`. No locally-reproduced numbers shipped yet (`eval/` is empty);
  populated by `openral benchmark run --suite robotwin` once the SAPIEN sidecar is
  provisioned.

> **STATE NOTE:** the checkpoint `config.json` declares `observation.state` shape
> `(6,)`, which disagrees with the dataset's `(14,)` and the 14-DoF embodiment. The
> loaded SmolVLA policy is authoritative; the single-task live run reconciles it
> (ADR-0061 §Live verification).
