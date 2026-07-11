---
language:
- en
license: apache-2.0
library_name: lerobot
pipeline_tag: robotics
tags:
- OpenRAL
- rskill
- smolvla
- lerobot
- vision-language-action
- so101_follower
datasets:
- sapanostic/pen-placement-task
inference: false
---

# rskill-smolvla-so101-pen-bf16

> **OpenRAL rSkill** — [SmolVLA](https://arxiv.org/abs/2506.01844) finetuned for
> pen pick-and-place on a **real SO-101 follower arm**, packaged for `OpenRAL`.

This package wraps
[`sapanostic/so_101_smolvla_pen_placement`](https://huggingface.co/sapanostic/so_101_smolvla_pen_placement)
with a `rskill.yaml` manifest that adds capability checking, license surfacing,
latency budgets, the joint-units contract, and local registry integration. It
does **not** copy model weights.

## Quick start

```python
from openral_rskill.loader import rSkill
pkg = rSkill.from_yaml("rskills/smolvla-so101-pen/rskill.yaml")
```

```bash
# Real SO-101 deploy (the upstream weights repo went gated after this host cached
# the snapshot — deploy from the local cache):
HF_HUB_OFFLINE=1 uv run openral deploy run --robot so101 --rskill rskills/smolvla-so101-pen
```

## Upstream model / training

| Field | Value |
| --- | --- |
| Source repo | [`sapanostic/so_101_smolvla_pen_placement`](https://huggingface.co/sapanostic/so_101_smolvla_pen_placement) |
| Base model | [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) |
| Paper | [arXiv:2506.01844](https://arxiv.org/abs/2506.01844) — *SmolVLA* |
| Training dataset | [`sapanostic/pen-placement-task`](https://huggingface.co/datasets/sapanostic/pen-placement-task) |
| Architecture | SmolVLA (~0.45 B) — VLM backbone + flow-matching action expert |
| Precision | bf16 |
| License | Apache-2.0 (code + weights) |

## Supported robots / embodiments

`so101_follower` — the 6-DoF SO-101 follower arm (5 arm joints + 1 gripper).
The checkpoint drives absolute joint positions.

> **Joint units — degrees.** This checkpoint's state and action are in **degrees**
> (verified: `observation.state` normalizer spans ±100). openral's `JointState` /
> `Action` contract is radians, so the skill_runner converts deg↔rad at the policy
> boundary. The manifest declares `action_contract.joint_units: degrees` explicitly
> (issue #135) — a wrong guess would drive the arm into its joint limits.

## Sensors / observation contract

Two RGB streams, aliased to the checkpoint's `observation.images.*` inputs:

| Manifest key | Scene sensor | Checkpoint view |
| --- | --- | --- |
| `observation.images.camera1` | `side` | side / overview |
| `observation.images.camera2` | `wrist` | wrist |

Both are 224×224. Proprioception is the 6-D joint-position vector.

## Manifest summary

| Field | Value |
| --- | --- |
| `model_family` | `smolvla` |
| `role` | `s1` |
| `chunk_size` / `n_action_steps` | 50 / 50 |
| `action_contract` | 6-D `joint_positions`, `joint_units: degrees` |
| `latency_budget` | 400 ms/chunk (~190 ms measured, RTX 4070 Laptop, bf16) |
| Actions | pick · place · pick_and_place (object: pen) |

## License

**Apache-2.0** (code and weights). OpenRAL's packaging is Apache-2.0;
the upstream checkpoint and dataset are Apache-2.0 as published by the author.
