---
language:
- en
license: apache-2.0
pipeline_tag: robotics
tags:
- OpenRAL
- rskill
- lingbot_vla_v2
- vision-language-action
- aloha_agilex
- rskill-kind:vla
- rskill-role:s1
- embodiment:aloha_agilex
- rskill-license:apache-2.0
- openral
- robotwin
base_model:
- robbyant/lingbot-vla-v2-6b-robotwin
base_model_relation: finetune
inference: false
openral:
  description: LingBot-VLA 2.0 (6.38 B, Qwen3-VL-4B backbone + sparse-MoE flow-matching
    action expert), RoboTwin 2.0 post-train, run in-process through lerobot's vendored
    policy. Predicts 50-step chunks (25-step replay) of a 14-DoF dual-arm joint command
    for the AgileX Cobot Magic from three RGB views (head + per-wrist) and a language
    instruction. bf16, needs a ≥16 GB GPU.
  model_family: lingbot_vla_v2
  min_vram_gb: 12.8
  latency_ms: 1500.0
  runtime: pytorch
  dtype: bf16
library_name: lerobot
---

# rskill-lingbot_vla_v2-aloha_agilex-robotwin-bf16

> **OpenRAL rSkill** — LingBot-VLA 2.0's RoboTwin 2.0 post-train, run **in-process**
> through lerobot's vendored `LingbotVLAV2Policy` for the AgileX Cobot Magic dual-arm
> (`aloha_agilex`).

This package contains the OpenRAL manifest only. The weights are
[`lerobot/lingbot_vla_v2_robotwin`](https://huggingface.co/lerobot/lingbot_vla_v2_robotwin),
pinned by commit: upstream
[`robbyant/lingbot-vla-v2-6b-robotwin`](https://huggingface.co/robbyant/lingbot-vla-v2-6b-robotwin)
(`checkpoints/global_step_50000`) unchanged, plus a LeRobot config and
processor pipeline.

## Preview

No rollout media ships with this package yet. Two `robotwin/place_empty_cup` episodes
were run on an RTX 5090 (see [Evaluation](#evaluation)).

## What this skill does

A generalist dual-arm manipulation policy for the RoboTwin 2.0 task family
(pick / place / handover / pouring on blocks, cups, bottles, bowls). It predicts
50-step chunks of a 14-DoF absolute joint command from three RGB views and a
language instruction, and replays 25 steps per chunk.

It is the same model as [`lingbot-vla2-robotwin`](../lingbot-vla2-robotwin/), which
runs the upstream-format *foundation* checkpoint in a pinned sidecar venv. This skill
runs the RoboTwin *post-train* in the workspace interpreter: no sidecar, no Triton-MoE
kernels, no `transformers==4.57` pin.

## How it works

`openral_sim.policies.lingbot_vla_v2` loads the pinned snapshot, validates every step of
its `policy_{pre,post}processor.json`, builds the policy on the host and casts it to
bf16, then moves it to the GPU. The checkpoint's own pipeline does quantile
normalisation, maps the 14-D state/action onto the 55-D canonical slots, and handles
Qwen3-VL image patching, the chat template and tokenisation. Inference runs under bf16
autocast, because lerobot's vendored vision tower interpolates its position embeddings
with fp32 weights.

### Observation → action contract

| In | Shape | Notes |
| --- | --- | --- |
| `observation.images.cam_high` / `cam_left_wrist` / `cam_right_wrist` | HWC uint8 RGB | slots `camera1/2/3` via `image_preprocessing.aliases`; resized with padding to 256x256 |
| `observation.state` | `(14,)` float32 | `[arm_l(6) grip_l(1) arm_r(6) grip_r(1)]`, radians, grippers 0–1 |
| `task` | str | the RoboTwin instruction |

Out: a 14-D absolute joint-position action per step (same layout as the state).

## Upstream model / training

| | |
| --- | --- |
| Model | LingBot-VLA 2.0, 6.38 B (Qwen3-VL-4B backbone + sparse-MoE Qwen2 action expert, flow matching, 10 denoising steps) |
| Checkpoint | RoboTwin 2.0 post-train, `checkpoints/global_step_50000` |
| Paper | [From Foundation to Application: Improving VLA Models in Practice](https://huggingface.co/papers/2607.06403) |
| Code | [Robbyant/lingbot-vla-v2](https://github.com/Robbyant/lingbot-vla-v2), lerobot port `lerobot.policies.lingbot_vla_v2` (lerobot>=0.7) |

## Supported robots

| Robot | Embodiment tag | Status | Notes |
| --- | --- | --- | --- |
| AgileX Cobot Magic (RoboTwin 2.0, SAPIEN) | `aloha_agilex` | sim | two 6-DoF arms + two grippers |

## Sensors required

| Key | Modality | Min resolution | Format |
| --- | --- | --- | --- |
| `observation.images.camera1` | RGB (head) | 256 x 256 | HWC uint8 |
| `observation.images.camera2` | RGB (left wrist) | 256 x 256 | HWC uint8 |
| `observation.images.camera3` | RGB (right wrist) | 256 x 256 | HWC uint8 |

## Manifest summary

| Field | Value |
| --- | --- |
| `name` | `OpenRAL/rskill-lingbot_vla_v2-aloha_agilex-robotwin-bf16` |
| `model_family` | `lingbot_vla_v2` |
| `license` | `apache-2.0` (code and weights) |
| `quantization.dtype` | `bf16` |
| `min_vram_gb` | bf16 12.8, fp32 25.5 |
| `weights_uri` | `hf://lerobot/lingbot_vla_v2_robotwin@54f79a4b6e9166bda91b5010088097ff3191bca2` |
| `chunk_size` / `n_action_steps` | 50 / 25 |
| `state_contract.dim` / `action_contract.dim` | 14 / 14 |

## Quick start

```bash
just sync --group sim
OPENRAL_ROBOTWIN_AUTO_PROVISION=1 openral benchmark scene \
  --config scenes/benchmark/robotwin_place_empty_cup.yaml \
  --rskill rskills/lingbot-vla-v2-robotwin-bf16 \
  --no-write-eval --n-episodes 2
```

Needs a CUDA GPU with at least 16 GB (bf16 weights plus the SAPIEN renderer).

## Reproduction

The command above is the run recorded under [Evaluation](#evaluation). A full
`openral benchmark run --suite robotwin --rskill rskills/lingbot-vla-v2-robotwin-bf16`
populates `eval/` on a ≥16 GB card.

## Evaluation

| Task | Episodes | Success | Mean step latency | Host |
| --- | --- | --- | --- | --- |
| `robotwin/place_empty_cup` | 2 (seeds 0, 1) | 2/2 | 105 ms | RTX 5090, bf16 |

The same two seeds through the `lingbot_vla2` sidecar on the upstream checkpoint (bf16)
also scored 2/2, at 123 ms. This is a smoke check of the port, not a benchmark claim;
`eval/` is empty until a full suite run.

## License

Apache-2.0, code and weights, as the upstream release. Fine-tuning with lerobot's
dual-query distillation downloads frozen teachers under their own licenses (including
DINOv3); inference does not use them.

## See also

- [`lingbot-vla2-robotwin`](../lingbot-vla2-robotwin/) — same model, upstream foundation checkpoint, sidecar runtime, 8 GB NF4 path.
- [`docs/reference/vla_compatibility.md`](../../docs/reference/vla_compatibility.md)
