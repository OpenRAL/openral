---
language:
- en
license: apache-2.0
library_name: lerobot
pipeline_tag: robotics
tags:
- OpenRAL
- rskill
- act
- lerobot
- vision-language-action
- franka_panda
- libero
inference: false
---

# rskill-act-franka_panda-libero-fp32

> **OpenRAL rSkill** — Action Chunking Transformer
> ([Zhao et al., 2023](https://arxiv.org/abs/2304.13705)) fine-tuned on
> `HuggingFaceVLA/libero`, packaged for the LIBERO Franka-Panda
> embodiment.

## Upstream model

| Field        | Value                                                       |
|--------------|-------------------------------------------------------------|
| Weights      | `hf://Deepkar/libero-test-act`                              |
| Architecture | ResNet-18 backbone · 4+1 encoder/decoder · latent VAE · `chunk_size=100` |
| Action       | 7-D delta-EEF + gripper                                     |
| Dataset      | `HuggingFaceVLA/libero` (Apache-2.0)                        |
| License      | Apache-2.0                                                  |
| Paper        | [arxiv:2304.13705](https://arxiv.org/abs/2304.13705) — *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* |

## Supported robots

| Robot | Embodiment tag | Status | Notes |
| --- | --- | --- | --- |
| Franka Panda (LIBERO sim) | `franka_panda` (LIBERO embodiment tag) | ✓ sim | Native training target; closed-loop spot-check under [Run](#run). |

## Sensors required

| Key | Modality | Resolution | Format |
| --- | --- | --- | --- |
| `observation.images.image`  | RGB | 256 × 256 | `float32` |
| `observation.images.image2` | RGB | 256 × 256 | `float32` |
| `observation.state`         | proprioception | (8,) | `float32` (LIBERO Franka layout: `pos3 + axisangle3 + grip2`) |

## Manifest summary

| Field | Value |
| --- | --- |
| `name` | `OpenRAL/rskill-act-franka_panda-libero-fp32` |
| `version` | `0.1.0` |
| `license` | `apache-2.0` |
| `role` | `s1` |
| `embodiment_tags` | `franka_panda` |
| `runtime` / `quantization.dtype` | `pytorch` / `fp32` |
| `weights_uri` | `hf://Deepkar/libero-test-act` |
| `chunk_size` | 100 |
| `is_commercial_use_allowed` (derived from `license`) | `true` |

Full schema: [`openral_core.schemas.RSkillManifest`](../../python/core/src/openral_core/schemas.py).

## Run

```bash
CC=/usr/bin/gcc just sync --group libero    # first time only
just sim-act-libero
# which runs:
#   openral sim run --config scenes/sim/libero_spatial.yaml \
#     --rskill rskill://rskills/act-libero --save-video example_videos
```

The shipped sim YAML pins `libero_spatial/0` for a 100-step
single-episode rollout. Sweep tasks with `--task libero_spatial/<n>`.
A spot-check on `libero_spatial/2` reaches `is_success=True` in
~91 steps (reward 1.0) on a single seed.

## Camera & state contract

LIBERO emits `images={"camera1": agentview, "camera2": eye_in_hand}`
while this checkpoint's input features are
`observation.images.image` / `observation.images.image2`. The
manifest's `image_preprocessing` block rewrites the batch keys at step
time:

```yaml
image_preprocessing:
  flip_180: true            # HuggingFaceVLA/libero is captured rotated 180°
  aliases:
    camera1: image
    camera2: image2
```

`state_contract.dim: 8` declares the proprio width. The upstream training set,
`HuggingFaceVLA/libero`, is the dataset the smolvla / pi05 / xvla LIBERO
checkpoints here were finetuned on, so the state semantics (pos3 + axisangle3 +
grip2) match OpenRAL's LIBERO backend end-to-end (no quat-vs-axisangle
mismatch).

## Benchmarks

[`eval/libero_spatial.json`](eval/libero_spatial.json) (`reproduced_locally:
true`) is a smoke-scale run: **1 episode per task** (seed 0, max 220 steps)
over the 10 LIBERO-Spatial tasks, `avg_success_rate` **0.5** (tasks 2–6
succeed). One episode per task is not the published protocol; treat it as a
coverage check, not a headline number. Reproduce with
`openral benchmark run --suite libero_spatial --rskill rskills/act-libero`.

## License

Apache-2.0 — both the wrapping rSkill package (`rskill.yaml`,
`README.md`) and the wrapped upstream weights at
`hf://Deepkar/libero-test-act`. Commercial use is allowed
(`is_commercial_use_allowed` is `true`).
