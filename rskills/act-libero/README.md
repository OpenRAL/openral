---
tags:
  - OpenRAL
  - rskill
  - act
  - libero
license: apache-2.0
language:
  - en
---

# rskill-act-libero

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
| Franka Panda (LIBERO sim) | `franka_panda` (LIBERO embodiment tag) | ✓ sim | Native training target; closed-loop rollout reaches `is_success=True` on `libero_spatial/2` in ~91 steps on a single seed. |

## Sensors required

| Key | Modality | Resolution | Format |
| --- | --- | --- | --- |
| `observation.images.image`  | RGB | 256 × 256 | `float32` |
| `observation.images.image2` | RGB | 256 × 256 | `float32` |
| `observation.state`         | proprioception | (8,) | `float32` (LIBERO Franka layout: `pos3 + axisangle3 + grip2`) |

## Manifest summary

| Field | Value |
| --- | --- |
| `name` | `OpenRAL/rskill-act-libero` |
| `version` | `0.1.0` |
| `license` | `apache-2.0` |
| `role` | `s1` |
| `embodiment_tags` | `franka_panda` |
| `runtime` / `quantization.dtype` | `pytorch` / `fp32` |
| `weights_uri` | `hf://Deepkar/libero-test-act` |
| `chunk_size` | 100 |
| `commercial_use_allowed` | `true` |

Full schema: [`openral_core.schemas.RSkillManifest`](../../python/core/src/openral_core/schemas.py).

## Run

```bash
CC=/usr/bin/gcc uv sync --group libero      # first time only
openral sim run --config scenes/benchmark/libero_spatial.yaml --rskill rskills/act-libero \
           --rskill rskills/act-libero
```

The shipped sim YAML pins `libero_spatial/0` for a 200-step
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

The `state_contract.dim: 8` declaration confirms the proprio width.
Because the upstream training set is `HuggingFaceVLA/libero` — the same
dataset the smolvla / pi05 / xvla LIBERO checkpoints in this repo were
finetuned on — the state semantics (pos3 + axisangle3 + grip2) line up
with OpenRAL's LIBERO backend end-to-end, with no quat-vs-axisangle
mismatch.

## Benchmarks

None measured yet. Populate `eval/` with `openral benchmark run` JSON
fixtures before publishing a headline number.

## License

Apache-2.0 — both the wrapping rSkill package (`rskill.yaml`,
`README.md`) and the wrapped upstream weights at
`hf://Deepkar/libero-test-act`. Commercial use is allowed
(`commercial_use_allowed: true`).
