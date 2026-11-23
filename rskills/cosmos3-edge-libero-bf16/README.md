---
language:
- en
license: other
license_name: openmdw-1.1
library_name: lerobot
pipeline_tag: robotics
tags:
- OpenRAL
- rskill
- cosmos3
- lerobot
- vision-language-action
- franka_panda
- rskill-kind:vla
- rskill-role:s1
- embodiment:franka_panda
- rskill-license:openmdw-1.1
base_model:
- nvidia/Cosmos3-Edge
base_model_relation: finetune
datasets:
- lerobot/LIBERO-Cosmos3
inference: false
openral:
  description: 'NVIDIA Cosmos 3 Edge world action model fine-tuned on all four LIBERO
    suites, run in-process through lerobot''s Cosmos3Policy as an action policy. From
    the task instruction and the third-person + wrist views it samples (UniPC, 30
    steps) a 16-step chunk of end-effector deltas for the Franka Panda together with
    the future video, which is discarded. Model card: 97.95 % average over LIBERO
    Spatial / Object / Goal / 10 (MuJoCo 2.3.7). bf16.'
  model_family: cosmos3
  min_vram_gb: 14.1
  latency_ms: 1500.0
  runtime: pytorch
  dtype: bf16
---

# rskill-cosmos3-franka_panda-libero-bf16

> **OpenRAL rSkill** — NVIDIA Cosmos 3 Edge fine-tuned on LIBERO, run **in-process**
> through lerobot's `Cosmos3Policy` as an action policy for the Franka Panda
> (`franka_panda`).

This package contains the OpenRAL manifest only. The weights are
[`lerobot/Cosmos3-Edge-Policy-LIBERO`](https://huggingface.co/lerobot/Cosmos3-Edge-Policy-LIBERO),
pinned by commit: [`nvidia/Cosmos3-Edge`](https://huggingface.co/nvidia/Cosmos3-Edge)
converted to LeRobot format and fine-tuned by the lerobot team on the four LIBERO suites
([`lerobot/LIBERO-Cosmos3`](https://huggingface.co/datasets/lerobot/LIBERO-Cosmos3)).

## Preview

No rollout media ships with this package yet. Three `libero_spatial` episodes were run
on an RTX 5090 (see [Evaluation](#evaluation)).

## What this skill does

A LIBERO tabletop manipulation policy for the Franka Panda: pick-and-place, drawers and
the other LIBERO Spatial / Object / Goal / 10 tasks, from a third-person view, a wrist
view and a language instruction. It samples a 16-step chunk of end-effector deltas and
executes the whole chunk before sampling the next one.

Cosmos 3 is a world action model: it denoises the action chunk together with the future
video of the scene. This skill keeps only the actions; the video is never decoded. It is
not the `cosmos3-edge` reasoner, which serves the base model's text tower for S2
planning.

## How it works

`openral_sim.policies.cosmos3` loads the pinned snapshot, validates every step of its
`policy_{pre,post}processor.json`, and builds lerobot's `Cosmos3Policy` in bf16 on the
GPU. The checkpoint's own pipeline builds the side-by-side two-view video canvas, the
prompt and the tokens. Its postprocessor unnormalises the 10-D
`[dpos(3), rot6d(6), gripper]` action, and lerobot's `Cosmos3LiberoActionProcessorStep`
turns it into the 7-D LIBERO command (axis-angle, gripper −1 open / +1 closed), as
`lerobot-eval` does for a Cosmos 3 policy on LIBERO. Sampling: UniPC, 30 steps, no
classifier-free guidance.

### Observation → action contract

| In | Shape | Notes |
| --- | --- | --- |
| `observation.images.image` / `image2` | HWC uint8 RGB | slots `camera1` (third person) / `camera2` (wrist) via `image_preprocessing.aliases`; rotated 180° (`flip_180`) as the dataset holds them; 256x256 |
| `observation.state` | `(8,)` float32 | LIBERO `eef_pos(3) | axis-angle(3) | gripper_qpos(2)`; declared by the checkpoint, not used by the model (`use_state: false`) |
| `task` | str | the LIBERO instruction |

Out: a 7-D LIBERO OSC delta command per step `[dpos(3), daxis-angle(3), gripper]`.

## Upstream model / training

| | |
| --- | --- |
| Model | NVIDIA Cosmos 3 Edge: Nemotron 3 Dense VL Mixture-of-Transformers (28 layers, width 2048) + Wan2.2 VAE |
| Checkpoint | LIBERO fine-tune (all four suites), EMA weights at optimizer update 4000 |
| Paper | [arXiv 2606.02800](https://arxiv.org/abs/2606.02800) |
| Code | [NVIDIA/cosmos-framework](https://github.com/NVIDIA/cosmos-framework), lerobot port `lerobot.policies.cosmos3` (lerobot>=0.7) |

The model card reports 97.95 % average success over LIBERO Spatial / Object / Goal / 10
(500 episodes per suite, MuJoCo 2.3.7, A800). Those numbers are the upstream's, not
reproduced here.

## Supported robots

| Robot | Embodiment tag | Status | Notes |
| --- | --- | --- | --- |
| Franka Panda (LIBERO, MuJoCo) | `franka_panda` | sim | OSC delta control |

## Sensors required

| Key | Modality | Min resolution | Format |
| --- | --- | --- | --- |
| `observation.images.camera1` | RGB (third person) | 256 x 256 | HWC uint8 |
| `observation.images.camera2` | RGB (wrist) | 256 x 256 | HWC uint8 |

## Manifest summary

| Field | Value |
| --- | --- |
| `name` | `OpenRAL/rskill-cosmos3-franka_panda-libero-bf16` |
| `model_family` | `cosmos3` |
| `license` | `openmdw-1.1` (weights); OpenRAL code Apache-2.0 |
| `quantization.dtype` | `bf16` |
| `min_vram_gb` | bf16 14.1 (measured peak, renderer included) |
| `weights_uri` | `hf://lerobot/Cosmos3-Edge-Policy-LIBERO@9d4de4350e416c12babc9a42db01ab0c5df9f82f` |
| `chunk_size` / `n_action_steps` | 16 / 16 |
| `state_contract.dim` / `action_contract.dim` | 8 / 7 |

## Quick start

```bash
just sync --group libero
MUJOCO_GL=egl openral benchmark run --suite libero_spatial \
  --rskill rskills/cosmos3-edge-libero-bf16 \
  --task libero_spatial/0 --n-episodes 1 --no-update-manifest
```

Needs a CUDA GPU with at least 16 GB.

## Reproduction

The command above, for tasks 0–2, is the run recorded under [Evaluation](#evaluation).
`openral benchmark run --suite libero_spatial --rskill rskills/cosmos3-edge-libero-bf16`
over the full suite populates `eval/`.

## Evaluation

| Task | Episodes | Success | Steps | Mean step latency | Host |
| --- | --- | --- | --- | --- | --- |
| `libero_spatial/0` | 1 (seed 0) | 1/1 | 75 | 49.8 ms | RTX 5090, bf16 |
| `libero_spatial/1` | 1 (seed 0) | 1/1 | 114 | 50.8 ms | RTX 5090, bf16 |
| `libero_spatial/2` | 1 (seed 0) | 1/1 | 100 | 52.4 ms | RTX 5090, bf16 |

Peak GPU memory 14.1 GiB. This is a smoke check of the port, not a benchmark claim;
`eval/` is empty until a full suite run. The run used the workspace's MuJoCo 3.x; the
model card warns that contact-rich LIBERO tasks differ under MuJoCo 3.x versus the
2.3.7 it was measured with. The card's numbers also ran the preprocessor's video-canvas
step on the CPU; this adapter runs the whole preprocessor on the policy device, which the
card notes shifts the inputs and the sampling by small numerical amounts. The 1500 ms
`per_chunk_ms` budget is an estimate from the mean step time, not a timed chunk.

## License

Weights: [OpenMDW 1.1](https://openmdw.ai/license/1-1/) — commercial and non-commercial
use permitted; keep the license and NOTICE when redistributing. OpenRAL's adapter code is
Apache-2.0.

## See also

- [`xvla-libero`](../xvla-libero/) — another LIBERO policy on the same scene contract.
- [`docs/reference/vla_compatibility.md`](../../docs/reference/vla_compatibility.md)
