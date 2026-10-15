---
language:
- en
license: other
license_name: nvidia-open-model-license
pipeline_tag: robotics
tags:
- OpenRAL
- rskill
- gr00t
- vision-language-action
- nf4
- 4-bit
- franka_panda
- nvidia
- vla
- franka
- libero
- manipulation
base_model:
- nvidia/GR00T-N1.7-LIBERO
base_model_relation: quantized
inference: false
---

# rskill-gr00t_n17-franka_panda-libero_spatial-bf16

> **OpenRAL rSkill** — NVIDIA Isaac **GR00T N1.7** (3B) finetuned on the
> [LIBERO](https://libero-project.github.io/) benchmark, packaged for the
> [OpenRAL](https://github.com/OpenRAL/openral) robot agent framework.

This package wraps [`nvidia/GR00T-N1.7-LIBERO`](https://huggingface.co/nvidia/GR00T-N1.7-LIBERO)
with a `rskill.yaml` manifest that adds capability checking, license
surfacing, latency budgets, and local registry integration. Its `weights_uri`
is an OpenRAL root-level repackage of the upstream `libero_spatial/`
subfolder checkpoint (the upstream repo has no root `model.safetensors`).

> **Runtime status.** The manifest, license posture, and `model_family: gr00t`
> are wired and tested. `openral_sim.policies.gr00t` runs GR00T N1.7
> **in-process** under the workspace's Python 3.12 via lerobot 0.6.0's native
> `GrootPolicy` (backbone-only NF4 quantization), like the SmolVLA/π0.5
> adapters; there is no out-of-process sidecar and no `tools/gr00t_sidecar.py`.
> (`tools/behavior_groot_sidecar.py` is unrelated: it drives the external
> Python 3.10 `wensi-ai/Isaac-GR00T` runtime for the official 2026 BEHAVIOR-1K
> organizer checkpoint.) `eval/libero_spatial.json` carries
> `reproduced_locally: true`: 1 episode per task over the 10 LIBERO-Spatial
> tasks (seed 0, max 280 steps), `avg_success_rate` 0.9.

## Upstream model, architecture & training

| Field | Value |
|---|---|
| Source repo | [`nvidia/GR00T-N1.7-LIBERO`](https://huggingface.co/nvidia/GR00T-N1.7-LIBERO) |
| Base model | [`nvidia/GR00T-N1.7-3B`](https://huggingface.co/nvidia/GR00T-N1.7-3B) |
| VLM backbone | Cosmos-Reason2-2B (SigLip2 vision encoder) |
| Paper | [arXiv:2503.14734](https://arxiv.org/abs/2503.14734) — *GR00T N1: An Open Foundation Model for Generalist Humanoid Robots* |
| Parameters | ~3.1 B |
| License | NVIDIA Open Model License Agreement (**commercial use permitted**) |
| Pretraining | 20K hours EgoScale human video + diverse robot demonstrations |
| Finetune | LIBERO task suite (Franka Panda, MuJoCo via robosuite) |

GR00T is a cross-embodiment foundation model with variable-dimension
proprioception and per-embodiment action heads. This checkpoint specializes
the N1.7 base on the LIBERO Franka embodiment.

## Supported robots / embodiments

| Robot | Embodiment tag | Status | Notes |
|---|---|---|---|
| Franka Panda (LIBERO sim) | `franka_panda` | ✓ sim | Native finetune embodiment; see `eval/libero_spatial.json` |

GR00T uses its own embodiment namespace: the manifest pins
`policy_extras.embodiment_tag: libero_sim` (override with
`OPENRAL_GR00T_EMBODIMENT_TAG`), while the OpenRAL capability gate matches the
canonical `franka_panda` tag from `robots/`.

## Sensors / observation contract

| Key | Type | Min resolution | Description |
|---|---|---|---|
| `observation.images.camera1` | RGB camera | 224 × 224 | Agentview / overhead |
| `observation.images.camera2` | RGB camera | 224 × 224 | Wrist / end-effector |
| state | Proprioception | (8,) | End-effector pose + gripper, LIBERO layout |

The policy emits a 16-step action chunk; each action is 7-D
(`delta_ee_6d_plus_gripper`). Images and state are normalized inside the
GR00T checkpoint's own `experiment_cfg` metadata rather than a lerobot
processor pipeline — hence no `processors` block in the manifest.

## Manifest summary

| Field | Value |
|---|---|
| `name` | `OpenRAL/rskill-gr00t_n17-franka_panda-libero_spatial-bf16` |
| `version` | `0.1.0` |
| `license` | `nvidia_open_model` (commercial OK) |
| `role` | `s1` |
| `model_family` | `gr00t` |
| `embodiment_tags` | `franka_panda` |
| `runtime` | `pytorch` (in-process lerobot `GrootPolicy`) |
| `quantization.dtype` | `int4` (NF4 at load; stored bf16 → `quantization.extra.stored_dtype`) |
| `weights_uri` | `hf://OpenRAL/rskill-gr00t_n17-franka_panda-libero_spatial-bf16` |
| `chunk_size` | 16 |
| `state_contract.dim` / `action_contract.dim` | 8 / 7 |
| `latency_budget.per_chunk_ms` | 1500 ms |

Full schema: [`openral_core.schemas.RSkillManifest`](../../python/core/src/openral_core/schemas.py).

## Hardware

GR00T N1.7-3B unpacked (~6 GB) does not co-fit an 8 GB GPU, so the in-process
adapter NF4-packs the Qwen3-VL backbone at load (~5.2 GiB peak per the
manifest; the diffusion action head stays bf16). A ≥ 16 GB GPU can skip packing
with `OPENRAL_QUANTIZATION_DTYPE=fp32` (the unpacked load runs fp32 params;
the adapter accepts only `nf4` / `fp32` / `none`, so there is no bf16 path).

## License

This rSkill package (`rskill.yaml`, `README.md`) is **Apache-2.0**.

The wrapped model weights ([`nvidia/GR00T-N1.7-LIBERO`](https://huggingface.co/nvidia/GR00T-N1.7-LIBERO))
are governed by the **NVIDIA Open Model License Agreement**, which permits
commercial use. This is the key distinction from GR00T N1 / N1.5 / N1.6,
which ship under the NVIDIA OneWay Noncommercial License and are blocked in
commercial deployments by the OpenRAL loader unless
`OPENRAL_ALLOW_NONCOMMERCIAL=1` is set (CLAUDE.md §3).
