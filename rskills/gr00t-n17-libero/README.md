---
tags:
  - OpenRAL
  - rskill
  - gr00t
  - nvidia
  - vla
  - franka
  - libero
  - manipulation
license: other
language:
  - en
---

# rskill-gr00t-n17-libero

> **OpenRAL rSkill** — NVIDIA Isaac **GR00T N1.7** (3B) finetuned on the
> [LIBERO](https://libero-project.github.io/) benchmark, packaged for the
> [OpenRAL](https://github.com/OpenRAL/openral) robot agent framework.

This package wraps [`nvidia/GR00T-N1.7-LIBERO`](https://huggingface.co/nvidia/GR00T-N1.7-LIBERO)
with a `rskill.yaml` manifest that adds capability checking, license
surfacing, latency budgets, and local registry integration. It does **not**
copy model weights.

> **Runtime status.** This is a *packaging-and-validation* slice (ADR-0046
> PR1): the manifest, license posture, and `model_family: gr00t` are wired
> and tested. The out-of-process runtime adapter (`openral_sim.policies.gr00t`)
> and the `tools/gr00t_sidecar.py` boot helper land in ADR-0046 PR2, which
> also produces the locally-reproduced LIBERO eval numbers. Until then the
> skill packages and validates but is gracefully dropped from a live policy
> palette with an install hint.

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
| Franka Panda (LIBERO sim) | `franka_panda` | packaged | Native finetune embodiment; live eval in ADR-0046 PR2 |

GR00T exposes a `LIBERO_PANDA` embodiment tag internally; OpenRAL maps it to
the canonical `franka_panda` embodiment from `robots/`.

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
| `name` | `OpenRAL/rskill-gr00t-n17-libero` |
| `version` | `0.1.0` |
| `license` | `nvidia_open_model` (commercial OK) |
| `role` | `s1` |
| `model_family` | `gr00t` |
| `embodiment_tags` | `franka_panda` |
| `runtime` | `pytorch` (out-of-process sidecar, ADR-0046) |
| `quantization.dtype` | `bf16` |
| `weights_uri` | `hf://nvidia/GR00T-N1.7-LIBERO` |
| `chunk_size` | 16 |
| `state_contract.dim` / `action_contract.dim` | 8 / 7 |
| `latency_budget.per_chunk_ms` | 1500 ms (sidecar round-trip + 3B inference) |

Full schema: [`openral_core.schemas.RSkillManifest`](../../python/core/src/openral_core/schemas.py).

## Hardware

GR00T N1.7-3B (bf16, ~6 GB weights) plus the Cosmos-Reason VLM does not fit
on an 8 GB GPU without NF4 quantization; the sidecar (ADR-0046 PR2) follows
the NF4 isolated-venv recipe used by the RLDX and detector sidecars. A
≥ 16 GB GPU runs bf16 directly.

## License

This rSkill package (`rskill.yaml`, `README.md`) is **Apache-2.0**.

The wrapped model weights ([`nvidia/GR00T-N1.7-LIBERO`](https://huggingface.co/nvidia/GR00T-N1.7-LIBERO))
are governed by the **NVIDIA Open Model License Agreement**, which permits
commercial use. This is the key distinction from GR00T N1 / N1.5 / N1.6,
which ship under the NVIDIA OneWay Noncommercial License and are blocked in
commercial deployments by the OpenRAL loader unless
`OPENRAL_ALLOW_NONCOMMERCIAL=1` is set (CLAUDE.md §3, ADR-0046).
