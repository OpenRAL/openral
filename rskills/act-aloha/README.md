---
language:
- en
license: mit
library_name: lerobot
pipeline_tag: robotics
tags:
- OpenRAL
- rskill
- act
- lerobot
- vision-language-action
- aloha
- bimanual
- manipulation
inference: false
---

# rskill-act-aloha-aloha_transfer_cube-fp32

> **OpenRAL rSkill** — ACT (Action Chunking Transformer) finetuned on
> the ALOHA bimanual cube-transfer task, packaged for `OpenRAL`.

This package wraps
[`lerobot/act_aloha_sim_transfer_cube_human`](https://huggingface.co/lerobot/act_aloha_sim_transfer_cube_human)
with a `rskill.yaml` manifest that adds capability checking, license
surfacing, latency budgets, and local registry integration. It does
**not** copy model weights.

## Upstream model

| Field | Value |
| --- | --- |
| Source repo | [`lerobot/act_aloha_sim_transfer_cube_human`](https://huggingface.co/lerobot/act_aloha_sim_transfer_cube_human) |
| Paper | [arxiv:2304.13705](https://arxiv.org/abs/2304.13705) — *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (Zhao et al., 2023) |
| License | MIT |
| Parameters | ~52 M (transformer encoder-decoder) |
| Action chunk | 100 |
| Benchmark | ALOHA bimanual cube-transfer (`gym-aloha`) |

> **Note.** The published checkpoint predates lerobot's
> `PolicyProcessorPipeline` migration: its mean/std normalisation buffers live
> inside `model.safetensors`, modern lerobot `ACTPolicy` drops them on load, and
> the adapter re-applies them. See `tests/sim/test_aloha_bimanual_act_aloha.py`.

## Supported robots

| Robot | Embodiment tag | Status | Notes |
| --- | --- | --- | --- |
| ALOHA bimanual (Trossen) — `gym-aloha` MuJoCo | `aloha` | ✓ sim | 14-DoF (2 × 7-DoF arms with parallel grippers) |

## Sensors required

| Key | Type | Resolution | Format |
| --- | --- | --- | --- |
| `observation.images.top` | RGB camera | 640 × 480 | `float32` |

ACT for ALOHA cube-transfer ships with a single top-down RGB stream. No
wrist or third-person view.

## Manifest summary

| Field | Value |
| --- | --- |
| `name` | `OpenRAL/rskill-act-aloha-aloha_transfer_cube-fp32` |
| `version` | `0.1.0` |
| `license` | `mit` |
| `role` | `s1` |
| `embodiment_tags` | `aloha` |
| `runtime` / `quantization.dtype` | `pytorch` / `fp32` |
| `weights_uri` | `hf://lerobot/act_aloha_sim_transfer_cube_human` |
| `latency_budget.per_chunk_ms` | 25 ms (warm; bf16 autocast ≈ 12 ms on RTX 4070 Laptop) |
| `is_commercial_use_allowed` (derived from `license`) | `true` |

Full schema: `openral_core.RSkillManifest` —
`python/core/src/openral_core/schemas.py`.

## Reproduction

```bash
git clone https://github.com/OpenRAL/openral && cd openral
just bootstrap && just sync --group sim

# One-episode BenchmarkScene demo:
just sim-act-aloha
# which runs:
#     openral benchmark scene --config scenes/benchmark/aloha_transfer_cube.yaml \
#       --rskill rskill://rskills/act-aloha --no-update-manifest --n-episodes 1 --save-dir example_videos

# Full 50-episode protocol (eval/aloha_transfer_cube.json):
openral benchmark run --suite aloha --rskill act-aloha

# Sim test (real gym-aloha MuJoCo with contact dynamics):
uv run pytest tests/sim/test_aloha_bimanual_act_aloha.py -v -m sim
```

## License

OpenRAL wrapper files follow the project Apache-2.0 license. The upstream
weights are **MIT**, so the manifest declares `license: mit` and commercial use
is allowed (`is_commercial_use_allowed` is `true`).

## See also

- [`robots/aloha_bimanual/README.md`](../../robots/aloha_bimanual/README.md) — RobotDescription manifest.
- [`scenes/benchmark/aloha_transfer_cube.yaml`](../../scenes/benchmark/aloha_transfer_cube.yaml) — paired BenchmarkScene config (pass `--rskill rskills/act-aloha`).
- [`docs/reference/vla_compatibility.md`](../../docs/reference/vla_compatibility.md) — VLA × Robot × Sim matrix.
