---
tags:
  - OpenRAL
  - rskill
  - act
  - lerobot
  - aloha
  - bimanual
  - manipulation
license: mit
language:
  - en
---

# rskill-act-aloha

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
> `PolicyProcessorPipeline` migration and ships **without normalisation
> buffers**. See `tests/sim/test_aloha_bimanual_act_aloha.py` for the resulting
> numerical-contract caveats.

## Supported robots

| Robot | Embodiment tag | Status | Notes |
| --- | --- | --- | --- |
| ALOHA bimanual (Trossen) — `gym-aloha` MuJoCo | `aloha`, `lerobot` | ✓ sim | 14-DoF (2 × 7-DoF arms with parallel grippers) |

## Sensors required

| Key | Type | Resolution | Format |
| --- | --- | --- | --- |
| `observation.images.top` | RGB camera | 640 × 480 | `float32` |

ACT for ALOHA cube-transfer ships with a single top-down RGB stream. No
wrist or third-person view.

## Manifest summary

| Field | Value |
| --- | --- |
| `name` | `OpenRAL/rskill-act-aloha` |
| `version` | `0.1.0` |
| `license` | `mit` |
| `role` | `s1` |
| `embodiment_tags` | `aloha`, `lerobot` |
| `runtime` / `quantization.dtype` | `pytorch` / `fp32` |
| `weights_uri` | `hf://lerobot/act_aloha_sim_transfer_cube_human` |
| `latency_budget.per_chunk_ms` | 25 ms (warm; bf16 autocast ≈ 12 ms on RTX 4070 Laptop) |
| `latency_budget.warmup_ms` | 5 000 ms |
| `latency_budget.load_ms` | 10 000 ms |
| `commercial_use_allowed` | `true` |

Full schema: `openral_core.RSkillManifest` —
`python/core/src/openral_core/schemas.py`.

## Reproduction

```bash
git clone https://github.com/OpenRAL/openral && cd OpenRAL
just bootstrap && uv sync --all-packages --group sim

# End-to-end via the canonical SimEnvironment config:
just sim-act-aloha
# which runs:
#     openral sim run --config scenes/benchmark/aloha_transfer_cube.yaml --rskill rskills/act-aloha --save-video

# Sim test (real gym-aloha MuJoCo with contact dynamics):
uv run pytest tests/sim/test_aloha_bimanual_act_aloha.py -v -m sim
```

## License

This rSkill package (`rskill.yaml`, `README.md`) is **MIT** to match the
upstream weights. Commercial use is allowed
(`commercial_use_allowed: true`).

## See also

- [`robots/aloha_bimanual/README.md`](../../robots/aloha_bimanual/README.md) — RobotDescription manifest.
- [`scenes/benchmark/aloha_transfer_cube.yaml`](../../scenes/benchmark/aloha_transfer_cube.yaml) — paired BenchmarkScene config (pass `--rskill rskills/act-aloha`).
- [`docs/reference/vla_compatibility.md`](../../docs/reference/vla_compatibility.md) — VLA × Robot × Sim matrix.
