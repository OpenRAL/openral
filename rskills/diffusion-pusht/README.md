---
tags:
  - OpenRAL
  - rskill
  - diffusion-policy
  - lerobot
  - pusht
  - manipulation
license: apache-2.0
language:
  - en
---

# rskill-diffusion-pusht

> **OpenRAL rSkill** — Diffusion Policy (Chi et al., 2023) trained on
> the PushT 2-D pushing benchmark, packaged for `OpenRAL`.

This package wraps [`lerobot/diffusion_pusht`](https://huggingface.co/lerobot/diffusion_pusht)
with a `rskill.yaml` manifest. It does **not** copy model weights.

## Upstream model

| Field | Value |
| --- | --- |
| Source repo | [`lerobot/diffusion_pusht`](https://huggingface.co/lerobot/diffusion_pusht) |
| Paper | [arxiv:2303.04137](https://arxiv.org/abs/2303.04137) — *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion* (Chi et al., 2023) |
| License | Apache-2.0 |
| Parameters | ~263 M (1-D U-Net) |
| Action chunk | 8 (within horizon 16) |
| Denoising | 100 DDPM steps per chunk |
| Benchmark | PushT (`gym_pusht`, `pymunk` 2-D rigid-body) |

Per-chunk inference is dominated by the 100-step denoising loop; cached
pops are essentially free, so this is the extreme test of the
queue-drain contract in `ChunkedExecutor`.

## Supported robots

| Robot | Embodiment tag | Status | Notes |
| --- | --- | --- | --- |
| PushT 2-D pseudo-robot (`gym_pusht/PushT-v0`) | `pusht`, `lerobot` | ✓ sim | 2-D end-effector pushing a T block on a 512 × 512 px canvas |

## Sensors required

| Key | Type | Resolution | Format |
| --- | --- | --- | --- |
| `observation.image` | RGB camera | 96 × 96 | `float32` |

PushT predates the multi-cam `observation.images.cameraN` convention and
exposes the raw key `observation.image`.

## Manifest summary

| Field | Value |
| --- | --- |
| `name` | `OpenRAL/rskill-diffusion-pusht` |
| `version` | `0.1.0` |
| `license` | `apache-2.0` |
| `role` | `s1` |
| `embodiment_tags` | `pusht`, `lerobot` |
| `runtime` / `quantization.dtype` | `pytorch` / `fp32` |
| `weights_uri` | `hf://lerobot/diffusion_pusht` |
| `latency_budget.per_chunk_ms` | 1 250 ms (warm full-chunk ≈ 1 756 ms on RTX 4070 Laptop, dominated by DDPM) |
| `latency_budget.warmup_ms` | 10 000 ms |
| `latency_budget.load_ms` | 30 000 ms |
| `commercial_use_allowed` | `true` |

Full schema: `openral_core.RSkillManifest` —
`python/core/src/openral_core/schemas.py`.

## Reproduction

```bash
git clone https://github.com/OpenRAL/openral && cd OpenRAL
just bootstrap && uv sync --all-packages --group sim

# End-to-end via the canonical SimEnvironment config (CPU is enough):
just sim-diffusion-pusht
# which runs:
#     openral sim run --config scenes/benchmark/pusht.yaml --rskill rskills/diffusion-pusht --save-video

# Sim test (gym_pusht + pymunk):
uv run pytest tests/sim/test_pusht_2d_diffusion_pusht.py -v -m sim
```

## License

This rSkill package (`rskill.yaml`, `README.md`) is **Apache-2.0** to
match the upstream weights. Commercial use is allowed
(`commercial_use_allowed: true`).

## See also

- [`robots/pusht_2d/README.md`](../../robots/pusht_2d/README.md) — RobotDescription manifest.
- [`scenes/benchmark/pusht.yaml`](../../scenes/benchmark/pusht.yaml) — paired BenchmarkScene config (pass `--rskill rskills/diffusion-pusht`).
- [`docs/reference/vla_compatibility.md`](../../docs/reference/vla_compatibility.md) — VLA × Robot × Sim matrix.
