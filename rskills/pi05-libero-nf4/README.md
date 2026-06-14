---
tags:
  - OpenRAL
  - rskill
  - pi05
  - lerobot
  - vla
  - libero
  - manipulation
license: other
license_name: permissive-research
language:
  - en
---

# rskill-pi05-libero-nf4

> **OpenRAL rSkill** — π0.5 (3 B PaliGemma backbone, flow-matching
> action head) finetuned on the [LIBERO](https://libero-project.github.io/)
> benchmark. **Non-commercial weights** (Physical Intelligence permissive
> research license).

> ⚠ **License gate** — π0.5 weights are *not* Apache-2.0. The
> OpenRAL loader pins `commercial_use_allowed: false`; commercial
> deployment requires a separate agreement with Physical Intelligence
> (see CLAUDE.md §7.4 / Operating Principle 9). The loader requires
> `OPENRAL_ALLOW_NONCOMMERCIAL=1` (or the `--non-commercial` flag
> on `openral skill install`) to activate this skill.

## Quick start

```python
import os
os.environ["OPENRAL_ALLOW_NONCOMMERCIAL"] = "1"

from openral_rskill.loader import rSkill
pkg = rSkill.from_yaml("rskills/pi05-libero-nf4/rskill.yaml")
```

```bash
# CLI (will prompt to accept the non-permissive license unless --yes is passed):
uv run openral skill install OpenRAL/rskill-pi05-libero-nf4 --non-commercial --yes

# LIBERO closed-loop sim:
just sim-pi05-libero --no-run    # validate wiring only
just sim-pi05-libero             # full run (≥ 8 GB VRAM)
```

## Upstream model

| Field | Value |
| --- | --- |
| Source repo | [`lerobot/pi05_libero_finetuned_v044`](https://huggingface.co/lerobot/pi05_libero_finetuned_v044) |
| Base model | [`lerobot/pi05_base`](https://huggingface.co/lerobot/pi05_base) |
| Paper | [arxiv:2410.24164](https://arxiv.org/abs/2410.24164) — *π0: A Vision-Language-Action Flow Model for General Robot Control* |
| Architecture | PaliGemma 3 B backbone + flow-matching action head |
| Code license | Apache-2.0 |
| Weights license | **Physical Intelligence permissive research** (non-commercial) |
| Parameters | ~3 B |
| Benchmark | LIBERO |

## Supported robots

| Robot | Embodiment tag | Status | Notes |
| --- | --- | --- | --- |
| Franka Panda (LIBERO sim) | `libero`, `franka_panda` | ✓ matches | Native training embodiment. |
| Other 7-DoF arms | — | requires obs-format adapter | State dim is 8-D LIBERO-style. |

State dimensions, camera names, and normalisation statistics for this
checkpoint have **not** been locally verified — see
`eval/libero.json::status: pending` and the `notes` field of
`rskill.yaml`.

## Sensors required

| Key | Modality | Min resolution |
| --- | --- | --- |
| `observation.images.camera1` | RGB | 224 × 224 |
| `observation.images.camera2` | RGB | 224 × 224 |

## Manifest summary

| Field | Value |
| --- | --- |
| `name` | `OpenRAL/rskill-pi05-libero-nf4` |
| `version` | `0.1.0` |
| `license` | `permissive_research` |
| `role` | `s1` |
| `runtime` / `quantization.dtype` | `pytorch` / `bf16` |
| `weights_uri` | `hf://lerobot/pi05_libero_finetuned_v044` |
| `latency_budget.per_chunk_ms` | 200 ms (3 B model is heavier than SmolVLA) |
| `latency_budget.warmup_ms` / `load_ms` | 15 000 ms / 60 000 ms |
| `commercial_use_allowed` | **`false`** |

Full schema: `openral_core.RSkillManifest`.

## Evaluation

`eval/libero.json::status` is **pending** — no locally-reproduced numbers
yet. The reproduction CLI placeholder is recorded in
`eval/libero.json::source.reproduction_cli.placeholder`; do not add
numbers to this README without either a locally-verified run or a
precise paper citation with table reference.

## Reproduction

```bash
just bootstrap && uv sync --all-packages

# Validate wiring (downloads the manifest + 3 B weights; no rollout):
just sim-pi05-libero --no-run

# Full LIBERO closed-loop run (requires ≥ 8 GB VRAM):
OPENRAL_ALLOW_NONCOMMERCIAL=1 just sim-pi05-libero
```

## License

This rSkill package (`rskill.yaml`, `README.md`, `eval/libero.json`)
is **Apache-2.0**. The wrapped weights are released under Physical
Intelligence's *permissive research* license — review the upstream
license file before any deployment beyond research.

## See also

- [`rskills/smolvla-libero/README.md`](../smolvla-libero/README.md) — Apache-2.0 LIBERO alternative.
- [`rskills/xvla-libero/README.md`](../xvla-libero/README.md) — xVLA LIBERO finetune.
- [`docs/reference/vla_compatibility.md`](../../docs/reference/vla_compatibility.md) §3.1 — VLA × Robot × Sim matrix.
- CLAUDE.md §7.4 — VLA license matrix and install-time guard rules.
