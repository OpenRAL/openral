---
language:
- en
license: other
license_name: permissive-research
library_name: lerobot
pipeline_tag: robotics
tags:
- OpenRAL
- rskill
- pi05
- lerobot
- vision-language-action
- int8
- 8-bit
- franka_panda
- nf4
- 4-bit
- vla
- libero
- manipulation
inference: false
base_model:
- lerobot/pi05_libero_finetuned_v044
base_model_relation: quantized
---

# rskill-pi05-franka_panda-libero_spatial-int8

> **OpenRAL rSkill** — π0.5 (3 B PaliGemma backbone, flow-matching
> action head) finetuned on the [LIBERO](https://libero-project.github.io/)
> benchmark, loaded bf16 and **LLM.int8-quantized in place** so it runs on
> an 8 GB consumer GPU. **Non-commercial weights** (Physical Intelligence
> permissive research license).

> ⚠ **License gate** — π0.5 weights are *not* Apache-2.0. The
> OpenRAL loader pins `commercial_use_allowed: false`; commercial
> deployment requires a separate agreement with Physical Intelligence
> (see CLAUDE.md §7.4 / Operating Principle 9). The loader requires
> `OPENRAL_ALLOW_NONCOMMERCIAL=1` (or the `--non-commercial` flag
> on `openral skill install`) to activate this skill.

## Why int8 (and not NF4)

π0.5's 3.4 B backbone is **too large to run bf16 on 8 GB** (~13.6 GiB peak → OOM),
but **NF4 4-bit is too lossy for it** — 4-bit quantization scores **0/5** on
`libero_spatial` because it destroys the policy. **LLM.int8**
(bitsandbytes `Linear8bitLt` on every Linear ≥ 4 M weight elements) is the sweet
spot: it both fits 8 GB *and* preserves competence — **~0.5–0.7 success** on
`libero_spatial` across 10-episode runs on an RTX 4070 (`eval/scene_libero_spatial.json`
records 0.5). Quantization runs at load from the bf16 checkpoint (no prequantized pack).

## Quick start

```python
import os
os.environ["OPENRAL_ALLOW_NONCOMMERCIAL"] = "1"

from openral_rskill.loader import rSkill
pkg = rSkill.from_yaml("rskills/pi05-libero-int8/rskill.yaml")
```

```bash
# CLI (will prompt to accept the non-permissive license unless --yes is passed):
uv run openral skill install OpenRAL/rskill-pi05-franka_panda-libero_spatial-int8 --non-commercial --yes

# LIBERO closed-loop sim (int8 fits 8 GB):
PYTORCH_ALLOC_CONF=expandable_segments:True \
  openral benchmark scene --config scenes/benchmark/libero_spatial.yaml \
    --rskill rskills/pi05-libero-int8
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
| Franka Panda (LIBERO sim) | `franka_panda` | ✓ matches | Native training embodiment. |
| Other 7-DoF arms | — | requires obs-format adapter | State dim is 8-D LIBERO-style. |

## Sensors required

| Key | Modality | Min resolution |
| --- | --- | --- |
| `observation.images.camera1` | RGB | 224 × 224 |
| `observation.images.camera2` | RGB | 224 × 224 |

## Manifest summary

| Field | Value |
| --- | --- |
| `name` | `OpenRAL/rskill-pi05-franka_panda-libero_spatial-int8` |
| `version` | `0.1.0` |
| `license` | `permissive_research` |
| `role` | `s1` |
| `runtime` / `quantization.dtype` | `pytorch` / `int8` |
| `weights_uri` | `hf://lerobot/pi05_libero_finetuned_v044` |
| `latency_budget.per_chunk_ms` | 200 ms (3 B model is heavier than SmolVLA) |
| `commercial_use_allowed` | **`false`** |

Full schema: `openral_core.RSkillManifest`.

## Evaluation

`eval/scene_libero_spatial.json` holds the locally-reproduced result:
**`libero_spatial` = 0.50 (5/10 episodes)**, int8, RTX 4070 8 GB (a separate
10-episode run scored 0.70; the policy sits around 0.5–0.7). Re-run with:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True OPENRAL_ALLOW_NONCOMMERCIAL=1 \
  openral benchmark scene --config scenes/benchmark/libero_spatial.yaml \
    --rskill rskills/pi05-libero-int8 --n-episodes 10 --write-eval
```

## License

This rSkill package (`rskill.yaml`, `README.md`, `eval/scene_libero_spatial.json`)
is **Apache-2.0**. The wrapped weights are released under Physical
Intelligence's *permissive research* license — review the upstream
license file before any deployment beyond research.

## See also

- [`rskills/smolvla-libero/README.md`](../smolvla-libero/README.md) — Apache-2.0 LIBERO alternative.
- [`docs/reference/vla_compatibility.md`](../../docs/reference/vla_compatibility.md) §3.1 — VLA × Robot × Sim matrix.
- CLAUDE.md §7.4 — VLA license matrix and install-time guard rules.
