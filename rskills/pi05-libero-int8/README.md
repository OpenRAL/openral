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

π0.5's 3.4 B backbone **does not fit bf16 on an 8 GB card** — bf16 peaks at
**8.06 GB** of device memory, and an 8 GB GPU has less than that free once the
driver takes its cut. **NF4 4-bit** fits but is too lossy for this backbone: it
scores **0/5** on `libero_spatial`, destroying the policy outright. **LLM.int8**
(bitsandbytes `Linear8bitLt` on every Linear ≥ 4 M weight elements) fits at
**6.35 GB** and keeps the policy working. Quantization runs at load from the
bf16 checkpoint (no prequantized pack).

### …but int8 is not free — prefer bf16 if it fits

int8 is a *memory* fit. It costs accuracy and speed, and neither cost was
recorded here before. Measured on a GB10 — `libero_spatial` tasks 0/1/2, seed
42, `max_steps=220`, **n=50 per task per arm (300 episodes)**, same episodes and
protocol, the only difference being `quantization.dtype`:

| dtype | task 0 | task 1 | task 2 | **Total** | Mean step latency | Peak device memory |
|---|---|---|---|---|---|---|
| **bf16** | 98% | 94% | 94% | **143/150 = 95%** | **9.6 ms** | 8.06 GB |
| int8 | 76% | **22%** | 82% | 93/150 = 62% | 24.8 ms | **6.35 GB** |

int8 costs **~33 points of task success** (Fisher exact **p = 4e-13**) and runs
**~2.6× slower per step** — LLM.int8's mixed-precision decomposition is slower
than a plain bf16 GEMM.

**Read the per-task columns, not just the total.** bf16 is steady (98/94/94);
int8 swings between 22% and 82%. The damage is *task-dependent*, so a
single-task spot-check can land on 82% and look acceptable while task 1 is
nearly non-functional. That variance is the real hazard — more than the average.

What int8 buys is **1.7 GB**, and on an 8 GB card that is decisive: bf16 will
not load at all, and 62% beats not running.

**On any host with ≳9 GB of VRAM, use bf16.** No manifest edit is needed:

```bash
openral sim run --config scenes/sim/libero_spatial.yaml \
    --rskill rskills/pi05-libero-int8 --vla-extra dtype=bf16
```

Faster still on NVIDIA hardware with TensorRT: an FP8 engine reaches **78.9 ms**
per policy call against bf16 eager's 190 ms and scores **49/50 — identical to
bf16**, so FP8 costs no task success at all. See the `openral-pro` TensorRT
runtime (`OPENRAL_PI05_TRT=1`, `OPENRAL_PI05_TRT_PRECISION=fp8_bf16_attn`).

Scope: one task of LIBERO-Spatial's ten, so treat 98%/76% as a like-for-like
*comparison* rather than a suite score. The older "0.5–0.7 across 10-episode
runs" figure is not contradicted so much as too noisy to use — at n=10 the
binomial SD is ~0.16, wide enough to cover both arms above.

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
10-episode run scored 0.70; the policy sits around 0.5–0.7).

⚠️ **n=10 is too small to compare anything with.** Its binomial SD is ~0.16, so
that 0.50 and the 0.76 measured at n=50 above are the same result, not a
regression or an improvement. Prefer the n=50 numbers in [Why int8](#why-int8-and-not-nf4)
when deciding a dtype, and treat the recorded 0.50 as provenance for the
committed eval artifact rather than as this policy's success rate.

Re-run with:

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
