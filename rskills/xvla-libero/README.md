---
tags:
  - OpenRAL
  - rskill
  - xvla
  - lerobot
  - vla
  - libero
  - manipulation
license: apache-2.0
language:
  - en
---

# rskill-xvla-libero

> **OpenRAL rSkill** — xVLA (lerobot ecosystem VLA) finetuned on the
> [LIBERO](https://libero-project.github.io/) benchmark. Apache-2.0
> weights; locally **unverified** (TBD checkpoint inspection — see
> `eval/libero.json`).

## Quick start

```python
from openral_rskill.loader import rSkill
pkg = rSkill.from_yaml("rskills/xvla-libero/rskill.yaml")
```

```bash
just sim-xvla-libero --no-run   # validate wiring only
just sim-xvla-libero            # full run (LIBERO sim; needs GPU + MUJOCO_GL)
```

## Upstream model

| Field | Value |
| --- | --- |
| Source repo | [`lerobot/xvla-libero`](https://huggingface.co/lerobot/xvla-libero) |
| Base model | [`lerobot/xvla-base`](https://huggingface.co/lerobot/xvla-base) |
| Paper | TBD |
| Code license | Apache-2.0 |
| Weights license | Apache-2.0 |
| Benchmark | LIBERO |

State dimensions, camera names, normalisation statistics, and the
underlying paper are TBD pending checkpoint inspection — see
`eval/libero.json::status: pending` and
`docs/reference/vla_compatibility.md` §3.1.

## Supported robots

| Robot | Embodiment tag | Status | Notes |
| --- | --- | --- | --- |
| Franka Panda (LIBERO sim) | `libero`, `franka_panda` | ✓ matches manifest | Native training embodiment (assumed). |
| Other 7-DoF arms | — | needs adapter | A third 224×224 zero-tensor slot is filled in-process by the runner; it does not correspond to a physical sensor. |

## Sensors required

| Key | Modality | Min resolution |
| --- | --- | --- |
| `observation.images.camera1` | RGB | 224 × 224 |
| `observation.images.camera2` | RGB | 224 × 224 |

## Manifest summary

| Field | Value |
| --- | --- |
| `name` | `OpenRAL/rskill-xvla-libero` |
| `version` | `0.1.0` |
| `license` | `apache-2.0` |
| `role` | `s1` |
| `runtime` / `quantization.dtype` | `pytorch` / `bf16` |
| `weights_uri` | `hf://lerobot/xvla-libero` |
| `latency_budget.per_chunk_ms` | 200 ms (conservative — update after local profiling) |
| `latency_budget.warmup_ms` / `load_ms` | 15 000 ms / 60 000 ms |
| `commercial_use_allowed` | `true` |

Full schema: `openral_core.RSkillManifest`.

## Evaluation

`eval/libero.json::status` is **pending** — no locally-reproduced numbers
and no paper to cite. Do not populate this README's numbers without
either a locally-verified run or a precise paper citation with table
reference (per CLAUDE.md operating principle 2: *truth over plausibility*).

## License

This rSkill package (`rskill.yaml`, `README.md`, `eval/libero.json`)
is **Apache-2.0**. The wrapped weights are also Apache-2.0 per the
upstream repo. Commercial use is allowed.

## See also

- [`rskills/smolvla-libero/README.md`](../smolvla-libero/README.md) — gold-standard LIBERO finetune (locally verified).
- [`rskills/pi05-libero-nf4/README.md`](../pi05-libero-nf4/README.md) — π0.5 LIBERO finetune (non-commercial).
- [`docs/reference/vla_compatibility.md`](../../docs/reference/vla_compatibility.md) §3.1.
