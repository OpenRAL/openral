---
tags:
  - OpenRAL
  - rskill
  - molmoact2
  - transformers
  - vla
  - libero
  - manipulation
license: apache-2.0
language:
  - en
---

# rskill-molmoact2-libero-nf4

> **OpenRAL rSkill** — MolmoAct2 (Ai2's open action reasoning model: a
> Molmo2-ER embodied-reasoning VLM backbone with a flow-matching
> continuous-action expert) finetuned on the
> [LIBERO](https://libero-project.github.io/) benchmark and NF4-quantized so
> the ~5.5 B-param model fits an 8 GB GPU. Robot: Franka Panda in simulation.
> **Apache-2.0 weights** — commercial use permitted.

This package wraps `hf://OpenRAL/rskill-molmoact2-libero-nf4` (an
NF4-quantized mirror of `allenai/MolmoAct2-LIBERO`) with a `rskill.yaml`
manifest that adds capability checking, license surfacing, latency budgets,
and local registry integration. It does **not** copy model weights — they
live on the Hub.

## What this skill does

Performs LIBERO table-top manipulation — picking, placing, opening, and
closing on bowls, cups, drawers, and miscellaneous objects — on a Franka
Panda arm in the MuJoCo-backed LIBERO simulator. The MolmoAct2 backbone
reasons about the scene in 3D and the flow-matching action expert emits a
continuous action chunk that the adapter replays one step at a time.

| Field | Value |
| --- | --- |
| Actions | pick, place, open, close |
| Objects | bowl, cup, drawer, object |
| Scenes  | tabletop, kitchen |
| Embodiment | `franka_panda` |

## How it works

MolmoAct2 grafts a modern DiT-style flow-matching continuous-action expert
onto the Molmo2-ER discrete-token VLM via per-layer KV-cache conditioning
(arXiv:2605.02881). It ships as a transformers **custom-code** model
(`trust_remote_code`, `auto_map` → `MolmoAct2ForConditionalGeneration`), not a
lerobot policy. The OpenRAL `molmoact2` adapter
(`python/sim/src/openral_sim/policies/molmoact2.py`) loads it via
`AutoModelForImageTextToText.from_pretrained` + `AutoProcessor` from the
manifest's `source_repo`, NF4-quantizes every Linear with ≥4M weight elements
via bitsandbytes, overlays the prequantized pack from `weights_uri`, then drives
it through the checkpoint's own `predict_action(...)` continuous-action API. Two
RGB camera streams (ordered `[agentview, wrist]`) plus an 8-D proprio state go
in; an `(n_action_steps, 7)` end-effector action chunk comes out — sliced from
the checkpoint's padded 32-D action down to the embodiment's 7-D — replayed one
step at a time and re-inferred when the queue empties. The chunk length is the
checkpoint's `action_horizon` (LIBERO = 10). Images are rotated 180° and the
`camera1`/`camera2` scene keys are aliased to the model's `image`/`image2`
input features.

Verified end-to-end on a single 8 GB RTX 4070 (LIBERO-Spatial task 0, NF4):
the rollout solves the task (`success=True`, reward 1.0).

### Observation → action contract

| Direction | Key | Shape | Notes |
| --- | --- | --- | --- |
| in | `observation.images.camera1` | `(1, 3, H, W) float32 [0,1]` | agentview (static) |
| in | `observation.images.camera2` | `(1, 3, H, W) float32 [0,1]` | eye-in-hand (wrist) |
| in | `observation.state`           | `(1, 8)` float32                | LIBERO 8-D proprio |
| out | action chunk                  | `(n_action_steps, 7)` float32   | 6-DoF EE delta + gripper (chunk = 10) |

## Upstream model / training

The wrapped weights come from Ai2's `allenai/MolmoAct2-LIBERO` checkpoint —
the base `allenai/MolmoAct2` foundation model finetuned on the full LIBERO
training mixture (Spatial + Object + Goal + Long). This rSkill repackages an
NF4-quantized mirror of those weights; it does **not** retrain or copy the
full-precision weights.

| Field | Value |
| --- | --- |
| Source repo | [`allenai/MolmoAct2-LIBERO`](https://huggingface.co/allenai/MolmoAct2-LIBERO) |
| Base model  | [`allenai/MolmoAct2`](https://huggingface.co/allenai/MolmoAct2) |
| Paper       | [arxiv:2605.02881](https://arxiv.org/abs/2605.02881) — *MolmoAct2: Action Reasoning Models for Real-world Deployment* |
| License     | apache-2.0 (code + weights) |
| Parameters  | ~5.49 B |
| Training data | LIBERO training mixture (Spatial + Object + Goal + Long) |

## Supported robots

| Robot | Embodiment tag | Status | Notes |
| --- | --- | --- | --- |
| Franka Panda (LIBERO sim) | `franka_panda` | ⚡ experimental | Native training embodiment; numbers paper-cited, not yet locally reproduced. |

## Sensors required

| Key | Modality | Min resolution | Format |
| --- | --- | --- | --- |
| `observation.images.camera1` | RGB | 224 × 224 | `float32` |
| `observation.images.camera2` | RGB | 224 × 224 | `float32` |
| `observation.state`          | proprioception | (8,) | `float32` |

## Manifest summary

| Field | Value |
| --- | --- |
| `name` | `OpenRAL/rskill-molmoact2-libero-nf4` |
| `version` | `0.1.0` |
| `license` | `apache-2.0` |
| `role` | `s1` |
| `embodiment_tags` | `["franka_panda"]` |
| `runtime` / `quantization.dtype` | `pytorch` / `int4` (NF4) |
| `weights_uri` | `hf://OpenRAL/rskill-molmoact2-libero-nf4` |
| `chunk_size` / `n_action_steps` | 10 / 10 (= checkpoint `action_horizon`) |
| `latency_budget.per_chunk_ms` | 1000 ms (flow-matching sampling; measured ~80–90 ms/step NF4) |
| `commercial_use_allowed` | `true` (Apache-2.0) |

Full schema: [`openral_core.schemas.RSkillManifest`](../../python/core/src/openral_core/schemas.py).

## Quick start

```python
from openral_rskill.loader import rSkill

pkg = rSkill.from_yaml("rskills/molmoact2-libero-nf4/rskill.yaml")
print(pkg.manifest.name, pkg.manifest.version)
```

```bash
# CLI:
uv run openral rskill install OpenRAL/rskill-molmoact2-libero-nf4
uv run openral rskill check                # does this host meet the requirements?
```

## Reproduction

```bash
just bootstrap && uv sync --all-packages --group libero

# LIBERO-Spatial closed-loop rollout (NF4 weights fit an 8 GB GPU):
openral sim run --config scenes/benchmark/libero_spatial.yaml --rskill rskills/molmoact2-libero-nf4 \
                --rskill rskills/molmoact2-libero-nf4
```

Producing / refreshing the NF4 weights on the Hub (one-shot, needs a CUDA
host):

```bash
HF_TOKEN=<write-token> uv run python tools/quantize_rskill.py \
    --source allenai/MolmoAct2-LIBERO \
    --target OpenRAL/rskill-molmoact2-libero-nf4 \
    --loader transformers --trust-remote-code
```

## Evaluation

`eval/libero.json::status` is **pending** — the success rates carried there
are the paper-cited numbers (`reproduced_locally: false`), not a local
reproduction. The reproduction command is recorded in
`eval/libero.json::source.reproduction_cli`. A full local rerun needs a GPU
and is documented but not yet run.

| Benchmark | Score | `reproduced_locally` | Config |
| --- | --- | --- | --- |
| LIBERO (avg) | 0.972 (paper) | false | `scenes/benchmark/libero_spatial.yaml` (with `--rskill rskills/molmoact2-libero-nf4`) |

## License

This rSkill package (`rskill.yaml`, `README.md`, `eval/libero.json`) is
**Apache-2.0**. The wrapped weights at
`hf://OpenRAL/rskill-molmoact2-libero-nf4` (NF4 mirror of
`allenai/MolmoAct2-LIBERO`) are also released under **Apache-2.0** by Ai2 —
commercial use is permitted; review the upstream LICENSE before deployment.

## See also

- [`robots/franka_panda/README.md`](../../robots/franka_panda/README.md) — RobotDescription manifest.
- [`scenes/benchmark/libero_spatial.yaml`](../../scenes/benchmark/libero_spatial.yaml) — canonical LIBERO-Spatial BenchmarkScene (pass `--rskill rskills/molmoact2-libero-nf4`).
- [`rskills/pi05-libero-nf4/README.md`](../pi05-libero-nf4/README.md) — π0.5 LIBERO alternative (the model MolmoAct2 outperforms).
- [`rskills/smolvla-libero/README.md`](../smolvla-libero/README.md) — Apache-2.0 LIBERO alternative.
- [CLAUDE.md §6.4](../../CLAUDE.md) — rSkill packaging contract.
