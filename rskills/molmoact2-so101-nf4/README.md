---
language:
- en
license: apache-2.0
pipeline_tag: robotics
tags:
- OpenRAL
- rskill
- molmoact2
- vision-language-action
- nf4
- 4-bit
- so100_follower
- so101_follower
- transformers
- vla
- so101
- so100
- manipulation
base_model:
- allenai/MolmoAct2-SO100_101
base_model_relation: quantized
inference: false
---

# rskill-molmoact2-multi-so101-nf4

> **OpenRAL rSkill** — MolmoAct2 (Ai2's open action reasoning model: a
> Molmo2-ER embodied-reasoning VLM backbone with a flow-matching
> continuous-action expert) finetuned on the
> [SO-100/SO-101](https://huggingface.co/allenai/MolmoAct2-SO100_101) teleop
> mixture and NF4-quantized so the ~5.5 B-param model fits an 8 GB GPU.
> Robots: SO-100 and SO-101 follower arms. **Apache-2.0 weights** — commercial
> use permitted.

This package wraps `hf://OpenRAL/rskill-molmoact2-multi-so101-nf4` (an
NF4-quantized mirror of `allenai/MolmoAct2-SO100_101`) with a `rskill.yaml`
manifest that adds capability checking, license surfacing, latency budgets,
and local registry integration. It does **not** copy model weights; they
live on the Hub.

> **Norm tag:** this checkpoint's normalization statistics are tagged
> `"so100_so101_molmoact2"`. The manifest carries it in
> `image_preprocessing.norm_tag`; the adapter resolves `vla.extra.norm_tag` →
> manifest → default `"libero"`, and the `"libero"` stats produce garbage
> actions for this checkpoint. Only set `vla.extra.norm_tag` to override.

## What this skill does

Performs tabletop manipulation — picking, placing, grasping, and transporting
objects — on the SO-100 and SO-101 follower arms. The MolmoAct2 backbone
reasons about the scene in 3D and the flow-matching action expert emits a
continuous absolute joint-position action chunk that the adapter replays one
step at a time.

| Field | Value |
| --- | --- |
| Actions | pick, place, pick_and_place, grasp |
| Objects | diverse tabletop objects |
| Scenes  | tabletop |
| Embodiments | `so100_follower`, `so101_follower` |

## How it works

MolmoAct2 grafts a modern DiT-style flow-matching continuous-action expert
onto the Molmo2-ER discrete-token VLM via per-layer KV-cache conditioning
(arXiv:2605.02881). Upstream ships it as transformers **custom code**
(`trust_remote_code`), not a lerobot policy. The OpenRAL `molmoact2` adapter
(`python/sim/src/openral_sim/policies/molmoact2.py`) instead builds the graph from
lerobot 0.6.0's vendored in-tree `MolmoAct2ForConditionalGeneration` and
`MolmoAct2Processor` (config, processor and `norm_stats.json` from the
manifest's `source_repo`), so no remote code executes and
`OPENRAL_ALLOW_REMOTE_CODE` is not needed. It NF4-quantizes the model via
bitsandbytes, overlays the prequantized pack from `weights_uri`, then drives it
through the checkpoint's own `predict_action(...)` continuous-action API. Two
RGB camera streams plus a 6-D proprio state go in; a `(chunk_size, 6)` absolute
joint-position chunk comes out, replayed one step at a time and re-inferred
when the queue empties.

### Observation → action contract

| Direction | Key | Shape | Notes |
| --- | --- | --- | --- |
| in | `observation.images.camera1` | `(1, 3, H, W) float32 [0,1]` | overhead view (→ model `top`) |
| in | `observation.images.camera2` | `(1, 3, H, W) float32 [0,1]` | wrist/side view (→ model `side`) |
| in | `observation.state`           | `(1, 6)` float32                | SO-101 6-D joint positions (rad) |
| out | action chunk                  | `(10, 6)` float32               | absolute joint-position targets |

**Camera aliases:** the manifest maps slot `camera1 → top` and
`camera2 → side`; in the `so101_box` scene those slots are `oak_top` and `wrist`.

## Upstream model / training

The weights come from Ai2's `allenai/MolmoAct2-SO100_101`: the base
`allenai/MolmoAct2` foundation model finetuned on the SO-100/SO-101 teleop
mixture with absolute joint-pose control and annotated language instructions.
This rSkill repackages an NF4-quantized mirror; it does **not** retrain or copy
the full-precision weights.

| Field | Value |
| --- | --- |
| Source repo | [`allenai/MolmoAct2-SO100_101`](https://huggingface.co/allenai/MolmoAct2-SO100_101) |
| Base model  | [`allenai/MolmoAct2`](https://huggingface.co/allenai/MolmoAct2) |
| Paper       | [arxiv:2605.02881](https://arxiv.org/abs/2605.02881) — *MolmoAct2: Action Reasoning Models for Real-world Deployment* |
| License     | apache-2.0 (code + weights) |
| Parameters  | ~5.5 B |
| Training data | SO-100/SO-101 teleop mixture (absolute joint-pose, annotated language) |
| norm_tag    | `"so100_so101_molmoact2"` — set in the manifest's `image_preprocessing` |

## Supported robots

| Robot | Embodiment tag | Status | Notes |
| --- | --- | --- | --- |
| SO-101 follower | `so101_follower` | ⚡ experimental | Native training embodiment; numbers not yet locally reproduced. |
| SO-100 follower | `so100_follower` | ⚡ experimental | Shares identical 6-DoF kinematics; covered by training mixture. |

## Sensors required

| Key | Modality | Min resolution | Format |
| --- | --- | --- | --- |
| `observation.images.camera1` | RGB | 224 × 224 | `float32` |
| `observation.images.camera2` | RGB | 224 × 224 | `float32` |
| `observation.state`          | proprioception | (6,) | `float32` |

## Manifest summary

| Field | Value |
| --- | --- |
| `name` | `OpenRAL/rskill-molmoact2-multi-so101-nf4` |
| `version` | `0.1.0` |
| `license` | `apache-2.0` |
| `role` | `s1` |
| `embodiment_tags` | `["so100_follower", "so101_follower"]` |
| `runtime` / `quantization.dtype` | `pytorch` / `int4` (NF4) |
| `weights_uri` | `hf://OpenRAL/rskill-molmoact2-multi-so101-nf4` |
| `chunk_size` / `n_action_steps` | 10 / unset (the adapter replays the full predicted chunk) |
| `latency_budget.per_chunk_ms` | 1000 ms |
| `is_commercial_use_allowed` (derived from `license`) | `true` (Apache-2.0) |
| `image_preprocessing.image_max_crops` | `4` (secondary vision lever; processor default is 8 — see Memory note) |
| `image_preprocessing.norm_tag` | `"so100_so101_molmoact2"` (override via `vla.extra.norm_tag`) |

Full schema: [`openral_core.schemas.RSkillManifest`](../../python/core/src/openral_core/schemas.py).

## Quick start

```python
from openral_rskill.loader import rSkill

pkg = rSkill.from_yaml("rskills/molmoact2-so101-nf4/rskill.yaml")
print(pkg.manifest.name, pkg.manifest.version)
```

```bash
# CLI:
uv run openral rskill install OpenRAL/rskill-molmoact2-multi-so101-nf4
uv run openral rskill check                # does this host meet the requirements?
```

### Sim config snippet

```yaml
vla:
  id: molmoact2
  weights_uri: rskills/molmoact2-so101-nf4
  extra:
    # norm_tag: "so100_so101_molmoact2"  # optional; the manifest already sets it
    # image_max_crops: 6                # optional secondary lever; manifest pins 4 (see note)
```

> **Memory note (measured on an 8 GiB RTX 4070, transformers 5.x).** NF4 makes
> the model ~6.0 GiB resident (bf16 vocab embeddings + vision tower dominate;
> the nf4 Linears are ~3.5 GiB) and it peaks **~7.63 GiB** during a chunk, at
> the edge of an 8 GiB card (~7.6 GiB usable). The decisive enabler is the
> **CUDA expandable-segments allocator**: without it the first forward's
> ~1.5 GiB embedding `cat` cannot be placed contiguously and OOMs. The adapter
> defaults it to `expandable_segments:True` (via `default_expandable_segments`)
> before its first CUDA allocation, on non-Jetson hosts and only when the
> variable is unset. If other GPU work in the process allocates before the
> policy loads, export it yourself: `PYTORCH_ALLOC_CONF` on torch ≥2.9,
> `PYTORCH_CUDA_ALLOC_CONF` on older torch (renamed in 2.9; the old name logs a
> deprecation warning).
> `image_max_crops` (pinned to 4 here) is a *secondary* lever: it bounds the
> vision tile count but does **not** by itself decide the 8 GiB fit, and
> transformers 5.x's fast image processor largely ignores it. Leave ~0.4 GiB of
> headroom: don't run other GPU processes alongside it.

## Reproduction

```bash
just bootstrap && just sync

# Closed-loop rollout against the SO-101 box scene (NF4 weights fit an 8 GB GPU):
openral sim run --config scenes/sim/so101_tube_insertion.yaml \
                --rskill rskills/molmoact2-so101-nf4
```

Producing / refreshing the NF4 weights on the Hub (one-shot, needs a CUDA
host):

```bash
# allenai is not a trusted remote-code org, so --trust-remote-code needs the ack:
HF_TOKEN=<write-token> OPENRAL_ALLOW_REMOTE_CODE=1 uv run python tools/quantize_rskill.py \
    --source allenai/MolmoAct2-SO100_101 \
    --target OpenRAL/rskill-molmoact2-multi-so101-nf4 \
    --loader transformers --trust-remote-code
```

## Evaluation

`eval/so101.json::status` is **pending**: no locally-reproduced numbers yet,
and no SO-101 benchmark suite is registered under `benchmarks/`, so its
recorded `reproduction_cli` (`--suite so101_tabletop`) cannot run yet.

## License

This rSkill package (`rskill.yaml`, `README.md`, `eval/so101.json`) is
**Apache-2.0**. The wrapped weights at
`hf://OpenRAL/rskill-molmoact2-multi-so101-nf4` (NF4 mirror of
`allenai/MolmoAct2-SO100_101`) are also **Apache-2.0** (Ai2); commercial use
is permitted. Review the upstream LICENSE before deployment.

## See also

- [`robots/so101_follower/README.md`](../../robots/so101_follower/README.md) — RobotDescription manifest.
- [`robots/so100_follower/README.md`](../../robots/so100_follower/README.md) — SO-100 variant.
- [`scenes/sim/so101_tube_insertion.yaml`](../../scenes/sim/so101_tube_insertion.yaml) — SO-101 sim scene config.
- [`rskills/molmoact2-libero-nf4/README.md`](../molmoact2-libero-nf4/README.md) — MolmoAct2 LIBERO variant (Franka Panda).
- [CLAUDE.md §3](../../CLAUDE.md) — rSkill packaging contract.
