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
- openarm
- robotics
- vla
base_model:
- qualiadev/pi05-openarm-restock-sequences
base_model_relation: finetune
inference: false
---

# rskill-pi05-openarm-restock_shelf-bf16

> **OpenRAL rSkill** — π0.5 fine-tuned for the Enactic OpenArm v2 bimanual
> station to restock boxes onto a shelf, from three RGB views. **Private**:
> Physical Intelligence permissive-research weight lineage, and the upstream
> checkpoint this was converted from has been deleted from the Hub.

## Preview

<!-- No demo media is shipped. No rollout of this checkpoint has been recorded
     on OpenRAL, in sim or on hardware, so there is no frame to show that would
     honestly depict this skill's behaviour. Add media/demo.png (or a start /
     middle / end triptych) after the first recorded rollout. -->

_No demo media yet — see [Evaluation](#evaluation) for why._

## What this skill does

Restocks boxes onto a shelf with a bimanual OpenArm v2: reaches into a box at
the front of the workspace, grasps an item, transfers it, and places it on the
shelf. Both arms are driven from one 16-DoF policy — this is a single bimanual
controller, not two independent arm policies.

| Field | Value |
| --- | --- |
| Actions | pick, place, pick_and_place, transfer |
| Objects | box, shelf |
| Scenes | shelf, warehouse |
| Embodiment | `openarm` |

## How it works

π0.5 — a PaliGemma 2 B vision-language backbone paired with a 300 M Gemma
action expert and a flow-matching action head. Three RGB views and the 16-D
joint state are tokenized alongside the instruction string; the action expert
denoises a `(35, 16)` action chunk in 10 flow-matching steps. State and action
are padded to 32-D internally and sliced back to 16 before the wire.

At 30 fps one action step is 33.3 ms, so a chunk covers 1.167 s of motion.
`chunk_size` is the checkpoint's baked `action_horizon` and is not tunable —
an engine built from it bakes the same number.

### Observation → action contract

| Direction | Key | Shape | Notes |
| --- | --- | --- | --- |
| in | `observation.images.context` | `(1, 3, 224, 224)` | workspace overview |
| in | `observation.images.wrist_left` | `(1, 3, 224, 224)` | left wrist |
| in | `observation.images.wrist_right` | `(1, 3, 224, 224)` | right wrist |
| in | `observation.state` | `(1, 16)` float32 | absolute joint positions, **radians** |
| out | action chunk | `(35, 16)` float32 | see the delta/absolute split below |

Camera order is load-bearing — cameras fill the policy's image slots in the
order above, matching training. The names are already the station's own sensor
names, so no alias remap is needed.

State and action are padded to 32-D inside π0.5 and sliced back to 16 before
the wire. Index layout, both for state and action:

```
 0- 6  left arm joints 1-7
 7     left gripper
 8-14  right arm joints 1-7
15     right gripper
```

### The arm joints are deltas; the grippers are absolute

This is the single most important fact for anyone driving this policy, and the
converted checkpoint carries no `metadata.pt` to record it. It is nevertheless
**verifiable from the shipped `norm_stats.json`**:

| dims | mean | q01 … q99 | reading |
| --- | --- | --- | --- |
| 0–6, 8–14 (arms) | ≈ 0.00 – 0.06 | symmetric about 0 | per-step **delta**, radians |
| 7 (left gripper) | 0.438 | 0.0000 … 0.7852 | **absolute** hinge position |
| 15 (right gripper) | −0.453 | −0.7854 … −0.0002 | **absolute** hinge position |

Those two gripper ranges reproduce the OpenArm v2 hinge limits in OpenRAL's
`robots/openarm/robot.yaml` exactly (left `[0, 0.7854]`, right `[-0.7854, 0]`),
which is independent evidence that the checkpoint was trained in that joint
convention.

This matches openpi's `relativize: true` and, on the lerobot side,
`PI05Config(use_relative_actions=True, relative_exclude_joints=["gripper"])`.
Whoever drives the policy owns the delta→absolute reconstruction; nothing in
this repo performs it.

### Normalization

π0.5 normalizes state and action by **quantile**, not mean/std:
`x → (x − q01) / (q99 − q01 + 1e-6) * 2 − 1`. In openpi this is
`use_quantile_norm = (model_type != PI0)`, i.e. true for pi05; lerobot's
`PI05Config` agrees, defaulting `STATE` and `ACTION` to `QUANTILES` and
`VISUAL` to `IDENTITY`. Use `q01`/`q99` from `norm_stats.json`, not `mean`/`std`.

### Prompt

One training instruction string, and it must be sent **verbatim**:

```
restock-shelf-from-front-box
```

A drifted prompt sends an out-of-distribution instruction to real arms.

> Not to be confused with `qualiadev/pi05-qua1507-restockshelfs`, a *different*
> OpenArm restocking checkpoint (20 fps, action_horizon 16, prompt
> `"restock boxes on the shelf"`). That one is still on the Hub. This is not it.

## Upstream model / training provenance

| Field | Value |
| --- | --- |
| Source repo | `qualiadev/pi05-openarm-restock-sequences` @ `7f96e610b6cb446e55301928f3d118b6591cded0` — **404, deleted from the Hub** |
| Training dataset | `qualiadev/openarm-restock-sequences-canonical-30fps` @ `f3415357965c8b97a4f99c4fffe4f4b90080c68a` — **404, deleted from the Hub** |
| Base model | Physical Intelligence π0.5 (`pi05_base`) |
| Parameters | 3.617 B, all BF16 (3.353 B excluding the unused action-expert `lm_head`) |
| Frame rate | 30 fps |
| Paper | [arxiv:2410.24164](https://arxiv.org/abs/2410.24164) |
| License | `permissive_research` (PI terms) |

The training run config — step count, batch size, episode count, optimizer —
lived only on the deleted model card and is **not reproduced here**, because
inventing it would be worse than its absence (CLAUDE.md §1.2). Everything
stated as fact in this README was read off the artifact itself or off the
conversion provenance records written on the lab Jetson AGX Thor
(`~/engines/openarm-restock-sequences/*.engine.provenance.json`).

A sibling checkpoint whose card *does* survive,
[`qualiadev/pi05-qua1507-restockshelfs`](https://huggingface.co/qualiadev/pi05-qua1507-restockshelfs),
documents the same station and task family at 20 fps with `action_horizon: 16`
and the prompt `"restock boxes on the shelf"`. It is a **different**
checkpoint — do not read its run config as this one's.

## Supported robots / embodiments

| Robot | Embodiment tag | Status | Notes |
| --- | --- | --- | --- |
| Enactic OpenArm v2 (bimanual) | `openarm` | ⚡ experimental | Contract matches `robots/openarm/robot.yaml` — 16 joints, gripper hinge ranges verified against the checkpoint's norm stats. No rollout recorded. |

## Sensors required

| Key | Modality | Min resolution | Format |
| --- | --- | --- | --- |
| `observation.images.context` | RGB | 224 × 224 | `float32` |
| `observation.images.wrist_left` | RGB | 224 × 224 | `float32` |
| `observation.images.wrist_right` | RGB | 224 × 224 | `float32` |
| `observation.state` | proprioception | `(16,)` | `float32`, radians |

Camera **order** is load-bearing — the three views fill the policy's image
slots in the order above, matching training.

## Manifest summary

| Field | Value |
| --- | --- |
| `name` | `OpenRAL/rskill-pi05-openarm-restock_shelf-bf16` |
| `version` | `0.1.0` |
| `license` | `permissive_research` |
| `role` / `kind` | `s1` / `vla` |
| `model_family` | `pi05` |
| `embodiment_tags` | `[openarm]` |
| `runtime` / `quantization.dtype` | `pytorch` / `bf16` |
| `weights_uri` | `hf://OpenRAL/rskill-pi05-openarm-restock_shelf-bf16` |
| `chunk_size` / `n_action_steps` | 35 / 35 |
| `latency_budget.per_chunk_ms` | 400.0 |
| `commercial_use_allowed` | **false** (derived from `license`) |

`min_vram_gb` is deliberately absent — see the manifest's comment. The hard
floor is the 6.74 GiB BF16 parameter footprint, before activations or KV cache.

## Quick start

```python
import os
os.environ["OPENRAL_ALLOW_NONCOMMERCIAL"] = "1"   # PI permissive-research weights

from openral_rskill import rSkill

skill = rSkill.from_yaml("rskills/rskill-pi05-openarm-restock_shelf-bf16/rskill.yaml")
print(skill.manifest.name, skill.manifest.chunk_size)
```

Send the training instruction **verbatim** — `restock-shelf-from-front-box`.

## Reproduction

No reproduction command is offered, because no evaluation has been run (see
below). The conversion that produced these weights is reproducible in shape but
not byte-for-byte:

```bash
# q-research, policies/openpi/convert — Stage 1 only.
ACTION_HORIZON=35 ./convert_finetune.sh   # openpi JAX/orbax -> PyTorch
```

## Evaluation

**No task-success evaluation has been run against this checkpoint on OpenRAL**,
in simulation or on hardware, so `eval/` ships no benchmark file and the
manifest carries no `benchmarks` block.

What does exist is the TensorRT conversion pipeline's own fidelity and latency
gates — see [Downstream validation](#downstream-validation-performed-on-this-checkpoint).
Those measure whether a quantized engine matches the PyTorch model. They say
nothing about whether the policy restocks a shelf.

## Timing

Trained at 30 fps, so one action step is **33.3 ms** and a 35-step chunk covers
1.167 s of motion. `action_horizon = 35` is baked into the checkpoint and into
any engine built from it; it is not a tunable.

## Reproducibility: one tensor differs between conversion runs

openpi's Stage-1 JAX→PyTorch conversion is **not byte-deterministic**. The same
source revision produced three different `converted_model_sha256` values across
three runs on this host, which looks alarming and is not.

Comparing the three 6.74 GiB artifacts tensor by tensor:

- **811 of 812 tensors are byte-identical** — the whole vision tower, both
  Gemma stacks, and every action projection.
- The sole difference is `paligemma_with_expert.gemma_expert.lm_head.weight`
  `[257152, 1024]`, where ~99.91% of elements differ.
- Its statistics in every copy: mean ≈ 0, std = 0.020, symmetric range ±0.115
  — a freshly random-initialized tensor.

The action expert never does token prediction (actions leave through
`action_out_proj`), so the JAX checkpoint carries no trained `lm_head` and the
converter initializes one that inference never reads. It is retained here only
so loaders that expect the key still work.

**Consequence:** do not use a whole-file hash to decide whether two conversions
of this checkpoint agree. Compare tensors, excluding
`gemma_expert.lm_head.weight`.

The copy published here has
`sha256 = 921c7a4fd43876e7c98e7e0afc5947bdc51ec03e4a8ab7d6433bfe0c894424a1`.

## Downstream validation performed on this checkpoint

Not task-success numbers — none were measured on OpenRAL. What exists is the
TensorRT conversion pipeline's own gates, run on a Jetson AGX Thor (`qthor`,
JetPack r38.4), recorded in the engine provenance records:

| Gate | Threshold | Result |
| --- | --- | --- |
| Mean cosine vs PyTorch reference, 8 seeds, real observation | ≥ 0.99 | passed (`accepted: true`) |
| Per-seed cosine | ≥ 0.98 | passed |
| Engine latency, fp8 + llm-nvfp4, 96 language tokens | target 52 ms | **59.68 ms** — target *not* met |
| Engine latency, fp8, 96 language tokens | — | 68.20 ms |

The 3-camera NVFP4 calibration used 32 real frames from the checkpoint's own
training dataset. Both engines were accepted on fidelity; neither hit the
latency target.

**No task-success evaluation has been run against this checkpoint on OpenRAL**,
in sim or on hardware. Do not infer competence from the fidelity gates — they
measure whether the quantized engine matches the PyTorch model, not whether the
policy does the job.

## Loading

This is an **openpi-format** checkpoint. OpenRAL's `pi05` adapter is
lerobot-based (`PI05Policy.from_pretrained`) and expects a lerobot `PI05Config`
plus `policy_preprocessor.json` / `policy_postprocessor.json`, none of which
this layout carries — so it does **not** load through OpenRAL's adapter as
published. It is served by openpi's own policy server.

The module names do line up (`paligemma_with_expert`, `action_in_proj`,
`action_out_proj`, `time_mlp_in`, `time_mlp_out` are lerobot's `PI05Pytorch`
submodules), so a faithful conversion is well specified — lerobot `PI05Config`
with `chunk_size=35`, `n_action_steps=35`, `dtype="bfloat16"`,
`tokenizer_max_length=96`, `use_relative_actions=True`, the default `QUANTILES`
normalization mapping, plus normalizer/unnormalizer processors built from
`norm_stats.json` and a `model.` state-dict prefix. That conversion has **not**
been performed or validated here.

## License

The wrapping rSkill package is Apache-2.0. The weights derive from Physical
Intelligence's π0.5 (`pi05_base`) under PI's **permissive research** terms —
NOT Apache-2.0. Commercial deployment is prohibited without a separate
agreement with Physical Intelligence. OpenRAL's loader derives
`is_commercial_use_allowed = False` from `license: permissive_research`;
activating commercially additionally requires `OPENRAL_ALLOW_NONCOMMERCIAL=1`
plus that agreement.

## See also

- `robots/openarm/robot.yaml` — the OpenRAL RobotDescription this policy targets.
- `python/hal/src/openral_hal/openarm_real.py` — the real-hardware HAL.
- [CLAUDE.md §6.4](https://github.com/OpenRAL/openral) — rSkill packaging contract.
