---
name: pi05-so101-pickplace-nf4
description: >-
  S1 Vision-Language-Action policy. Capabilities: pick, place, pick_and_place. Pre-quantized nf4 mirror of the community π0.5 SO-101 pick-place finetune HollyTan/pi05_so101_pick_place-v2.2basev2.4_abs_nofreeze_8b (4.14 B PaliGemma + action expert, trained on the HollyTan/so101_pick-place-v2.2-100eps dataset, 100 episodes). Emits 6-DoF absolute joint-position action chunks of length 50 for the so101_follower embodiment. Quantized to ~4 GiB so it loads on an 8 GiB GPU. Apache-2.0 weights + code. Discovery view of an OpenRAL rSkill — NOT directly runnable by an agent harness; it runs via rSkill.from_pretrained + the robot HAL.
metadata:
  openral_rskill: true            # generated discovery view of an rSkill
  schema_version: 0.1
  rskill_id: OpenRAL/rskill-pi05-so101-pickplace-nf4
  manifest: ./rskill.yaml
  role: s1
  kind: vla
  model_family: pi05
  embodiment_tags: [so101_follower]
  actions: [pick, place, pick_and_place]
  scenes: [tabletop]
  sensors_required: ['rgb:observation.images.camera1', 'rgb:observation.images.camera2']
  state_dim: 6
  action_dim: 6
  runtime: pytorch
  quantization: int4/pytorch
  min_vram_gb: {fp32: 18.0, bf16: 8.0, int4: 4.0}
  chunk_size: 50
  latency_budget: {per_chunk_ms: 2500.0}
  license_code: Apache-2.0
  license_weights: apache-2.0
  weights_uri: hf://OpenRAL/rskill-pi05-so101-pickplace-nf4
  source_repo: hf://HollyTan/pi05_so101_pick_place-v2.2basev2.4_abs_nofreeze_8b
  paper_url: https://arxiv.org/abs/2410.24164
---

# pi05-so101-pickplace-nf4 — rSkill discovery view

> **Generated view, not a hand-written skill.** This `SKILL.md` is a discovery-only
> mirror of [`rskill.yaml`](./rskill.yaml), produced by `tools/generate_rskill_skillmd.py`.
> It lets tools that read the standard agent-skill format find and reason about this
> OpenRAL rSkill. The `rskill.yaml` manifest is the single source of truth
> (CLAUDE.md §1.3). Do not edit by hand — edit the manifest and regenerate.

## What it is

An OpenRAL **Vision-Language-Action policy** (`role: s1`, `kind: vla`). Pre-quantized nf4 mirror of the community π0.5 SO-101 pick-place finetune HollyTan/pi05_so101_pick_place-v2.2basev2.4_abs_nofreeze_8b (4.14 B PaliGemma + action expert, trained on the HollyTan/so101_pick-place-v2.2-100eps dataset, 100 episodes). Emits 6-DoF absolute joint-position action chunks of length 50 for the so101_follower embodiment. Quantized to ~4 GiB so it loads on an 8 GiB GPU. Apache-2.0 weights + code.

## Capabilities

- **Verbs:** pick · place · pick_and_place
- **Scenes:** tabletop
- **Embodiments:** so101_follower

## Why this is discovery-only

An agent skill is natural-language instructions loaded into an LLM's context. An rSkill
is an executable artifact: it carries a typed capability/embodiment contract, model weights,
a runtime, and a license/provenance gate — none of which fit in freeform markdown. So an
agent can use this view to *select* the right skill, but cannot *execute* it by loading
this file. Execution always goes through the OpenRAL loader and the robot HAL.

## License

- **Code:** Apache-2.0.
- **Weights:** `apache-2.0` — permissive / commercial-use OK

## How to actually run it (not via an agent harness)

```python
from openral_rskill import rSkill

skill = rSkill.from_pretrained("OpenRAL/rskill-pi05-so101-pickplace-nf4")
# the loader validates embodiment / sensors / runtime / quantization against the target
# RobotDescription and enforces the weight-license gate before any weights load.
```

See [`rskill.yaml`](./rskill.yaml) for the authoritative, validated manifest.
