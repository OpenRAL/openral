---
name: molmoact2-so101-nf4
description: >-
  S1 Vision-Language-Action policy. Capabilities: pick, place, pick_and_place, grasp. MolmoAct2 (Ai2) finetuned on the SO-100/SO-101 teleop mixture, NF4-quantized for 8 GB GPUs. Emits 6-DoF absolute joint-position chunks (size 10) for the SO-100/SO-101 follower arm. Flow-matching action expert on Molmo2-ER VLM. Apache-2.0. norm_tag="so100_so101_molmoact2" travels in the manifest's image_preprocessing block (overridable via vla.extra.norm_tag). Discovery view of an OpenRAL rSkill — NOT directly runnable by an agent harness; it runs via rSkill.from_pretrained + the robot HAL.
metadata:
  openral_rskill: true            # generated discovery view of an rSkill
  schema_version: 0.1
  rskill_id: OpenRAL/rskill-molmoact2-so101-nf4
  manifest: ./rskill.yaml
  role: s1
  kind: vla
  model_family: molmoact2
  embodiment_tags: [so100_follower, so101_follower]
  actions: [pick, place, pick_and_place, grasp]
  scenes: [tabletop]
  sensors_required: ['rgb:observation.images.camera1', 'rgb:observation.images.camera2']
  state_dim: 6
  action_dim: 6
  runtime: pytorch
  quantization: int4/pytorch
  min_vram_gb: {fp32: 22.0, bf16: 11.0, int4: 4.0}
  chunk_size: 10
  latency_budget: {per_chunk_ms: 1000.0}
  license_code: Apache-2.0
  license_weights: apache-2.0
  weights_uri: hf://OpenRAL/rskill-molmoact2-so101-nf4
  source_repo: hf://allenai/MolmoAct2-SO100_101
  paper_url: https://arxiv.org/abs/2605.02881
---

# molmoact2-so101-nf4 — rSkill discovery view

> **Generated view, not a hand-written skill.** This `SKILL.md` is a discovery-only
> mirror of [`rskill.yaml`](./rskill.yaml), produced by `tools/generate_rskill_skillmd.py`.
> It lets tools that read the standard agent-skill format find and reason about this
> OpenRAL rSkill. The `rskill.yaml` manifest is the single source of truth
> (CLAUDE.md §1.3). Do not edit by hand — edit the manifest and regenerate.

## What it is

An OpenRAL **Vision-Language-Action policy** (`role: s1`, `kind: vla`). MolmoAct2 (Ai2) finetuned on the SO-100/SO-101 teleop mixture, NF4-quantized for 8 GB GPUs. Emits 6-DoF absolute joint-position chunks (size 10) for the SO-100/SO-101 follower arm. Flow-matching action expert on Molmo2-ER VLM. Apache-2.0. norm_tag="so100_so101_molmoact2" travels in the manifest's image_preprocessing block (overridable via vla.extra.norm_tag).

## Capabilities

- **Verbs:** pick · place · pick_and_place · grasp
- **Scenes:** tabletop
- **Embodiments:** so100_follower · so101_follower

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

skill = rSkill.from_pretrained("OpenRAL/rskill-molmoact2-so101-nf4")
# the loader validates embodiment / sensors / runtime / quantization against the target
# RobotDescription and enforces the weight-license gate before any weights load.
```

See [`rskill.yaml`](./rskill.yaml) for the authoritative, validated manifest.
