---
name: gr00t-n17-so101-fruit
description: >-
  S1 Vision-Language-Action policy. Capabilities: pick, place, pick_and_place, grasp on banana, orange, apple. GR00T N1.7 (3B, Cosmos-Reason2-2B backbone) community-finetuned on the SO-101 `so101_fruit` teleop dataset (200 eps, front+wrist RGB). Emits 6-D absolute joint chunks (5 arm + gripper) over a 16-step horizon under GR00T's `new_embodiment` tag. In-process lerobot GrootPolicy, whole-model NF4 — GPU-verified 5.8 GiB peak on 8 GB. NVIDIA Open Model License (commercial OK). Discovery view of an OpenRAL rSkill — NOT directly runnable by an agent harness; it runs via rSkill.from_pretrained + the robot HAL.
metadata:
  openral_rskill: true            # generated discovery view of an rSkill
  schema_version: 0.1
  rskill_id: OpenRAL/rskill-gr00t-n17-so101-fruit
  manifest: ./rskill.yaml
  role: s1
  kind: vla
  model_family: gr00t
  embodiment_tags: [so101_follower]
  actions: [pick, place, pick_and_place, grasp]
  objects: [banana, orange, apple]
  scenes: [tabletop]
  sensors_required: ['rgb:observation.images.camera1', 'rgb:observation.images.camera2']
  state_dim: 6
  action_dim: 6
  action_representation: joint_positions
  runtime: pytorch
  quantization: bf16/pytorch
  min_vram_gb: {bf16: 12.0, int4: 6.0}
  chunk_size: 16
  latency_budget: {per_chunk_ms: 1500.0}
  license_code: Apache-2.0
  license_weights: nvidia_open_model
  weights_uri: hf://aaronsu11/GR00T-N1.7-3B-SO101-FruitPicking@26b179c37f35168359ccdf2e51fed6c2690186cf
  source_repo: hf://aaronsu11/GR00T-N1.7-3B-SO101-FruitPicking
  paper_url: https://arxiv.org/abs/2503.14734
---

# gr00t-n17-so101-fruit — rSkill discovery view

> **Generated view, not a hand-written skill.** This `SKILL.md` is a discovery-only
> mirror of [`rskill.yaml`](./rskill.yaml), produced by `tools/generate_rskill_skillmd.py`.
> It lets tools that read the standard agent-skill format find and reason about this
> OpenRAL rSkill. The `rskill.yaml` manifest is the single source of truth
> (CLAUDE.md §1.3). Do not edit by hand — edit the manifest and regenerate.

## What it is

An OpenRAL **Vision-Language-Action policy** (`role: s1`, `kind: vla`). GR00T N1.7 (3B, Cosmos-Reason2-2B backbone) community-finetuned on the SO-101 `so101_fruit` teleop dataset (200 eps, front+wrist RGB). Emits 6-D absolute joint chunks (5 arm + gripper) over a 16-step horizon under GR00T's `new_embodiment` tag. In-process lerobot GrootPolicy, whole-model NF4 — GPU-verified 5.8 GiB peak on 8 GB. NVIDIA Open Model License (commercial OK).

## Capabilities

- **Verbs:** pick · place · pick_and_place · grasp
- **Objects:** banana · orange · apple
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
- **Weights:** `nvidia_open_model` — permissive / commercial-use OK

## How to actually run it (not via an agent harness)

```python
from openral_rskill import rSkill

skill = rSkill.from_pretrained("OpenRAL/rskill-gr00t-n17-so101-fruit")
# the loader validates embodiment / sensors / runtime / quantization against the target
# RobotDescription and enforces the weight-license gate before any weights load.
```

See [`rskill.yaml`](./rskill.yaml) for the authoritative, validated manifest.
