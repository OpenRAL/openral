---
name: pi05-metal-pick-place
description: >-
  S1 Vision-Language-Action policy. Capabilities: pick, place, pick_and_place on cube. π0.5 with MakerMods' LoRA for the MakerMods Metal arm: picks a cube and places it on a marked spot from a fixed table-facing and a wrist RGB view and the 7-D joint + gripper state, at 30 Hz in 50-step chunks of absolute joint positions (degrees in the checkpoint). The LoRA is merged into lerobot/pi05_base at load. PI permissive-research base weights: commercial use needs a vendor agreement. Discovery view of an OpenRAL rSkill — NOT directly runnable by an agent harness; it runs via rSkill.from_pretrained + the robot HAL.
metadata:
  openral_rskill: true            # generated discovery view of an rSkill
  schema_version: 0.1
  rskill_id: OpenRAL/rskill-pi05-makermods_metal-pick_place-bf16
  manifest: ./rskill.yaml
  role: s1
  kind: vla
  model_family: pi05
  embodiment_tags: [makermods_metal]
  actions: [pick, place, pick_and_place]
  objects: [cube]
  scenes: [tabletop]
  sensors_required: ['rgb:observation.images.top', 'rgb:observation.images.wrist']
  state_dim: 7
  action_dim: 7
  action_representation: joint_positions
  runtime: pytorch
  quantization: bf16/pytorch
  min_vram_gb: {bf16: 9.5}
  chunk_size: 50
  n_action_steps: 50
  latency_budget: {per_chunk_ms: 500.0}
  license_code: Apache-2.0
  license_weights: permissive_research   # NOT permissive — see License section
  weights_uri: hf://makermods/pi05_metal_pick_place_lora@87595e59df01dc78e794b41df756241cb5efd7b6
  source_repo: hf://makermods/pi05_metal_pick_place_lora
  paper_url: https://arxiv.org/abs/2504.16054
---

# pi05-metal-pick-place — rSkill discovery view

> **Generated view, not a hand-written skill.** This `SKILL.md` is a discovery-only
> mirror of [`rskill.yaml`](./rskill.yaml), produced by `tools/generate_rskill_skillmd.py`.
> It lets tools that read the standard agent-skill format find and reason about this
> OpenRAL rSkill. The `rskill.yaml` manifest is the single source of truth
> (CLAUDE.md §1.3). Do not edit by hand — edit the manifest and regenerate.

## What it is

An OpenRAL **Vision-Language-Action policy** (`role: s1`, `kind: vla`). π0.5 with MakerMods' LoRA for the MakerMods Metal arm: picks a cube and places it on a marked spot from a fixed table-facing and a wrist RGB view and the 7-D joint + gripper state, at 30 Hz in 50-step chunks of absolute joint positions (degrees in the checkpoint). The LoRA is merged into lerobot/pi05_base at load. PI permissive-research base weights: commercial use needs a vendor agreement.

## Capabilities

- **Verbs:** pick · place · pick_and_place
- **Objects:** cube
- **Scenes:** tabletop
- **Embodiments:** makermods_metal

## Why this is discovery-only

An agent skill is natural-language instructions loaded into an LLM's context. An rSkill
is an executable artifact: it carries a typed capability/embodiment contract, model weights,
a runtime, and a license/provenance gate — none of which fit in freeform markdown. So an
agent can use this view to *select* the right skill, but cannot *execute* it by loading
this file. Execution always goes through the OpenRAL loader and the robot HAL.

## License

- **Code:** Apache-2.0.
- **Weights:** `permissive_research` — **NOT** fully permissive. The loader surfaces this posture and enforces the non-commercial guard (`OPENRAL_ALLOW_NONCOMMERCIAL=1`) where applicable. Commercial use may require a separate upstream agreement. This is third-party weight lineage; OpenRAL's own code is Apache-2.0.

## How to actually run it (not via an agent harness)

```python
from openral_rskill import rSkill

skill = rSkill.from_pretrained("OpenRAL/rskill-pi05-makermods_metal-pick_place-bf16")
# the loader validates embodiment / sensors / runtime / quantization against the target
# RobotDescription and enforces the weight-license gate before any weights load.
```

See [`rskill.yaml`](./rskill.yaml) for the authoritative, validated manifest.
