---
name: rskill-pi05-openarm-restock_shelf-bf16
description: >-
  S1 Vision-Language-Action policy. Capabilities: pick, place, pick_and_place, transfer on box, shelf. π0.5 (PaliGemma + flow matching, 3.617 B) fine-tuned for the Enactic OpenArm v2 bimanual station to restock boxes onto a shelf. Three RGB views and 16-D joint state in; 35-step chunks of 16-D actions out — arm joints as per-step deltas in radians, both grippers absolute. Trained at 30 fps. ONE instruction string, "restock-shelf-from-front-box"; deploy it verbatim. Upstream checkpoint and dataset are both deleted from the Hub, so this is the surviving copy — see the README. Discovery view of an OpenRAL rSkill — NOT directly runnable by an agent harness; it runs via rSkill.from_pretrained + the robot HAL.
metadata:
  openral_rskill: true            # generated discovery view of an rSkill
  schema_version: 0.1
  rskill_id: OpenRAL/rskill-pi05-openarm-restock_shelf-bf16
  manifest: ./rskill.yaml
  role: s1
  kind: vla
  model_family: pi05
  embodiment_tags: [openarm]
  actions: [pick, place, pick_and_place, transfer]
  objects: [box, shelf]
  scenes: [shelf, warehouse]
  sensors_required: ['rgb:observation.images.context', 'rgb:observation.images.wrist_left', 'rgb:observation.images.wrist_right']
  state_dim: 16
  action_dim: 16
  runtime: pytorch
  quantization: bf16/pytorch
  min_vram_gb: {bf16: 9.2}
  chunk_size: 35
  n_action_steps: 35
  latency_budget: {per_chunk_ms: 600.0}
  license_code: Apache-2.0
  license_weights: permissive_research   # NOT permissive — see License section
  weights_uri: hf://OpenRAL/rskill-pi05-openarm-restock_shelf-bf16
  source_repo: hf://qualiadev/pi05-openarm-restock-sequences@7f96e610b6cb446e55301928f3d118b6591cded0
  paper_url: https://arxiv.org/abs/2410.24164
---

# rskill-pi05-openarm-restock_shelf-bf16 — rSkill discovery view

> **Generated view, not a hand-written skill.** This `SKILL.md` is a discovery-only
> mirror of [`rskill.yaml`](./rskill.yaml), produced by `tools/generate_rskill_skillmd.py`.
> It lets tools that read the standard agent-skill format find and reason about this
> OpenRAL rSkill. The `rskill.yaml` manifest is the single source of truth
> (CLAUDE.md §1.3). Do not edit by hand — edit the manifest and regenerate.

## What it is

An OpenRAL **Vision-Language-Action policy** (`role: s1`, `kind: vla`). π0.5 (PaliGemma + flow matching, 3.617 B) fine-tuned for the Enactic OpenArm v2 bimanual station to restock boxes onto a shelf. Three RGB views and 16-D joint state in; 35-step chunks of 16-D actions out — arm joints as per-step deltas in radians, both grippers absolute. Trained at 30 fps. ONE instruction string, "restock-shelf-from-front-box"; deploy it verbatim. Upstream checkpoint and dataset are both deleted from the Hub, so this is the surviving copy — see the README.

## Capabilities

- **Verbs:** pick · place · pick_and_place · transfer
- **Objects:** box · shelf
- **Scenes:** shelf · warehouse
- **Embodiments:** openarm

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

skill = rSkill.from_pretrained("OpenRAL/rskill-pi05-openarm-restock_shelf-bf16")
# the loader validates embodiment / sensors / runtime / quantization against the target
# RobotDescription and enforces the weight-license gate before any weights load.
```

See [`rskill.yaml`](./rskill.yaml) for the authoritative, validated manifest.
