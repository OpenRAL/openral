---
name: open-vision-banana-4b
description: >-
  S1 object detector. Capabilities: detect on open-vocabulary object, referring expression. Open Vision Banana 4B (zero-shot) — FLUX.2 [klein] fine-tuned to paint perception answers — packaged as an on-demand referring-segmentation locator. Given an RGB frame and a free-text description it paints the matching region, which is decoded into one box per connected region. No actuators. Discovery view of an OpenRAL rSkill — NOT directly runnable by an agent harness; it runs via rSkill.from_pretrained + the robot HAL.
metadata:
  openral_rskill: true            # generated discovery view of an rSkill
  schema_version: 0.1
  rskill_id: OpenRAL/rskill-open_vision_banana_4b-any-refseg-bf16
  manifest: ./rskill.yaml
  role: s1
  kind: detector
  embodiment_tags: [any]
  actions: [detect]
  objects: [open-vocabulary object, referring expression]
  scenes: [tabletop, kitchen, indoor]
  sensors_required: [rgb]
  runtime: pytorch
  quantization: bf16/pytorch
  min_vram_gb: {bf16: 17.0}
  chunk_size: 1
  latency_budget: {per_chunk_ms: 30000.0}
  license_code: Apache-2.0
  license_weights: flux_non_commercial   # NOT permissive — see License section
  weights_uri: hf://FunAILab/Open-Vision-Banana@be5da66250cc385ff9c75ebbe4114e13ec305848
  source_repo: hf://FunAILab/Open-Vision-Banana@be5da66250cc385ff9c75ebbe4114e13ec305848
  paper_url: https://arxiv.org/abs/2611.15844
---

# open-vision-banana-4b — rSkill discovery view

> **Generated view, not a hand-written skill.** This `SKILL.md` is a discovery-only
> mirror of [`rskill.yaml`](./rskill.yaml), produced by `tools/generate_rskill_skillmd.py`.
> It lets tools that read the standard agent-skill format find and reason about this
> OpenRAL rSkill. The `rskill.yaml` manifest is the single source of truth
> (CLAUDE.md §1.3). Do not edit by hand — edit the manifest and regenerate.

## What it is

An OpenRAL **object detector** (`role: s1`, `kind: detector`). Open Vision Banana 4B (zero-shot) — FLUX.2 [klein] fine-tuned to paint perception answers — packaged as an on-demand referring-segmentation locator. Given an RGB frame and a free-text description it paints the matching region, which is decoded into one box per connected region. No actuators.

## Capabilities

- **Verbs:** detect
- **Objects:** open-vocabulary object · referring expression
- **Scenes:** tabletop · kitchen · indoor
- **Embodiments:** any

## Why this is discovery-only

An agent skill is natural-language instructions loaded into an LLM's context. An rSkill
is an executable artifact: it carries a typed capability/embodiment contract, model weights,
a runtime, and a license/provenance gate — none of which fit in freeform markdown. So an
agent can use this view to *select* the right skill, but cannot *execute* it by loading
this file. Execution always goes through the OpenRAL loader and the robot HAL.

## License

- **Code:** Apache-2.0.
- **Weights:** `flux_non_commercial` — **NOT** fully permissive. The loader surfaces this posture and enforces the non-commercial guard (`OPENRAL_ALLOW_NONCOMMERCIAL=1`) where applicable. Commercial use may require a separate upstream agreement. This is third-party weight lineage; OpenRAL's own code is Apache-2.0.

## How to actually run it (not via an agent harness)

```python
from openral_rskill import rSkill

skill = rSkill.from_pretrained("OpenRAL/rskill-open_vision_banana_4b-any-refseg-bf16")
# the loader validates embodiment / sensors / runtime / quantization against the target
# RobotDescription and enforces the weight-license gate before any weights load.
```

See [`rskill.yaml`](./rskill.yaml) for the authoritative, validated manifest.
