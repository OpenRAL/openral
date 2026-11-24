---
name: rskill-da3metric_large-any-metric_depth-bf16
description: >-
  S1 monocular depth estimator. Capabilities: detect on scene geometry. Depth Anything 3 metric (large) packaged as a monocular depth estimator. Estimates canonical depth from one RGB frame; the depth provider turns it into metres with the camera's own calibrated focal and back-projects it into the world-collision cloud on a robot with no depth sensor. Wraps depth-anything/DA3METRIC-LARGE. An estimate, not a measurement. Emits no actions and drives no actuators. Discovery view of an OpenRAL rSkill — NOT directly runnable by an agent harness; it runs via rSkill.from_pretrained + the robot HAL.
metadata:
  openral_rskill: true            # generated discovery view of an rSkill
  schema_version: 0.2
  rskill_id: OpenRAL/rskill-da3metric_large-any-metric_depth-bf16
  manifest: ./rskill.yaml
  role: s1
  kind: depth_estimator
  embodiment_tags: [any]
  actions: [detect]
  objects: [scene geometry]
  scenes: [tabletop, kitchen, indoor, warehouse]
  sensors_required: [rgb]
  runtime: pytorch
  quantization: bf16/pytorch
  min_vram_gb: {bf16: 2.0}
  chunk_size: 1
  latency_budget: {per_chunk_ms: 150.0}
  license_code: Apache-2.0
  license_weights: apache-2.0
  weights_uri: hf://depth-anything/DA3METRIC-LARGE@4010e39f3634a45bc60553321fb49fb760bd594e
  source_repo: hf://depth-anything/DA3METRIC-LARGE
  paper_url: https://arxiv.org/abs/2511.10647
---

# rskill-da3metric_large-any-metric_depth-bf16 — rSkill discovery view

> **Generated view, not a hand-written skill.** This `SKILL.md` is a discovery-only
> mirror of [`rskill.yaml`](./rskill.yaml), produced by `tools/generate_rskill_skillmd.py`.
> It lets tools that read the standard agent-skill format find and reason about this
> OpenRAL rSkill. The `rskill.yaml` manifest is the single source of truth
> (CLAUDE.md §1.3). Do not edit by hand — edit the manifest and regenerate.

## What it is

An OpenRAL **monocular depth estimator** (`role: s1`, `kind: depth_estimator`). Depth Anything 3 metric (large) packaged as a monocular depth estimator. Estimates canonical depth from one RGB frame; the depth provider turns it into metres with the camera's own calibrated focal and back-projects it into the world-collision cloud on a robot with no depth sensor. Wraps depth-anything/DA3METRIC-LARGE. An estimate, not a measurement. Emits no actions and drives no actuators.

## Capabilities

- **Verbs:** detect
- **Objects:** scene geometry
- **Scenes:** tabletop · kitchen · indoor · warehouse
- **Embodiments:** any

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

skill = rSkill.from_pretrained("OpenRAL/rskill-da3metric_large-any-metric_depth-bf16")
# the loader validates embodiment / sensors / runtime / quantization against the target
# RobotDescription and enforces the weight-license gate before any weights load.
```

See [`rskill.yaml`](./rskill.yaml) for the authoritative, validated manifest.
