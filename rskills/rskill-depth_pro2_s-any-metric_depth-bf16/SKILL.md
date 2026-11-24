---
name: rskill-depth_pro2_s-any-metric_depth-bf16
description: >-
  S1 monocular depth estimator. Capabilities: detect on scene geometry. Apple Depth Pro 2 (S, 24M parameters) packaged as a monocular metric depth estimator. Given the camera's calibrated focal length it returns metres per pixel at the input's own resolution, which the depth provider back-projects into the world-collision cloud on a robot with no depth sensor. Wraps apple/DepthPro2-S. Research-only weights (Apple ML Research Model license). An estimate, not a measurement. Emits no actions and drives no actuators. Discovery view of an OpenRAL rSkill — NOT directly runnable by an agent harness; it runs via rSkill.from_pretrained + the robot HAL.
metadata:
  openral_rskill: true            # generated discovery view of an rSkill
  schema_version: 0.2
  rskill_id: OpenRAL/rskill-depth_pro2_s-any-metric_depth-bf16
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
  min_vram_gb: {bf16: 1.2}
  chunk_size: 1
  latency_budget: {per_chunk_ms: 60.0}
  license_code: Apache-2.0
  license_weights: apple-amlr   # NOT permissive — see License section
  weights_uri: hf://apple/DepthPro2-S@903a1c1ade918831d138adcc4aedbcc8f4da0078
  source_repo: hf://apple/DepthPro2-S
  paper_url: https://arxiv.org/abs/2611.26603
---

# rskill-depth_pro2_s-any-metric_depth-bf16 — rSkill discovery view

> **Generated view, not a hand-written skill.** This `SKILL.md` is a discovery-only
> mirror of [`rskill.yaml`](./rskill.yaml), produced by `tools/generate_rskill_skillmd.py`.
> It lets tools that read the standard agent-skill format find and reason about this
> OpenRAL rSkill. The `rskill.yaml` manifest is the single source of truth
> (CLAUDE.md §1.3). Do not edit by hand — edit the manifest and regenerate.

## What it is

An OpenRAL **monocular depth estimator** (`role: s1`, `kind: depth_estimator`). Apple Depth Pro 2 (S, 24M parameters) packaged as a monocular metric depth estimator. Given the camera's calibrated focal length it returns metres per pixel at the input's own resolution, which the depth provider back-projects into the world-collision cloud on a robot with no depth sensor. Wraps apple/DepthPro2-S. Research-only weights (Apple ML Research Model license). An estimate, not a measurement. Emits no actions and drives no actuators.

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
- **Weights:** `apple-amlr` — **NOT** fully permissive. The loader surfaces this posture and enforces the non-commercial guard (`OPENRAL_ALLOW_NONCOMMERCIAL=1`) where applicable. Commercial use may require a separate upstream agreement. This is third-party weight lineage; OpenRAL's own code is Apache-2.0.

## How to actually run it (not via an agent harness)

```python
from openral_rskill import rSkill

skill = rSkill.from_pretrained("OpenRAL/rskill-depth_pro2_s-any-metric_depth-bf16")
# the loader validates embodiment / sensors / runtime / quantization against the target
# RobotDescription and enforces the weight-license gate before any weights load.
```

See [`rskill.yaml`](./rskill.yaml) for the authoritative, validated manifest.
