---
name: rskill-internvla_n1-mobile_base-vln-nf4
description: >-
  S1 Vision-Language-Action policy. Capabilities: navigate, reach on room, hallway, door, kitchen, landmark. InternVLA-N1 / DualVLN — a dual-system Vision-Language Navigation foundation model (Qwen2.5-VL-7B waypoint planner + NavDP diffusion controller). Takes egocentric RGB-D plus a natural-language navigation instruction and drives a mobile base with body-twist velocity commands, emitting STOP on arrival. Zero-shot across wheeled and legged bases (paper: Unitree Go2 / H1). Discovery view of an OpenRAL rSkill — NOT directly runnable by an agent harness; it runs via rSkill.from_pretrained + the robot HAL.
metadata:
  openral_rskill: true            # generated discovery view of an rSkill
  schema_version: 0.1
  rskill_id: OpenRAL/rskill-internvla_n1-mobile_base-vln-nf4
  manifest: ./rskill.yaml
  role: s1
  kind: vla
  model_family: internvla_n1
  embodiment_tags: [mobile_base]
  actions: [navigate, reach]
  objects: [room, hallway, door, kitchen, landmark]
  scenes: [indoor, kitchen, home]
  sensors_required: ['rgb:observation.images.camera1']
  action_dim: 6
  runtime: pytorch
  quantization: int4/pytorch
  min_vram_gb: {int4: 7.0, bf16: 16.6}
  chunk_size: 1
  latency_budget: {per_chunk_ms: 2500.0, max_execution_s: 300.0}
  license_code: Apache-2.0
  license_weights: cc-by-nc-sa-4.0   # NOT permissive — see License section
  weights_uri: hf://InternRobotics/InternVLA-N1-DualVLN
  source_repo: hf://InternRobotics/InternVLA-N1-DualVLN
  paper_url: https://arxiv.org/abs/2512.08186
---

# rskill-internvla_n1-mobile_base-vln-nf4 — rSkill discovery view

> **Generated view, not a hand-written skill.** This `SKILL.md` is a discovery-only
> mirror of [`rskill.yaml`](./rskill.yaml), produced by `tools/generate_rskill_skillmd.py`.
> It lets tools that read the standard agent-skill format find and reason about this
> OpenRAL rSkill. The `rskill.yaml` manifest is the single source of truth
> (CLAUDE.md §1.3). Do not edit by hand — edit the manifest and regenerate.

## What it is

An OpenRAL **Vision-Language-Action policy** (`role: s1`, `kind: vla`). InternVLA-N1 / DualVLN — a dual-system Vision-Language Navigation foundation model (Qwen2.5-VL-7B waypoint planner + NavDP diffusion controller). Takes egocentric RGB-D plus a natural-language navigation instruction and drives a mobile base with body-twist velocity commands, emitting STOP on arrival. Zero-shot across wheeled and legged bases (paper: Unitree Go2 / H1).

## Capabilities

- **Verbs:** navigate · reach
- **Objects:** room · hallway · door · kitchen · landmark
- **Scenes:** indoor · kitchen · home
- **Embodiments:** mobile_base

## Why this is discovery-only

An agent skill is natural-language instructions loaded into an LLM's context. An rSkill
is an executable artifact: it carries a typed capability/embodiment contract, model weights,
a runtime, and a license/provenance gate — none of which fit in freeform markdown. So an
agent can use this view to *select* the right skill, but cannot *execute* it by loading
this file. Execution always goes through the OpenRAL loader and the robot HAL.

## License

- **Code:** Apache-2.0.
- **Weights:** `cc-by-nc-sa-4.0` — **NOT** fully permissive. The loader surfaces this posture and enforces the non-commercial guard (`OPENRAL_ALLOW_NONCOMMERCIAL=1`) where applicable. Commercial use may require a separate upstream agreement. This is third-party weight lineage; OpenRAL's own code is Apache-2.0.

## How to actually run it (not via an agent harness)

```python
from openral_rskill import rSkill

skill = rSkill.from_pretrained("OpenRAL/rskill-internvla_n1-mobile_base-vln-nf4")
# the loader validates embodiment / sensors / runtime / quantization against the target
# RobotDescription and enforces the weight-license gate before any weights load.
```

See [`rskill.yaml`](./rskill.yaml) for the authoritative, validated manifest.
