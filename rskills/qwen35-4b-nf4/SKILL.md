---
name: qwen35-4b-nf4
description: >-
  S2 vision-language model. Capabilities: query on open-vocabulary object, text, scene region, spatial relation. Qwen3.5-4B natively-multimodal video-language model packaged as an NF4 bitsandbytes vlm rSkill. Accepts RGB image or video frames plus a natural-language query; returns a text answer. Embodiment-agnostic. No actuators. Apache-2.0. ADR-0047. Discovery view of an OpenRAL rSkill — NOT directly runnable by an agent harness; it runs via rSkill.from_pretrained + the robot HAL.
metadata:
  openral_rskill: true            # generated discovery view of an rSkill
  schema_version: 0.1
  rskill_id: OpenRAL/rskill-qwen35-4b-nf4
  manifest: ./rskill.yaml
  role: s2
  kind: vlm
  embodiment_tags: [any]
  actions: [query]
  objects: [open-vocabulary object, text, scene region, spatial relation]
  scenes: [tabletop, kitchen, indoor, outdoor, warehouse, driving]
  sensors_required: [rgb]
  runtime: pytorch
  quantization: int4/pytorch
  min_vram_gb: {fp32: 16.0, bf16: 8.0, int4: 2.5}
  chunk_size: 1
  latency_budget: {per_chunk_ms: 3000.0}
  license_code: Apache-2.0
  license_weights: apache-2.0
  weights_uri: hf://OpenRAL/rskill-qwen35-4b-nf4
  source_repo: hf://Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
---

# qwen35-4b-nf4 — rSkill discovery view

> **Generated view, not a hand-written skill.** This `SKILL.md` is a discovery-only
> mirror of [`rskill.yaml`](./rskill.yaml), produced by `tools/generate_rskill_skillmd.py`.
> It lets tools that read the standard agent-skill format find and reason about this
> OpenRAL rSkill. The `rskill.yaml` manifest is the single source of truth
> (CLAUDE.md §1.3). Do not edit by hand — edit the manifest and regenerate.

## What it is

An OpenRAL **vision-language model** (`role: s2`, `kind: vlm`). Qwen3.5-4B natively-multimodal video-language model packaged as an NF4 bitsandbytes vlm rSkill. Accepts RGB image or video frames plus a natural-language query; returns a text answer. Embodiment-agnostic. No actuators. Apache-2.0. ADR-0047.

## Capabilities

- **Verbs:** query
- **Objects:** open-vocabulary object · text · scene region · spatial relation
- **Scenes:** tabletop · kitchen · indoor · outdoor · warehouse · driving
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

skill = rSkill.from_pretrained("OpenRAL/rskill-qwen35-4b-nf4")
# the loader validates embodiment / sensors / runtime / quantization against the target
# RobotDescription and enforces the weight-license gate before any weights load.
```

See [`rskill.yaml`](./rskill.yaml) for the authoritative, validated manifest.
