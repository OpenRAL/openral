---
name: omdet-turbo-locator
description: >-
  S1 object detector. Capabilities: detect on open-vocabulary queried object. OmDet-Turbo (Swin-tiny) on-demand open-vocabulary locator. The reasoner prompts it via the read-only locate_in_view tool to find a specific object in the current frame — a lightweight, real-time, Apache-2.0 in-process alternative to the 3B LocateAnything VLM for simple "find X" queries. Wraps omlab/omdet-turbo-swin-tiny-hf. See ADR-0051 for the continuous-vs-on_demand detector mode split. Discovery view of an OpenRAL rSkill — NOT directly runnable by an agent harness; it runs via rSkill.from_pretrained + the robot HAL.
metadata:
  openral_rskill: true            # generated discovery view of an rSkill
  schema_version: 0.1
  rskill_id: OpenRAL/rskill-omdet-turbo-locator
  manifest: ./rskill.yaml
  role: s1
  kind: detector
  actions: [detect]
  objects: [open-vocabulary queried object]
  scenes: [tabletop, kitchen, indoor, household, office]
  sensors_required: [rgb]
  runtime: pytorch
  quantization: fp16/pytorch
  min_vram_gb: {fp32: 1.0, fp16: 0.6}
  chunk_size: 1
  latency_budget: {per_chunk_ms: 200.0}
  license_code: Apache-2.0
  license_weights: apache-2.0
  weights_uri: hf://omlab/omdet-turbo-swin-tiny-hf
  source_repo: hf://omlab/omdet-turbo-swin-tiny-hf
  paper_url: https://arxiv.org/abs/2403.06892
---

# omdet-turbo-locator — rSkill discovery view

> **Generated view, not a hand-written skill.** This `SKILL.md` is a discovery-only
> mirror of [`rskill.yaml`](./rskill.yaml), produced by `tools/generate_rskill_skillmd.py`.
> It lets tools that read the standard agent-skill format find and reason about this
> OpenRAL rSkill. The `rskill.yaml` manifest is the single source of truth
> (CLAUDE.md §1.3). Do not edit by hand — edit the manifest and regenerate.

## What it is

An OpenRAL **object detector** (`role: s1`, `kind: detector`). OmDet-Turbo (Swin-tiny) on-demand open-vocabulary locator. The reasoner prompts it via the read-only locate_in_view tool to find a specific object in the current frame — a lightweight, real-time, Apache-2.0 in-process alternative to the 3B LocateAnything VLM for simple "find X" queries. Wraps omlab/omdet-turbo-swin-tiny-hf. See ADR-0051 for the continuous-vs-on_demand detector mode split.

## Capabilities

- **Verbs:** detect
- **Objects:** open-vocabulary queried object
- **Scenes:** tabletop · kitchen · indoor · household · office

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

skill = rSkill.from_pretrained("OpenRAL/rskill-omdet-turbo-locator")
# the loader validates embodiment / sensors / runtime / quantization against the target
# RobotDescription and enforces the weight-license gate before any weights load.
```

See [`rskill.yaml`](./rskill.yaml) for the authoritative, validated manifest.
