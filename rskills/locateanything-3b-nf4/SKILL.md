---
name: locateanything-3b-nf4
description: >-
  S1 object detector. Capabilities: detect on open-vocabulary object, text, gui element, point target. NVIDIA LocateAnything-3B open-vocabulary grounding detector packaged as an NF4 bitsandbytes PyTorch rSkill. It localizes queried objects, text, GUI elements, and points from RGB images without commanding actuators. Discovery view of an OpenRAL rSkill — NOT directly runnable by an agent harness; it runs via rSkill.from_pretrained + the robot HAL.
metadata:
  openral_rskill: true            # generated discovery view of an rSkill
  schema_version: 0.1
  rskill_id: OpenRAL/rskill-locateanything-3b-nf4
  manifest: ./rskill.yaml
  role: s1
  kind: detector
  actions: [detect]
  objects: [open-vocabulary object, text, gui element, point target]
  scenes: [tabletop, kitchen, indoor, document, gui, driving]
  sensors_required: [rgb]
  runtime: pytorch
  quantization: int4/pytorch
  min_vram_gb: {fp32: 17.0, bf16: 8.5, int4: 5.0}
  chunk_size: 1
  latency_budget: {per_chunk_ms: 1000.0}
  license_code: Apache-2.0
  license_weights: nvidia_non_commercial   # NOT permissive — see License section
  weights_uri: hf://OpenRAL/rskill-locateanything-3b-nf4
  source_repo: hf://nvidia/LocateAnything-3B@7a81d810571dc5f244b2f0b6868128f24b1cbd85
  paper_url: https://arxiv.org/abs/2605.27365
---

# locateanything-3b-nf4 — rSkill discovery view

> **Generated view, not a hand-written skill.** This `SKILL.md` is a discovery-only
> mirror of [`rskill.yaml`](./rskill.yaml), produced by `tools/generate_rskill_skillmd.py`.
> It lets tools that read the standard agent-skill format find and reason about this
> OpenRAL rSkill. The `rskill.yaml` manifest is the single source of truth
> (CLAUDE.md §1.3). Do not edit by hand — edit the manifest and regenerate.

## What it is

An OpenRAL **object detector** (`role: s1`, `kind: detector`). NVIDIA LocateAnything-3B open-vocabulary grounding detector packaged as an NF4 bitsandbytes PyTorch rSkill. It localizes queried objects, text, GUI elements, and points from RGB images without commanding actuators.

## Capabilities

- **Verbs:** detect
- **Objects:** open-vocabulary object · text · gui element · point target
- **Scenes:** tabletop · kitchen · indoor · document · gui · driving

## Why this is discovery-only

An agent skill is natural-language instructions loaded into an LLM's context. An rSkill
is an executable artifact: it carries a typed capability/embodiment contract, model weights,
a runtime, and a license/provenance gate — none of which fit in freeform markdown. So an
agent can use this view to *select* the right skill, but cannot *execute* it by loading
this file. Execution always goes through the OpenRAL loader and the robot HAL.

## License

- **Code:** Apache-2.0.
- **Weights:** `nvidia_non_commercial` — **NOT** fully permissive. The loader surfaces this posture and enforces the non-commercial guard (`OPENRAL_ALLOW_NONCOMMERCIAL=1`) where applicable. Commercial use may require a separate upstream agreement. This is third-party weight lineage; OpenRAL's own code is Apache-2.0.

## How to actually run it (not via an agent harness)

```python
from openral_rskill import rSkill

skill = rSkill.from_pretrained("OpenRAL/rskill-locateanything-3b-nf4")
# the loader validates embodiment / sensors / runtime / quantization against the target
# RobotDescription and enforces the weight-license gate before any weights load.
```

See [`rskill.yaml`](./rskill.yaml) for the authoritative, validated manifest.
