---
name: rtdetr-coco-r18
description: >-
  S1 object detector. Capabilities: detect on person, cup, bottle, bowl, chair, table. RT-DETR-L (Real-Time DEtection TRansformer, large variant) trained on COCO and exported to ONNX. Runs on the camera tee and publishes ObjectsMetadata to /openral/perception/objects. 80 COCO categories. Apache-2.0 weights. Reference latency ~20 ms on GPU, ~45 ms on CPU. See ADR-0037 for the detector rSkill kind contract. Discovery view of an OpenRAL rSkill — NOT directly runnable by an agent harness; it runs via rSkill.from_pretrained + the robot HAL.
metadata:
  openral_rskill: true            # generated discovery view of an rSkill
  schema_version: 0.1
  rskill_id: OpenRAL/rskill-rtdetr-coco-r18
  manifest: ./rskill.yaml
  role: s1
  kind: detector
  actions: [detect]
  objects: [person, cup, bottle, bowl, chair, table]
  scenes: [tabletop, kitchen, indoor]
  sensors_required: [rgb]
  runtime: onnx
  quantization: fp32/onnx
  chunk_size: 1
  latency_budget: {per_chunk_ms: 50.0}
  license_code: Apache-2.0
  license_weights: apache-2.0
  weights_uri: local://rskills/rtdetr-coco-r18
  source_repo: hf://PekingU/rtdetr_r18vd_coco_o365
  paper_url: https://arxiv.org/abs/2304.08069
---

# rtdetr-coco-r18 — rSkill discovery view

> **Generated view, not a hand-written skill.** This `SKILL.md` is a discovery-only
> mirror of [`rskill.yaml`](./rskill.yaml), produced by `tools/generate_rskill_skillmd.py`.
> It lets tools that read the standard agent-skill format find and reason about this
> OpenRAL rSkill. The `rskill.yaml` manifest is the single source of truth
> (CLAUDE.md §1.3). Do not edit by hand — edit the manifest and regenerate.

## What it is

An OpenRAL **object detector** (`role: s1`, `kind: detector`). RT-DETR-L (Real-Time DEtection TRansformer, large variant) trained on COCO and exported to ONNX. Runs on the camera tee and publishes ObjectsMetadata to /openral/perception/objects. 80 COCO categories. Apache-2.0 weights. Reference latency ~20 ms on GPU, ~45 ms on CPU. See ADR-0037 for the detector rSkill kind contract.

## Capabilities

- **Verbs:** detect
- **Objects:** person · cup · bottle · bowl · chair · table
- **Scenes:** tabletop · kitchen · indoor

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

skill = rSkill.from_pretrained("OpenRAL/rskill-rtdetr-coco-r18")
# the loader validates embodiment / sensors / runtime / quantization against the target
# RobotDescription and enforces the weight-license gate before any weights load.
```

See [`rskill.yaml`](./rskill.yaml) for the authoritative, validated manifest.
