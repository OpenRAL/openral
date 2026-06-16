---
name: omdet-turbo-indoor
description: >-
  S1 object detector. Capabilities: detect on kitchenware (cup, mug, bottle, bowl, plate, pot, pan, kettle, utensils), food (fruit, vegetables, bread, packaged food), appliances (fridge, microwave, oven, toaster, kettle, TV), electronics (laptop, monitor, keyboard, phone, remote, charger, cables), furniture (chair, sofa, table, desk, bed, shelf, cabinet, drawer), bathroom items (toilet, sink, towel, soap, toothbrush, mirror). OmDet-Turbo (Swin-tiny) real-time open-vocabulary detector, run in-process over a fixed curated indoor vocabulary (~230 household/kitchen/office/ manipulation classes). An unprompted background producer: publishes ObjectsMetadata to /openral/perception/objects every frame without reasoner prompting, giving the world model far more classes than the 80 COCO categories. Apache-2.0 weights from omlab/omdet-turbo-swin-tiny-hf. See ADR-0037 (2026-06-12 amendment) for the zeroshot_hf detector engine. Discovery view of an OpenRAL rSkill — NOT directly runnable by an agent harness; it runs via rSkill.from_pretrained + the robot HAL.
metadata:
  openral_rskill: true            # generated discovery view of an rSkill
  schema_version: 0.1
  rskill_id: OpenRAL/rskill-omdet-turbo-indoor
  manifest: ./rskill.yaml
  role: s1
  kind: detector
  actions: [detect]
  objects: ['kitchenware (cup, mug, bottle, bowl, plate, pot, pan, kettle, utensils)', 'food
    (fruit, vegetables, bread, packaged food)', 'appliances (fridge, microwave, oven,
    toaster, kettle, TV)', 'electronics (laptop, monitor, keyboard, phone, remote,
    charger, cables)', 'furniture (chair, sofa, table, desk, bed, shelf, cabinet,
    drawer)', 'bathroom items (toilet, sink, towel, soap, toothbrush, mirror)', 'office
    and stationery (book, pen, scissors, stapler, folder, notebook)', 'cleaning and
    laundry (broom, mop, bucket, sponge, spray bottle, hamper)', 'containers and personal
    items (box, basket, bag, backpack, keys, glasses)', 'tools and hardware (hammer,
    screwdriver, drill, switch, outlet, door handle)', 'clothing and wearables (shoe,
    hat, glove, jacket)', 'toys (teddy bear, doll, ball, blocks)']
  scenes: [tabletop, kitchen, indoor, household, office, bathroom]
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

# omdet-turbo-indoor — rSkill discovery view

> **Generated view, not a hand-written skill.** This `SKILL.md` is a discovery-only
> mirror of [`rskill.yaml`](./rskill.yaml), produced by `tools/generate_rskill_skillmd.py`.
> It lets tools that read the standard agent-skill format find and reason about this
> OpenRAL rSkill. The `rskill.yaml` manifest is the single source of truth
> (CLAUDE.md §1.3). Do not edit by hand — edit the manifest and regenerate.

## What it is

An OpenRAL **object detector** (`role: s1`, `kind: detector`). OmDet-Turbo (Swin-tiny) real-time open-vocabulary detector, run in-process over a fixed curated indoor vocabulary (~230 household/kitchen/office/ manipulation classes). An unprompted background producer: publishes ObjectsMetadata to /openral/perception/objects every frame without reasoner prompting, giving the world model far more classes than the 80 COCO categories. Apache-2.0 weights from omlab/omdet-turbo-swin-tiny-hf. See ADR-0037 (2026-06-12 amendment) for the zeroshot_hf detector engine.

## Capabilities

- **Verbs:** detect
- **Objects:** kitchenware (cup, mug, bottle, bowl, plate, pot, pan, kettle, utensils) · food (fruit, vegetables, bread, packaged food) · appliances (fridge, microwave, oven, toaster, kettle, TV) · electronics (laptop, monitor, keyboard, phone, remote, charger, cables) · furniture (chair, sofa, table, desk, bed, shelf, cabinet, drawer) · bathroom items (toilet, sink, towel, soap, toothbrush, mirror) · office and stationery (book, pen, scissors, stapler, folder, notebook) · cleaning and laundry (broom, mop, bucket, sponge, spray bottle, hamper) · containers and personal items (box, basket, bag, backpack, keys, glasses) · tools and hardware (hammer, screwdriver, drill, switch, outlet, door handle) · clothing and wearables (shoe, hat, glove, jacket) · toys (teddy bear, doll, ball, blocks)
- **Scenes:** tabletop · kitchen · indoor · household · office · bathroom

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

skill = rSkill.from_pretrained("OpenRAL/rskill-omdet-turbo-indoor")
# the loader validates embodiment / sensors / runtime / quantization against the target
# RobotDescription and enforces the weight-license gate before any weights load.
```

See [`rskill.yaml`](./rskill.yaml) for the authoritative, validated manifest.
