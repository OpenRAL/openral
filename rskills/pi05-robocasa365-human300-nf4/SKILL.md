---
name: pi05-robocasa365-human300-nf4
description: >-
  S1 Vision-Language-Action policy. Capabilities: generalist, pick, place, open, close, pour. Pre-quantized nf4 mirror of the openpi-converted robocasa/robocasa365_checkpoints/pi05_pretrain_human300 checkpoint (PaliGemma 3.4 B, 300 atomic+composite RoboCasa365 tasks). The pi05 adapter detects the quantization_metadata.json sentinel and skips the bf16->nf4 conversion, dropping warm-up to ~20 s on a 4070-mobile. Image input is vertically flipped before ingestion to match openpi-robocasa eval; state is the 16-D human300 layout. Discovery view of an OpenRAL rSkill — NOT directly runnable by an agent harness; it runs via rSkill.from_pretrained + the robot HAL.
metadata:
  openral_rskill: true            # generated discovery view of an rSkill
  schema_version: 0.1
  rskill_id: OpenRAL/rskill-pi05-robocasa365-human300-nf4
  manifest: ./rskill.yaml
  role: s1
  kind: vla
  model_family: pi05
  embodiment_tags: [panda_mobile]
  actions: [generalist, pick, place, open, close, pour, wipe, push]
  scenes: [kitchen]
  sensors_required: ['rgb:observation.images.camera1', 'rgb:observation.images.camera2']
  state_dim: 16
  action_dim: 12
  runtime: pytorch
  quantization: int4/pytorch
  chunk_size: 50
  latency_budget: {per_chunk_ms: 5000.0}
  license_code: Apache-2.0
  license_weights: apache-2.0
  weights_uri: hf://OpenRAL/rskill-pi05-robocasa365-human300-nf4
  source_repo: hf://OpenRAL/rskill-pi05-robocasa365-human300-nf4
---

# pi05-robocasa365-human300-nf4 — rSkill discovery view

> **Generated view, not a hand-written skill.** This `SKILL.md` is a discovery-only
> mirror of [`rskill.yaml`](./rskill.yaml), produced by `tools/generate_rskill_skillmd.py`.
> It lets tools that read the standard agent-skill format find and reason about this
> OpenRAL rSkill. The `rskill.yaml` manifest is the single source of truth
> (CLAUDE.md §1.3). Do not edit by hand — edit the manifest and regenerate.

## What it is

An OpenRAL **Vision-Language-Action policy** (`role: s1`, `kind: vla`). Pre-quantized nf4 mirror of the openpi-converted robocasa/robocasa365_checkpoints/pi05_pretrain_human300 checkpoint (PaliGemma 3.4 B, 300 atomic+composite RoboCasa365 tasks). The pi05 adapter detects the quantization_metadata.json sentinel and skips the bf16->nf4 conversion, dropping warm-up to ~20 s on a 4070-mobile. Image input is vertically flipped before ingestion to match openpi-robocasa eval; state is the 16-D human300 layout.

## Capabilities

- **Verbs:** generalist · pick · place · open · close · pour · wipe · push
- **Scenes:** kitchen
- **Embodiments:** panda_mobile

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

skill = rSkill.from_pretrained("OpenRAL/rskill-pi05-robocasa365-human300-nf4")
# the loader validates embodiment / sensors / runtime / quantization against the target
# RobotDescription and enforces the weight-license gate before any weights load.
```

See [`rskill.yaml`](./rskill.yaml) for the authoritative, validated manifest.
