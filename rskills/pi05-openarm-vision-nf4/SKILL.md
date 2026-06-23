---
name: pi05-openarm-vision-nf4
description: >-
  S1 Vision-Language-Action policy. Capabilities: pick, place, pick_and_place. π0.5 OpenArm v2 bimanual fine-tune trained on mddoai/openarm_2026-05-14_clean (89 episodes), NF4-quantized for the 8 GiB 4070-mobile dev target. Emits 16-D absolute joint-position actions in LEFT-FIRST order, in radians. Drops in to the OpenArm URDF without permutation or rad↔deg conversion. Headline numbers remain reproduced_locally=false pending `openral benchmark run`. Discovery view of an OpenRAL rSkill — NOT directly runnable by an agent harness; it runs via rSkill.from_pretrained + the robot HAL.
metadata:
  openral_rskill: true            # generated discovery view of an rSkill
  schema_version: 0.1
  rskill_id: OpenRAL/rskill-pi05-openarm-vision-nf4
  manifest: ./rskill.yaml
  role: s1
  kind: vla
  model_family: pi05
  embodiment_tags: [openarm]
  actions: [pick, place, pick_and_place]
  sensors_required: ['rgb:observation.images.base', 'rgb:observation.images.left_wrist', 'rgb:observation.images.right_wrist']
  state_dim: 16
  action_dim: 16
  runtime: pytorch
  quantization: int4/pytorch
  min_vram_gb: {fp32: 18.0, bf16: 8.0, int4: 4.0}
  chunk_size: 50
  latency_budget: {per_chunk_ms: 1000.0}
  license_code: Apache-2.0
  license_weights: apache-2.0
  weights_uri: hf://OpenRAL/rskill-pi05-openarm-vision-nf4
  source_repo: hf://mddoai/pi05_openarm_vision
  paper_url: https://www.physicalintelligence.company/blog/pi05
---

# pi05-openarm-vision-nf4 — rSkill discovery view

> **Generated view, not a hand-written skill.** This `SKILL.md` is a discovery-only
> mirror of [`rskill.yaml`](./rskill.yaml), produced by `tools/generate_rskill_skillmd.py`.
> It lets tools that read the standard agent-skill format find and reason about this
> OpenRAL rSkill. The `rskill.yaml` manifest is the single source of truth
> (CLAUDE.md §1.3). Do not edit by hand — edit the manifest and regenerate.

## What it is

An OpenRAL **Vision-Language-Action policy** (`role: s1`, `kind: vla`). π0.5 OpenArm v2 bimanual fine-tune trained on mddoai/openarm_2026-05-14_clean (89 episodes), NF4-quantized for the 8 GiB 4070-mobile dev target. Emits 16-D absolute joint-position actions in LEFT-FIRST order, in radians. Drops in to the OpenArm URDF without permutation or rad↔deg conversion. Headline numbers remain reproduced_locally=false pending `openral benchmark run`.

## Capabilities

- **Verbs:** pick · place · pick_and_place
- **Embodiments:** openarm

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

skill = rSkill.from_pretrained("OpenRAL/rskill-pi05-openarm-vision-nf4")
# the loader validates embodiment / sensors / runtime / quantization against the target
# RobotDescription and enforces the weight-license gate before any weights load.
```

See [`rskill.yaml`](./rskill.yaml) for the authoritative, validated manifest.
