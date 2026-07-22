---
name: gr00t-n17-b1k-turning-on-radio
description: >-
  S1 Vision-Language-Action policy. Capabilities: rotate, push on radio, dial, button. GR00T N1.7 finetuned for the 2026 BEHAVIOR-1K turning_on_radio task on the simulated Galaxea R1 Pro. Consumes head plus dual-wrist RGB and 61-D proprioception; emits the official mixed 23-D mobile-bimanual action. Discovery view of an OpenRAL rSkill — NOT directly runnable by an agent harness; it runs via rSkill.from_pretrained + the robot HAL.
metadata:
  openral_rskill: true            # generated discovery view of an rSkill
  schema_version: 0.1
  rskill_id: OpenRAL/rskill-gr00t-n17-b1k-turning-on-radio
  manifest: ./rskill.yaml
  role: s1
  kind: vla
  model_family: gr00t
  embodiment_tags: [custom]
  actions: [rotate, push]
  objects: [radio, dial, button]
  scenes: [household, living_room]
  sensors_required: ['rgb:observation.images.head', 'rgb:observation.images.left_wrist', 'rgb:observation.images.right_wrist']
  state_dim: 61
  action_dim: 23
  runtime: pytorch
  quantization: bf16/pytorch
  chunk_size: 16
  n_action_steps: 1
  latency_budget: {per_chunk_ms: 1500.0}
  license_code: Apache-2.0
  license_weights: unknown   # NOT permissive — see License section
  weights_uri: local://checkpoints/behavior-groot-turning-on-radio
  paper_url: https://arxiv.org/abs/2503.14734
---

# gr00t-n17-b1k-turning-on-radio — rSkill discovery view

> **Generated view, not a hand-written skill.** This `SKILL.md` is a discovery-only
> mirror of [`rskill.yaml`](./rskill.yaml), produced by `tools/generate_rskill_skillmd.py`.
> It lets tools that read the standard agent-skill format find and reason about this
> OpenRAL rSkill. The `rskill.yaml` manifest is the single source of truth
> (CLAUDE.md §1.3). Do not edit by hand — edit the manifest and regenerate.

## What it is

An OpenRAL **Vision-Language-Action policy** (`role: s1`, `kind: vla`). GR00T N1.7 finetuned for the 2026 BEHAVIOR-1K turning_on_radio task on the simulated Galaxea R1 Pro. Consumes head plus dual-wrist RGB and 61-D proprioception; emits the official mixed 23-D mobile-bimanual action.

## Capabilities

- **Verbs:** rotate · push
- **Objects:** radio · dial · button
- **Scenes:** household · living_room
- **Embodiments:** custom

## Why this is discovery-only

An agent skill is natural-language instructions loaded into an LLM's context. An rSkill
is an executable artifact: it carries a typed capability/embodiment contract, model weights,
a runtime, and a license/provenance gate — none of which fit in freeform markdown. So an
agent can use this view to *select* the right skill, but cannot *execute* it by loading
this file. Execution always goes through the OpenRAL loader and the robot HAL.

## License

- **Code:** Apache-2.0.
- **Weights:** `unknown` — **NOT** fully permissive. The loader surfaces this posture and enforces the non-commercial guard (`OPENRAL_ALLOW_NONCOMMERCIAL=1`) where applicable. Commercial use may require a separate upstream agreement. This is third-party weight lineage; OpenRAL's own code is Apache-2.0.

## How to actually run it (not via an agent harness)

```python
from openral_rskill import rSkill

skill = rSkill.from_pretrained("OpenRAL/rskill-gr00t-n17-b1k-turning-on-radio")
# the loader validates embodiment / sensors / runtime / quantization against the target
# RobotDescription and enforces the weight-license gate before any weights load.
```

See [`rskill.yaml`](./rskill.yaml) for the authoritative, validated manifest.
