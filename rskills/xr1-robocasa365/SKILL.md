---
name: xr1-robocasa365
description: >-
  S1 Vision-Language-Action policy. Capabilities: generalist, pick, place, open, close on kitchen_object. Apache-2.0 XR-1 RoboCasa365 checkpoint. The adapter keeps seven frames, samples four at interval two, converts OpenRAL's 16-D quaternion layout to XR-1's 14-D axis-angle state, and replays sixteen decoded actions per query. Discovery view of an OpenRAL rSkill — NOT directly runnable by an agent harness; it runs via rSkill.from_pretrained + the robot HAL.
metadata:
  openral_rskill: true            # generated discovery view of an rSkill
  schema_version: 0.1
  rskill_id: OpenRAL/rskill-xr1-panda_mobile-robocasa365-nf4
  manifest: ./rskill.yaml
  role: s1
  kind: vla
  model_family: xr1
  embodiment_tags: [panda_mobile]
  actions: [generalist, pick, place, open, close]
  objects: [kitchen_object]
  scenes: [kitchen]
  sensors_required: ['rgb:observation.images.camera1', 'rgb:observation.images.camera2', 'rgb:observation.images.camera3']
  state_dim: 16
  action_dim: 12
  runtime: pytorch
  quantization: int4/pytorch
  chunk_size: 16
  n_action_steps: 16
  latency_budget: {per_chunk_ms: 120000.0}
  license_code: Apache-2.0
  license_weights: apache-2.0
  weights_uri: hf://OpenRAL/rskill-xr1-panda_mobile-robocasa365-nf4@045c87106a8a6a98fcdb431a976f18f45215cae2
  source_repo: hf://XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365@0d1aa76d0d82debc9b611e4d1e231096434d5be4
  paper_url: https://arxiv.org/abs/2607.15330
---

# xr1-robocasa365 — rSkill discovery view

> **Generated view, not a hand-written skill.** This `SKILL.md` is a discovery-only
> mirror of [`rskill.yaml`](./rskill.yaml), produced by `tools/generate_rskill_skillmd.py`.
> It lets tools that read the standard agent-skill format find and reason about this
> OpenRAL rSkill. The `rskill.yaml` manifest is the single source of truth
> (CLAUDE.md §1.3). Do not edit by hand — edit the manifest and regenerate.

## What it is

An OpenRAL **Vision-Language-Action policy** (`role: s1`, `kind: vla`). Apache-2.0 XR-1 RoboCasa365 checkpoint. The adapter keeps seven frames, samples four at interval two, converts OpenRAL's 16-D quaternion layout to XR-1's 14-D axis-angle state, and replays sixteen decoded actions per query.

## Capabilities

- **Verbs:** generalist · pick · place · open · close
- **Objects:** kitchen_object
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

skill = rSkill.from_pretrained("OpenRAL/rskill-xr1-panda_mobile-robocasa365-nf4")
# the loader validates embodiment / sensors / runtime / quantization against the target
# RobotDescription and enforces the weight-license gate before any weights load.
```

See [`rskill.yaml`](./rskill.yaml) for the authoritative, validated manifest.
