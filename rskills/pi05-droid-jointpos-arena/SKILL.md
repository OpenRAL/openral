---
name: pi05-droid-jointpos-arena
description: >-
  S1 Vision-Language-Action policy. Capabilities: pick, place, pick_and_place on bowl, object. π0.5 DROID joint-position policy (openpi pi05_droid_jointpos, the checkpoint Isaac Lab-Arena evaluates) for a Franka Panda + Robotiq 2F-85: picks a tabletop object and places it in a container from one exterior and one wrist RGB view and the 8-D joint + gripper state, at 15 Hz in 15-step chunks of 7 joint deltas plus an absolute gripper. PI permissive-research weights: commercial use needs a vendor agreement. Discovery view of an OpenRAL rSkill — NOT directly runnable by an agent harness; it runs via rSkill.from_pretrained + the robot HAL.
metadata:
  openral_rskill: true            # generated discovery view of an rSkill
  schema_version: 0.1
  rskill_id: OpenRAL/rskill-pi05-droid_franka_robotiq-droid_jointpos-bf16
  manifest: ./rskill.yaml
  role: s1
  kind: vla
  model_family: pi05
  embodiment_tags: [droid_franka_robotiq]
  actions: [pick, place, pick_and_place]
  objects: [bowl, object]
  scenes: [tabletop]
  sensors_required: ['rgb:observation.images.camera1', 'rgb:observation.images.camera2']
  state_dim: 8
  action_dim: 8
  action_representation: joint_positions
  runtime: pytorch
  quantization: bf16/pytorch
  min_vram_gb: {bf16: 8.1}
  chunk_size: 15
  n_action_steps: 15
  latency_budget: {per_chunk_ms: 400.0}
  license_code: Apache-2.0
  license_weights: permissive_research   # NOT permissive — see License section
  weights_uri: hf://DAVIAN-Robotics/pi05_droid_jointpos@66f1ce6e5d199eb7f8f266bc5ed9d67644fbb6f1
  source_repo: hf://DAVIAN-Robotics/pi05_droid_jointpos
  paper_url: https://arxiv.org/abs/2504.16054
---

# pi05-droid-jointpos-arena — rSkill discovery view

> **Generated view, not a hand-written skill.** This `SKILL.md` is a discovery-only
> mirror of [`rskill.yaml`](./rskill.yaml), produced by `tools/generate_rskill_skillmd.py`.
> It lets tools that read the standard agent-skill format find and reason about this
> OpenRAL rSkill. The `rskill.yaml` manifest is the single source of truth
> (CLAUDE.md §1.3). Do not edit by hand — edit the manifest and regenerate.

## What it is

An OpenRAL **Vision-Language-Action policy** (`role: s1`, `kind: vla`). π0.5 DROID joint-position policy (openpi pi05_droid_jointpos, the checkpoint Isaac Lab-Arena evaluates) for a Franka Panda + Robotiq 2F-85: picks a tabletop object and places it in a container from one exterior and one wrist RGB view and the 8-D joint + gripper state, at 15 Hz in 15-step chunks of 7 joint deltas plus an absolute gripper. PI permissive-research weights: commercial use needs a vendor agreement.

## Capabilities

- **Verbs:** pick · place · pick_and_place
- **Objects:** bowl · object
- **Scenes:** tabletop
- **Embodiments:** droid_franka_robotiq

## Why this is discovery-only

An agent skill is natural-language instructions loaded into an LLM's context. An rSkill
is an executable artifact: it carries a typed capability/embodiment contract, model weights,
a runtime, and a license/provenance gate — none of which fit in freeform markdown. So an
agent can use this view to *select* the right skill, but cannot *execute* it by loading
this file. Execution always goes through the OpenRAL loader and the robot HAL.

## License

- **Code:** Apache-2.0.
- **Weights:** `permissive_research` — **NOT** fully permissive. The loader surfaces this posture and enforces the non-commercial guard (`OPENRAL_ALLOW_NONCOMMERCIAL=1`) where applicable. Commercial use may require a separate upstream agreement. This is third-party weight lineage; OpenRAL's own code is Apache-2.0.

## How to actually run it (not via an agent harness)

```python
from openral_rskill import rSkill

skill = rSkill.from_pretrained("OpenRAL/rskill-pi05-droid_franka_robotiq-droid_jointpos-bf16")
# the loader validates embodiment / sensors / runtime / quantization against the target
# RobotDescription and enforces the weight-license gate before any weights load.
```

See [`rskill.yaml`](./rskill.yaml) for the authoritative, validated manifest.
