---
name: smolvla-maniskill-franka
description: >-
  S1 Vision-Language-Action policy. Capabilities: pick, grasp on cube. SmolVLA (0.45 B, lerobot/smolvla_base) finetuned on Calvert0921/SmolVLA_LiftCube_Franka_1000 (1000 demos of a Franka Panda lifting a cube in ManiSkill3 SAPIEN). Action chunks of length 50 across overhead + wrist RGB views and a 9-D Franka qpos state. Runs end-to-end on ManiSkill3 PickCube-v1 with a live SAPIEN viewer via `openral sim run --view`. Discovery view of an OpenRAL rSkill — NOT directly runnable by an agent harness; it runs via rSkill.from_pretrained + the robot HAL.
metadata:
  openral_rskill: true            # generated discovery view of an rSkill
  schema_version: 0.1
  rskill_id: OpenRAL/rskill-smolvla-maniskill-franka
  manifest: ./rskill.yaml
  role: s1
  kind: vla
  model_family: smolvla
  embodiment_tags: [franka_panda]
  actions: [pick, grasp]
  objects: [cube]
  scenes: [tabletop]
  sensors_required: ['rgb:observation.images.camera1', 'rgb:observation.images.camera2']
  state_dim: 9
  action_dim: 8
  runtime: pytorch
  quantization: bf16/pytorch
  chunk_size: 50
  n_action_steps: 50
  latency_budget: {per_chunk_ms: 200.0}
  license_code: Apache-2.0
  license_weights: apache-2.0
  weights_uri: hf://Calvert0921/smolvla_franka_liftcube_1000
  source_repo: hf://Calvert0921/smolvla_franka_liftcube_1000
  paper_url: https://arxiv.org/abs/2506.01844
---

# smolvla-maniskill-franka — rSkill discovery view

> **Generated view, not a hand-written skill.** This `SKILL.md` is a discovery-only
> mirror of [`rskill.yaml`](./rskill.yaml), produced by `tools/generate_rskill_skillmd.py`.
> It lets tools that read the standard agent-skill format find and reason about this
> OpenRAL rSkill. The `rskill.yaml` manifest is the single source of truth
> (CLAUDE.md §1.3). Do not edit by hand — edit the manifest and regenerate.

## What it is

An OpenRAL **Vision-Language-Action policy** (`role: s1`, `kind: vla`). SmolVLA (0.45 B, lerobot/smolvla_base) finetuned on Calvert0921/SmolVLA_LiftCube_Franka_1000 (1000 demos of a Franka Panda lifting a cube in ManiSkill3 SAPIEN). Action chunks of length 50 across overhead + wrist RGB views and a 9-D Franka qpos state. Runs end-to-end on ManiSkill3 PickCube-v1 with a live SAPIEN viewer via `openral sim run --view`.

## Capabilities

- **Verbs:** pick · grasp
- **Objects:** cube
- **Scenes:** tabletop
- **Embodiments:** franka_panda

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

skill = rSkill.from_pretrained("OpenRAL/rskill-smolvla-maniskill-franka")
# the loader validates embodiment / sensors / runtime / quantization against the target
# RobotDescription and enforces the weight-license gate before any weights load.
```

See [`rskill.yaml`](./rskill.yaml) for the authoritative, validated manifest.
