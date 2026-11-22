---
name: cosmos3-edge-libero-bf16
description: >-
  S1 Vision-Language-Action policy. Capabilities: pick, place, open, close, push on bowl, plate, drawer, object, stove. NVIDIA Cosmos 3 Edge world action model fine-tuned on all four LIBERO suites, run in-process through lerobot's Cosmos3Policy as an action policy. From the task instruction and the third-person + wrist views it samples (UniPC, 30 steps) a 16-step chunk of end-effector deltas for the Franka Panda together with the future video, which is discarded. Model card: 97.95 % average over LIBERO Spatial / Object / Goal / 10 (MuJoCo 2.3.7). bf16. Discovery view of an OpenRAL rSkill — NOT directly runnable by an agent harness; it runs via rSkill.from_pretrained + the robot HAL.
metadata:
  openral_rskill: true            # generated discovery view of an rSkill
  schema_version: 0.1
  rskill_id: OpenRAL/rskill-cosmos3-franka_panda-libero-bf16
  manifest: ./rskill.yaml
  role: s1
  kind: vla
  model_family: cosmos3
  embodiment_tags: [franka_panda]
  actions: [pick, place, open, close, push]
  objects: [bowl, plate, drawer, object, stove]
  scenes: [tabletop, kitchen]
  sensors_required: ['rgb:observation.images.camera1', 'rgb:observation.images.camera2']
  state_dim: 8
  action_dim: 7
  action_representation: delta_ee_6d_plus_gripper
  runtime: pytorch
  quantization: bf16/pytorch
  min_vram_gb: {bf16: 14.1}
  chunk_size: 16
  n_action_steps: 16
  latency_budget: {per_chunk_ms: 1500.0}
  license_code: Apache-2.0
  license_weights: openmdw-1.1   # NOT permissive — see License section
  weights_uri: hf://lerobot/Cosmos3-Edge-Policy-LIBERO@9d4de4350e416c12babc9a42db01ab0c5df9f82f
  source_repo: hf://nvidia/Cosmos3-Edge
  paper_url: https://arxiv.org/abs/2606.02800
---

# cosmos3-edge-libero-bf16 — rSkill discovery view

> **Generated view, not a hand-written skill.** This `SKILL.md` is a discovery-only
> mirror of [`rskill.yaml`](./rskill.yaml), produced by `tools/generate_rskill_skillmd.py`.
> It lets tools that read the standard agent-skill format find and reason about this
> OpenRAL rSkill. The `rskill.yaml` manifest is the single source of truth
> (CLAUDE.md §1.3). Do not edit by hand — edit the manifest and regenerate.

## What it is

An OpenRAL **Vision-Language-Action policy** (`role: s1`, `kind: vla`). NVIDIA Cosmos 3 Edge world action model fine-tuned on all four LIBERO suites, run in-process through lerobot's Cosmos3Policy as an action policy. From the task instruction and the third-person + wrist views it samples (UniPC, 30 steps) a 16-step chunk of end-effector deltas for the Franka Panda together with the future video, which is discarded. Model card: 97.95 % average over LIBERO Spatial / Object / Goal / 10 (MuJoCo 2.3.7). bf16.

## Capabilities

- **Verbs:** pick · place · open · close · push
- **Objects:** bowl · plate · drawer · object · stove
- **Scenes:** tabletop · kitchen
- **Embodiments:** franka_panda

## Why this is discovery-only

An agent skill is natural-language instructions loaded into an LLM's context. An rSkill
is an executable artifact: it carries a typed capability/embodiment contract, model weights,
a runtime, and a license/provenance gate — none of which fit in freeform markdown. So an
agent can use this view to *select* the right skill, but cannot *execute* it by loading
this file. Execution always goes through the OpenRAL loader and the robot HAL.

## License

- **Code:** Apache-2.0.
- **Weights:** `openmdw-1.1` — **NOT** fully permissive. The loader surfaces this posture and enforces the non-commercial guard (`OPENRAL_ALLOW_NONCOMMERCIAL=1`) where applicable. Commercial use may require a separate upstream agreement. This is third-party weight lineage; OpenRAL's own code is Apache-2.0.

## How to actually run it (not via an agent harness)

```python
from openral_rskill import rSkill

skill = rSkill.from_pretrained("OpenRAL/rskill-cosmos3-franka_panda-libero-bf16")
# the loader validates embodiment / sensors / runtime / quantization against the target
# RobotDescription and enforces the weight-license gate before any weights load.
```

See [`rskill.yaml`](./rskill.yaml) for the authoritative, validated manifest.
