---
name: smolvla-robotwin
description: >-
  S1 Vision-Language-Action policy. Capabilities: generalist, pick, place, transfer on block, pot, cup, hammer. SmolVLA (0.45 B, lerobot/smolvla_base) finetuned on the RoboTwin 2.0 unified dataset (50 dual-arm tasks, aloha-agilex embodiment, SAPIEN). Multi-task: action chunks of length 50 across three RGB views (head + per-wrist) driving a 14-DoF dual-arm joint command. Runs on the RoboTwin scene backend through the out-of-process SAPIEN sidecar (ADR-0061). Discovery view of an OpenRAL rSkill — NOT directly runnable by an agent harness; it runs via rSkill.from_pretrained + the robot HAL.
metadata:
  openral_rskill: true            # generated discovery view of an rSkill
  schema_version: 0.1
  rskill_id: OpenRAL/rskill-smolvla-robotwin
  manifest: ./rskill.yaml
  role: s1
  kind: vla
  model_family: smolvla
  embodiment_tags: [aloha_agilex]
  actions: [generalist, pick, place, transfer]
  objects: [block, pot, cup, hammer]
  scenes: [tabletop]
  sensors_required: ['rgb:observation.images.camera1', 'rgb:observation.images.camera2', 'rgb:observation.images.camera3']
  state_dim: 14
  action_dim: 14
  runtime: pytorch
  quantization: bf16/pytorch
  chunk_size: 50
  n_action_steps: 50
  latency_budget: {per_chunk_ms: 250.0}
  license_code: Apache-2.0
  license_weights: apache-2.0
  weights_uri: hf://lerobot/smolvla_robotwin
  source_repo: hf://lerobot/smolvla_robotwin
  paper_url: https://arxiv.org/abs/2506.18088
---

# smolvla-robotwin — rSkill discovery view

> **Generated view, not a hand-written skill.** This `SKILL.md` is a discovery-only
> mirror of [`rskill.yaml`](./rskill.yaml), produced by `tools/generate_rskill_skillmd.py`.
> It lets tools that read the standard agent-skill format find and reason about this
> OpenRAL rSkill. The `rskill.yaml` manifest is the single source of truth
> (CLAUDE.md §1.3). Do not edit by hand — edit the manifest and regenerate.

## What it is

An OpenRAL **Vision-Language-Action policy** (`role: s1`, `kind: vla`). SmolVLA (0.45 B, lerobot/smolvla_base) finetuned on the RoboTwin 2.0 unified dataset (50 dual-arm tasks, aloha-agilex embodiment, SAPIEN). Multi-task: action chunks of length 50 across three RGB views (head + per-wrist) driving a 14-DoF dual-arm joint command. Runs on the RoboTwin scene backend through the out-of-process SAPIEN sidecar (ADR-0061).

## Capabilities

- **Verbs:** generalist · pick · place · transfer
- **Objects:** block · pot · cup · hammer
- **Scenes:** tabletop
- **Embodiments:** aloha_agilex

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

skill = rSkill.from_pretrained("OpenRAL/rskill-smolvla-robotwin")
# the loader validates embodiment / sensors / runtime / quantization against the target
# RobotDescription and enforces the weight-license gate before any weights load.
```

See [`rskill.yaml`](./rskill.yaml) for the authoritative, validated manifest.
