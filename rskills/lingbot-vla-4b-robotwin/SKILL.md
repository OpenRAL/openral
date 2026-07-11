---
name: lingbot-vla-4b-robotwin
description: >-
  S1 Vision-Language-Action policy. Capabilities: generalist, pick, place, grasp, transfer on pot, block, bottle, cup, bowl. LingBot-VLA 1.0 (4 B, Qwen2.5-VL-3B + dense Qwen2 flow-matching expert) POST-TRAINED on RoboTwin for the AgileX Cobot Magic dual-arm embodiment. Predicts 50-step chunks from three RGB views (head + per-wrist) driving a 14-DoF dual-arm joint command (25-step replay). Runs in a torch-2.8 ZMQ sidecar; NF4 fits the 4 B model in 3.65 GB, co-resident with SAPIEN on an 8 GB GPU. Verified: open-loop MAE 0.023 vs GT lift_pot (not mean-collapsed). Discovery view of an OpenRAL rSkill — NOT directly runnable by an agent harness; it runs via rSkill.from_pretrained + the robot HAL.
metadata:
  openral_rskill: true            # generated discovery view of an rSkill
  schema_version: 0.1
  rskill_id: OpenRAL/rskill-lingbot-vla-4b-robotwin
  manifest: ./rskill.yaml
  role: s1
  kind: vla
  model_family: lingbot_vla
  embodiment_tags: [aloha_agilex]
  actions: [generalist, pick, place, grasp, transfer]
  objects: [pot, block, bottle, cup, bowl]
  scenes: [tabletop]
  sensors_required: ['rgb:observation.images.camera1', 'rgb:observation.images.camera2', 'rgb:observation.images.camera3']
  state_dim: 14
  action_dim: 14
  action_representation: joint_positions
  runtime: pytorch
  quantization: int4/pytorch
  min_vram_gb: {fp32: 16.8, bf16: 8.4, int4: 4.5}
  chunk_size: 50
  n_action_steps: 25
  latency_budget: {per_chunk_ms: 1000.0}
  license_code: Apache-2.0
  license_weights: apache-2.0
  weights_uri: hf://robbyant/lingbot-vla-4b-posttrain-robotwin@fb71a2c9749ccfedbb7290c2c3f0e5e7c7305c9e
  source_repo: hf://robbyant/lingbot-vla-4b-posttrain-robotwin
  paper_url: https://arxiv.org/abs/2601.18692
---

# lingbot-vla-4b-robotwin — rSkill discovery view

> **Generated view, not a hand-written skill.** This `SKILL.md` is a discovery-only
> mirror of [`rskill.yaml`](./rskill.yaml), produced by `tools/generate_rskill_skillmd.py`.
> It lets tools that read the standard agent-skill format find and reason about this
> OpenRAL rSkill. The `rskill.yaml` manifest is the single source of truth
> (CLAUDE.md §1.3). Do not edit by hand — edit the manifest and regenerate.

## What it is

An OpenRAL **Vision-Language-Action policy** (`role: s1`, `kind: vla`). LingBot-VLA 1.0 (4 B, Qwen2.5-VL-3B + dense Qwen2 flow-matching expert) POST-TRAINED on RoboTwin for the AgileX Cobot Magic dual-arm embodiment. Predicts 50-step chunks from three RGB views (head + per-wrist) driving a 14-DoF dual-arm joint command (25-step replay). Runs in a torch-2.8 ZMQ sidecar; NF4 fits the 4 B model in 3.65 GB, co-resident with SAPIEN on an 8 GB GPU. Verified: open-loop MAE 0.023 vs GT lift_pot (not mean-collapsed).

## Capabilities

- **Verbs:** generalist · pick · place · grasp · transfer
- **Objects:** pot · block · bottle · cup · bowl
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

skill = rSkill.from_pretrained("OpenRAL/rskill-lingbot-vla-4b-robotwin")
# the loader validates embodiment / sensors / runtime / quantization against the target
# RobotDescription and enforces the weight-license gate before any weights load.
```

See [`rskill.yaml`](./rskill.yaml) for the authoritative, validated manifest.
