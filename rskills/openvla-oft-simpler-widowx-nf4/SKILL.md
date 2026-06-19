---
name: openvla-oft-simpler-widowx-nf4
description: >-
  S1 Vision-Language-Action policy. Capabilities: pick, place on object. OpenVLA-OFT (RLinf, PPO on ManiSkill3 PutOnPlateInScene25) WidowX bridge policy, evaluated on SimplerEnv WidowX carrot-on-plate. Loaded in-process via transformers custom-code (trust_remote_code), NF4 for 8 GB hosts. MIT license. Discovery view of an OpenRAL rSkill — NOT directly runnable by an agent harness; it runs via rSkill.from_pretrained + the robot HAL.
metadata:
  openral_rskill: true            # generated discovery view of an rSkill
  schema_version: 0.1
  rskill_id: OpenRAL/rskill-openvla-oft-simpler-widowx-nf4
  manifest: ./rskill.yaml
  role: s1
  kind: vla
  model_family: openvla
  embodiment_tags: [widowx]
  actions: [pick, place]
  objects: [object]
  scenes: [tabletop]
  sensors_required: ['rgb:observation.images.camera1']
  state_dim: 8
  action_dim: 7
  action_representation: delta_ee_6d_plus_gripper
  runtime: pytorch
  quantization: int4/pytorch
  min_vram_gb: {bf16: 16.8, int4: 7.0}
  chunk_size: 8
  n_action_steps: 8
  latency_budget: {per_chunk_ms: 2000.0}
  license_code: Apache-2.0
  license_weights: mit
  weights_uri: hf://RLinf/RLinf-OpenVLAOFT-PPO-ManiSkill3-25ood
  source_repo: hf://RLinf/RLinf-OpenVLAOFT-PPO-ManiSkill3-25ood
  paper_url: https://huggingface.co/RLinf/RLinf-OpenVLAOFT-PPO-ManiSkill3-25ood
---

# openvla-oft-simpler-widowx-nf4 — rSkill discovery view

> **Generated view, not a hand-written skill.** This `SKILL.md` is a discovery-only
> mirror of [`rskill.yaml`](./rskill.yaml), produced by `tools/generate_rskill_skillmd.py`.
> It lets tools that read the standard agent-skill format find and reason about this
> OpenRAL rSkill. The `rskill.yaml` manifest is the single source of truth
> (CLAUDE.md §1.3). Do not edit by hand — edit the manifest and regenerate.

## What it is

An OpenRAL **Vision-Language-Action policy** (`role: s1`, `kind: vla`). OpenVLA-OFT (RLinf, PPO on ManiSkill3 PutOnPlateInScene25) WidowX bridge policy, evaluated on SimplerEnv WidowX carrot-on-plate. Loaded in-process via transformers custom-code (trust_remote_code), NF4 for 8 GB hosts. MIT license.

## Capabilities

- **Verbs:** pick · place
- **Objects:** object
- **Scenes:** tabletop
- **Embodiments:** widowx

## Why this is discovery-only

An agent skill is natural-language instructions loaded into an LLM's context. An rSkill
is an executable artifact: it carries a typed capability/embodiment contract, model weights,
a runtime, and a license/provenance gate — none of which fit in freeform markdown. So an
agent can use this view to *select* the right skill, but cannot *execute* it by loading
this file. Execution always goes through the OpenRAL loader and the robot HAL.

## License

- **Code:** Apache-2.0.
- **Weights:** `mit` — permissive / commercial-use OK

## How to actually run it (not via an agent harness)

```python
from openral_rskill import rSkill

skill = rSkill.from_pretrained("OpenRAL/rskill-openvla-oft-simpler-widowx-nf4")
# the loader validates embodiment / sensors / runtime / quantization against the target
# RobotDescription and enforces the weight-license gate before any weights load.
```

See [`rskill.yaml`](./rskill.yaml) for the authoritative, validated manifest.
