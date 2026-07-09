---
name: act-libero
description: >-
  S1 Vision-Language-Action policy. Capabilities: pick, place, open, close on bowl, cup, drawer, object. ACT (Zhao et al., 2023) on HuggingFaceVLA/libero. ResNet-18, 4+1 enc/dec, latent VAE, chunk_size=100. Two 256x256 RGB (image / image2) + 8-D state + 7-D action; plain chunked replay. State layout matches openral_sim's LIBERO backend. Discovery view of an OpenRAL rSkill — NOT directly runnable by an agent harness; it runs via rSkill.from_pretrained + the robot HAL.
metadata:
  openral_rskill: true            # generated discovery view of an rSkill
  schema_version: 0.1
  rskill_id: OpenRAL/rskill-act-libero
  manifest: ./rskill.yaml
  role: s1
  kind: vla
  model_family: act
  embodiment_tags: [franka_panda]
  actions: [pick, place, open, close]
  objects: [bowl, cup, drawer, object]
  scenes: [tabletop, kitchen]
  sensors_required: ['rgb:observation.images.camera1', 'rgb:observation.images.camera2']
  state_dim: 8
  action_dim: 7
  action_representation: delta_ee_6d_plus_gripper
  runtime: pytorch
  quantization: fp32/pytorch
  chunk_size: 100
  latency_budget: {per_chunk_ms: 100.0}
  license_code: Apache-2.0
  license_weights: apache-2.0
  weights_uri: hf://Deepkar/libero-test-act
  source_repo: hf://Deepkar/libero-test-act
  paper_url: https://arxiv.org/abs/2304.13705
---

# act-libero — rSkill discovery view

> **Generated view, not a hand-written skill.** This `SKILL.md` is a discovery-only
> mirror of [`rskill.yaml`](./rskill.yaml), produced by `tools/generate_rskill_skillmd.py`.
> It lets tools that read the standard agent-skill format find and reason about this
> OpenRAL rSkill. The `rskill.yaml` manifest is the single source of truth
> (CLAUDE.md §1.3). Do not edit by hand — edit the manifest and regenerate.

## What it is

An OpenRAL **Vision-Language-Action policy** (`role: s1`, `kind: vla`). ACT (Zhao et al., 2023) on HuggingFaceVLA/libero. ResNet-18, 4+1 enc/dec, latent VAE, chunk_size=100. Two 256x256 RGB (image / image2) + 8-D state + 7-D action; plain chunked replay. State layout matches openral_sim's LIBERO backend.

## Capabilities

- **Verbs:** pick · place · open · close
- **Objects:** bowl · cup · drawer · object
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
- **Weights:** `apache-2.0` — permissive / commercial-use OK

## How to actually run it (not via an agent harness)

```python
from openral_rskill import rSkill

skill = rSkill.from_pretrained("OpenRAL/rskill-act-libero")
# the loader validates embodiment / sensors / runtime / quantization against the target
# RobotDescription and enforces the weight-license gate before any weights load.
```

See [`rskill.yaml`](./rskill.yaml) for the authoritative, validated manifest.
