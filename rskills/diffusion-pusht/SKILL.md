---
name: diffusion-pusht
description: >-
  S1 Vision-Language-Action policy. Capabilities: push on t_shape. Diffusion Policy (~263M-param U-Net with 100-step DDPM denoiser) for the PushT 2-DoF pushing benchmark. Action chunks of length 8 within a horizon of 16. The chunk inference cost is dominated by the denoising loop, so cached pops are essentially free — this is the extreme test of the queue-drain contract. Discovery view of an OpenRAL rSkill — NOT directly runnable by an agent harness; it runs via rSkill.from_pretrained + the robot HAL.
metadata:
  openral_rskill: true            # generated discovery view of an rSkill
  schema_version: 0.1
  rskill_id: OpenRAL/rskill-diffusion-pusht
  manifest: ./rskill.yaml
  role: s1
  kind: vla
  model_family: diffusion
  embodiment_tags: [pusht]
  actions: [push]
  objects: [t_shape]
  scenes: [tabletop_2d]
  sensors_required: ['rgb:observation.image']
  state_dim: 2
  action_dim: 2
  action_representation: joint_positions
  runtime: pytorch
  quantization: fp32/pytorch
  chunk_size: 8
  latency_budget: {per_chunk_ms: 1250.0}
  license_code: Apache-2.0
  license_weights: apache-2.0
  weights_uri: hf://lerobot/diffusion_pusht
  source_repo: hf://lerobot/diffusion_pusht
  paper_url: https://arxiv.org/abs/2303.04137
---

# diffusion-pusht — rSkill discovery view

> **Generated view, not a hand-written skill.** This `SKILL.md` is a discovery-only
> mirror of [`rskill.yaml`](./rskill.yaml), produced by `tools/generate_rskill_skillmd.py`.
> It lets tools that read the standard agent-skill format find and reason about this
> OpenRAL rSkill. The `rskill.yaml` manifest is the single source of truth
> (CLAUDE.md §1.3). Do not edit by hand — edit the manifest and regenerate.

## What it is

An OpenRAL **Vision-Language-Action policy** (`role: s1`, `kind: vla`). Diffusion Policy (~263M-param U-Net with 100-step DDPM denoiser) for the PushT 2-DoF pushing benchmark. Action chunks of length 8 within a horizon of 16. The chunk inference cost is dominated by the denoising loop, so cached pops are essentially free — this is the extreme test of the queue-drain contract.

## Capabilities

- **Verbs:** push
- **Objects:** t_shape
- **Scenes:** tabletop_2d
- **Embodiments:** pusht

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

skill = rSkill.from_pretrained("OpenRAL/rskill-diffusion-pusht")
# the loader validates embodiment / sensors / runtime / quantization against the target
# RobotDescription and enforces the weight-license gate before any weights load.
```

See [`rskill.yaml`](./rskill.yaml) for the authoritative, validated manifest.
