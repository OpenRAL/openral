---
name: act-so101-pen
description: >-
  S1 Vision-Language-Action policy. Capabilities: pick, place, pick_and_place, transfer on pen. ACT (Zhao et al., 2023) finetuned for "pass the pen" pick-and-place on a real SO-101 follower arm (Apache-2.0). ResNet-18 backbone, 4+1 enc/dec, latent VAE, chunk_size=100. Emits 6-DoF absolute joint-position chunks (in joint degrees) from two RGB views (front/overview + wrist). A smaller, faster, ONNX/TensorRT-friendly sibling of rskill-smolvla-so101-pen-bf16; the whole model exports to a single ONNX graph (OPENRAL_ACT_TRT=1). Discovery view of an OpenRAL rSkill — NOT directly runnable by an agent harness; it runs via rSkill.from_pretrained + the robot HAL.
metadata:
  openral_rskill: true            # generated discovery view of an rSkill
  schema_version: 0.1
  rskill_id: OpenRAL/rskill-act-so101-pen-fp32
  manifest: ./rskill.yaml
  role: s1
  kind: vla
  model_family: act
  embodiment_tags: [so101_follower]
  actions: [pick, place, pick_and_place, transfer]
  objects: [pen]
  scenes: [tabletop]
  sensors_required: ['rgb:observation.images.camera1', 'rgb:observation.images.camera2']
  state_dim: 6
  action_dim: 6
  action_representation: joint_positions
  runtime: pytorch
  quantization: fp32/pytorch
  min_vram_gb: {fp32: 0.5}
  chunk_size: 100
  latency_budget: {per_chunk_ms: 100.0}
  license_code: Apache-2.0
  license_weights: apache-2.0
  weights_uri: hf://gabrycina/so101-passing-pen-policy
  source_repo: hf://gabrycina/so101-passing-pen-policy
  paper_url: https://arxiv.org/abs/2304.13705
---

# act-so101-pen — rSkill discovery view

> **Generated view, not a hand-written skill.** This `SKILL.md` is a discovery-only
> mirror of [`rskill.yaml`](./rskill.yaml), produced by `tools/generate_rskill_skillmd.py`.
> It lets tools that read the standard agent-skill format find and reason about this
> OpenRAL rSkill. The `rskill.yaml` manifest is the single source of truth
> (CLAUDE.md §1.3). Do not edit by hand — edit the manifest and regenerate.

## What it is

An OpenRAL **Vision-Language-Action policy** (`role: s1`, `kind: vla`). ACT (Zhao et al., 2023) finetuned for "pass the pen" pick-and-place on a real SO-101 follower arm (Apache-2.0). ResNet-18 backbone, 4+1 enc/dec, latent VAE, chunk_size=100. Emits 6-DoF absolute joint-position chunks (in joint degrees) from two RGB views (front/overview + wrist). A smaller, faster, ONNX/TensorRT-friendly sibling of rskill-smolvla-so101-pen-bf16; the whole model exports to a single ONNX graph (OPENRAL_ACT_TRT=1).

## Capabilities

- **Verbs:** pick · place · pick_and_place · transfer
- **Objects:** pen
- **Scenes:** tabletop
- **Embodiments:** so101_follower

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

skill = rSkill.from_pretrained("OpenRAL/rskill-act-so101-pen-fp32")
# the loader validates embodiment / sensors / runtime / quantization against the target
# RobotDescription and enforces the weight-license gate before any weights load.
```

See [`rskill.yaml`](./rskill.yaml) for the authoritative, validated manifest.
