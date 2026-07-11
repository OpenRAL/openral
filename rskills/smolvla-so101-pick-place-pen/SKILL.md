---
name: smolvla-so101-pick-place-pen
description: >-
  S1 Vision-Language-Action policy. Capabilities: pick, place, pick_and_place on pen. SmolVLA finetuned for pen pick-and-place on a real SO-101 follower arm (Apache-2.0). Emits 6-DoF absolute joint-position chunks (size 50, in joint degrees) from two RGB views (fixed overview + arm-mounted). The checkpoint declares three camera slots inherited from smolvla_base but trains and runs on two (empty_cameras=0 drops the third); the TRT export mirrors that with a 2-camera engine, verified bit-exact against native torch inference. Discovery view of an OpenRAL rSkill — NOT directly runnable by an agent harness; it runs via rSkill.from_pretrained + the robot HAL.
metadata:
  openral_rskill: true            # generated discovery view of an rSkill
  schema_version: 0.1
  rskill_id: OpenRAL/rskill-smolvla-so101-pick_place_pen-bf16
  manifest: ./rskill.yaml
  role: s1
  kind: vla
  model_family: smolvla
  embodiment_tags: [so101_follower]
  actions: [pick, place, pick_and_place]
  objects: [pen]
  scenes: [tabletop]
  sensors_required: ['rgb:observation.images.camera1', 'rgb:observation.images.camera2']
  state_dim: 6
  action_dim: 6
  action_representation: joint_positions
  runtime: pytorch
  quantization: bf16/pytorch
  min_vram_gb: {bf16: 1.5}
  chunk_size: 50
  n_action_steps: 50
  latency_budget: {per_chunk_ms: 400.0}
  license_code: Apache-2.0
  license_weights: apache-2.0
  weights_uri: hf://OpenRAL/rskill-smolvla-so101-pick_place_pen-bf16
  source_repo: hf://nota-gmbh/so101_pick_place_pen_smolvla
  paper_url: https://arxiv.org/abs/2506.01844
---

# smolvla-so101-pick-place-pen — rSkill discovery view

> **Generated view, not a hand-written skill.** This `SKILL.md` is a discovery-only
> mirror of [`rskill.yaml`](./rskill.yaml), produced by `tools/generate_rskill_skillmd.py`.
> It lets tools that read the standard agent-skill format find and reason about this
> OpenRAL rSkill. The `rskill.yaml` manifest is the single source of truth
> (CLAUDE.md §1.3). Do not edit by hand — edit the manifest and regenerate.

## What it is

An OpenRAL **Vision-Language-Action policy** (`role: s1`, `kind: vla`). SmolVLA finetuned for pen pick-and-place on a real SO-101 follower arm (Apache-2.0). Emits 6-DoF absolute joint-position chunks (size 50, in joint degrees) from two RGB views (fixed overview + arm-mounted). The checkpoint declares three camera slots inherited from smolvla_base but trains and runs on two (empty_cameras=0 drops the third); the TRT export mirrors that with a 2-camera engine, verified bit-exact against native torch inference.

## Capabilities

- **Verbs:** pick · place · pick_and_place
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

skill = rSkill.from_pretrained("OpenRAL/rskill-smolvla-so101-pick_place_pen-bf16")
# the loader validates embodiment / sensors / runtime / quantization against the target
# RobotDescription and enforces the weight-license gate before any weights load.
```

See [`rskill.yaml`](./rskill.yaml) for the authoritative, validated manifest.
