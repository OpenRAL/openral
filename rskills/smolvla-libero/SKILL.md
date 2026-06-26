---
name: smolvla-libero
description: >-
  S1 Vision-Language-Action policy. Capabilities: pick, place, open, close on bowl, cup, drawer, object. SmolVLA finetuned on the LIBERO task suite (Apache-2.0). Action chunks of length 16 across two RGB camera views (wrist + overhead) matching the original LIBERO dataset convention. The lerobot checkpoint wrapped here matches the paper's reported configuration on all five architecture fields — see header comment for the rejected sibling. Discovery view of an OpenRAL rSkill — NOT directly runnable by an agent harness; it runs via rSkill.from_pretrained + the robot HAL.
metadata:
  openral_rskill: true            # generated discovery view of an rSkill
  schema_version: 0.1
  rskill_id: OpenRAL/rskill-smolvla-libero
  manifest: ./rskill.yaml
  role: s1
  kind: vla
  model_family: smolvla
  embodiment_tags: [franka_panda]
  actions: [pick, place, open, close]
  objects: [bowl, cup, drawer, object]
  scenes: [tabletop, kitchen]
  sensors_required: ['rgb:observation.images.camera1', 'rgb:observation.images.camera2']
  state_dim: 8
  action_dim: 7
  action_representation: delta_ee_6d_plus_gripper
  runtime: pytorch
  quantization: bf16/pytorch
  chunk_size: 16
  n_action_steps: 25
  latency_budget: {per_chunk_ms: 150.0, max_execution_s: 60.0}
  license_code: Apache-2.0
  license_weights: apache-2.0
  weights_uri: hf://lerobot/smolvla_libero
  source_repo: hf://lerobot/smolvla_libero
  paper_url: https://arxiv.org/abs/2506.01844
---

# smolvla-libero — rSkill discovery view

> **Generated view, not a hand-written skill.** This `SKILL.md` is a discovery-only
> mirror of [`rskill.yaml`](./rskill.yaml), produced by `tools/generate_rskill_skillmd.py`.
> It lets tools that read the standard agent-skill format find and reason about this
> OpenRAL rSkill. The `rskill.yaml` manifest is the single source of truth
> (CLAUDE.md §1.3). Do not edit by hand — edit the manifest and regenerate.

## What it is

An OpenRAL **Vision-Language-Action policy** (`role: s1`, `kind: vla`). SmolVLA finetuned on the LIBERO task suite (Apache-2.0). Action chunks of length 16 across two RGB camera views (wrist + overhead) matching the original LIBERO dataset convention. The lerobot checkpoint wrapped here matches the paper's reported configuration on all five architecture fields — see header comment for the rejected sibling.

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

skill = rSkill.from_pretrained("OpenRAL/rskill-smolvla-libero")
# the loader validates embodiment / sensors / runtime / quantization against the target
# RobotDescription and enforces the weight-license gate before any weights load.
```

See [`rskill.yaml`](./rskill.yaml) for the authoritative, validated manifest.
