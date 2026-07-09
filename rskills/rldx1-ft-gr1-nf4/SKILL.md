---
name: rldx1-ft-gr1-nf4
description: >-
  S1 Vision-Language-Action policy. Capabilities: generalist, pick, place. RLDX-1 fine-tuned on the Fourier GR-1 humanoid tabletop tasks (RoboCasa GR1 fork, 24-task suite). 29-D action space (per-arm joint deltas + waist + Fourier dexhand grasps); 39-D proprio (joint_pos + per-hand gripper_qpos). Shares the out-of-process rldx sidecar runtime with the LIBERO finetune. Non-commercial license. Discovery view of an OpenRAL rSkill — NOT directly runnable by an agent harness; it runs via rSkill.from_pretrained + the robot HAL.
metadata:
  openral_rskill: true            # generated discovery view of an rSkill
  schema_version: 0.1
  rskill_id: OpenRAL/rskill-rldx1-ft-gr1-nf4
  manifest: ./rskill.yaml
  role: s1
  kind: vla
  model_family: rldx
  embodiment_tags: [gr1]
  actions: [generalist, pick, place]
  scenes: [tabletop]
  sensors_required: ['rgb:observation.images.camera1']
  state_dim: 29
  action_dim: 29
  action_representation: joint_positions
  runtime: pytorch
  quantization: int4/pytorch
  min_vram_gb: {bf16: 18.0, int4: 7.0}
  chunk_size: 16
  latency_budget: {per_chunk_ms: 1500.0}
  license_code: Apache-2.0
  license_weights: rlwrld_non_commercial   # NOT permissive — see License section
  weights_uri: hf://RLWRLD/RLDX-1-FT-GR1
  source_repo: hf://RLWRLD/RLDX-1-FT-GR1
  paper_url: https://huggingface.co/RLWRLD/RLDX-1-FT-GR1
---

# rldx1-ft-gr1-nf4 — rSkill discovery view

> **Generated view, not a hand-written skill.** This `SKILL.md` is a discovery-only
> mirror of [`rskill.yaml`](./rskill.yaml), produced by `tools/generate_rskill_skillmd.py`.
> It lets tools that read the standard agent-skill format find and reason about this
> OpenRAL rSkill. The `rskill.yaml` manifest is the single source of truth
> (CLAUDE.md §1.3). Do not edit by hand — edit the manifest and regenerate.

## What it is

An OpenRAL **Vision-Language-Action policy** (`role: s1`, `kind: vla`). RLDX-1 fine-tuned on the Fourier GR-1 humanoid tabletop tasks (RoboCasa GR1 fork, 24-task suite). 29-D action space (per-arm joint deltas + waist + Fourier dexhand grasps); 39-D proprio (joint_pos + per-hand gripper_qpos). Shares the out-of-process rldx sidecar runtime with the LIBERO finetune. Non-commercial license.

## Capabilities

- **Verbs:** generalist · pick · place
- **Scenes:** tabletop
- **Embodiments:** gr1

## Why this is discovery-only

An agent skill is natural-language instructions loaded into an LLM's context. An rSkill
is an executable artifact: it carries a typed capability/embodiment contract, model weights,
a runtime, and a license/provenance gate — none of which fit in freeform markdown. So an
agent can use this view to *select* the right skill, but cannot *execute* it by loading
this file. Execution always goes through the OpenRAL loader and the robot HAL.

## License

- **Code:** Apache-2.0.
- **Weights:** `rlwrld_non_commercial` — **NOT** fully permissive. The loader surfaces this posture and enforces the non-commercial guard (`OPENRAL_ALLOW_NONCOMMERCIAL=1`) where applicable. Commercial use may require a separate upstream agreement. This is third-party weight lineage; OpenRAL's own code is Apache-2.0.

## How to actually run it (not via an agent harness)

```python
from openral_rskill import rSkill

skill = rSkill.from_pretrained("OpenRAL/rskill-rldx1-ft-gr1-nf4")
# the loader validates embodiment / sensors / runtime / quantization against the target
# RobotDescription and enforces the weight-license gate before any weights load.
```

See [`rskill.yaml`](./rskill.yaml) for the authoritative, validated manifest.
