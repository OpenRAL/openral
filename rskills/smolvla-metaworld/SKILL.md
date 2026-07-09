---
name: smolvla-metaworld
description: >-
  S1 Vision-Language-Action policy. Capabilities: generalist, reach, push, pick, place, open. SmolVLA (0.45 B) finetuned on MetaWorld MT50 — 50 manipulation tasks on a Rethink Sawyer arm (MuJoCo via lerobot). Runs the MT10/MT50 suites (benchmarks/metaworld_mt{10,50}.yaml) and 5 demo scenes (scenes/benchmark/metaworld_*.yaml). Locally reproduced on MT50: 16/50 solved at 1 ep/seed-0 (avg 0.30); see eval/metaworld_mt50.json. 4-D proprio / camera1 contract verified against the checkpoint (no adapter flip — lerobot's MetaworldEnv already corrects the corner camera's 180° inversion). Discovery view of an OpenRAL rSkill — NOT directly runnable by an agent harness; it runs via rSkill.from_pretrained + the robot HAL.
metadata:
  openral_rskill: true            # generated discovery view of an rSkill
  schema_version: 0.1
  rskill_id: OpenRAL/rskill-smolvla-metaworld
  manifest: ./rskill.yaml
  role: s1
  kind: vla
  model_family: smolvla
  embodiment_tags: [sawyer]
  actions: [generalist, reach, push, pick, place, open, close, insert, slide]
  scenes: [tabletop]
  sensors_required: ['rgb:observation.images.camera1']
  state_dim: 4
  action_dim: 4
  action_representation: delta_ee_3d_plus_gripper
  runtime: pytorch
  quantization: bf16/pytorch
  chunk_size: 16
  latency_budget: {per_chunk_ms: 150.0}
  license_code: Apache-2.0
  license_weights: apache-2.0
  weights_uri: hf://lerobot/smolvla_metaworld
  source_repo: hf://lerobot/smolvla_metaworld
  paper_url: https://arxiv.org/abs/2506.01844
---

# smolvla-metaworld — rSkill discovery view

> **Generated view, not a hand-written skill.** This `SKILL.md` is a discovery-only
> mirror of [`rskill.yaml`](./rskill.yaml), produced by `tools/generate_rskill_skillmd.py`.
> It lets tools that read the standard agent-skill format find and reason about this
> OpenRAL rSkill. The `rskill.yaml` manifest is the single source of truth
> (CLAUDE.md §1.3). Do not edit by hand — edit the manifest and regenerate.

## What it is

An OpenRAL **Vision-Language-Action policy** (`role: s1`, `kind: vla`). SmolVLA (0.45 B) finetuned on MetaWorld MT50 — 50 manipulation tasks on a Rethink Sawyer arm (MuJoCo via lerobot). Runs the MT10/MT50 suites (benchmarks/metaworld_mt{10,50}.yaml) and 5 demo scenes (scenes/benchmark/metaworld_*.yaml). Locally reproduced on MT50: 16/50 solved at 1 ep/seed-0 (avg 0.30); see eval/metaworld_mt50.json. 4-D proprio / camera1 contract verified against the checkpoint (no adapter flip — lerobot's MetaworldEnv already corrects the corner camera's 180° inversion).

## Capabilities

- **Verbs:** generalist · reach · push · pick · place · open · close · insert · slide
- **Scenes:** tabletop
- **Embodiments:** sawyer

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

skill = rSkill.from_pretrained("OpenRAL/rskill-smolvla-metaworld")
# the loader validates embodiment / sensors / runtime / quantization against the target
# RobotDescription and enforces the weight-license gate before any weights load.
```

See [`rskill.yaml`](./rskill.yaml) for the authoritative, validated manifest.
