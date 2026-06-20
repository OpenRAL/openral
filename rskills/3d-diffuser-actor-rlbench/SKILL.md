---
name: 3d-diffuser-actor-rlbench
description: >-
  S1 Vision-Language-Action policy. Capabilities: generalist, open, close, pick, place. 3D Diffuser Actor (Ke et al., 2024) — a diffusion policy over end-effector keyposes fusing multi-view RGB-D into a 3D scene representation, on the RLBench PerAct 18-task benchmark. Shares the out-of-process CoppeliaSim/PyRep sidecar with the rlbench scene backend (ADR-0061). MIT code + checkpoints. The PerAct checkpoint is loaded verbatim; ships three live-verified starter tasks. Discovery view of an OpenRAL rSkill — NOT directly runnable by an agent harness; it runs via rSkill.from_pretrained + the robot HAL.
metadata:
  openral_rskill: true            # generated discovery view of an rSkill
  schema_version: 0.1
  rskill_id: OpenRAL/rskill-3d-diffuser-actor-rlbench
  manifest: ./rskill.yaml
  role: s1
  kind: vla
  model_family: diffuser_actor
  embodiment_tags: [franka_panda]
  actions: [generalist, open, close, pick, place]
  scenes: [tabletop]
  sensors_required: [rgb]
  action_dim: 8
  runtime: pytorch
  min_vram_gb: {bf16: 2.0, fp32: 2.0}
  chunk_size: 1
  latency_budget: {per_chunk_ms: 3000.0}
  license_code: Apache-2.0
  license_weights: mit
  weights_uri: hf://katefgroup/3d_diffuser_actor
  source_repo: hf://katefgroup/3d_diffuser_actor
  paper_url: https://arxiv.org/abs/2402.10885
---

# 3d-diffuser-actor-rlbench — rSkill discovery view

> **Generated view, not a hand-written skill.** This `SKILL.md` is a discovery-only
> mirror of [`rskill.yaml`](./rskill.yaml), produced by `tools/generate_rskill_skillmd.py`.
> It lets tools that read the standard agent-skill format find and reason about this
> OpenRAL rSkill. The `rskill.yaml` manifest is the single source of truth
> (CLAUDE.md §1.3). Do not edit by hand — edit the manifest and regenerate.

## What it is

An OpenRAL **Vision-Language-Action policy** (`role: s1`, `kind: vla`). 3D Diffuser Actor (Ke et al., 2024) — a diffusion policy over end-effector keyposes fusing multi-view RGB-D into a 3D scene representation, on the RLBench PerAct 18-task benchmark. Shares the out-of-process CoppeliaSim/PyRep sidecar with the rlbench scene backend (ADR-0061). MIT code + checkpoints. The PerAct checkpoint is loaded verbatim; ships three live-verified starter tasks.

## Capabilities

- **Verbs:** generalist · open · close · pick · place
- **Scenes:** tabletop
- **Embodiments:** franka_panda

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

skill = rSkill.from_pretrained("OpenRAL/rskill-3d-diffuser-actor-rlbench")
# the loader validates embodiment / sensors / runtime / quantization against the target
# RobotDescription and enforces the weight-license gate before any weights load.
```

See [`rskill.yaml`](./rskill.yaml) for the authoritative, validated manifest.
