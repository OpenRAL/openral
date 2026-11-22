---
name: nat-vla-so101-pretrained-bf16
description: >-
  S1 Vision-Language-Action policy. Capabilities: generalist, pick, place. NaT VLA (450M, LiquidAI LFM2-VL-450M writing NAT action tokens) multi-robot pretrained checkpoint, run zero-shot on the SO-101 follower: 6-DoF absolute joint chunks (32, 16 replayed) in LeRobot servo degrees from a fixed overview and a wrist camera. Not fine-tuned on an OpenRAL task; LFM Open License v1.0 weights (commercial use only under $10M revenue). Discovery view of an OpenRAL rSkill — NOT directly runnable by an agent harness; it runs via rSkill.from_pretrained + the robot HAL.
metadata:
  openral_rskill: true            # generated discovery view of an rSkill
  schema_version: 0.1
  rskill_id: OpenRAL/rskill-nat_vla-so101-pretrained-bf16
  manifest: ./rskill.yaml
  role: s1
  kind: vla
  model_family: nat_vla
  embodiment_tags: [so101_follower]
  actions: [generalist, pick, place]
  scenes: [tabletop]
  sensors_required: ['rgb:observation.images.camera1', 'rgb:observation.images.camera2']
  state_dim: 6
  action_dim: 6
  action_representation: joint_positions
  runtime: pytorch
  quantization: bf16/pytorch
  min_vram_gb: {bf16: 1.5}
  chunk_size: 32
  n_action_steps: 16
  latency_budget: {per_chunk_ms: 500.0}
  license_code: Apache-2.0
  license_weights: lfm1.0   # NOT permissive — see License section
  weights_uri: hf://ahad-j/nat-vla-450m@a5c2e13d45832050e364265962f6ad536460e955
  source_repo: hf://ahad-j/nat-vla-450m
  paper_url: https://ahadjawaid.com/nat-paper
---

# nat-vla-so101-pretrained-bf16 — rSkill discovery view

> **Generated view, not a hand-written skill.** This `SKILL.md` is a discovery-only
> mirror of [`rskill.yaml`](./rskill.yaml), produced by `tools/generate_rskill_skillmd.py`.
> It lets tools that read the standard agent-skill format find and reason about this
> OpenRAL rSkill. The `rskill.yaml` manifest is the single source of truth
> (CLAUDE.md §1.3). Do not edit by hand — edit the manifest and regenerate.

## What it is

An OpenRAL **Vision-Language-Action policy** (`role: s1`, `kind: vla`). NaT VLA (450M, LiquidAI LFM2-VL-450M writing NAT action tokens) multi-robot pretrained checkpoint, run zero-shot on the SO-101 follower: 6-DoF absolute joint chunks (32, 16 replayed) in LeRobot servo degrees from a fixed overview and a wrist camera. Not fine-tuned on an OpenRAL task; LFM Open License v1.0 weights (commercial use only under $10M revenue).

## Capabilities

- **Verbs:** generalist · pick · place
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
- **Weights:** `lfm1.0` — **NOT** fully permissive. The loader surfaces this posture and enforces the non-commercial guard (`OPENRAL_ALLOW_NONCOMMERCIAL=1`) where applicable. Commercial use may require a separate upstream agreement. This is third-party weight lineage; OpenRAL's own code is Apache-2.0.

## How to actually run it (not via an agent harness)

```python
from openral_rskill import rSkill

skill = rSkill.from_pretrained("OpenRAL/rskill-nat_vla-so101-pretrained-bf16")
# the loader validates embodiment / sensors / runtime / quantization against the target
# RobotDescription and enforces the weight-license gate before any weights load.
```

See [`rskill.yaml`](./rskill.yaml) for the authoritative, validated manifest.
