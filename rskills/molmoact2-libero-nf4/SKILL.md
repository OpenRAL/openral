---
name: molmoact2-libero-nf4
description: >-
  S1 Vision-Language-Action policy. Capabilities: pick, place, open, close on bowl, cup, drawer, object. MolmoAct2 (Ai2) finetuned on the full LIBERO training mixture, NF4-quantized via tools/quantize_rskill.py and re-hosted at OpenRAL/rskill-molmoact2-libero-nf4 so 8 GB GPUs can run the rollout without OOM. MolmoAct2 grafts a flow-matching continuous-action expert onto the Molmo2-ER embodied-reasoning VLM via per-layer KV-cache conditioning. Reported LIBERO success: 97.2% (98.1% for the -Think depth-reasoning variant). Weights are Apache-2.0 — commercial use permitted. See eval/libero.json. Discovery view of an OpenRAL rSkill — NOT directly runnable by an agent harness; it runs via rSkill.from_pretrained + the robot HAL.
metadata:
  openral_rskill: true            # generated discovery view of an rSkill
  schema_version: 0.1
  rskill_id: OpenRAL/rskill-molmoact2-libero-nf4
  manifest: ./rskill.yaml
  role: s1
  kind: vla
  model_family: molmoact2
  embodiment_tags: [franka_panda]
  actions: [pick, place, open, close]
  objects: [bowl, cup, drawer, object]
  scenes: [tabletop, kitchen]
  sensors_required: ['rgb:observation.images.camera1', 'rgb:observation.images.camera2']
  state_dim: 8
  action_dim: 7
  action_representation: delta_ee_6d_plus_gripper
  runtime: pytorch
  quantization: int4/pytorch
  min_vram_gb: {fp32: 22.0, bf16: 11.0, int4: 4.0}
  chunk_size: 10
  n_action_steps: 10
  latency_budget: {per_chunk_ms: 1000.0}
  license_code: Apache-2.0
  license_weights: apache-2.0
  weights_uri: hf://OpenRAL/rskill-molmoact2-libero-nf4
  source_repo: hf://allenai/MolmoAct2-LIBERO
  paper_url: https://arxiv.org/abs/2605.02881
---

# molmoact2-libero-nf4 — rSkill discovery view

> **Generated view, not a hand-written skill.** This `SKILL.md` is a discovery-only
> mirror of [`rskill.yaml`](./rskill.yaml), produced by `tools/generate_rskill_skillmd.py`.
> It lets tools that read the standard agent-skill format find and reason about this
> OpenRAL rSkill. The `rskill.yaml` manifest is the single source of truth
> (CLAUDE.md §1.3). Do not edit by hand — edit the manifest and regenerate.

## What it is

An OpenRAL **Vision-Language-Action policy** (`role: s1`, `kind: vla`). MolmoAct2 (Ai2) finetuned on the full LIBERO training mixture, NF4-quantized via tools/quantize_rskill.py and re-hosted at OpenRAL/rskill-molmoact2-libero-nf4 so 8 GB GPUs can run the rollout without OOM. MolmoAct2 grafts a flow-matching continuous-action expert onto the Molmo2-ER embodied-reasoning VLM via per-layer KV-cache conditioning. Reported LIBERO success: 97.2% (98.1% for the -Think depth-reasoning variant). Weights are Apache-2.0 — commercial use permitted. See eval/libero.json.

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

skill = rSkill.from_pretrained("OpenRAL/rskill-molmoact2-libero-nf4")
# the loader validates embodiment / sensors / runtime / quantization against the target
# RobotDescription and enforces the weight-license gate before any weights load.
```

See [`rskill.yaml`](./rskill.yaml) for the authoritative, validated manifest.
