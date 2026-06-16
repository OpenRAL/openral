---
name: pi05-libero-nf4
description: >-
  S1 Vision-Language-Action policy. Capabilities: pick, place, open, close on bowl, cup, drawer, object. π0.5 fine-tuned on LIBERO (v0.44), NF4-quantized via tools/quantize_rskill.py and re-hosted at OpenRAL/rskill-pi05-libero-nf4 so 8 GB GPUs can run the rollout without OOM. π0.5 uses a PaliGemma 3B backbone with a flow-matching head. Weights are PI permissive-research — commercial use needs a vendor agreement. See eval/libero.json (pending). Discovery view of an OpenRAL rSkill — NOT directly runnable by an agent harness; it runs via rSkill.from_pretrained + the robot HAL.
metadata:
  openral_rskill: true            # generated discovery view of an rSkill
  schema_version: 0.1
  rskill_id: OpenRAL/rskill-pi05-libero-nf4
  manifest: ./rskill.yaml
  role: s1
  kind: vla
  model_family: pi05
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
  min_vram_gb: {fp32: 14.0, bf16: 7.0, int4: 4.0}
  chunk_size: 50
  n_action_steps: 25
  latency_budget: {per_chunk_ms: 200.0}
  license_code: Apache-2.0
  license_weights: permissive_research   # NOT permissive — see License section
  weights_uri: hf://OpenRAL/rskill-pi05-libero-nf4
  source_repo: hf://lerobot/pi05_libero_finetuned_v044
  paper_url: https://arxiv.org/abs/2410.24164
---

# pi05-libero-nf4 — rSkill discovery view

> **Generated view, not a hand-written skill.** This `SKILL.md` is a discovery-only
> mirror of [`rskill.yaml`](./rskill.yaml), produced by `tools/generate_rskill_skillmd.py`.
> It lets tools that read the standard agent-skill format find and reason about this
> OpenRAL rSkill. The `rskill.yaml` manifest is the single source of truth
> (CLAUDE.md §1.3). Do not edit by hand — edit the manifest and regenerate.

## What it is

An OpenRAL **Vision-Language-Action policy** (`role: s1`, `kind: vla`). π0.5 fine-tuned on LIBERO (v0.44), NF4-quantized via tools/quantize_rskill.py and re-hosted at OpenRAL/rskill-pi05-libero-nf4 so 8 GB GPUs can run the rollout without OOM. π0.5 uses a PaliGemma 3B backbone with a flow-matching head. Weights are PI permissive-research — commercial use needs a vendor agreement. See eval/libero.json (pending).

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
- **Weights:** `permissive_research` — **NOT** fully permissive. The loader surfaces this posture and enforces the non-commercial guard (`OPENRAL_ALLOW_NONCOMMERCIAL=1`) where applicable. Commercial use may require a separate upstream agreement. This is third-party weight lineage; OpenRAL's own code is Apache-2.0.

## How to actually run it (not via an agent harness)

```python
from openral_rskill import rSkill

skill = rSkill.from_pretrained("OpenRAL/rskill-pi05-libero-nf4")
# the loader validates embodiment / sensors / runtime / quantization against the target
# RobotDescription and enforces the weight-license gate before any weights load.
```

See [`rskill.yaml`](./rskill.yaml) for the authoritative, validated manifest.
