---
name: smolvla-vlabench
description: >-
  S1 Vision-Language-Action policy. Capabilities: pick, place, grasp on fruit, object. SmolVLA (~0.5 B) finetuned on VLABench (lerobot/vlabench_unified), lerobot-native, runs in-process on lerobot 0.6.0 (bf16, fits 8 GB). VALIDATED INTEGRATION BASELINE, not a passing policy: 0/3 on six diverse primitives (identical to lerobot's own lerobot-eval reference — the 0% is the policy, not the wiring). The only >50% VLABench policy (pi0-fast-ft 51.2% primitive avg) is openpi/JAX needing conversion. See README + project memory. Use this to exercise the VLABench backend. Discovery view of an OpenRAL rSkill — NOT directly runnable by an agent harness; it runs via rSkill.from_pretrained + the robot HAL.
metadata:
  openral_rskill: true            # generated discovery view of an rSkill
  schema_version: 0.1
  rskill_id: OpenRAL/rskill-smolvla-vlabench
  manifest: ./rskill.yaml
  role: s1
  kind: vla
  model_family: smolvla
  embodiment_tags: [franka_panda]
  actions: [pick, place, grasp]
  objects: [fruit, object]
  scenes: [tabletop]
  sensors_required: ['rgb:observation.images.camera1', 'rgb:observation.images.camera2', 'rgb:observation.images.camera3']
  state_dim: 7
  action_dim: 7
  action_representation: delta_ee_6d_plus_gripper
  runtime: pytorch
  quantization: bf16/pytorch
  min_vram_gb: {fp32: 4.0, bf16: 2.0}
  chunk_size: 50
  n_action_steps: 25
  latency_budget: {per_chunk_ms: 100.0}
  license_code: Apache-2.0
  license_weights: apache-2.0
  weights_uri: hf://lerobot/smolvla_vlabench
  source_repo: hf://lerobot/smolvla_vlabench
  paper_url: https://arxiv.org/abs/2412.18194
---

# smolvla-vlabench — rSkill discovery view

> **Generated view, not a hand-written skill.** This `SKILL.md` is a discovery-only
> mirror of [`rskill.yaml`](./rskill.yaml), produced by `tools/generate_rskill_skillmd.py`.
> It lets tools that read the standard agent-skill format find and reason about this
> OpenRAL rSkill. The `rskill.yaml` manifest is the single source of truth
> (CLAUDE.md §1.3). Do not edit by hand — edit the manifest and regenerate.

## What it is

An OpenRAL **Vision-Language-Action policy** (`role: s1`, `kind: vla`). SmolVLA (~0.5 B) finetuned on VLABench (lerobot/vlabench_unified), lerobot-native, runs in-process on lerobot 0.6.0 (bf16, fits 8 GB). VALIDATED INTEGRATION BASELINE, not a passing policy: 0/3 on six diverse primitives (identical to lerobot's own lerobot-eval reference — the 0% is the policy, not the wiring). The only >50% VLABench policy (pi0-fast-ft 51.2% primitive avg) is openpi/JAX needing conversion. See README + project memory. Use this to exercise the VLABench backend.

## Capabilities

- **Verbs:** pick · place · grasp
- **Objects:** fruit · object
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
- **Weights:** `apache-2.0` — permissive / commercial-use OK

## How to actually run it (not via an agent harness)

```python
from openral_rskill import rSkill

skill = rSkill.from_pretrained("OpenRAL/rskill-smolvla-vlabench")
# the loader validates embodiment / sensors / runtime / quantization against the target
# RobotDescription and enforces the weight-license gate before any weights load.
```

See [`rskill.yaml`](./rskill.yaml) for the authoritative, validated manifest.
