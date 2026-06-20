---
name: act-aloha-insertion
description: >-
  S1 Vision-Language-Action policy. Capabilities: insert, pick, place on peg, socket. ACT (~52M params, chunk=100) finetuned on the ALOHA bimanual sim-insertion demonstration set. Insertion is the harder ALOHA task in the original paper; the 0.20 success rate reflects that. See the norm-stats note above re: the legacy safetensors-resident buffers. Discovery view of an OpenRAL rSkill — NOT directly runnable by an agent harness; it runs via rSkill.from_pretrained + the robot HAL.
metadata:
  openral_rskill: true            # generated discovery view of an rSkill
  schema_version: 0.1
  rskill_id: OpenRAL/rskill-act-aloha-insertion
  manifest: ./rskill.yaml
  role: s1
  kind: vla
  model_family: act
  embodiment_tags: [aloha]
  actions: [insert, pick, place]
  objects: [peg, socket]
  scenes: [tabletop]
  sensors_required: ['rgb:observation.images.top']
  state_dim: 14
  action_dim: 14
  runtime: pytorch
  quantization: fp32/pytorch
  chunk_size: 100
  latency_budget: {per_chunk_ms: 25.0}
  license_code: Apache-2.0
  license_weights: mit
  weights_uri: hf://lerobot/act_aloha_sim_insertion_human
  source_repo: hf://lerobot/act_aloha_sim_insertion_human
  paper_url: https://arxiv.org/abs/2304.13705
---

# act-aloha-insertion — rSkill discovery view

> **Generated view, not a hand-written skill.** This `SKILL.md` is a discovery-only
> mirror of [`rskill.yaml`](./rskill.yaml), produced by `tools/generate_rskill_skillmd.py`.
> It lets tools that read the standard agent-skill format find and reason about this
> OpenRAL rSkill. The `rskill.yaml` manifest is the single source of truth
> (CLAUDE.md §1.3). Do not edit by hand — edit the manifest and regenerate.

## What it is

An OpenRAL **Vision-Language-Action policy** (`role: s1`, `kind: vla`). ACT (~52M params, chunk=100) finetuned on the ALOHA bimanual sim-insertion demonstration set. Insertion is the harder ALOHA task in the original paper; the 0.20 success rate reflects that. See the norm-stats note above re: the legacy safetensors-resident buffers.

## Capabilities

- **Verbs:** insert · pick · place
- **Objects:** peg · socket
- **Scenes:** tabletop
- **Embodiments:** aloha

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

skill = rSkill.from_pretrained("OpenRAL/rskill-act-aloha-insertion")
# the loader validates embodiment / sensors / runtime / quantization against the target
# RobotDescription and enforces the weight-license gate before any weights load.
```

See [`rskill.yaml`](./rskill.yaml) for the authoritative, validated manifest.
