---
name: topreward-qwen3vl-4b-nf4
description: >-
  S2 task-progress / reward monitor. Capabilities: monitor on task progress, task success. TOPReward (arXiv 2602.19313) as an NF4 reward rSkill on lerobot 0.6.0. A ZERO-SHOT reward: asks an off-the-shelf Qwen3-VL-4B VLM how likely the task instruction is given the rollout video, reading log P("True") as the signal. Per-frame progress (0-1) comes from lerobot's prefix sweep. Runs parallel to a VLA, queried on demand by the Reasoner. Advisory-only — never gates motors. Embodiment-agnostic. Discovery view of an OpenRAL rSkill — NOT directly runnable by an agent harness; it runs via rSkill.from_pretrained + the robot HAL.
metadata:
  openral_rskill: true            # generated discovery view of an rSkill
  schema_version: 0.1
  rskill_id: OpenRAL/rskill-topreward-qwen3vl-4b-nf4
  manifest: ./rskill.yaml
  role: s2
  kind: reward
  embodiment_tags: [any]
  actions: [monitor]
  objects: [task progress, task success]
  scenes: [tabletop, kitchen, indoor, manipulation]
  sensors_required: [rgb]
  runtime: pytorch
  quantization: int4/pytorch
  min_vram_gb: {fp32: 18.0, bf16: 9.0, int4: 3.2}
  chunk_size: 1
  latency_budget: {per_chunk_ms: 3000.0}
  license_code: Apache-2.0
  license_weights: apache-2.0
  weights_uri: hf://OpenRAL/rskill-topreward-qwen3vl-4b-nf4
  source_repo: hf://Qwen/Qwen3-VL-4B-Instruct
  paper_url: https://arxiv.org/abs/2602.19313
---

# topreward-qwen3vl-4b-nf4 — rSkill discovery view

> **Generated view, not a hand-written skill.** This `SKILL.md` is a discovery-only
> mirror of [`rskill.yaml`](./rskill.yaml), produced by `tools/generate_rskill_skillmd.py`.
> It lets tools that read the standard agent-skill format find and reason about this
> OpenRAL rSkill. The `rskill.yaml` manifest is the single source of truth
> (CLAUDE.md §1.3). Do not edit by hand — edit the manifest and regenerate.

## What it is

An OpenRAL **task-progress / reward monitor** (`role: s2`, `kind: reward`). TOPReward (arXiv 2602.19313) as an NF4 reward rSkill on lerobot 0.6.0. A ZERO-SHOT reward: asks an off-the-shelf Qwen3-VL-4B VLM how likely the task instruction is given the rollout video, reading log P("True") as the signal. Per-frame progress (0-1) comes from lerobot's prefix sweep. Runs parallel to a VLA, queried on demand by the Reasoner. Advisory-only — never gates motors. Embodiment-agnostic.

## Capabilities

- **Verbs:** monitor
- **Objects:** task progress · task success
- **Scenes:** tabletop · kitchen · indoor · manipulation
- **Embodiments:** any

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

skill = rSkill.from_pretrained("OpenRAL/rskill-topreward-qwen3vl-4b-nf4")
# the loader validates embodiment / sensors / runtime / quantization against the target
# RobotDescription and enforces the weight-license gate before any weights load.
```

See [`rskill.yaml`](./rskill.yaml) for the authoritative, validated manifest.
