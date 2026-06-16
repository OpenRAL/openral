---
name: robometer-4b
description: >-
  S2 task-progress / reward monitor. Capabilities: monitor on task progress, task success. Robometer-4B (Qwen3-VL-4B robotic reward foundation model, arXiv 2603.02115) as an NF4 reward rSkill. Runs parallel to a VLA: given rollout frames + the task instruction it emits per-frame normalized progress (0-1) and success probability, queried on demand by the Reasoner. Advisory-only — never gates motors. Embodiment-agnostic. Apache-2.0. ADR-0057. Discovery view of an OpenRAL rSkill — NOT directly runnable by an agent harness; it runs via rSkill.from_pretrained + the robot HAL.
metadata:
  openral_rskill: true            # generated discovery view of an rSkill
  schema_version: 0.1
  rskill_id: OpenRAL/rskill-robometer-4b-nf4
  manifest: ./rskill.yaml
  role: s2
  kind: reward
  actions: [monitor]
  objects: [task progress, task success]
  scenes: [tabletop, kitchen, indoor, manipulation]
  sensors_required: [rgb]
  runtime: pytorch
  quantization: int4/pytorch
  min_vram_gb: {fp32: 18.0, bf16: 9.0, int4: 3.6}
  chunk_size: 1
  latency_budget: {per_chunk_ms: 3000.0}
  license_code: Apache-2.0
  license_weights: apache-2.0
  weights_uri: hf://OpenRAL/rskill-robometer-4b-nf4
  source_repo: hf://robometer/Robometer-4B@beef63bc914c5c189329d49c6d712d96d632aa34
---

# robometer-4b — rSkill discovery view

> **Generated view, not a hand-written skill.** This `SKILL.md` is a discovery-only
> mirror of [`rskill.yaml`](./rskill.yaml), produced by `tools/generate_rskill_skillmd.py`.
> It lets tools that read the standard agent-skill format find and reason about this
> OpenRAL rSkill. The `rskill.yaml` manifest is the single source of truth
> (CLAUDE.md §1.3). Do not edit by hand — edit the manifest and regenerate.

## What it is

An OpenRAL **task-progress / reward monitor** (`role: s2`, `kind: reward`). Robometer-4B (Qwen3-VL-4B robotic reward foundation model, arXiv 2603.02115) as an NF4 reward rSkill. Runs parallel to a VLA: given rollout frames + the task instruction it emits per-frame normalized progress (0-1) and success probability, queried on demand by the Reasoner. Advisory-only — never gates motors. Embodiment-agnostic. Apache-2.0. ADR-0057.

## Capabilities

- **Verbs:** monitor
- **Objects:** task progress · task success
- **Scenes:** tabletop · kitchen · indoor · manipulation

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

skill = rSkill.from_pretrained("OpenRAL/rskill-robometer-4b-nf4")
# the loader validates embodiment / sensors / runtime / quantization against the target
# RobotDescription and enforces the weight-license gate before any weights load.
```

See [`rskill.yaml`](./rskill.yaml) for the authoritative, validated manifest.
