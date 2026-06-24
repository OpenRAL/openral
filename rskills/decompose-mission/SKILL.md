---
name: decompose-mission
description: >-
  S2 decision-procedure playbook (weightless). Capabilities: plan on compound task. S2 decision procedure: break a compound, multi-step instruction into an ordered list of subtasks, each with its own verifiable done-condition (an internal TODO list), executing and verifying them in order. Composes execute_rskill, query_scene, query_task_progress, memory_write and memory_search. ADR-0071. Discovery view of an OpenRAL rSkill — NOT directly runnable by an agent harness; it runs via rSkill.from_pretrained + the robot HAL.
metadata:
  openral_rskill: true            # generated discovery view of an rSkill
  schema_version: 0.1
  rskill_id: OpenRAL/rskill-decompose-mission
  manifest: ./rskill.yaml
  role: s2
  kind: playbook
  embodiment_tags: [any]
  actions: [plan]
  objects: [compound task]
  scenes: [kitchen, indoor, household]
  chunk_size: 1
  latency_budget: {per_chunk_ms: 5000.0}
  license_code: Apache-2.0
  license_weights: apache-2.0
  paper_url: https://github.com/OpenRAL/openral/blob/master/docs/adr/0071-reasoner-playbooks-and-self-maintained-memory.md
---

# decompose-mission — rSkill discovery view

> **Generated view, not a hand-written skill.** This `SKILL.md` is a discovery-only
> mirror of [`rskill.yaml`](./rskill.yaml), produced by `tools/generate_rskill_skillmd.py`.
> It lets tools that read the standard agent-skill format find and reason about this
> OpenRAL rSkill. The `rskill.yaml` manifest is the single source of truth
> (CLAUDE.md §1.3). Do not edit by hand — edit the manifest and regenerate.

## What it is

An OpenRAL **decision-procedure playbook (weightless)** (`role: s2`, `kind: playbook`). S2 decision procedure: break a compound, multi-step instruction into an ordered list of subtasks, each with its own verifiable done-condition (an internal TODO list), executing and verifying them in order. Composes execute_rskill, query_scene, query_task_progress, memory_write and memory_search. ADR-0071.

## Capabilities

- **Verbs:** plan
- **Objects:** compound task
- **Scenes:** kitchen · indoor · household
- **Embodiments:** any

## Why this is discovery-only

An agent skill is natural-language instructions loaded into an LLM's context. An rSkill
is an executable artifact: it carries a typed capability/embodiment contract
a runtime, and a license/provenance gate — none of which fit in freeform markdown. So an
agent can use this view to *select* the right skill, but cannot *execute* it by loading
this file. Execution always goes through the OpenRAL loader and the robot HAL.

## License

- **Code:** Apache-2.0. This is a weightless rSkill (the manifest *is* the artifact).

## How to actually run it (not via an agent harness)

```python
from openral_rskill import rSkill

skill = rSkill.from_pretrained("OpenRAL/rskill-decompose-mission")
# the loader validates embodiment / sensors / runtime / quantization against the target
# RobotDescription and enforces the weight-license gate before any weights load.
```

See [`rskill.yaml`](./rskill.yaml) for the authoritative, validated manifest.
