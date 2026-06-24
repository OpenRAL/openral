---
name: preflight-reach
description: >-
  S2 decision-procedure playbook (weightless). Capabilities: plan on manipulation target. S2 decision procedure: before dispatching a manipulation skill, check the target is within the robot's reachable workspace (using the robot self-model + object pose) and stage the base or hand off if it is not. Composes recall_object, resolve_place, query_scene and NAVIGATE skills. ADR-0071. Discovery view of an OpenRAL rSkill — NOT directly runnable by an agent harness; it runs via rSkill.from_pretrained + the robot HAL.
metadata:
  openral_rskill: true            # generated discovery view of an rSkill
  schema_version: 0.1
  rskill_id: OpenRAL/rskill-preflight-reach
  manifest: ./rskill.yaml
  role: s2
  kind: playbook
  embodiment_tags: [any]
  actions: [plan]
  objects: [manipulation target]
  scenes: [kitchen, indoor]
  chunk_size: 1
  latency_budget: {per_chunk_ms: 5000.0}
  license_code: Apache-2.0
  license_weights: apache-2.0
  paper_url: https://github.com/OpenRAL/openral/blob/master/docs/adr/0071-reasoner-playbooks-and-self-maintained-memory.md
---

# preflight-reach — rSkill discovery view

> **Generated view, not a hand-written skill.** This `SKILL.md` is a discovery-only
> mirror of [`rskill.yaml`](./rskill.yaml), produced by `tools/generate_rskill_skillmd.py`.
> It lets tools that read the standard agent-skill format find and reason about this
> OpenRAL rSkill. The `rskill.yaml` manifest is the single source of truth
> (CLAUDE.md §1.3). Do not edit by hand — edit the manifest and regenerate.

## What it is

An OpenRAL **decision-procedure playbook (weightless)** (`role: s2`, `kind: playbook`). S2 decision procedure: before dispatching a manipulation skill, check the target is within the robot's reachable workspace (using the robot self-model + object pose) and stage the base or hand off if it is not. Composes recall_object, resolve_place, query_scene and NAVIGATE skills. ADR-0071.

## Capabilities

- **Verbs:** plan
- **Objects:** manipulation target
- **Scenes:** kitchen · indoor
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

skill = rSkill.from_pretrained("OpenRAL/rskill-preflight-reach")
# the loader validates embodiment / sensors / runtime / quantization against the target
# RobotDescription and enforces the weight-license gate before any weights load.
```

See [`rskill.yaml`](./rskill.yaml) for the authoritative, validated manifest.
