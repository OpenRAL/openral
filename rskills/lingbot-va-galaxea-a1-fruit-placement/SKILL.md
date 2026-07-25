---
name: lingbot-va-galaxea-a1-fruit-placement
description: >-
  S1 Vision-Language-Action policy. Capabilities: pick, place on fruit, mango, bowl, plate. LingBot-VA fruit-placement policy for the Galaxea A1. It consumes the synchronized front and wrist RGB views, predicts episode-relative EEF pose and continuous gripper chunks, and uses the tracked A1 Runtime IK contract before OpenRAL validates and executes the resulting joint/gripper actions. Discovery view of an OpenRAL rSkill — NOT directly runnable by an agent harness; it runs via rSkill.from_pretrained + the robot HAL.
metadata:
  openral_rskill: true            # generated discovery view of an rSkill
  schema_version: 0.1
  rskill_id: OpenRAL/rskill-lingbot_va_a1-galaxea_a1-fruit_placement-bf16
  manifest: ./rskill.yaml
  role: s1
  kind: vla
  model_family: lingbot_va_a1
  embodiment_tags: [galaxea_a1]
  actions: [pick, place]
  objects: [fruit, mango, bowl, plate]
  scenes: [tabletop]
  sensors_required: ['rgb:observation.images.front', 'rgb:observation.images.wrist']
  state_dim: 6
  action_dim: 7
  runtime: pytorch
  quantization: bf16/pytorch
  chunk_size: 16
  n_action_steps: 8
  latency_budget: {per_chunk_ms: 6000.0, max_execution_s: 420.0}
  license_code: Apache-2.0
  license_weights: apache-2.0
  weights_uri: hf://pengyue-polaron/lingbot-va-galaxea-a1-fruit-placement-eef@90e017bdbc6afac2e441b4634c9192776bbcb8b7
  source_repo: hf://robbyant/lingbot-va-base
---

# lingbot-va-galaxea-a1-fruit-placement — rSkill discovery view

> **Generated view, not a hand-written skill.** This `SKILL.md` is a discovery-only
> mirror of [`rskill.yaml`](./rskill.yaml), produced by `tools/generate_rskill_skillmd.py`.
> It lets tools that read the standard agent-skill format find and reason about this
> OpenRAL rSkill. The `rskill.yaml` manifest is the single source of truth
> (CLAUDE.md §1.3). Do not edit by hand — edit the manifest and regenerate.

## What it is

An OpenRAL **Vision-Language-Action policy** (`role: s1`, `kind: vla`). LingBot-VA fruit-placement policy for the Galaxea A1. It consumes the synchronized front and wrist RGB views, predicts episode-relative EEF pose and continuous gripper chunks, and uses the tracked A1 Runtime IK contract before OpenRAL validates and executes the resulting joint/gripper actions.

## Capabilities

- **Verbs:** pick · place
- **Objects:** fruit · mango · bowl · plate
- **Scenes:** tabletop
- **Embodiments:** galaxea_a1

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

skill = rSkill.from_pretrained("OpenRAL/rskill-lingbot_va_a1-galaxea_a1-fruit_placement-bf16")
# the loader validates embodiment / sensors / runtime / quantization against the target
# RobotDescription and enforces the weight-license gate before any weights load.
```

See [`rskill.yaml`](./rskill.yaml) for the authoritative, validated manifest.
