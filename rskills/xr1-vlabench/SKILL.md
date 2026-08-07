---
name: xr1-vlabench
description: >-
  S1 Vision-Language-Action policy. Capabilities: pick, place, grasp on fruit, object. Apache-2.0 XR-1 VLABench checkpoint. It consumes the benchmark's front, second, and wrist 480x480 views plus 7-D relative EE state. OpenRAL integrates the decoded deltas into absolute VLABench targets and replans after five of each ten predicted actions. Discovery view of an OpenRAL rSkill — NOT directly runnable by an agent harness; it runs via rSkill.from_pretrained + the robot HAL.
metadata:
  openral_rskill: true            # generated discovery view of an rSkill
  schema_version: 0.1
  rskill_id: OpenRAL/rskill-xr1-franka_panda-vlabench-nf4
  manifest: ./rskill.yaml
  role: s1
  kind: vla
  model_family: xr1
  embodiment_tags: [franka_panda]
  actions: [pick, place, grasp]
  objects: [fruit, object]
  scenes: [tabletop]
  sensors_required: ['rgb:observation.images.camera1', 'rgb:observation.images.camera2', 'rgb:observation.images.camera3']
  state_dim: 7
  action_dim: 7
  action_representation: cartesian_pose
  runtime: pytorch
  quantization: int4/pytorch
  min_vram_gb: {int4: 4.0}
  chunk_size: 10
  n_action_steps: 5
  latency_budget: {per_chunk_ms: 2000.0}
  license_code: Apache-2.0
  license_weights: apache-2.0
  weights_uri: hf://OpenRAL/rskill-xr1-franka_panda-vlabench-nf4
  source_repo: hf://XiaomiRobotics/Xiaomi-Robotics-1-VLABench@2dfc33b390478f71737eacb4748333e6d8638a06
  paper_url: https://arxiv.org/abs/2607.15330
---

# xr1-vlabench — rSkill discovery view

> **Generated view, not a hand-written skill.** This `SKILL.md` is a discovery-only
> mirror of [`rskill.yaml`](./rskill.yaml), produced by `tools/generate_rskill_skillmd.py`.
> It lets tools that read the standard agent-skill format find and reason about this
> OpenRAL rSkill. The `rskill.yaml` manifest is the single source of truth
> (CLAUDE.md §1.3). Do not edit by hand — edit the manifest and regenerate.

## What it is

An OpenRAL **Vision-Language-Action policy** (`role: s1`, `kind: vla`). Apache-2.0 XR-1 VLABench checkpoint. It consumes the benchmark's front, second, and wrist 480x480 views plus 7-D relative EE state. OpenRAL integrates the decoded deltas into absolute VLABench targets and replans after five of each ten predicted actions.

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

skill = rSkill.from_pretrained("OpenRAL/rskill-xr1-franka_panda-vlabench-nf4")
# the loader validates embodiment / sensors / runtime / quantization against the target
# RobotDescription and enforces the weight-license gate before any weights load.
```

See [`rskill.yaml`](./rskill.yaml) for the authoritative, validated manifest.
