---
name: rskill-moveit-joints
description: >-
  S1 ROS action skill (weightless). Capabilities: reach. Plan and execute a collision-free motion to a target JOINT configuration via MoveIt's MoveGroup (self + planning-scene collision checked). Provide one target angle per planning-group joint. Defaults target the Franka Panda home pose; override the manifest to retarget. Use when you have a joint-space goal (e.g. a policy's in-distribution starting pose); for a Cartesian end-effector target use rskill-moveit-eef-pose instead. Discovery view of an OpenRAL rSkill — NOT directly runnable by an agent harness; it runs via rSkill.from_pretrained + the robot HAL.
metadata:
  openral_rskill: true            # generated discovery view of an rSkill
  schema_version: 0.1
  rskill_id: OpenRAL/rskill-moveit-joints
  manifest: ./rskill.yaml
  role: s1
  kind: ros_action
  embodiment_tags: [franka_panda, ur5e, ur10e, so100_follower, openarm, rizon4, sawyer, widowx]
  actions: [reach]
  chunk_size: 1
  latency_budget: {per_chunk_ms: 2000.0}
  license_code: Apache-2.0
  license_weights: apache-2.0
  paper_url: https://moveit.picknik.ai/
---

# rskill-moveit-joints — rSkill discovery view

> **Generated view, not a hand-written skill.** This `SKILL.md` is a discovery-only
> mirror of [`rskill.yaml`](./rskill.yaml), produced by `tools/generate_rskill_skillmd.py`.
> It lets tools that read the standard agent-skill format find and reason about this
> OpenRAL rSkill. The `rskill.yaml` manifest is the single source of truth
> (CLAUDE.md §1.3). Do not edit by hand — edit the manifest and regenerate.

## What it is

An OpenRAL **ROS action skill (weightless)** (`role: s1`, `kind: ros_action`). Plan and execute a collision-free motion to a target JOINT configuration via MoveIt's MoveGroup (self + planning-scene collision checked). Provide one target angle per planning-group joint. Defaults target the Franka Panda home pose; override the manifest to retarget. Use when you have a joint-space goal (e.g. a policy's in-distribution starting pose); for a Cartesian end-effector target use rskill-moveit-eef-pose instead.

## Capabilities

- **Verbs:** reach
- **Embodiments:** franka_panda · ur5e · ur10e · so100_follower · openarm · rizon4 · sawyer · widowx

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

skill = rSkill.from_pretrained("OpenRAL/rskill-moveit-joints")
# the loader validates embodiment / sensors / runtime / quantization against the target
# RobotDescription and enforces the weight-license gate before any weights load.
```

See [`rskill.yaml`](./rskill.yaml) for the authoritative, validated manifest.
