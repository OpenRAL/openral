---
name: rskill-nav2-navigate-to-pose
description: >-
  S1 ROS action skill (weightless). Capabilities: navigate. Navigate the mobile base to a target pose via Nav2's NavigateToPose. Supports BOTH absolute and relative goals via pose.header.frame_id: "map" = absolute world coordinates (drive to (3.5, 2.1)); "base_link" = relative to current pose, used for turns / forward / back-up (Nav2 transforms via tf2 on goal accept — the LLM does NOT need to compose quaternions against the live pose). Result-only mode: Nav2 publishes cmd_vel directly; collision avoidance relies on its costmap. Discovery view of an OpenRAL rSkill — NOT directly runnable by an agent harness; it runs via rSkill.from_pretrained + the robot HAL.
metadata:
  openral_rskill: true            # generated discovery view of an rSkill
  schema_version: 0.1
  rskill_id: OpenRAL/rskill-nav2-navigate-to-pose
  manifest: ./rskill.yaml
  role: s1
  kind: ros_action
  embodiment_tags: [mobile_base]
  actions: [navigate]
  scenes: [indoor]
  chunk_size: 1
  latency_budget: {per_chunk_ms: 60000.0}
  license_code: Apache-2.0
  license_weights: apache-2.0
  paper_url: https://docs.nav2.org/
---

# rskill-nav2-navigate-to-pose — rSkill discovery view

> **Generated view, not a hand-written skill.** This `SKILL.md` is a discovery-only
> mirror of [`rskill.yaml`](./rskill.yaml), produced by `tools/generate_rskill_skillmd.py`.
> It lets tools that read the standard agent-skill format find and reason about this
> OpenRAL rSkill. The `rskill.yaml` manifest is the single source of truth
> (CLAUDE.md §1.3). Do not edit by hand — edit the manifest and regenerate.

## What it is

An OpenRAL **ROS action skill (weightless)** (`role: s1`, `kind: ros_action`). Navigate the mobile base to a target pose via Nav2's NavigateToPose. Supports BOTH absolute and relative goals via pose.header.frame_id: "map" = absolute world coordinates (drive to (3.5, 2.1)); "base_link" = relative to current pose, used for turns / forward / back-up (Nav2 transforms via tf2 on goal accept — the LLM does NOT need to compose quaternions against the live pose). Result-only mode: Nav2 publishes cmd_vel directly; collision avoidance relies on its costmap.

## Capabilities

- **Verbs:** navigate
- **Scenes:** indoor
- **Embodiments:** mobile_base

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

skill = rSkill.from_pretrained("OpenRAL/rskill-nav2-navigate-to-pose")
# the loader validates embodiment / sensors / runtime / quantization against the target
# RobotDescription and enforces the weight-license gate before any weights load.
```

See [`rskill.yaml`](./rskill.yaml) for the authoritative, validated manifest.
