# Architecture Overview

OpenRAL has seven layers. Each has one responsibility and talks only to adjacent layers through typed contracts. The [repo state map](repo-state-map.html) is the interactive per-module canvas (status colors, inputs, outputs, schemas); the status marks below match it: **✓ shipped** = source + tests on disk, **🟡 partial** = some pieces shipped, **⏳ planned** = not yet on disk.

```
0  HAL                  python/hal/, packages/openral_hal_node/     ✓ shipped (12 robots: SO-100/101, Franka, UR5e/10e, ALOHA, OpenArm, Anvil-OpenArm-v2, Galaxea A1, Rizon4, H1, G1, panda_mobile)
1  Sensors              python/sensors/                             ✓ shipped (catalog + vendor adapters)
2  World State          python/world_state/, packages/world_state/  ✓ shipped (aggregator + lifecycle node)
3  rSkill (S1)           python/rskill/, packages/openral_rskill_ros/ ✓ shipped (Python ABC + rSkill loader + openral_rskill_ros action server)
4  Reasoning (S2)       python/reasoner/, packages/openral_reasoner_ros/  ✓ shipped (ReasonerCore + reasoner/prompt-router nodes + typed ReasonerToolCall dispatch; full replanning ladder in progress)
5  Safety               packages/openral_safety/, cpp/openral_safety_kernel/  🟡 partial (Python supervisor + an independent deadman watchdog and human-E-stop forwarder, launched by the deploy graph; hardware-pendant driver and certifiable C++ kernel planned)
6  Observability        python/observability/                       ✓ shipped (OTel SDK + OTLP exporter + structlog↔OTel bridge)
```

Layers 0–3 and Observability (Layer 6) ship today; the `rSkill` loader's
policy adapters cover SmolVLA, ACT, Diffusion Policy, π0.5, and xVLA.
Reasoning (Layer 4) and Safety (Layer 5) have initial ROS 2
implementations; the certifiable C++ safety kernel (Layer 5) is still
planned. The prose below describes their target shape.

The cross-cutting eval layer (`python/sim/`) ships and drives closed-loop
sim via `openral sim run` against the configs under `scenes/`.

## Layer contracts

Pydantic schemas in `python/core/` (module `openral_core`) mediate every layer interaction; the normative ROS 2 IDL lives in `packages/msgs/` (ROS package `openral_msgs`). Crossing a layer boundary without an ADR is rejected in review.

## Dual-system pattern

Every robot agent has:
- **S1** — fast policy (VLA, 30–200 Hz), action-chunked.
- **S2** — slow reasoning (event-driven, ~0.2 Hz heartbeat). The reasoner emits
  typed `ReasonerToolCall` structured tool-calls (`ExecuteRskillTool`,
  `LifecycleTransitionTool`, `EmitPromptTool`, …) as its sole planner output — the
  direct typed-dispatch surface.
- **S0** (humanoids only) — cerebellar layer (500–1000 Hz, C++, inside ros2_control).

## Safety architecture

The safety kernel is a separate, deny-by-default C++ process with a watchdog. Python proposes actions; C++ disposes them. `ROSSafetyViolation` is never silently caught.

Full discipline: [CLAUDE.md](https://github.com/OpenRAL/openral/blob/master/CLAUDE.md) §1 (operating principles) and §3 (architecture discipline).
