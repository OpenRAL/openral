# Architecture Overview

OpenRAL has seven layers. Each has one responsibility and talks only to adjacent layers through typed contracts. The [repo state map](repo-state-map.html) is the interactive per-module canvas (status colors, inputs, outputs, schemas); the status marks below match it: **✓ shipped** = source + tests on disk, **🟡 partial** = some pieces shipped, **⏳ planned** = not yet on disk.

```
0  HAL                  python/hal/, packages/openral_hal_node/     ✓ shipped (manifest-driven; one RobotDescription per robot under robots/)
1  Sensors              python/sensors/                             ✓ shipped (catalog + vendor adapters)
2  World State          python/world_state/, packages/world_state/  ✓ shipped (aggregator + lifecycle node)
3  rSkill (S1)           python/rskill/, packages/openral_rskill_ros/ ✓ shipped (Python ABC + rSkill loader + openral_rskill_ros action server)
4  Reasoning (S2)       python/reasoner/, packages/openral_reasoner_ros/  ✓ shipped (ReasonerCore + reasoner/prompt-router nodes + typed ReasonerToolCall dispatch; full replanning ladder in progress)
5  Safety               packages/openral_safety/, cpp/openral_safety_kernel/  🟡 partial (C++ deny-by-default kernel + deadman watchdog + human-E-stop forwarder, launched by the deploy graph; no hardware-pendant driver ships; certification pending)
6  Observability        python/observability/                       ✓ shipped (OTel SDK + OTLP exporter + structlog↔OTel bridge)
```

Layers 0–3 and Observability (Layer 6) ship today. The registered policy
adapters (`openral_sim.POLICIES`) cover SmolVLA, ACT, Diffusion Policy, π0.5,
xVLA, MolmoAct2, RLDX-1, GR00T (N1.7 and the BEHAVIOR-1K checkpoint), OpenVLA,
Xiaomi XR-1, LingBot-VLA 1/2, LingBot-VA, InternVLA-N1 and 3D Diffuser Actor.
Reasoning (Layer 4) and Safety (Layer 5) run in the deploy graph; the C++
safety kernel is implemented and tested, certification pending.

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
