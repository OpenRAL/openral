# Architecture Overview

> **See also:** [`repo-state-map.html`](repo-state-map.html) — a one-page interactive canvas of every module across the eight layers, color-coded by build status (working / in-development / planned / out-of-scope), with click-through inputs, outputs, and schemas. The fastest way to build a mental model of what exists today vs. what is still spec-only.

OpenRAL uses an eight-layer architecture. Each layer has a single responsibility and communicates with adjacent layers through typed contracts. Status pills below match the [repo state map](repo-state-map.html): **✓ shipped** = source + tests on disk, **🟡 partial** = some pieces shipped, **⏳ planned** = not yet on disk.

```
0  HAL                  python/hal/, packages/openral_hal_*/        ✓ shipped (10 robots: SO-100/101, Franka, UR5e/10e, ALOHA, OpenArm, Rizon4, H1, G1, panda_mobile)
1  Sensors              python/sensors/                             ✓ shipped (catalog + vendor adapters)
2  World State          python/world_state/, packages/world_state/  ✓ shipped (aggregator + lifecycle node)
3  rSkill (S1)           python/rskill/, packages/openral_skill/     ✓ shipped (Python ABC + rSkill loader + openral_rskill_ros action server)
4  Reasoning (S2)       python/reasoner/, packages/openral_reasoner_ros/  ✓ shipped (ReasonerCore + reasoner/prompt-router nodes + typed ReasonerToolCall dispatch; full replanning ladder in progress)
5  World Action Model   python/wam/                                 ⏳ planned (optional layer)
6  Safety               packages/openral_safety/, cpp/openral_safety_kernel/  🟡 partial (Python supervisor + deadman/E-stop forwarders ship; certifiable C++ kernel planned)
7  Observability        python/observability/                       ✓ shipped (OTel SDK + OTLP exporter + structlog↔OTel bridge)
```

Layers 0–3 and Observability (Layer 7) ship today: the Python HAL/sensors
adapters, the World State aggregator (plus its ROS 2 lifecycle node), the
`Skill` ABC, the `rSkill` loader (manifest validation, license guard,
capability matching) with policy adapters for SmolVLA, ACT, Diffusion
Policy, π0.5, and xVLA, the `openral_rskill_ros` action server, and the
OpenTelemetry instrumentation. Reasoning (Layer 4) and Safety (Layer 6)
have **initial ROS 2 implementations** — an LLM reasoner/supervisor graph
(`openral_reasoner_ros` + `openral_prompt_router`) and a Python safety
supervisor with deadman/E-stop forwarders (`openral_safety`) — with the
certifiable **C++ safety kernel** (Layer 6) and the **WAM** (Layer 5) still
planned; the prose below describes their target shape. The cross-cutting
eval layer (`python/sim/`, ADR-0002) is shipped and drives the closed-loop
sim today via `openral sim run` against the configs under `scenes/`.

## Layer contracts

Every layer interaction is mediated by Pydantic schemas in `python/core/` (module `openral_core`).
The normative ROS 2 IDL lives in `packages/msgs/` (ROS package `openral_msgs`).

Crossing a layer boundary without an ADR is rejected in review.

## Dual-system pattern

Every robot agent has:
- **S1** — fast policy (VLA, 30–200 Hz), action-chunked.
- **S2** — slow reasoning (event-driven, ~0.2 Hz heartbeat). The reasoner emits
  typed `ReasonerToolCall` structured tool-calls (`ExecuteSkill`,
  `LifecycleTransition`, `EmitPrompt`, …) as its sole planner output — the
  direct typed-dispatch surface.
- **S0** (humanoids only) — cerebellar layer (500–1000 Hz, C++, inside ros2_control).

## Safety architecture

The safety kernel is a **separate C++ process** with a watchdog. It is deny-by-default.
Python proposes actions; C++ disposes them. `ROSSafetyViolation` is never silently caught.

See [CLAUDE.md](https://github.com/OpenRAL/openral/blob/master/CLAUDE.md) (§1 operating principles, §3 architecture discipline) for the full architectural discipline and the eight operating principles.
