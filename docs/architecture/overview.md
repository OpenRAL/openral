# Architecture Overview

OpenRAL has seven layers. Each has one responsibility and talks only to adjacent layers through typed contracts. The [repo state map](repo-state-map.html) is the interactive per-module canvas; the marks below match it: **✓ shipped** = source and tests on disk, **🟡 partial** = some pieces shipped.

| # | Layer | Code | Status |
|---|---|---|---|
| 0 | HAL | `python/hal/`, `packages/openral_hal_node/` | ✓ manifest-driven; one `RobotDescription` per robot under `robots/` |
| 1 | Sensors | `python/sensors/` | ✓ catalog and vendor adapters |
| 2 | World State | `python/world_state/`, `packages/world_state/` | ✓ aggregator and lifecycle node |
| 3 | rSkill (S1) | `python/rskill/`, `packages/openral_rskill_ros/` | ✓ rSkill loader, runtimes, `ExecuteRskill` action server |
| 4 | Reasoning (S2) | `python/reasoner/`, `packages/openral_reasoner_ros/` | ✓ `ReasonerCore`, typed tool dispatch, bounded replanning |
| 5 | Safety | `packages/openral_safety/`, `cpp/openral_safety_kernel/` | 🟡 C++ deny-by-default kernel, deadman watchdog and human E-stop run in the deploy graph; no hardware-pendant driver ships; certification pending |
| 6 | Observability | `python/observability/`, `packages/openral_compliance/` | ✓ OTel SDK and exporter, structlog bridge, robot dashboard, compliance recorder |

The cross-cutting eval layer (`python/sim/`) drives closed-loop simulation through `openral sim run` and `openral benchmark run`. Registered policy families (`openral_sim.POLICIES`): SmolVLA, ACT, Diffusion Policy, pi0.5, xVLA, MolmoAct2, RLDX-1, GR00T (N1.7 and the BEHAVIOR-1K checkpoint), OpenVLA, NaT VLA, Cosmos 3, Xiaomi XR-1, LingBot-VLA 1/2, LingBot-VA, InternVLA-N1 and 3D Diffuser Actor.

## Layer contracts

Pydantic schemas in `python/core/` (module `openral_core`) mediate every layer interaction. The normative ROS 2 IDL lives in `packages/msgs/` (ROS package `openral_msgs`). Moving a responsibility across a layer needs a recorded decision before code (see [Design decisions](../decisions.md)).

## Dual-system pattern

- **S1**: fast policy (a VLA, 30-200 Hz), action-chunked, run as an rSkill.
- **S2**: slow reasoning (event-driven, ~0.2 Hz heartbeat). The reasoner emits typed `ReasonerToolCall`s (`execute_rskill`, `lifecycle_transition`, `emit_prompt`, plus read-only queries) as its only planner output.
- **S0** (humanoids only): cerebellar layer (500-1000 Hz, C++, inside `ros2_control`).

An rSkill's `kind` decides which slot it runs in. The reasoner can only dispatch installed, capability-matched, licensed rSkills.

## Safety architecture

The safety kernel is a separate, deny-by-default C++ process with an independent deadman watchdog. Python proposes actions; C++ disposes them. `ROSSafetyViolation` is never silently caught. Real-robot ROS graphs are authenticated with SROS2: [ROS graph security](../reference/ros-graph-security.md). Engineering rules: [CLAUDE.md](https://github.com/OpenRAL/openral/blob/master/CLAUDE.md) sections 1 and 3.

## Multi-robot deploy sims

Several robots in one scene start with `openral deploy fleet`, part of OpenRAL Pro. `deploy sim` and `deploy run` start one robot: a scene that lists `robots:` refuses and names `deploy fleet`. Each robot's graph gets its own `ROS_DOMAIN_ID` and dashboard (see [Dashboard](../quickstart/dashboard.md)). Each robot's safety kernel also checks its moves against the other robots, published on `/openral/peer_voxels`, with no exemption (ADR-0137); a robot that shares a scene but has no world-voxel check refuses to launch. Scenes: [Sim environments](../reference/sim-environments.md).

## Open and Pro seams

Commercial runtimes plug into the open code through four entry-point groups: `openral.runtime_backends`, `openral.policy_attach_hooks`, `openral.detector_tiers` and `openral.pro_modules`. The safety kernel's relaxing exemptions use a separate provider interface. What is Pro: [Design decisions](../decisions.md#licensing-and-commercial-boundary).
