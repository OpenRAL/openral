# Roadmap

Live status of OpenRAL development. For detailed architecture and module-by-module canvas, see [`../architecture/repo-state-map.html`](../architecture/repo-state-map.html).

---

## Snapshot

| Area | Status |
|---|---|
| Schemas (`openral_core`) | ✅ shipped — Pydantic v2, hypothesis fuzz, JSON Schema export, CI drift check |
| HAL Protocol + ROS bridge | ✅ shipped — `RosControlHAL`, mock unit tests, exception hierarchy |
| LeRobot SO-100/SO-101 HAL | ✅ shipped — real + MuJoCo `SO100FollowerHAL`, `SO100_DESCRIPTION`, `openral connect` |
| Franka / UR5e / UR10e HALs (sim) | ✅ shipped — over `MujocoArmHAL`; ROS package skeletons; hardware bring-up deferred to M3 |
| Humanoid + bimanual HALs (sim) | ✅ shipped — `G1MujocoHAL`, `H1MujocoHAL`, `AlohaMujocoHAL`, `OpenArmMujocoHAL`, `Rizon4MujocoHAL`, `PandaMobileHAL` with lifecycle nodes (ADR-0029) |
| Sensor adapters | 🟡 in flight — `openral_sensors` covers RealSense / Orbbec / Hokuyo / Livox / Ouster / SLAMTec / IMU / force-torque / tactile / USB UVC |
| World State aggregator | ✅ shipped — 30 Hz tf2-aware snapshot, stale-sensor diagnostics, detected-objects lift |
| Skill base + runtimes | ✅ shipped — lifecycle node, `PyTorchRuntime`, `ONNXRuntime`, quantization registry, engine cache |
| rSkill manifest + loader | ✅ shipped — HF Hub packaging, `rSkill.from_pretrained`, license surface, sigstore verify-only stub |
| SmolVLA, π0.5, xVLA, ACT, DP, MolmoAct2, RLDX-1 adapters | ✅ shipped — loaded, tested, embodiment-tag gated |
| End-to-end sim demo (SO-100 + SmolVLA) | ✅ shipped — smoketest + GIF; full LIBERO rollout verified |
| Configurable sim/eval harness | ✅ shipped — `openral sim run`, scene/task/policy registries (ADR-0002) |
| OpenTelemetry instrumentation | ✅ shipped — OTel SDK + OTLP exporter, `skill_span` / `inference_span` / `safety_span`, `openral dashboard` live OTLP receiver (ADR-0017) |
| rosbag2 ↔ LeRobotDataset v3 bridge | ✅ shipped — `RolloutRecorder`, `LeRobotDatasetSink`, `openral dataset {push,from-bag}` (ADR-0019) |
| Reasoner (S2 LLM supervisor) | 🟡 in flight — `openral_reasoner` core + `openral_reasoner_ros` node; provider-selected LLM; bounded replanning ladder (ADR-0018) |
| Deploy ROS graph | ✅ shipped — `openral deploy {run,sim}` with HAL + safety + reasoner + world_state; dynamic skill dispatch (ADR-0031/0032/0034) |
| Object detection + spatial lift | ✅ shipped — `RosImageObjectDetectorNode`, 2D→3D object lift, RT-DETR detector rSkills, GStreamer perception bus; LocateAnything packaged as a private NF4 detector artifact pending adapter (ADR-0035/0037) |
| Geometric safety + watchdog | 🟡 in flight — self/world/voxel collision checking (ADR-0030), deadman + E-stop forwarders |
| BehaviorTree v4 executor | 🔵 planned — future option behind `bt_executor_node` (ADR-0018 §4) |
| C++ safety kernel | 🔵 planned — deny-by-default allocation-free validator (ADR-0020) |
| Org / publishing | 🔴 outstanding — branch protection, PyPI trusted publishing, GHCR org packages |

Legend: ✅ done, 🟡 in flight, 🔵 planned, 🔴 blocked / outstanding.

---

## Next phases

- **M2** — Unitree G1 HAL + cerebellar (S0) C++ controller examples.
- **M3** — UR5e and Franka FR3 HALs on real hardware; sensor bring-up (D435 HIL smoke test).
- **v0.3** — WAM adapters (Cosmos Predict, UnifoLM-WMA-0, IRASim) behind the `WorldModel` Protocol.
- **v1.0** — Failure-anticipation as first-class; safety kernel hardening; certifiable build.
- **HAL consolidation** — unify per-robot nodes into one `robot.yaml`-driven node (ADR-0029, ~10–12 d).

See [repo state map](../architecture/repo-state-map.html) for detailed per-module status and cross-layer dependencies.
