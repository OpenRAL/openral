# Roadmap

Live status of OpenRAL development. For detailed architecture and module-by-module canvas, see [`../architecture/repo-state-map.html`](../architecture/repo-state-map.html).

---

## Snapshot

| Area | Status |
|---|---|
| Schemas (`openral_core`) | ✅ shipped — Pydantic v2, hypothesis fuzz, JSON Schema export, CI drift check |
| HAL Protocol + ROS bridge | ✅ shipped — `RosControlHAL`, mock unit tests, exception hierarchy |
| LeRobot SO-100/SO-101 HAL | ✅ shipped — real + MuJoCo `SO100FollowerHAL`, `SO100_DESCRIPTION`, `openral connect` |
| Franka / UR5e / UR10e HALs | ✅ sim shipped over `MujocoArmHAL`; real-HW adapters landed (UR5e/UR10e via `ur_robot_driver`, Franka via FCI, Sawyer, ALOHA) — HIL gated on lab runners (M3) |
| Humanoid + bimanual HALs (sim) | ✅ shipped — `G1MujocoHAL`, `H1MujocoHAL`, `AlohaMujocoHAL`, `OpenArmMujocoHAL`, `Rizon4MujocoHAL`, `PandaMobileHAL` with lifecycle nodes (ADR-0029) |
| Sensor adapters | 🟡 in flight — `openral_sensors` catalog (RealSense D435/D435i/D415, Logitech UVC, Luxonis OAK-D Pro, Robotiq FT 300-S) + launch-gen + ROS image publisher; full perception-head ROS package still planned |
| World State aggregator | ✅ shipped — 30 Hz tf2-aware snapshot, stale-sensor diagnostics, detected-objects lift |
| Persistent spatial memory (scene graph) | 🟡 in flight — durable advisory object/place/room/agent graph the S2 reasoner queries (recall/resolve) + CLIP open-vocab match; sqlite-vec persistence + ROS feeder pending (ADR-0038/0039) |
| Skill base + runtimes | ✅ shipped — lifecycle node, `PyTorchRuntime`, `ONNXRuntime`, quantization registry, engine cache |
| rSkill manifest + loader | ✅ shipped — HF Hub packaging, `rSkill.from_pretrained`, license surface; sigstore provenance not yet implemented (unverified-provenance warning + `OPENRAL_REQUIRE_SIGNED_SKILLS` fail-closed gate) |
| SmolVLA, π0.5, xVLA, ACT, DP, MolmoAct2, RLDX-1 adapters | ✅ shipped — loaded, tested, embodiment-tag gated; GR00T N1.7 (ADR-0046) via out-of-process ZMQ sidecar (🟡 live eval operator-run) |
| End-to-end sim demo (SO-100 + SmolVLA) | ✅ shipped — smoketest + GIF; full LIBERO rollout verified |
| Configurable sim/eval harness | ✅ shipped — `openral sim run` + `benchmark run`, three-tier scene registry (ADR-0041); LIBERO / MetaWorld / gym-aloha / gym-pusht / ManiSkill3 / SimplerEnv / RoboCasa backends + Isaac Sim sidecar (ADR-0045) |
| OpenTelemetry instrumentation | ✅ shipped — OTel SDK + OTLP exporter, `skill_span` / `inference_span` / `safety_span`, `openral dashboard` live OTLP receiver (ADR-0017) |
| rosbag2 ↔ LeRobotDataset v3 bridge | ✅ shipped — `RolloutRecorder`, `LeRobotDatasetSink`, `openral dataset {push,from-bag}` (ADR-0019) |
| Reasoner (S2 LLM supervisor) | ✅ shipped — `openral_reasoner` core + `openral_reasoner_ros` node + `openral_prompt_router`; provider-selected LLM, typed tool dispatch, event-driven ticks; bounded-retry replanning live (full substitute/replan ladder partial) (ADR-0018) |
| Deploy ROS graph | ✅ shipped — `openral deploy {run,sim}` with HAL + safety + reasoner + world_state; dynamic skill dispatch + `/clock` publisher (ADR-0031/0032/0034/0048) |
| Navigation stack (SLAM + Nav2) | ✅ shipped — `openral_slam_bringup` (slam_toolbox) + `openral_nav2_bringup` as reasoner-managed background services; `cmd_vel` → mobile-base HAL (ADR-0025) |
| World Action Model (WAM) | 🟡 protocol shipped — `WorldModel` Protocol + `NullWorldModel`; Cosmos / UnifoLM-WMA-0 / IRASim adapters planned (v0.3) |
| Object detection + spatial lift | ✅ shipped — `RosImageObjectDetectorNode`, 2D→3D object lift, RT-DETR + OmDet-Turbo detector rSkills, GStreamer perception bus; LocateAnything-3B wired via VLM sidecar + `locate_in_view` on-demand tool; scene-VLM `kind:vlm` (ADR-0035/0037/0043/0047/0051) |
| Geometric safety + watchdog | 🟡 in flight — chunk-rate safety pass-through + envelope checks (✅), deadman/E-stop forwarders + human-estop (✅ `openral_safety_watchdog` / `openral_human_estop`); self/world/voxel collision + OctoMap→voxel bridge in dev (ADR-0030/0040) |
| BehaviorTree v4 executor | 🔵 planned — future option behind `bt_executor_node` (ADR-0018 §4) |
| C++ safety kernel | 🟡 in flight — deny-by-default allocation-free validator landed (n_dof / position / velocity / torque / cartesian / ee-speed + geometric collision), OTel spans; sim/HIL gate + LTTng pending (ADR-0020) |
| Org / publishing | 🟡 in flight — public repo + `master` branch protection ✅; PyPI trusted-publishing + GHCR images wired (`release-pypi.yml` / `release.yml`) but nothing published yet |

Legend: ✅ done, 🟡 in flight, 🔵 planned, 🔴 blocked / outstanding.

---

## Next phases

- **M2** — Unitree G1 real-HW HAL (`unitree_sdk2`) + cerebellar (S0) C++ controller; `rt_bridge` shared-memory ring.
- **M3** — HIL bring-up on lab runners: UR5e/UR10e/Franka/Sawyer/ALOHA real-HW adapters + D435 smoke test (adapters landed; runners not yet registered).
- **v0.3** — WAM adapters (Cosmos Predict, UnifoLM-WMA-0, IRASim) behind the `WorldModel` Protocol; spatial-memory ROS feeder + sqlite-vec persistence.
- **v1.0** — Failure-anticipation as first-class; C++ safety-kernel sim/HIL hardening + LTTng; certifiable build.

See [repo state map](../architecture/repo-state-map.html) for detailed per-module status and cross-layer dependencies.
