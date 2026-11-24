# Roadmap

Live status of OpenRAL development, one line per area. Per-module status and dependencies: [repo state map](../architecture/repo-state-map.html). Released versions: [CHANGELOG.md](https://github.com/OpenRAL/openral/blob/master/CHANGELOG.md).

Legend: ✅ done, 🟡 in flight, 🔵 planned, 🔴 blocked.

## Snapshot

| Area | Status |
|---|---|
| Schemas (`openral_core`) | ✅ Pydantic v2, hypothesis fuzz, JSON Schema export, CI drift check |
| HAL Protocol and ROS bridge | ✅ `RosControlHAL`, one manifest-driven lifecycle node for every robot |
| Sim HALs (arms, humanoids, bimanual, mobile) | ✅ SO-100/101, Franka, UR5e/UR10e, ALOHA, OpenArm, Rizon4, Metal, H1/H2/G1, OpenAMRobot 2.0, panda_mobile, Galaxea A1 and R1 Pro twins; per robot in [Robots](../reference/robots.md) |
| Real-hardware HALs | 🟡 adapters landed for SO-100/101, Franka, UR5e/UR10e, Sawyer, ALOHA, OpenArm, Galaxea A1, Flexiv Rizon and Unitree hg; only OpenArm and Galaxea A1 have run on hardware |
| Enactic OpenArm v2 real | 🟡 attended deploys have driven the arms; the commanded-motion HIL gate (double-gated on `OPENRAL_OPENARM_ALLOW_MOTION=1` and `OPENRAL_OPENARM_ATTENDED=1`) has no recorded run; [runbook](../tutorials/deploy/openarm-real-world-voxel-check.md) |
| Sensor adapters | 🟡 RealSense D435/D435i/D415, ZED mini, Arducam B0495, UVC (C920), OAK-D Pro, Robotiq FT-300; lidar, tactile and a perception-head package are planned |
| World State aggregator | ✅ 30 Hz tf2-aware snapshot, staleness latching, detected-object lift |
| Persistent spatial memory | 🟡 object/place/room graph with recall and resolve tools; sqlite-vec persistence pending |
| Skill base and runtimes | ✅ lifecycle node, PyTorch and ONNX runtimes, engine cache |
| rSkill manifest and loader | ✅ HF Hub packaging, license surface; sigstore provenance not implemented (unverified-provenance warning, `OPENRAL_REQUIRE_SIGNED_SKILLS` gate) |
| Policy adapters | ✅ SmolVLA, pi0.5, xVLA, ACT, Diffusion Policy, MolmoAct2, RLDX-1, GR00T N1.7, OpenVLA, NaT VLA, Cosmos 3, LingBot-VLA 1/2 and VA, XR-1, InternVLA-N1, 3D Diffuser Actor |
| Sim and eval harness | ✅ `openral sim run`, `benchmark run`, three-tier scene registry; [backends](../reference/sim-environments.md) |
| Multi-robot deploy sim | ✅ several robots in one scene, each with its own domain and dashboard |
| Observability | ✅ OTel spans and metrics, `openral dashboard`, compliance recorder ([Telemetry](../reference/telemetry.md)) |
| Dashboard remote link | ✅ `openral robot link` to openral.com; console is a prebuilt bundle ([ADR-0131](../decisions.md)) |
| ROS graph security | ✅ SROS2 enclaves via `openral security init` ([guide](../reference/ros-graph-security.md)) |
| Dataset bridge | ✅ `RolloutRecorder`, `LeRobotDatasetSink`, `openral dataset {push,from-bag}` |
| Reasoner (S2) | ✅ provider-selected LLM, typed tool dispatch, event-driven ticks; bounded retry live, full substitute/replan ladder partial |
| Deploy ROS graph | ✅ `openral deploy {run,sim}` with HAL, safety, reasoner, world state |
| Navigation (SLAM, Nav2) | ✅ reasoner-managed `slam_toolbox` and Nav2; `cmd_vel` to mobile-base HAL |
| Object detection and spatial lift | ✅ RT-DETR and OmDet-Turbo detectors, 2D to 3D lift, LocateAnything-3B on demand, scene VLM |
| Safety supervisor and watchdog | 🟡 envelope checks, deadman and human E-stop launched by the deploy graph (✅); hardware pendant needs a per-vendor driver |
| C++ safety kernel | ✅ deny-by-default validator with geometric collision, OTel spans; LTTng and formal proofs out of scope |
| Weight quantization | ✅ open runs fp32/fp16/bf16; 8-bit and below is OpenRAL Pro ([ADR-0123](../decisions.md)) |
| Publishing | ✅ lockstep SemVer by `release-please.yml` ([releasing](../contributing/releasing.md)); PyPI trusted publishing live (0.4.0 latest) |

## Next phases

- **M2: Unitree real hardware.** `packages/openral_hal_unitree` and `openral_hal.unitree_real.UnitreeHgRealHAL` reuse NVIDIA's Isaac ROS stack (S0 runs in `ros2_control`; OpenRAL commands the upper body and base twist). Not yet run on a physical G1 or H2. The 500-1000 Hz S0 target remains open.
- **M3: HIL bring-up on lab runners.** OpenArm v2 and Galaxea A1 have run by hand on a lab host. UR5e/UR10e, Franka, Sawyer and ALOHA adapters have HIL files but never met a physical arm. No CI workflow runs `tests/hil/`, so no HIL gate is enforced in CI.
- **Spatial memory.** sqlite-vec persistence for the scene graph.
- **v1.0.** Failure anticipation as a first-class feature; safety-kernel hardening and LTTng; a certifiable build.
