# Roadmap

Live status of OpenRAL development. For detailed architecture and module-by-module canvas, see [`../architecture/repo-state-map.html`](../architecture/repo-state-map.html).

---

## Snapshot

| Area | Status |
|---|---|
| Schemas (`openral_core`) | ✅ shipped — Pydantic v2, hypothesis fuzz, JSON Schema export, CI drift check |
| HAL Protocol + ROS bridge | ✅ shipped — `RosControlHAL`, mock unit tests, exception hierarchy |
| LeRobot SO-100/SO-101 HAL | ✅ shipped — real + MuJoCo `SO100FollowerHAL`, `SO100_DESCRIPTION`, `openral connect` |
| Franka / UR5e / UR10e HALs | ✅ sim shipped over `MujocoArmHAL`; real-HW adapters landed (UR5e/UR10e via `ur_robot_driver`, Franka via FCI, Sawyer, ALOHA) — HIL files exist but none has met a physical rig (M3) |
| Humanoid + bimanual HALs (sim) | ✅ shipped — `G1MujocoHAL`, `H1MujocoHAL`, `AlohaMujocoHAL`, `OpenArmMujocoHAL`, `Rizon4MujocoHAL`, `PandaMobileHAL` with lifecycle nodes |
| Enactic OpenArm v2 real HW | 🟡 in flight — `OpenArmRealHAL` fans a 16-DoF action across `openarm_bringup`'s four ros2_control controllers; Damiao CAN FD at 400 Hz stays in C++. `tests/hil/test_openarm_can_live.py` passes both transport gates and the motor round-trip on the wired arm, but that round-trip is a read: it never energises. The command→motion gate is written (`tests/hil/test_openarm_slot_group_motion.py`, double-gated on `OPENRAL_OPENARM_ALLOW_MOTION=1` + `OPENRAL_OPENARM_ATTENDED=1`) and has no recorded run, so no `send_action` has reached a motor yet. Run by hand on the lab host — nothing in CI runs `tests/hil/` — and no real-HW OpenArm deploy scene ships in-tree (host-specific CAN/camera names) |
| Sensor adapters | 🟡 in flight — `openral_sensors` catalog (RealSense D435/D435i/D415, Logitech UVC, Luxonis OAK-D Pro, Robotiq FT 300-S) + launch-gen + ROS image publisher; full perception-head ROS package still planned |
| World State aggregator | ✅ shipped — 30 Hz tf2-aware snapshot, stale-sensor diagnostics, detected-objects lift |
| Persistent spatial memory (scene graph) | 🟡 in flight — durable advisory object/place/room/agent graph the S2 reasoner queries (recall/resolve) + CLIP open-vocab match; sqlite-vec persistence pending (ROS feeder shipped) |
| Skill base + runtimes | ✅ shipped — lifecycle node, `PyTorchRuntime`, `ONNXRuntime`, quantization registry, engine cache |
| rSkill manifest + loader | ✅ shipped — HF Hub packaging, `rSkill.from_pretrained`, license surface; sigstore provenance not yet implemented (unverified-provenance warning + `OPENRAL_REQUIRE_SIGNED_SKILLS` fail-closed gate) |
| SmolVLA, π0.5, xVLA, ACT, DP, MolmoAct2, RLDX-1 adapters | ✅ shipped — loaded, tested, embodiment-tag gated; GR00T N1.7 in-process (lerobot 0.6.0 `GrootPolicy`, NF4 backbone) — ✅ live LIBERO-spatial 5/5 |
| End-to-end sim demo (SO-100 + SmolVLA) | ✅ shipped — smoketest + GIF; full LIBERO rollout verified |
| Configurable sim/eval harness | ✅ shipped — `openral sim run` + `benchmark run`, three-tier scene registry; LIBERO / MetaWorld / gym-aloha / gym-pusht / ManiSkill3 / SimplerEnv / RoboCasa backends + Isaac Sim sidecar |
| OpenTelemetry instrumentation | ✅ shipped — OTel SDK + OTLP exporter, `skill_span` / `inference_span` / `safety_span`, `openral dashboard` live OTLP receiver |
| rosbag2 ↔ LeRobotDataset v3 bridge | ✅ shipped — `RolloutRecorder`, `LeRobotDatasetSink`, `openral dataset {push,from-bag}` |
| Reasoner (S2 LLM supervisor) | ✅ shipped — `openral_reasoner` core + `openral_reasoner_ros` node + `openral_prompt_router`; provider-selected LLM, typed tool dispatch, event-driven ticks; bounded-retry replanning live (full substitute/replan ladder partial) |
| Deploy ROS graph | ✅ shipped — `openral deploy {run,sim}` with HAL + safety + reasoner + world_state; dynamic skill dispatch + `/clock` publisher |
| Navigation stack (SLAM + Nav2) | ✅ shipped — `openral_slam_bringup` (slam_toolbox) + `openral_nav2_bringup` as reasoner-managed background services; `cmd_vel` → mobile-base HAL |
| Object detection + spatial lift | ✅ shipped — `RosImageObjectDetectorNode`, 2D→3D object lift, RT-DETR + OmDet-Turbo detector rSkills, GStreamer perception bus; LocateAnything-3B wired via VLM sidecar + `locate_in_view` on-demand tool; scene-VLM `kind:vlm` |
| Geometric safety + watchdog | 🟡 in flight — chunk-rate safety pass-through + envelope checks (✅), deadman/E-stop forwarders + human-estop (✅ `openral_safety_watchdog` / `openral_human_estop`); self/world/voxel collision + OctoMap→voxel bridge in dev |
| C++ safety kernel | ✅ deny-by-default allocation-free validator landed (n_dof / position / velocity / torque / cartesian / ee-speed + geometric collision), OTel spans; geometric collision verified end-to-end through the real kernel in the sim tier (issue #77); LTTng + formal proofs remain out of scope |
| Org / publishing | 🟡 in flight — public repo + `master` branch protection ✅; lockstep SemVer computed from Conventional Commits by `release-please.yml` ✅ ([releasing](../contributing/releasing.md)); PyPI trusted-publishing live — 0.1.0 and 0.2.0 published for all 14 packages ✅. `release.yml` was dropped 2026-06-17; the GHCR release-image path returns as a purpose-built `release-image.yml` once ROS-in-CI infra exists |

Legend: ✅ done, 🟡 in flight, 🔵 planned, 🔴 blocked / outstanding.

---

## Next phases

- **M2** — Unitree G1 real-HW HAL (`unitree_sdk2`) + cerebellar (S0) C++ controller; `rt_bridge` shared-memory ring.
- **M3** — HIL bring-up on lab runners. Exercised on real hardware today, by hand on a lab host: **OpenArm v2** (`tests/hil/test_openarm_can_live.py` — CAN transport + a read-only motor round-trip; commanded motion is still an unrun gate) and **Galaxea A1** (`tests/hil/test_galaxea_a1*.py` — observation, hold, bounded joint and gripper round trips, and `candidate_action` → C++ kernel → `safe_action` → real HAL). Adapters and `tests/hil/` files landed but never met a physical arm: UR5e/UR10e/Franka/Sawyer/ALOHA — there is no physical Franka, Sawyer or ALOHA to validate against, and the UR pair is gated on `UR5E_HOST` / `UR10E_HOST` with no recorded run. Blocked on runners: no workflow runs `tests/hil/` at all, so no HIL gate is enforced in CI. The SO-100/SO-101 and RealSense (D435) HIL gates were removed until matching lab hardware exists.
- **v0.3** — spatial-memory ROS feeder + sqlite-vec persistence.
- **v1.0** — Failure-anticipation as first-class; C++ safety-kernel sim/HIL hardening + LTTng; certifiable build.

See [repo state map](../architecture/repo-state-map.html) for detailed per-module status and cross-layer dependencies.
