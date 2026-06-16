# Architecture Decision Records — index

This directory holds OpenRAL's Architecture Decision Records (ADRs). The format and
ground rules are set by [ADR-0001](0001-record-architecture-decisions.md):

> One file per decision, **monotonically numbered**. Immutable once accepted — to reverse
> or contradict a decision you write a **new** ADR that marks the prior one *Superseded*;
> you never renumber, merge, or delete an accepted record. Factual corrections and status
> updates may be added in place as dated **Amendments**.

Because the log is append-only, the file count only ever grows. The point of this index is
to keep that growing log **navigable**: ADRs are grouped below by topic cluster, each with
its current status and its relationships to other ADRs. When a decision evolves, look for
the highest-numbered ADR in its cluster — that is usually the live one.

**Legend** — Status: `Accepted` (decided; may be partly or fully implemented) · `Proposed`
(written, not yet ratified/landed) · `Superseded` (replaced by a later ADR). Relations:
*extends / builds on* (additive), *amends* (in-place correction of an earlier ADR),
*supersedes* (replaces an earlier decision or sub-decision).

> **Maintainers:** this index is hand-maintained. When you add an ADR, add a row to the
> right cluster here **and** a nav entry in `mkdocs.yml`. Per [ADR-0001](0001-record-architecture-decisions.md)
> the source ADR files are the normative record; this table is a convenience map over them.

---

## A · Process & foundations

| ADR | Title | Status | Relations |
|-----|-------|--------|-----------|
| [0001](0001-record-architecture-decisions.md) | Record architecture decisions | Accepted | — |
| [0003](0003-pydantic-over-dataclasses.md) | Pydantic v2 over `@dataclass` for all schemas/contracts | Accepted | — |
| [0004](0004-monorepo-over-polyrepo.md) | Single monorepo over poly-repo for the open-core | Accepted | — |
| [0012](0012-open-core-licensing.md) | Licensing — uniform Apache-2.0, no commercial tier | Accepted | — |
| [0016](0016-multi-platform-support.md) | Multi-platform support — x86 (CUDA + CPU) and L4T (Jetson) | Accepted | refs 0010 amendment |
| [0021](0021-curl-installer-cli-rename-and-pypi-release.md) | Curl-bash installer, CLI rename, multi-package PyPI scaffold | Accepted | — |

## B · rSkill packaging, manifest & action contracts

| ADR | Title | Status | Relations |
|-----|-------|--------|-----------|
| [0006](0006-hf-hub-skill-packaging.md) | Hugging Face Hub as the rSkill packaging substrate | Accepted | — |
| [0013](0013-rskill-manifest-actuators-and-processors.md) | rSkill manifest — actuators, custom-embodiment hatch, processors | Accepted | — |
| [0022](0022-rskill-action-vocabulary.md) | rSkill action vocabulary for the reasoner LLM tool palette | Proposed | extended by 0026 |
| [0024](0024-ros-wrapped-rskills.md) | ROS-wrapped rSkills (`kind: ros_action` / `ros_service`) | Proposed | amended by 0030 (out-of-scope note) |
| [0026](0026-rskill-structured-goal-parameters.md) | rSkill structured goal parameters (`goal_params_json`) | Proposed | extends 0022 |
| [0027](0027-rskill-state-contract-bindings.md) | rSkill state-contract bindings + layout adapter registry | Proposed | — |
| [0028](0028-rskill-action-contract-slots.md) | Action-contract slots + robot.yaml gripper convention (**0028a foundation**) | Proposed | split → 0028b/0028c/0028d |
| [0028b](0028b-rskill-action-contract-slots-dispatch.md) | rSkill action-contract slot dispatch (`action_contract.slots`) | Proposed | part of 0028 split; extended by 0036 |
| [0028c](0028c-panda-mobile-hal-cartesian-gripper-handlers.md) | panda_mobile HAL CARTESIAN_DELTA + GRIPPER_POSITION handlers | Proposed | part of 0028 split |
| [0028d](0028d-panda-mobile-hal-joint-velocity-torso-handlers.md) | panda_mobile HAL JOINT_VELOCITY + COMPOSITE_MODE handlers | Proposed | part of 0028 split |
| [0036](0036-osc-action-contracts-deploy-path-gate.md) | Cartesian/OSC action contracts + deploy-path-aware palette gate | Accepted | extends 0028b; amended for 0041 |
| [0047](0047-vlm-rskill-kind.md) | `vlm` rSkill kind for video-language scene-understanding models | Accepted | — |
| [0055](0055-rskill-registry-model-and-discoverability.md) | rSkill registry model + discoverability (`rskill search`) | Proposed | — |

## C · Reasoner & S2 planning

| ADR | Title | Status | Relations |
|-----|-------|--------|-----------|
| [0005](0005-bt-llm-not-langgraph.md) | BehaviorTree.CPP v4 XML + typed LLM tool palette, not LangGraph | Accepted | — |
| [0018](0018-ros2-reasoner-supervisor.md) | ROS 2 reasoner + supervisor graph | Accepted | extended by 0039 (§4 palette) |
| [0025](0025-reasoner-managed-background-services.md) | Reasoner-managed background services (SLAM, perception trees) | Accepted | amended 2026-06-08 for 0041 |
| [0039](0039-llm-task-planning-active-search.md) | LLM task planning + active object search over the scene graph | Proposed | extends 0018 §4; depends 0038 |
| [0043](0043-locate-in-view-reasoner-tool.md) | `locate_in_view` — on-demand live-detector query for the reasoner | Accepted | related 0037, 0039; reconciled by 0051 |

## D · HAL & robot description

| ADR | Title | Status | Relations |
|-----|-------|--------|-----------|
| [0007](0007-robot-sim-split.md) | Separate physical robot manifests from simulator IO contracts | Accepted | — |
| [0008](0008-auto-provisioning-detect-package.md) | Auto-provisioning — `python/detect/` as a top-level package | Accepted | — |
| [0023](0023-data-driven-mujoco-hal.md) | Data-driven MuJoCo HAL — per-robot constants into `RobotDescription.sim` | Proposed | — |
| [0029](0029-unified-hal-lifecycle-node.md) | Unify per-robot HAL lifecycle nodes into one robot.yaml-driven node | Accepted | — |
| [0031](0031-sim-real-hal-separation.md) | Explicit sim/real HAL separation with deterministic command routing | Proposed | built on by 0032 |
| [0049](0049-hal-multithreaded-executor-proprio-snapshot.md) | Dedicated HAL publisher thread + proprio snapshot (odom-starvation fix) | Proposed | — |

## E · Sim, eval & benchmarking

| ADR | Title | Status | Relations |
|-----|-------|--------|-----------|
| [0002](0002-eval-and-sim-environments.md) | Configurable sim environments for rSkill validation | Accepted | naming superseded by 0009 |
| [0009](0009-separate-sim-and-benchmarking.md) | Separate simulation from benchmarking | Accepted | supersedes 0002 naming; reconciled w/ 0010 |
| [0014](0014-maniskill3-simpler-env-backends.md) | ManiSkill3 and SimplerEnv as opt-in sim backends | Accepted | — |
| [0015](0015-robocasa-isolated-backend-lazy-assets.md) | RoboCasa as an isolated `openral sim` backend with lazy assets | Accepted | — |
| [0033](0033-robot-parameterized-native-scenes.md) | Robot-parameterized native sim scenes (Effort 3) | Proposed | §Decision-4 superseded by 0034 |
| [0041](0041-scene-three-tier-hierarchy.md) | Scene three-tier hierarchy (Deploy/Sim/Benchmark Scene) | Accepted | item-10 superseded by 0042 |
| [0042](0042-drop-benchmarkspec.md) | Drop `BenchmarkSpec` for a bare `list[BenchmarkScene]` | Accepted | supersedes 0041 item 10 |
| [0045](0045-isaac-sim-backend-integration.md) | NVIDIA Isaac Sim as an optional sim backend | Proposed | refs 0034 idle-step amendment |

## F · Deploy path & runtime

| ADR | Title | Status | Relations |
|-----|-------|--------|-----------|
| [0010](0010-inference-runner.md) | Inference runner for hardware deployments | Accepted | amended (NVMM 0011, dataset 0019) |
| [0032](0032-deploy-run-ros-graph.md) | `deploy run` runs the ROS launch graph against real hardware | Proposed | builds on 0031 |
| [0034](0034-deploy-sim-scene-attach-for-arms.md) | Deploy-sim scene-attach + sim-sensor bridge for manifest-driven arms | Accepted | supersedes 0033 §Decision-4 |
| [0046](0046-nvidia-gr00t-backend.md) | NVIDIA Isaac GR00T as an out-of-process VLA backend | Accepted | reuses 0010 RLDX-1 sidecar; refs 0019 contract dims |
| [0048](0048-deploy-sim-clock-publisher.md) | A sim `/clock` publisher for the deploy-sim graph | Proposed | refs 0034 idle-stepper; safety-WG-gated |
| [0050](0050-single-resident-skill-vram-eviction.md) | Single-resident-skill VRAM eviction (unload-on-switch) | Proposed | — |

## G · Safety & collision

| ADR | Title | Status | Relations |
|-----|-------|--------|-----------|
| [0020](0020-cpp-safety-kernel.md) | C++ safety kernel | Proposed | — |
| [0030](0030-geometric-safety-collision-checking.md) | Geometric safety — self/world-collision checking in the kernel | Proposed | amends 0024; extended by 0040, 0053 |
| [0040](0040-geometric-collision-all-control-modes.md) | Geometric collision checking for every control mode | Proposed | extends 0030 |
| [0053](0053-collision-aware-approach-to-pose.md) | Collision-aware "approach to pose" before rSkill activation | Accepted | builds on 0030; related 0054 |

## H · Perception & spatial memory

| ADR | Title | Status | Relations |
|-----|-------|--------|-----------|
| [0035](0035-perception-spatial-memory-object-lift.md) | Perception → spatial-memory object lift (2D→3D) | Accepted | refined by 0052 |
| [0037](0037-gstreamer-perception-bus-object-detection.md) | GStreamer perception bus — tee consumers + object-detection rSkill | Proposed | related 0043, 0051 |
| [0038](0038-persistent-semantic-spatial-memory.md) | Persistent spatial memory — object-centric recall, scene graph | Proposed | feeds 0039 |
| [0051](0051-detector-invocation-mode.md) | Detector invocation mode — continuous producers vs on-demand locators | Accepted | reconciles 0037, 0043 |
| [0052](0052-cross-frame-object-lift.md) | Cross-frame object-lift (RGB optical TF + octomap/kernel decoupling) | Proposed | refines 0035 |

## I · Sensors & observability

| ADR | Title | Status | Relations |
|-----|-------|--------|-----------|
| [0011](0011-nvmm-handoff.md) | NVMM → CUDA tensor handoff across Sensors / Skill boundary | Proposed | extends 0010 |
| [0017](0017-dashboard-otlp-receiver.md) | `openral dashboard` — embedded OTLP/HTTP receiver | Accepted | — |
| [0019](0019-rosbag2-lerobot-dataset-bridge.md) | rosbag2 ↔ LeRobotDataset v3 bridge | Accepted | amends 0010 |

## J · Motion / MoveIt goal building

| ADR | Title | Status | Relations |
|-----|-------|--------|-----------|
| [0044](0044-look-at-skill-grid-refined-approach.md) | `look_at` skill + occupancy-grid-refined approach poses | Accepted | amended/extended by 0054 |
| [0054](0054-moveit-goal-builder-library.md) | `goal_builder` — joint/pose/look_at library over `ROSActionRskill` | Accepted | amends/extends 0044 |

---

## Notes for future consolidation

These are observations for maintainers — **not** licence to renumber or delete records
(forbidden by [ADR-0001](0001-record-architecture-decisions.md)). Consolidation here means
*better cross-links and overviews*, not fewer files.

- **The `0028` family** (`0028`, `0028b`, `0028c`, `0028d`) is one decision split across four
  files; `0028` is internally labelled "0028a". When this work is fully landed, a short
  amendment at the top of `0028` summarising the final shape would let readers skip the split.
- **Collision** (`0020` → `0030` → `0040` → `0053`) and **spatial memory** (`0035` → `0038`
  → `0052`, with `0037`/`0051` on detection) are the two longest evolution chains; the
  highest number in each is the current design.
- Many cluster-B/D ADRs are still `Proposed` despite landed implementations. A status-only
  amendment pass (additive, dated) would make the index reflect reality without touching
  decisions.
