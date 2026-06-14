# Repository map

The canonical visual map is [`repo-state-map.html`](repo-state-map.html). This page is the textual companion — the directory layout and the external-repo boundaries.

## Layout

```
openral/                      ← THIS monorepo
├─ python/                        ← uv workspace; pure-Python libs
│  ├─ core/         (openral_core)        ← Pydantic v2 schemas (normative) ✓
│  ├─ cli/          (openral_cli)         ← `openral` entry point (bare → REPL, with args → one-shot; ADR-0021) ✓
│  ├─ detect/       (openral_detect)      ← `openral detect` auto-provisioning → robot.yaml (ADR-0008) ✓
│  ├─ hal/          (openral_hal)         ← `HAL` Protocol + manifest-driven `MujocoArmHAL` adapters (SO-100/101, Franka, UR5e/10e, ALOHA, OpenArm, Rizon4, H1, G1, panda_mobile) ✓
│  ├─ sensors/      (openral_sensors)     ← Sensor catalog + vendor adapters ✓
│  ├─ world_state/  (openral_world_state) ← `WorldStateAggregator` (30 Hz snapshot, staleness latching, detected-objects fold-in) ✓
│  ├─ rskill/       (openral_rskill)      ← `rSkillBase` ABC, `rSkill` loader, runtime adapters (PyTorch/ONNX/TensorRT), VLA + detector adapters ✓
│  ├─ state_adapter/ (openral_state_adapter) ← rSkill state-contract bindings (ADR-0027) ✓
│  ├─ sim/          (openral_sim)         ← `SimRunner` + `openral sim run` / `openral benchmark run` registry; LIBERO/MetaWorld/RoboCasa/ManiSkill3/SimplerEnv/gym-aloha/gym-pusht/IsaacSim (ADR-0045 sidecar) adapters ✓
│  ├─ runner/       (openral_runner)      ← `InferenceRunner` Protocol + `HardwareRunner` + `GStreamerSensorReader` + `NullSafetyClient`; ADR-0010 ✓
│  ├─ reasoner/     (openral_reasoner)    ← S2 LLM reasoner/supervisor core — `ReasonerCore`, `ReasonerToolCall` palette, bounded replanning ladder (ADR-0018) ✓
│  ├─ dataset/      (openral_dataset)     ← rosbag2 ↔ LeRobotDataset v3 bridge (ADR-0019) ✓
│  ├─ wam/          (openral_wam)         ← World Action Model layer (mental-simulation gating) — scaffold ◐
│  └─ observability/ (openral_observability) ← OTel SDK + OTLP exporter, span helpers, structlog↔OTel bridge ✓
│  Planned: dispatcher (edge/cloud/split)
├─ packages/                      ← ROS 2 packages (colcon build)
│  ├─ msgs/         (openral_msgs)        ← IDL (.msg, .action) — normative; WorldStateStamped carries detected_objects (ADR-0035) ✓
│  ├─ world_state/  (openral_world_state_ros) ← lifecycle node wrapping the aggregator ✓
│  ├─ openral_hal_so100/      ← SO-100 / SO-101 lifecycle node ✓
│  ├─ openral_hal_franka/     ← Franka lifecycle node ✓
│  ├─ openral_hal_ur5e/  · openral_hal_ur10e/  ← UR5e / UR10e lifecycle nodes ✓
│  ├─ openral_hal_aloha/  · openral_hal_openarm/  ← bimanual lifecycle nodes ✓
│  ├─ openral_hal_rizon4/  · openral_hal_g1/  · openral_hal_h1/  · openral_hal_panda_mobile/  ← per-robot lifecycle nodes (ADR-0029 unified base) ✓
│  ├─ openral_reasoner_ros/   ← ADR-0018 F4 reasoner_node (LLM ReasonerToolCall dispatch) ✓
│  ├─ openral_prompt_router/  ← ADR-0018 F10 prompt fan-in lifecycle node ✓
│  ├─ openral_rskill_ros/      ← ADR-0018 F1 rskill_runner_node + ExecuteSkill action server ✓
│  ├─ openral_safety/         ← ADR-0018 F5 safety_node (geometric collision checking, ADR-0030) ✓
│  ├─ openral_safety_watchdog/ · openral_human_estop/ ← deadman watchdog + human E-stop forwarders ✓
│  ├─ openral_perception_ros/ ← RosImageObjectDetectorNode (Image → ObjectsMetadata → 2D→3D lift, ADR-0035) ✓
│  ├─ openral_octomap_bridge/ ← OctoMap → safety-kernel OccupancyVoxels lowering (ADR-0030) ✓
│  └─ openral_nav2_bringup/ · openral_slam_bringup/ ← reasoner-managed Nav2 / slam_toolbox services (ADR-0025) ✓
│  Planned: core_ros, sensors_ros, dispatcher_ros, launch
├─ cpp/                           ← openral_safety_kernel — C++ deny-by-default kernel (ADR-0020, planned) ◐
├─ rskills/                       ← rSkill packages (manifest + weights + eval/) — VLA + detector kinds ✓
├─ scenes/                        ← SimEnvironment YAMLs (`sim run`) + native scenes (ADR-0002/0033) ✓
├─ benchmarks/                    ← benchmark suite definitions ✓
├─ deployments/                   ← RobotEnvironment deploy YAMLs (`openral deploy run`, ADR-0032) ✓
├─ robots/                        ← canonical RobotDescription manifests ✓
├─ tests/{unit,integration,sim,hil}/  ← all four trees ✓
├─ docs/                          ← mkdocs-material; ADRs in docs/adr/ ✓
├─ tools/                         ← schema_export.py, skill_publisher.py ✓
├─ scripts/                       ← bootstrap_*.sh, install.sh ✓
├─ Justfile                       ← canonical task runner ✓
├─ pyproject.toml + uv.lock       ← ✓
└─ .github/workflows/             ← CI ✓
```

Convention: directory names use short forms (`core/`, `cli/`, `msgs/`); Python module names and PyPI package names keep the `openral_` / `openral-` prefix. ✓ = shipped; ◐ = scaffold / planned; items without a mark are planned — see [`repo-state-map.html`](repo-state-map.html) for the per-module canvas.

## External, separate repos

Don't put their code in this monorepo:

- `huggingface.co/openral/skill-*` — skill weights & manifests.
- `huggingface.co/openral/dataset-*` — LeRobotDatasets.
- `openral/cloud` — BSL-1.1 hosted observability/fleet control plane.
- `openral/contrib-closed-shims` — private, NDA-restricted vendor shims.
- `openral/awesome-ros` — community curation.
