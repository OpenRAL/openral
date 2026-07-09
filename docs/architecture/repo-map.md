# Repository map

The canonical visual map is [`repo-state-map.html`](repo-state-map.html). This page is the textual companion — the directory layout and the external-repo boundaries.

## Layout

```
openral/                      ← THIS monorepo
├─ python/                        ← uv workspace; pure-Python libs
│  ├─ core/         (openral_core)        ← Pydantic v2 schemas (normative) ✓
│  ├─ cli/          (openral_cli)         ← `openral` entry point (bare → REPL, with args → one-shot) ✓
│  ├─ detect/       (openral_detect)      ← `openral detect` auto-provisioning → robot.yaml ✓
│  ├─ hal/          (openral_hal)         ← `HAL` Protocol + manifest-driven `MujocoArmHAL` adapters (SO-100/101, Franka, UR5e/10e, ALOHA, OpenArm, Anvil-OpenArm-v2, Rizon4, H1, G1, panda_mobile) ✓
│  ├─ sensors/      (openral_sensors)     ← Sensor catalog + vendor adapters ✓
│  ├─ world_state/  (openral_world_state) ← `WorldStateAggregator` (30 Hz snapshot, staleness latching, detected-objects fold-in) ✓
│  ├─ rskill/       (openral_rskill)      ← `rSkillBase` ABC, `rSkill` loader, runtime adapters (PyTorch/ONNX; TensorRT is an OpenRAL Pro plugin), VLA + detector adapters ✓
│  ├─ state_adapter/ (openral_state_adapter) ← rSkill state-contract bindings ✓
│  ├─ sim/          (openral_sim)         ← `SimRunner` + `openral sim run` / `openral benchmark run` registry; LIBERO/MetaWorld/RoboCasa/ManiSkill3/SimplerEnv/gym-aloha/gym-pusht/IsaacSim (sidecar) adapters ✓
│  ├─ runner/       (openral_runner)      ← `InferenceRunner` Protocol + `DeployRunner` + `GStreamerSensorReader` + `NullSafetyClient` ✓
│  ├─ reasoner/     (openral_reasoner)    ← S2 LLM reasoner/supervisor core — `ReasonerCore`, `ReasonerToolCall` palette, bounded replanning ladder ✓
│  ├─ dataset/      (openral_dataset)     ← rosbag2 ↔ LeRobotDataset v3 bridge ✓
│  ├─ wam/          (openral_wam)         ← World Action Model layer (mental-simulation gating) — scaffold ◐
│  └─ observability/ (openral_observability) ← OTel SDK + OTLP exporter, span helpers, structlog↔OTel bridge ✓
│  Planned: dispatcher (edge/cloud/split)
├─ packages/                      ← ROS 2 packages (colcon build)
│  ├─ msgs/         (openral_msgs)        ← IDL (.msg, .action) — normative; WorldStateStamped carries detected_objects ✓
│  ├─ world_state/  (openral_world_state_ros) ← lifecycle node wrapping the aggregator ✓
│  ├─ openral_hal_so100/      ← SO-100 / SO-101 lifecycle node ✓
│  ├─ openral_hal_franka/     ← Franka lifecycle node ✓
│  ├─ openral_hal_ur5e/  · openral_hal_ur10e/  ← UR5e / UR10e lifecycle nodes ✓
│  ├─ openral_hal_aloha/  · openral_hal_openarm/  ← bimanual lifecycle nodes ✓
│  ├─ openral_hal_rizon4/  · openral_hal_g1/  · openral_hal_h1/  · openral_hal_panda_mobile/  ← per-robot lifecycle nodes (unified base) ✓
│  ├─ openral_hal_scene_attached/ ← scene-attached sim HAL node (`deploy sim`) ✓
│  ├─ openral_reasoner_ros/   ← reasoner_node (LLM ReasonerToolCall dispatch) ✓
│  ├─ openral_prompt_router/  ← prompt fan-in lifecycle node ✓
│  ├─ openral_rskill_ros/      ← rskill_runner_node + ExecuteSkill action server ✓
│  ├─ openral_safety/         ← safety_node (geometric collision checking) ✓
│  ├─ openral_safety_watchdog/ · openral_human_estop/ ← deadman watchdog + human E-stop forwarders ✓
│  ├─ openral_perception_ros/ ← RosImageObjectDetectorNode (Image → ObjectsMetadata → 2D→3D lift) ✓
│  ├─ openral_octomap_bridge/ ← OctoMap → safety-kernel OccupancyVoxels lowering ✓
│  ├─ openral_nav2_bringup/ · openral_slam_bringup/ ← reasoner-managed Nav2 / slam_toolbox services ✓
│  └─ openral_foxglove_bringup/ ← read-only Foxglove live-scene bridge + Bucket-2 converter + MCAP ✓
│  Planned: core_ros, sensors_ros, dispatcher_ros, launch
├─ cpp/                           ← openral_safety_kernel — C++ deny-by-default kernel (implemented + tested, certification pending) ✓
├─ rskills/                       ← rSkill packages (manifest + weights + eval/) — VLA + detector kinds ✓
├─ scenes/                        ← SimEnvironment YAMLs (`sim run`) + native scenes ✓
├─ benchmarks/                    ← benchmark suite definitions ✓
├─ deployments/                   ← retired; deploy configs live in scenes/deploy ✓
├─ robots/                        ← canonical RobotDescription manifests ✓
├─ tests/{unit,integration,sim,hil}/  ← all four trees ✓
├─ docs/                          ← mkdocs-material; decision log lives in the private OpenRAL/management repo ✓
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
- `openral/cloud` — hosted observability/fleet control plane (separate repo).
- `openral/contrib-closed-shims` — adapters for closed third-party vendor SDKs (the SDK is closed, not OpenRAL).
- `openral/awesome-ros` — community curation.
