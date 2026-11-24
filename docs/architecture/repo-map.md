# Repository map

The visual map is [`repo-state-map.html`](repo-state-map.html). This page is the textual companion: where each thing lives. Directory names use short forms (`core/`, `cli/`, `msgs/`); Python modules and PyPI packages keep the `openral_` / `openral-` prefix.

```
openral/
├─ python/                  uv workspace, 13 pure-Python packages
│  ├─ core/           openral_core            Pydantic v2 schemas (normative)
│  ├─ cli/            openral_cli             the `openral` command (bare = REPL)
│  ├─ detect/         openral_detect          hardware probing, robot.yaml wizard
│  ├─ hal/            openral_hal             HAL Protocol; sim and real adapters (see reference/robots.md)
│  ├─ sensors/        openral_sensors         sensor catalog and vendor adapters
│  ├─ world_state/    openral_world_state     30 Hz WorldStateAggregator
│  ├─ rskill/         openral_rskill          rSkill loader, runtimes, quantization gate
│  ├─ state_adapter/  openral_state_adapter   per-checkpoint state-vector layouts
│  ├─ sim/            openral_sim             SimRunner, scene registry, policy adapters
│  ├─ runner/         openral_runner          InferenceRunner, DeployRunner
│  ├─ reasoner/       openral_reasoner        S2 ReasonerCore, tool palette, replanning
│  ├─ dataset/        openral_dataset         rosbag2 / LeRobotDataset v3 bridge
│  └─ observability/  openral_observability   OTel, dashboard server
├─ packages/                ROS 2 packages (colcon)
│  ├─ msgs/                   openral_msgs IDL (.msg, .srv, .action)
│  ├─ world_state/            world-state lifecycle node
│  ├─ openral_hal_node/       the one manifest-driven HAL lifecycle node
│  ├─ openral_hal_openarm/    OpenArm real-hardware bringup
│  ├─ openral_hal_rizon/      dual Flexiv Rizon 4s real-hardware bringup
│  ├─ openral_hal_unitree/    Unitree hg (G1, H2) real-hardware bringup
│  ├─ openral_reasoner_ros/   reasoner node
│  ├─ openral_prompt_router/  operator-prompt fan-in node
│  ├─ openral_rskill_ros/     rskill_runner_node, ExecuteRskill action server
│  ├─ openral_safety/         safety supervisor, collision lowering helpers
│  ├─ openral_safety_watchdog/  deadman watchdog, hardware E-stop bridge
│  ├─ openral_human_estop/    human E-stop forwarder
│  ├─ openral_compliance/     compliance recorder (OTLP log records)
│  ├─ openral_perception_ros/ detector and segmenter nodes
│  ├─ openral_octomap_bridge/ OctoMap to safety-kernel voxels, self-filter
│  ├─ openral_nav2_bringup/ · openral_slam_bringup/   reasoner-managed Nav2, slam_toolbox
│  └─ openral_foxglove_bringup/  read-only Foxglove live-scene bridge
├─ cpp/                     openral_safety_kernel (C++ kernel), opentelemetry_cpp_vendor
├─ rskills/                 example rSkill packages (manifest, eval/); the catalog is on the Hub
├─ scenes/                  {deploy,sim,benchmark}/ scene YAMLs
├─ robots/                  RobotDescription manifests + per-host units/<unit>.yaml overlays
├─ benchmarks/              benchmark suite definitions
├─ docker/                  inference/ (deploy image), otelcol/, galaxea_a1_sidecar/
├─ tests/{unit,integration,sim,hil}/
├─ docs/                    mkdocs-material site
├─ tools/ · scripts/        schema export, publisher, topic-graph generator; install.sh
├─ Justfile · pyproject.toml · uv.lock
└─ .github/workflows/       CI
```

The bootstrap scripts live in `python/cli/src/openral_cli/bootstrap/` so `openral install ros` works from the wheel without a clone.

## Outside this repo

- `huggingface.co/OpenRAL`: rSkill repos (`OpenRAL/rskill-*`) and datasets.
- `OpenRAL/openral-pro` (private): Pro runtimes, see [Design decisions](../decisions.md).
- `OpenRAL/platform` (private): the robot console source and openral.com ([ADR-0131](../decisions.md)).
- `OpenRAL/management` (private): the ADR log and safety hazard log.
