# CLI reference

Every `openral` command, grouped by task. Run any command with `--help` for its flags;
from a checkout, prefix it with `uv run`.

```bash
uv run openral --help
uv run openral deploy sim --help
```

## Host setup and diagnosis

| Command | What it does |
|---|---|
| `openral doctor` | Diagnose the host: Python, OS, ROS 2 distro, GPU, USB devices. See [Quickstart](../quickstart/index.md). |
| `openral install list` | List every dependency group the installer knows about. |
| `openral install sim` | Install the `sim` group (gym-aloha, gym-pusht, MuJoCo). |
| `openral install libero` / `robocasa` | Install LIBERO or RoboCasa. The two are mutually exclusive. |
| `openral install maniskill3` / `metaworld` / `simpler-env` | Install the ManiSkill3, MetaWorld or SimplerEnv backends. |
| `openral install rldx` | Install the RLDX-1 sidecar client. |
| `openral install ros` | Re-run the ROS 2 and system-package bootstrap (sudo, Linux only). |

## Robots and sensors

| Command | What it does |
|---|---|
| `openral detect` | Probe the host and write a complete `robot.yaml` ([guide](https://github.com/OpenRAL/openral/blob/master/python/detect/README.md)). |
| `openral connect` | Open a HAL connection, read one joint state, disconnect. |
| `openral check` | Cross-validate every robot, rSkill and scene manifest. |
| `openral collision lower` | Lower a URDF/SRDF to the kernel's collision model; prints a diff, writes only with `--write`. |
| `openral collision check` | Exit 1 if any manifest drifts from its lowered collision model. |
| `openral robot vendor-urdf` | Expand an upstream description into a flat, committed URDF. |
| `openral sensor list` / `show` | Browse the sensor catalog ([Sensor catalog](sensors_landscape.md)). |
| `openral calibrate camera` | Calibrate a camera with ROS 2 `camera_calibration`. |

## rSkills

| Command | What it does |
|---|---|
| `openral rskill search` | Search the OpenRAL Hugging Face org for installable rSkills. |
| `openral rskill install` | Download an rSkill, validate it and register it locally. |
| `openral rskill list` | List in-tree and installed rSkills. |
| `openral rskill check` | Report whether one (or every) rSkill will run on this host. |
| `openral rskill new` | Scaffold a new rSkill from `rskills/template/`. |

Guides: [Write & publish an rSkill](../tutorials/rskill/write-and-publish-an-rskill.md), [rSkills](rskills.md).

## Simulation and benchmarks

| Command | What it does |
|---|---|
| `openral sim list` | List every scene under `scenes/`. |
| `openral sim run` | Run a sim rollout from a SimScene YAML (`--config`) and an rSkill (`--rskill`). |
| `openral benchmark list` | List benchmark suite ids. |
| `openral benchmark run` | Run a suite and write a validated `RSkillEvalResult` JSON. |
| `openral benchmark scene` | Run one benchmark scene and write its eval JSON. |
| `openral benchmark report` | Roll up every `<skill>/eval/*.json`. |
| `openral behavior serve` | Serve an rSkill to BEHAVIOR-1K's official evaluator. |

Guides: [First sim rollout](../tutorials/sim/first-rollout.md), [Run a benchmark](../tutorials/benchmark/run-a-benchmark.md), [Sim environments](sim-environments.md).

## Deploying

| Command | What it does |
|---|---|
| `openral deploy list` | List deploy scenes under `scenes/deploy/`. |
| `openral deploy validate` | Readiness check for `deploy run`; no hardware, no ROS launch. |
| `openral deploy sim` | Boot the full ROS graph against a digital-twin HAL from a DeployScene YAML. |
| `openral deploy run` | Run an rSkill on real hardware through the production ROS graph. |
| `openral prompt` | Send a one-shot operator prompt to the prompt router (needs a sourced ROS 2). |
| `openral estop reset` | Reset the safety kernel, then broadcast `/openral/estop_cleared`. |
| `openral security init` | Generate the robot's SROS2 keystore ([ROS graph security](ros-graph-security.md)). |
| `openral robot link` | Link this robot to openral.com and install its tunnel ([Dashboard](../quickstart/dashboard.md)). |

Guide: [Deploy on a robot](../tutorials/deploy/deploy-run-and-dashboard.md).

## Observability, data and compliance

| Command | What it does |
|---|---|
| `openral dashboard` | Serve the robot dashboard, its API and an OTLP/HTTP receiver ([Dashboard](../quickstart/dashboard.md)). |
| `openral record` | Run `ros2 bag record` for the OpenRAL graph with a slim or full profile. |
| `openral replay` | Join a rosbag2/MCAP file with OTel spans into a JSON timeline. |
| `openral profile session` | Start, stop or view an LTTng session for the realtime hot path (needs `OPENRAL_ROS2_TRACING=1` on the traced process). |
| `openral dataset from-bag` | Convert a recorded MCAP bag into a LeRobotDataset v3. |
| `openral dataset push` | Upload a LeRobotDataset v3 to the Hub (private, consent-gated). |
| `openral compliance verify` | Walk every hash chain of the compliance log; exit 1 on tampering. |
| `openral compliance inventory` | List the software installed on this unit, safety software marked. |
| `openral compliance hold` | Freeze the whole log during an incident investigation. |
| `openral compliance export` | Bundle whole segments for an authority's request. |
| `openral compliance ship` | Send records not yet shipped; resumes where the last run stopped. |
| `openral compliance prune` | Delete segments past their retention floor; each deletion is recorded. |

Details: [Telemetry](telemetry.md), [Compliance log](compliance-log.md).
