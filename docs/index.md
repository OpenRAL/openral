# OpenRAL

**OpenRAL** is the open-source Robot Agentic Layer: one typed, traceable, safety-first runtime that unifies fast policies, slow reasoning, reward signals, perception and classical control for deployable robot agents. A fixed harness runs swappable **rSkills**, which are packaged capabilities installed from the Hugging Face Hub.

Seven layers, each with one job: HAL, Sensors, World State, rSkill (S1), Reasoning (S2), Safety, Observability. Pydantic v2 schemas are the contract between them.

## Start here

1. [Quickstart](quickstart/index.md): install and run `openral doctor`.
2. [Your first sim rollout](tutorials/sim/first-rollout.md): a real policy driving a simulated robot, no GPU.
3. [Deploy on a robot & open the dashboard](tutorials/deploy/deploy-run-and-dashboard.md): the real-hardware graph.

```bash
git clone https://github.com/OpenRAL/openral && cd openral
just bootstrap && just sync --group sim
uv run openral doctor
```

## Site map

### Tutorials

- [Your first sim rollout](tutorials/sim/first-rollout.md)
- [Create a sim environment](tutorials/sim/create-a-sim-environment.md): author a scene, robot or adapter
- [Run a benchmark](tutorials/benchmark/run-a-benchmark.md): reproducible numbers
- [Write & publish an rSkill](tutorials/rskill/write-and-publish-an-rskill.md)
- [Quantize an rSkill](tutorials/rskill/quantize-an-rskill.md)
- [Deploy on a robot & open the dashboard](tutorials/deploy/deploy-run-and-dashboard.md)
- [OpenArm real-cell world-voxel check](tutorials/deploy/openarm-real-world-voxel-check.md)

### Quickstart

- [Quickstart](quickstart/index.md) and the [robot dashboard](quickstart/dashboard.md)

### Architecture

- [Overview](architecture/overview.md): layers, dual system, safety
- [Repo map](architecture/repo-map.md) and the interactive [repo state map](architecture/repo-state-map.html)
- [Methods inventory](METHODS.md): every public symbol, by layer
- [ROS 2 topic graph](topics/README.md)
- [Roadmap](roadmap/index.md)

### Reference

- Core: [API](reference/api.md), [Glossary](reference/glossary.md), [Design decisions](decisions.md)
- Robots and skills: [Robot descriptions](reference/robots.md), [rSkills](reference/rskills.md), [VLA × Robot × Sim compatibility](reference/vla_compatibility.md), [Sensor catalog](reference/sensors_landscape.md)
- Reasoner: [Reasoner (S2)](reference/reasoner.md), [design](reference/reasoner-design.md), [Cosmos 3 Edge](reference/cosmos3-edge-reasoner.md)
- Runtime and ops: [Sim environments](reference/sim-environments.md), [Telemetry](reference/telemetry.md), [Compliance log](reference/compliance-log.md), [ROS graph security](reference/ros-graph-security.md), [DDS large messages](reference/dds-large-messages.md), [aarch64 hosts](reference/aarch64-support.md)
- Safety evidence: the collision-stack pages listed under Reference in the site navigation

### Contributing

- [Development](contributing/development.md), [Toolchain](contributing/toolchain.md), [Releasing](contributing/releasing.md), [Selective testing](contributing/selective-testing.md), [Test-suite audit](contributing/test-audit.md), [Validation matrix](contributing/validation-matrix.md)
