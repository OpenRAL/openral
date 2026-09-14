# OpenRAL

An open-source operating layer for embodied AI, **OpenRAL** unifies fast policies, slow reasoning, and classical control into one typed, traceable, safety-first runtime for deployable robot agents.

## Quick start

```bash
git clone https://github.com/OpenRAL/openral && cd OpenRAL
just bootstrap          # installs uv, ROS 2, system deps
just sync               # install Python workspace (wraps uv sync + repair)
uv run openral doctor        # verify your environment
just test               # run the test suite
```

## What does it do?

- Load any VLA (SmolVLA, π0, GR00T N1, OpenVLA) on any robot (SO-100, G1, UR5e).
- Type-safe, layer-isolated architecture — HAL → Sensors → World State → Skill → Reasoning → Safety → Observability.
- Skill packaging on HuggingFace Hub with versioned, license-aware manifests (sigstore provenance signing is planned, not yet implemented).
- Full OpenTelemetry traces per execution.
- LeRobotDataset v3 flywheel — every execution becomes a training row.

## Navigation

**Get started**

- [Quickstart — `openral doctor`](quickstart/index.md)
- [Your first sim rollout](tutorials/sim/first-rollout.md) — a real policy driving a simulated robot, no GPU needed
- [Development setup](contributing/development.md)

**Tutorials**

- [Your first sim rollout](tutorials/sim/first-rollout.md) — start here
- [Create a sim environment](tutorials/sim/create-a-sim-environment.md) — author a scene, robot or adapter
- [Run a benchmark](tutorials/benchmark/run-a-benchmark.md) — reproducible numbers, and what publishing one commits you to
- [Write & publish an rSkill](tutorials/rskill/write-and-publish-an-rskill.md) — package a policy
- [Quantize an rSkill](tutorials/rskill/quantize-an-rskill.md) — fit a VLA on an 8 GB card
- [Deploy on a robot & open the dashboard](tutorials/deploy/deploy-run-and-dashboard.md) — real hardware

**Understand the system**

- [Architecture overview](architecture/overview.md)
- [Repo state map](architecture/repo-state-map.html) — interactive per-module status canvas (open in a browser; no build step)

**Plan & status**

- [Roadmap — Done / In flight / TODO](roadmap/index.md)

**Reference**

- [API](reference/api.md)
- [VLA × Robot × Sim compatibility](reference/vla_compatibility.md)
- [Sensor catalog & roadmap](reference/sensors_landscape.md)
- [Design decisions (ADRs)](decisions.md)
