# OpenRAL

An open-source operating layer for embodied AI, **OpenRAL** unifies fast policies, slow reasoning, and classical control into one typed, traceable, safety-first runtime for deployable robot agents.

## Quick start

```bash
git clone https://github.com/OpenRAL/openral && cd OpenRAL
just bootstrap          # installs uv, ROS 2, system deps
uv sync --all-packages  # install Python workspace
uv run openral doctor        # verify your environment
just test               # run the test suite
```

## What does it do?

- Load any VLA (SmolVLA, π0, GR00T N1, OpenVLA) on any robot (SO-100, G1, UR5e).
- Type-safe, layer-isolated architecture — HAL → Sensors → World State → Skill → Reasoning → Safety → Observability.
- Skill packaging on HuggingFace Hub with signature verification.
- Full OpenTelemetry traces per execution.
- LeRobotDataset v3 flywheel — every execution becomes a training row.

## Navigation

**Get started**

- [Quickstart — `openral doctor`](quickstart/index.md)
- [Development setup](contributing/development.md)

**Understand the system**

- [Architecture overview](architecture/overview.md)
- [Repo state map](architecture/repo-state-map.html) — interactive per-module status canvas (open in a browser; no build step)

**Plan & status**

- [Roadmap — Done / In flight / TODO](roadmap/index.md)

**Reference**

- [API](reference/api.md)
- [VLA × Robot × Sim compatibility](reference/vla_compatibility.md)
- [Sensor catalog & roadmap](reference/sensors_landscape.md)
- [ADRs](adr/0001-record-architecture-decisions.md)
