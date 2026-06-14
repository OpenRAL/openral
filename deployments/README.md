# `deployments/` — real-hardware `RobotEnvironment` configs

This is the on-disk home for **`RobotEnvironment`** YAMLs consumed by
`openral deploy run` (the real-hardware sibling of `openral sim run`). It is the
deployment counterpart to [`scenes/`](../scenes/README.md), which holds the
`SceneEnvironment` configs for simulation.

```
scenes/        SceneEnvironment  → openral sim run / openral deploy sim   (simulation)
deployments/   RobotEnvironment  → openral deploy run                 (real hardware)
```

A `RobotEnvironment` pins a real deployment: `(robot × HAL × sensors × task ×
VLA × safety)`. Per **ADR-0031**, `deploy run` is **real-hardware only** —
`build_runner` loads `robots/<robot_id>/robot.yaml` and constructs the HAL via
`openral_hal.build_hal(mode="real")` (the manifest's `hal.real` is the single
source of truth). A simulation-only robot raises `ROSCapabilityMismatch`; use
`openral deploy sim` with a `scenes/` config instead.

```bash
# List the deployment configs here (paste-able --config paths):
openral deploy list

# Run one against connected hardware:
openral deploy run --config deployments/<your-deployment>.yaml
```

`openral deploy list` walks this directory; it prints `<none>` until a deployment
config is committed here. Deployment configs are intentionally **not** shipped
in the open-core tree by default — they pin a specific lab's robot IP / FCI port
/ camera serial, which is site-specific. Add yours here following the
`RobotEnvironment` schema in `python/core/src/openral_core/schemas.py`.
