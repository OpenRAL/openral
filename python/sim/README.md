# openral-sim

OpenRAL sim eval — robot × scene × task × VLA factory and runner for rSkill validation before hardware deployment

Part of [**OpenRAL**](https://github.com/OpenRAL/openral) — the open Robot
Abstraction Layer for vision-language-action robotics. This package is one
member of the OpenRAL Python workspace; see the architecture overview and the
seven-layer model in the project docs.

- **Docs:** https://docs.openral.com/
- **Source:** https://github.com/OpenRAL/openral
- **License:** Apache-2.0

## Isaac Sim timing

Isaac scenes keep the configured physics dt and substep count for control. The
sidecar reports that nominal period on `ping`; after reset warmup it reports the
measured PhysX post-step callback interval multiplied by the configured substeps
as `sim_dt_per_tick_s`. That callback interval advances the same
`World.current_time` used for observation stamps. Deploy HAL validates and hands
the reset value to the self-filter before camera evidence is admitted. If the
callback cadence is unavailable or invalid, reset fails closed; no nominal
substitution or tolerance is used for the filter period.

> All OpenRAL workspace packages move in lockstep with the workspace SemVer
> version; the bump is computed from Conventional Commits by release-please.
