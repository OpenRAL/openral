# METHODS.md — Public Symbol Inventory (index)

> **Last cleaned: 2026-09-22** (openral PR #296).
>
> **Purpose.** A layer-ordered list of every class, function, method and
> module-level constant under `python/`, `packages/` and `tools/`, one line
> each, so you can find an existing helper before writing a new one.
>
> **How to search.** `grep -rn <symbol> docs/methods/`. Add, rename, move or
> remove a public symbol → update the matching `docs/methods/` file in the
> same PR.
>
> **Not normative.** The contracts are the `openral_core` schemas and the
> `openral_msgs` IDL; a stale entry here is a defect, not a source of truth.
> `tools/refresh_methods_linenos.py` refreshes the `(LNN)` line markers;
> `--check --coverage` (run by `just lint`) fails on a stale marker, an
> unresolvable entry, or a public symbol with no entry.
>
> **Format.** `name(args) -> ret` — one-line description. `(LNN)` is the
> source line. Decorators are tagged `[@…]`; Pydantic fields are listed
> inline so cross-model duplication is visible.

---

## Inventory files

| File | Scope |
|---|---|
| [00-core-schemas.md](methods/00-core-schemas.md) | Layer 0 — `openral_core` Pydantic schemas, loaders, URDF resolve, exception hierarchy |
| [01-hal.md](methods/01-hal.md) | Layer 1 — HAL Protocol + every robot adapter (real, MuJoCo, sim-attached, lifecycle, transports) |
| [02-sensors.md](methods/02-sensors.md) | Layer 2 — sensor catalog, `SensorSpec`/`SensorBundle` factories, ROS publisher, reader protocol |
| [03-world-state.md](methods/03-world-state.md) | Layer 3 — state adapter registry, world-state aggregator, spatial memory, geometry/grid, object lift |
| [04-rskill.md](methods/04-rskill.md) | Layer 4 — rSkill ABC, runtimes (PyTorch/ONNX; TensorRT via the private OpenRAL Pro plugin), loader, executor, VLA adapters |
| [05-inference-runner.md](methods/05-inference-runner.md) | Inference Runner — clocks, runner loop, sensor readers, dataset recording |
| [06-reasoning-wam-safety-observability.md](methods/06-reasoning-wam-safety-observability.md) | CLAUDE.md layers 4–6 — Reasoner core/tool-use, safety supervisor, observability |
| [07-eval-sim.md](methods/07-eval-sim.md) | Eval (sim) — scene/robot registries, SimRunner, scene + policy adapters, benchmark suites |
| [08-cli.md](methods/08-cli.md) | CLI — `openral` command tree |
| [09-auto-provisioning.md](methods/09-auto-provisioning.md) | Auto-provisioning (detection) — `python/detect` hardware probes (USB/CAN/DDS/GPU/camera/network), `RobotDescription` assembly, `openral detect`, rSkill compatibility checks |
| [10-tools.md](methods/10-tools.md) | Tools — `tools/*.py` dev utilities (quantization, sidecars, publishers, this file's refresher) |
| [11-ros2-nodes.md](methods/11-ros2-nodes.md) | ROS 2 lifecycle nodes (`packages/`) |
| [12-tests-hil.md](methods/12-tests-hil.md) | Tests · HIL bridges |
| [13-tests-sim-helpers.md](methods/13-tests-sim-helpers.md) | Tests · sim helpers |
| [14-duplication-watch.md](methods/14-duplication-watch.md) | **Duplication & Reuse Watch** — confirmed redundancy candidates; recheck before every PR |
