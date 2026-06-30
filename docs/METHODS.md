# METHODS.md — Public Symbol Inventory (index)

> **Last cleaned: 2026-06-11** — split the single-file inventory into
> per-layer files under [`docs/methods/`](https://github.com/OpenRAL/openral/tree/master/docs/methods/) and refreshed every
> `(LNN)` citation via `tools/refresh_methods_linenos.py`. (Previous pass
> 2026-05-16: repo-wide de-slop, ADR renumber, `skills/` → `rskills/`.)
>
> **Purpose.** A flat, layer-ordered list of every class, function, method,
> and module-level constant defined in the OpenRAL Python source tree
> (`python/`, `packages/`, `tools/`), with signatures and one-line
> descriptions, intended as a duplication / redundancy detector.
>
> **How to search.** Grep the folder, not this index:
> `grep -rn <symbol> docs/methods/`. Before writing a new helper, search
> here first (CLAUDE.md §1.13); add/rename/move/remove a public symbol →
> update the matching `docs/methods/` file in the same PR.
>
> **This inventory is hand-curated** (generated once via `ast` and then
> reorganised by hand). It is **not** normative — the authoritative
> contracts remain `openral_core` schemas (Pydantic) and `openral_msgs`
> IDL (per CLAUDE.md §1.3). When code drifts, the inventory drifts. Treat
> a stale entry as a defect, not a source of truth. `(LNN)` line markers
> are kept fresh with `python tools/refresh_methods_linenos.py`
> (`--check` reports drift without writing).
>
> **Format.** `name(args) -> ret` — first docstring line. `(LNN)` is the
> source line number. Decorators are tagged in `[@…]`. Pydantic field
> attrs are listed inline so cross-model duplication is visible.

---

## Inventory files

| File | Scope |
|---|---|
| [00-core-schemas.md](methods/00-core-schemas.md) | Layer 0 — `openral_core` Pydantic schemas, loaders, URDF resolve, exception hierarchy |
| [01-hal.md](methods/01-hal.md) | Layer 1 — HAL Protocol + every robot adapter (real, MuJoCo, sim-attached, lifecycle, transports) |
| [02-sensors.md](methods/02-sensors.md) | Layer 2 — sensor catalog, `SensorSpec`/`SensorBundle` factories, ROS publisher, reader protocol |
| [03-world-state.md](methods/03-world-state.md) | Layer 3 — state adapter registry, world-state aggregator, spatial memory, geometry/grid, object lift |
| [04-rskill.md](methods/04-rskill.md) | Layer 4 — rSkill ABC, runtimes (PyTorch/ONNX/TensorRT), loader, executor, VLA adapters |
| [05-inference-runner.md](methods/05-inference-runner.md) | Inference Runner (ADR-0010) — clocks, runner loop, sensor readers, dataset recording |
| [06-reasoning-wam-safety-observability.md](methods/06-reasoning-wam-safety-observability.md) | Layers 5–8 — Reasoner core/tool-use, WAM, safety supervisor, observability |
| [07-eval-sim.md](methods/07-eval-sim.md) | Eval (sim) — scene/robot registries, SimRunner, scene + policy adapters, benchmark suites |
| [08-cli.md](methods/08-cli.md) | CLI — `openral` command tree |
| [09-auto-provisioning.md](methods/09-auto-provisioning.md) | Auto-provisioning (detection) — GStreamer perception bus, detector tiers |
| [10-tools.md](methods/10-tools.md) | Tools — `tools/*.py` dev utilities (quantization, sidecars, publishers, this file's refresher) |
| [11-ros2-nodes.md](methods/11-ros2-nodes.md) | ROS 2 lifecycle nodes (`packages/`) |
| [12-tests-hil.md](methods/12-tests-hil.md) | Tests · HIL bridges |
| [13-tests-sim-helpers.md](methods/13-tests-sim-helpers.md) | Tests · sim helpers |
| [14-duplication-watch.md](methods/14-duplication-watch.md) | **Duplication & Reuse Watch** — confirmed redundancy candidates; recheck before every PR |
