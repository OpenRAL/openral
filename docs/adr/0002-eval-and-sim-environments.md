# ADR-0002: Configurable sim environments for rSkill validation

- Status: Accepted
- Date: 2026-05-24
- Amended: 2026-05-24 (see Amendments below)

## Context

CLAUDE.md §6 lists `python/eval/` as a planned package and §7.4 mandates that
every rSkill go through sim before hardware.  In practice, every example
under `examples/` was a hand-written `run.py` with a robot, scene, task, and
VLA hard-baked into one file — five scripts total at the time of writing
(`smolvla_libero_eval`, `pi05_libero_eval`, `xvla_libero_eval`,
`smolvla_metaworld_eval`, `so100_smolvla_smoketest`).  Switching any one of
those four axes meant copying a script.  That makes pre-deployment rSkill
validation tedious, error-prone, and impossible to drive from CI matrices.

The pieces we already had:

- `RobotDescription` + `RSkillManifest` Pydantic schemas in `openral_core`.
- `SO100DigitalTwin` and `SimTransport` in `openral_hal`.
- The `lerobot[libero]` and `lerobot.envs.metaworld` adapters living inline
  in each example script.

Pieces we did NOT have:

- A typed contract for "scene", "task", or "the swappable triple".
- A factory that, given a config, returns a runnable env + policy.
- A registry that lets contributors add new sim backends without touching
  the runner.
- A CLI that drives the above (the `ral` tool only had install / list).

## Decision

Introduce a new top-level Python package — `openral_eval` — and four new
Pydantic schemas in `openral_core`:

```
openral_core.PhysicsBackend          (enum)
openral_core.SceneSpec               (where the action happens)
openral_core.TaskSpec                (what the robot must achieve)
openral_core.VLASpec                 (which brain drives the body)
openral_core.SimEnvironment          (robot_id × scene × task × vla)
```

`SimEnvironment` is YAML-loadable and self-validates (`task.scene_id` must
equal `scene.id`).  Configs live under `examples/configs/`.  The eval
package exposes three registries (`SCENES`, `POLICIES`, `ROBOTS`) and three
public functions (`make_env`, `make_policy`, `run_evaluation`).  Built-in
adapters live under `openral_eval.adapters/` and lazy-import their
backends so installing the eval package does not pull `lerobot[libero]` or
torch transitively.

A new console entry point `ral-eval` (registered by
`openral-eval/pyproject.toml`) drives evaluations from either a YAML
config or explicit `--robot/--scene/--task/--vla` flags, and a new Justfile
target `just sim-eval <config>` is the canonical CI-friendly invocation.

### Why a registry, not entry-points?

Entry-point discovery (`importlib.metadata.entry_points`) lets third-party
packages register adapters without modifying this repo, but at the cost of
import-time complexity and surprising failure modes.  v0.1 ships a flat
`{name: factory}` dict; we can promote to entry-points later without
breaking the public API since `_Registry` already encapsulates lookup.

### Why decouple `Scene` from `Task`?

LIBERO ships 10 tasks per suite × 4 suites = 40 task / scene pairings; the
scene assets are reused across tasks.  Encoding both in a single
`SimEnvironment` field would force duplication.  Keeping them separate also
matches how LIBERO and MetaWorld already think (suite ↔ task).

### Why does `VLASpec` live in `openral_core`, not `openral_rskill`?

`VLASpec` is a serialisation/eval-config concern — it points at weights and
sets a runtime override.  The runtime `Skill` ABC and `RSkillManifest` stay
where they are.  `VLASpec.weights_uri` accepts bare rSkill references (name, path, or HF repo ID)
so existing rSkills plug in without duplication.

## Consequences

- **Pros**
  - One `examples/configs/*.yaml` per (robot, scene, task, vla) tuple
    replaces the ad-hoc per-example `run.py` scripts.
  - rSkill manifests can be validated in sim by pointing at one or more
    `SimEnvironment` configs — this is the gate we want before hardware.
  - Adding LIBERO suites, MetaWorld tasks, or new VLAs is a one-decorator
    change in `openral_eval.adapters`, no runner changes.
  - The eval package itself has only three runtime deps
    (`openral-core`, `openral-rskill`, `PyYAML`, `structlog`); heavy
    backends remain opt-in.
  - Schema changes are SemVer-tracked: `openral-core` 0.1.0 → 0.2.0.

- **Cons**
  - One more package in the workspace, one more layer to learn.
  - Existing `examples/*_eval/run.py` scripts now duplicate logic with the
    eval adapters.  We keep them for one minor release as a migration
    reference, then remove them.
  - The `success_key` indirection means scene authors must surface success
    in `info[...]`; this is already the gym convention but is now load-bearing.

## Migration

- `openral-core` 0.1.0 → 0.2.0; `tools/schema_export.py` re-runs cover
  the four new models.  No existing fields removed; this is purely additive.
- `examples/configs/` is the new canonical home.  `examples/*_eval/run.py`
  remain runnable but are deprecated; they will be removed once every
  Justfile sim target points at `ral-eval`.

## Future work (not in this ADR)

- ~~Register `RobotDescription` factories in `ROBOTS` so the runner can do
  capability-vs-rSkill checks before launching.~~ — landed in commit
  `f528f0b` (*feat(eval): wire RobotDescription manifests + rSkill compat
  into examples*); the canonical manifests now live under `robots/`.
- Persist `EpisodeResult` to OTel + LeRobotDataset v3 for the dataset
  flywheel (RFC §10).
- Add an `RSkillManifest.sim_validation` field listing required
  `SimEnvironment` configs; gate `ral skill install` on their success rate.
- Wire MJX (GPU-batched) as a `PhysicsBackend.MUJOCO_MJX` adapter for
  large-scale evaluation.

## Amendments

### 2026-05-08 — Reconciliation with shipped state

The "Cons" section originally claimed:

> Existing `examples/*_eval/run.py` scripts now duplicate logic with the
> eval adapters. We keep them for one minor release as a migration
> reference, then remove them.

In practice the per-example `run.py` scripts are still on disk
(`examples/{smolvla_libero,pi05_libero,xvla_libero,smolvla_metaworld}_eval/run.py`)
and the Justfile still drives them directly via
`just sim-libero` / `just sim-pi05-libero` / `just sim-xvla-libero` /
`just sim-metaworld`. Removal is **deferred** until each `run.py`'s
behaviour is fully expressible as a `SimEnvironment` config plus a
`ral-eval` invocation; until then the scripts double as integration
fixtures and CI smoke targets.

The `ral-eval` config-driven path (`just sim-eval <config>`) is the
canonical entry point for new evaluations.

### 2026-05-09 — Migration progress

Partial migration has now landed:

- `examples/pi05_libero_eval/` and `examples/so100_smolvla_smoketest/`
  have been **removed**. `just sim-pi05-libero` now invokes
  `ral-eval --config examples/configs/pi05_libero_spatial.yaml`; there is
  no longer a wiring/latency `just sim so100` smoketest (its job is
  covered by `tests/hil/test_so100.py` and the unit/integration HAL
  tests).
- New `SimEnvironment` configs added: `act_aloha_transfer_cube.yaml`,
  `diffusion_pusht.yaml`, `xvla_libero_spatial.yaml`. Two new
  `RobotDescription`s landed (`robots/aloha_bimanual/`,
  `robots/pusht_2d/`) and three new in-tree rSkills
  (`rskills/act-aloha/`, `rskills/diffusion-pusht/`, `rskills/pi05-so100/`).
- New eval adapters: `act.py`, `aloha.py`, `diffusion.py`, `pi05.py`,
  `pusht.py`, `xvla.py` plus the shared `_video_capture.py`. The
  per-example `_viz.py` modules are gone, replaced by `examples/_video.py`
  (subsequently moved into `openral_eval._video` — see the 2026-05-10
  "moved into the eval package" amendment below).
- Two new Justfile recipes — `just sim-act-aloha`,
  `just sim-diffusion-pusht` — exercise the new configs through
  `ral-eval --config ... --save-video`.

The remaining `examples/{smolvla_libero,xvla_libero,smolvla_metaworld}_eval/run.py`
scripts continue to back the corresponding `just sim-*` recipes — they
still expose per-policy debug-video / latency tooling that `ral-eval`
does not yet surface.

### 2026-05-10 — Migration complete

The remaining `examples/{smolvla_libero,xvla_libero,smolvla_metaworld}_eval/run.py`
wrappers have been **removed**. `ral-eval --save-video` now covers the
3-panel debug-MP4 path through `ral-eval._write_videos` →
`openral_eval._video:save_episode_mp4` (originally
`examples/_video.py`; relocated into the eval package on 2026-05-10),
so the per-example wrappers no longer add anything. `just sim-libero`,
`just sim-xvla-libero`, and `just sim-metaworld` invoke
`ral-eval --config examples/configs/<...>.yaml --save-video`; after the
relocation, the only artifact under `examples/` is `configs/` (plus a
top-level `README.md`).

Two operational consequences land with the same change:

- `ral-eval` now calls `configure_observability(service_name="ral-eval")`
  at startup and `shutdown_observability()` in a `finally` block. With
  `OTEL_EXPORTER_OTLP_ENDPOINT` set (e.g. `http://localhost:4317` for
  the docker-compose Jaeger), spans reach the collector before the
  process exits — earlier the example wrappers never initialised OTel,
  so `examples/smolvla_libero_eval/run.py` produced no traces no matter
  what env var was set.
- `openral_observability.shutdown_observability()` is the new public
  symbol that performs the flush + provider shutdown; it is also
  registered via `atexit` on first successful `configure_observability`
  call so callers that forget the explicit invocation still flush.

### 2026-05-10 — `_video.py` moved into the eval package

The shared 3-panel debug-MP4 helper has been **moved** from
`examples/_video.py` into `python/eval/src/openral_eval/_video.py`
and re-exported from `openral_eval` as `save_episode_mp4`. The
`examples/` tree now holds only `configs/` (per the 2026-05-10
"Migration complete" amendment above) plus a `README.md`.

Why
- The CLI (`ral-eval --save-video`) was the only consumer and was
  importing it via a `sys.path.insert(...)` hack against
  `<repo_root>/examples/`. Co-locating the helper with `EpisodeResult`
  (the type it consumes) removes the hack and makes the symbol
  available as a clean `from openral_eval import save_episode_mp4`
  for anyone wiring rollout video into their own driver.
- Decision text above is unchanged. This is an additive amendment per
  CLAUDE.md §7.9 (ADRs may be amended additively for factual
  reconciliation).

### 2026-05-11 — `ral-eval` folded into `ral` as `ral eval`

The standalone `ral-eval` console script and the `python -m openral_eval`
entry point have been **removed**. The same Typer app
(`openral_eval.cli.eval_app`) is now mounted on the top-level `ral`
Typer tree, so the canonical invocation is `ral eval --config …` (and the
8 `just sim-*` recipes have been updated accordingly).

Why
- Discoverability: `openral --help` now lists `eval` alongside `doctor`,
  `detect`, `connect`, `calibrate`, `skill`, `sensor`, `benchmark` —
  users no longer need to know that `ral-eval` was a separate binary.
- Single framework: the eval CLI was the only argparse holdout; porting
  to Typer (per the same PR) collapses the framework boundary and lets
  the existing `typer.testing.CliRunner`-based tests cover both surfaces.
- No new deps: `openral-cli` now depends on `openral-eval`
  (one-way ascending, no layer violation). The eval module's heavy sim
  imports (torch / mujoco / gymnasium / lerobot) stay lazy inside
  `_run()` so `openral doctor` startup is unaffected — guarded by
  `tests/unit/test_cli_eval.py::test_bh_cli_import_is_light`.

What's preserved
- All flag names and semantics are identical. The argparse
  `nargs="?"` shape of `--save-video` (bare flag → default directory)
  doesn't map cleanly into Typer, so the Justfile recipes now pass
  `--save-video example_videos` explicitly. User-supplied paths still
  win.
- `openral_eval.cli.main(argv)` stays as an in-process helper so
  the unit tests can drive the CLI without spawning subprocesses.
- The OTel service name remains `ral-eval` for trace continuity.

Decision text above (the ADR-0002 "Decision" section) is unchanged.
This is an additive amendment per CLAUDE.md §7.9.

### 2026-05-11 — sim/benchmark split (ADR-0009)

[ADR-0009](0009-separate-sim-and-benchmarking.md) splits the "eval"
responsibility this ADR introduced into two clearly named subsystems:

- The `SimEnvironment` schema, free-axis runner, and YAML directory
  are renamed to `sim` (`ral eval` → `openral sim run`, `openral_eval` →
  `openral_sim`, `examples/configs/` → `scenes/`).
- A new `BenchmarkSpec` schema in `openral_core` and a new
  `openral benchmark run` command own the fixed-axis-except-VLA case.
  Output: a validated `RSkillEvalResult` written to
  `rskills/<vla>/eval/<id>.json` with `reproduced_locally: true`,
  closing the "reproduction deferred — use external `lerobot-eval`"
  loop the original eval JSONs document.

ADR-0002's Decision text — `SimEnvironment`, `SceneSpec`, `TaskSpec`,
`VLASpec`, the three registries, the lazy-import discipline — is
preserved verbatim; ADR-0009 supersedes only the naming and the
absence of a benchmark suite object. Migration is phased; see
ADR-0009 §Migration.

### 2026-05-16 — `run_evaluation` retired; SimRunner is the loop

The original Decision text introduced three public functions —
`make_env`, `make_policy`, and `run_evaluation` — as the sim API. The
first two are unchanged. `run_evaluation` (and its inner
`run_episode`) have been deleted and replaced by
`openral_sim.SimRunner` (in `python/sim/src/openral_sim/sim_runner.py`),
a per-step `InferenceRunner` subclass that shares the loop, OTel
spans, and `TickResult` / `RunResult` shape with
`openral_runner.DeployRunner` (ADR-0010 amendment 1).

Callers go from:

```python
results = run_evaluation(env_cfg)
```

to:

```python
runner = SimRunner(env_cfg)
runner.activate()
runner.run(max_ticks=env_cfg.n_episodes * (env_cfg.task.max_steps + 1))
results = runner.episode_results
runner.deactivate()
```

`save_episode_mp4` and the rest of ADR-0002's surface are unchanged.
The `openral_eval` one-release deprecation shim referenced in this
ADR's earlier amendments was removed in the same PR — by the time the
unification landed the one-release clock had elapsed.

### 2026-05-22 — `base_pose` for free-axis robot mounting

The original ADR-0002 schemas left robot placement implicit: scene
adapters either embedded a fixed mounting pose in their MJCF (LIBERO,
MetaWorld, RoboCasa) or applied an URDF identity placement (mock,
maniskill3, simpler_env). There was no per-rollout knob — the same
`(scene_id, robot_id)` pair always produced the same
world → `base_frame` transform.

This was fine while every scene was either fixed-robot or owned by a
single demo, but it blocks two real cases:

1. The same free-axis scene used by two different robots: each has a
   different physical base size, so `pos="0 0 0"` is correct for one
   and unreachable for the other.
2. The same `(scene_id, robot_id)` used in two rollouts with different
   tabletop layouts (e.g. moving the robot 20 cm forward to clear a
   new object footprint).

Both were workable only through scene-side back doors like the
`backend_options.robot_lift_z` / `robot_forward_x` keys that sibling
branches have used on robosuite scenes — no schema-level contract, no
migration path.

#### Decision

Add `base_pose: Pose6D | None = None` to both `SceneEnvironment`
(YAML-side, `schemas.py:2420`) and `SimEnvironment` (runtime-side,
`schemas.py:2343`). The CLI composer (`_load_or_build_env` in
`python/sim/src/openral_sim/cli.py`) copies it through. Constraints
are enforced at compose-time:

- `base_pose` on a scene where `SCENES.fixed_robot(scene_id) is not
  None` raises `ROSConfigError` — those scenes ship their own MJCF and
  the field has no physical meaning there.
- `base_pose` on a free-axis scene is the world → `base_frame`
  transform for that rollout. The pose's `frame_id` is documented as
  `"world"`; adapters anchor on the robot manifest's existing
  `RobotDescription.base_frame` field — no robot-side schema change
  is needed.
- Default `None` preserves backward compatibility: every YAML under
  `scenes/` validates unchanged
  (`tests/unit/test_examples_sim_configs_load.py` is the gate).

#### Why `SceneEnvironment` and not `SceneSpec`

`SceneSpec` is the scene **definition** (assets, cameras, observation
size) — reusable across rollouts. A robot mounting pose is per-rollout
state, not scene-definition state. Co-locating `base_pose` with
`robot_id` on `SceneEnvironment` matches the symmetry of "this robot,
at this pose, in this scene." Two YAMLs sharing a scene id can declare
different `base_pose`s without redeclaring the scene.

#### Why no schema-version bump

The field is additive with a `None` default at the schema level — a
YAML that omits `base_pose` validates against `SceneEnvironment`
exactly as before, and free-axis scenes that don't need a non-trivial
mounting (mock, maniskill3, simpler_env) treat `None` as "use the
adapter's native placement". Scenes that *do* need a non-trivial
pose (currently just `openarm_tabletop_pnp`) raise
`ROSConfigError` at compose time when `base_pose` is unset.

#### openarm_robosuite — first adapter user

The openarm_robosuite adapter
(`python/sim/src/openral_sim/backends/openarm_robosuite/env.py`)
**requires** `env_cfg.base_pose` to be set on the YAML. The resolver
extracts `(x, y, z)` from the typed `Pose6D` and rejects non-zero `y`
and non-identity quaternion (the underlying `_lift_robot_bases`
helper is translation-only).

There is **no back-compat path**: the previous
`scene.backend_options.robot_lift_z` / `robot_forward_x` keys and
the hand-tuned `(0.55, 0.20)` defaults that the adapter used to
apply silently are both removed. A YAML that omits `base_pose` for
this scene fails loud with a `ROSConfigError` that includes the
canonical pose snippet. The three in-tree
`scenes/openarm_*.yaml` files carry the table-clearance pose
(`xyz: [0.20, 0.0, 0.55]`) explicitly.

Full 6-DOF mounting (Y translation + rotation) on openarm is a future
extension once the MJCF helper learns to rotate the bases.
