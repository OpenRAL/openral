# Scene YAMLs (`scenes/`)

This directory holds the three scene tiers from [ADR-0041](../docs/adr/0041-scene-three-tier-hierarchy.md).
Each tier is a Pydantic schema in `openral_core`; the directory layout matches
the schema, the CLI matches the directory, and every YAML is loaded through a
strict per-tier loader (`load_scene_strict(..., expect=<Tier>)`) that rejects
wrong-tier YAMLs at parse time.

```
DeployScene  ⊆  SimScene  ⊆  BenchmarkScene
   scenes/deploy/   scenes/sim/   scenes/benchmark/
```

| Tier              | What it pins                                                                 | CLI consumer                                   | Output                                  |
|-------------------|------------------------------------------------------------------------------|------------------------------------------------|-----------------------------------------|
| `DeployScene`     | scene only (env-only playground; reasoner picks the rSkill at runtime)       | `openral deploy sim --config scenes/deploy/…`  | Live ROS graph (HAL + reasoner + kernel) |
| `SimScene`        | scene + task (single rollout; policy supplied via `--rskill <name>`)         | `openral sim run --config scenes/sim/…`        | One or more `EpisodeResult`s            |
| `BenchmarkScene`  | scene + task + paper metadata + `n_episodes` + `seed` (paper-comparable)     | `openral benchmark scene --config scenes/benchmark/…` | `RSkillEvalResult` JSON         |

Sibling resources:

- [`benchmarks/`](../benchmarks/) — suite-style benchmark YAMLs (bare
  `list[BenchmarkScene]` at the YAML root, ADR-0042) that aggregate multiple
  `BenchmarkScene`s under uniform invariants.
  Consumed by `openral benchmark run --suite <id> --rskill <name>`.
- [`deployments/`](../deployments/) — `RobotEnvironment` YAMLs for real
  hardware. Consumed by `openral deploy run`.

## Choosing a tier

| Question                                                                  | Tier              | Notes                                              |
|---------------------------------------------------------------------------|-------------------|----------------------------------------------------|
| "Boot the full stack so the reasoner can pick its own rSkill"             | `DeployScene`     | Env-only; no task; no eval.                        |
| "Run one rollout with a specific rSkill" / "save a debug video"           | `SimScene`        | Single CLI invocation; sized for ad-hoc / smoke.   |
| "Reproduce a paper number for this rSkill" (one scene)                    | `BenchmarkScene`  | Writes a citable `RSkillEvalResult` JSON.          |
| "Reproduce a paper number for this rSkill" (suite of N scenes)            | `list[BenchmarkScene]` | Lives in [`benchmarks/`](../benchmarks/) (ADR-0042). |

A scene can have a sibling YAML at multiple tiers — e.g. `scenes/benchmark/libero_spatial.yaml`
(paper protocol) and `scenes/sim/libero_spatial.yaml` (ad-hoc smoke). The sim-tier
sibling is **not** valid for paper claims; the loader-strictness gate
(`load_scene_strict`) prevents accidental tier confusion.

## Run any of them

```bash
# DeployScene — env-only playground (reasoner picks the rSkill at runtime).
openral deploy sim --config scenes/deploy/openarm_tabletop.yaml

# SimScene — single rollout with an explicit rSkill (and any override flag).
MUJOCO_GL=egl uv run --group libero \
    openral sim run --config scenes/sim/libero_spatial.yaml \
                    --rskill smolvla-libero --task libero_spatial/3

# BenchmarkScene — paper-comparable single-scene eval; writes
# rskills/<vla>/eval/<scene_id>.json with reproduced_locally=true.
MUJOCO_GL=egl uv run --group libero \
    openral benchmark scene --config scenes/benchmark/libero_spatial.yaml \
                            --rskill smolvla-libero

# Benchmark suite — multi-scene aggregate (lives in benchmarks/, not scenes/).
MUJOCO_GL=egl uv run --group libero \
    openral benchmark run --suite libero_spatial --rskill smolvla-libero
```

## Swap any axis

The CLI accepts override flags on every tier (the YAML pins defaults; the flag
wins). Most useful:

```bash
# Pick a different task in the same scene/suite.
openral sim run --config scenes/sim/libero_spatial.yaml \
                --rskill smolvla-libero \
                --task libero_spatial/3 \
                --instruction "pick up the alphabet soup"

# Cap episode length without editing the YAML.
openral sim run --config scenes/sim/libero_spatial.yaml \
                --rskill smolvla-libero --max-steps 80 --n-episodes 1

# Override a free-axis scene's robot (only legal where the scene is not
# scene-fixed — see the table below).
openral sim run --config scenes/sim/tabletop_cube_push.yaml \
                --rskill <id> --robot ur5e
```

`openral benchmark scene` accepts the same overrides; `openral benchmark run`
deliberately does not (it must reproduce the suite verbatim).

## Justfile shortcuts

```bash
# SimScene-tier — `openral sim run --save-video`.
just sim-libero                     # SmolVLA × LIBERO        (GPU + MUJOCO_GL)
just sim-xvla-libero                # xVLA × LIBERO           (Florence-2)
just sim-pi05-libero                # π0.5 × LIBERO           (≥8 GB VRAM)
just sim-act-libero                 # ACT × LIBERO            (paper protocol)
just sim-pi05-robocasa              # π0.5 × RoboCasa kitchen (≥8 GB VRAM)

# BenchmarkScene-tier — `openral benchmark scene --no-update-manifest --n-episodes 1`.
just sim-metaworld --task metaworld/reach-v3
just sim-maniskill3                 # SAPIEN-backed PickCube-v1
just sim-simpler-widowx             # RLDX-1 × WidowX carrot-on-plate
just sim-act-aloha                  # ACT × gym-aloha bimanual cube-transfer
just sim-diffusion-pusht            # Diffusion Policy × gym-pusht (CPU)
just sim-custom                     # ACT × gym-aloha insertion (rskills/act-aloha-insertion)
```

`just sim-audit` runs `tools/audit_sim_configs.py` over the full per-tier
catalogue and reports row-by-row latency + success metrics.

## Adding a new YAML

See [Create a sim environment](../docs/tutorials/sim/create-a-sim-environment.md)
for the long-form tutorial covering YAML authoring, adding a new robot
manifest, and writing custom scene / policy adapters.

Quick reference:

- A `DeployScene` is `scene:` only (+ optional `robot_id`, `base_pose`).
- A `SimScene` is a `DeployScene` + required `task:` block (+ optional `seed`,
  `n_episodes`, `record_video`).
- A `BenchmarkScene` is a `SimScene` + required `metadata: BenchmarkMetadata`
  (paper URL + honest_scope string) + non-`None` `seed` and `n_episodes`.

The loader (`load_scene_strict(path, expect=<Tier>)`) refuses to load a
wrong-tier YAML and tells you which tier the file actually fits. A YAML that
still carries a `vla:` block raises `ROSConfigError` — policy is always
supplied at the CLI via `--rskill <name>`.

## Available scene IDs (`scene.id`)

| Backend             | Built-in scene IDs                                                                                                                                                                                                                                                                          | Adapter file                              |
|---------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------|
| LIBERO              | `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, `franka_libero_custom_bddl` (custom BDDL → robosuite `OffScreenRenderEnv`; see [`sim/franka_libero_pnp.yaml`](sim/franka_libero_pnp.yaml))                                                                                       | `python/sim/.../backends/libero*.py`      |
| MetaWorld           | `metaworld` (passes `<env_id>` through to `gym.make`)                                                                                                                                                                                                                                       | `python/sim/.../backends/metaworld.py`    |
| gym-aloha           | `aloha_transfer_cube`, `aloha_insertion`                                                                                                                                                                                                                                                    | `python/sim/.../backends/aloha.py`        |
| gym-pusht           | `pusht` (2-D pymunk)                                                                                                                                                                                                                                                                        | `python/sim/.../backends/pusht.py`        |
| RoboCasa (MuJoCo)   | `robocasa` (procedural) + ~19 curated kitchen tasks (e.g. `robocasa/PickPlaceCounterToCabinet`) + 24 GR1 tabletop tasks (e.g. `robocasa/gr1/PnPCupToDrawerClose`)                                                                                                                            | `python/sim/.../backends/robocasa.py`     |
| ManiSkill3 (SAPIEN) | `maniskill3` (free-axis; passes `<env_id>` to `gym.make`)                                                                                                                                                                                                                                   | `python/sim/.../backends/maniskill3.py`   |
| SimplerEnv (SAPIEN) | `simpler_env` (Bridge V2 digital twin: 4 WidowX tasks on MS3 v3.0.x)                                                                                                                                                                                                                        | `python/sim/.../backends/simpler_env.py`  |
| Custom OpenArm      | `openarm_tabletop_pnp` (bimanual; default top camera matches the mddoai dataset POV)                                                                                                                                                                                                        | `python/sim/.../backends/openarm_*/env.py`|
| Custom SO-101       | `so101_box` (100 × 61.5 × 75 cm box arena + OAK-D Pro overhead RGB-D + wrist camera + tube-insertion task — geometry/sensors/spawn ranges configurable via `BoxSceneOptions`)                                                                                                               | `python/sim/.../backends/so101_box/env.py`|
| Custom tabletop     | `tabletop_push` (robot-agnostic cube push-to-goal; free-axis — pass `--robot`)                                                                                                                                                                                                              | `python/sim/.../backends/tabletop_push/env.py` |

`openral sim list` walks both subdirectories and prints every scene + every
in-tree rSkill (paste-able `--rskill` tokens).

## Scene-fixed robots

Some scenes hard-wire the physics robot via `@SCENES.register(..., fixed_robot=...)`:

| Scene                                            | Fixed robot         |
|--------------------------------------------------|---------------------|
| `libero_spatial` / `libero_object` / `libero_goal` / `libero_10` / `franka_libero_custom_bddl` | `franka_panda`    |
| `metaworld`                                      | `sawyer`            |
| `pusht`                                          | `pusht_2d`          |
| `aloha_transfer_cube` / `aloha_insertion`        | `aloha_bimanual`    |
| `so101_box`                                      | `so101_follower`    |
| `robocasa/*` (kitchen)                           | `panda_mobile`      |
| `robocasa/gr1/*` (humanoid tabletop)             | `gr1`               |

Passing `--robot` (or authoring `robot_id:` in a YAML) with a value that
disagrees with the scene's `fixed_robot` raises `ROSConfigError` at
config-build time — the message tells you which robot the scene requires.
Free-axis scenes (`tabletop_push`, `maniskill3`, `simpler_env`,
`openarm_tabletop_pnp`) leave `--robot` user-controlled.

## Placing robots in free-axis scenes (`base_pose:`)

Free-axis scenes accept an optional `base_pose:` block that anchors the robot
in the scene's world frame. Adapters write the `world → base_frame` transform
(from the robot manifest's `RobotDescription.base_frame`) into the scene's
MJCF at load. Example:

```yaml
robot_id: so100_follower

scene:
  id: tabletop_push                 # or any free-axis backend
  backend: mujoco

task:
  id: tabletop_push/0
  scene_id: tabletop_push
  instruction: ""

base_pose:
  xyz: [0.0, 0.0, 0.0]              # world-frame position (m)
  quat_xyzw: [0.0, 0.0, 0.0, 1.0]   # world-frame orientation
  frame_id: world
```

Setting `base_pose:` on a fixed-robot scene is a `ROSConfigError` — those
scenes ship their own MJCF and the field has no physical meaning there. See
[ADR-0002 Amendment 3](../docs/adr/0002-eval-and-sim-environments.md#2026-05-22-base_pose-for-free-axis-robot-mounting)
for the rationale.

## rSkill compatibility check

The runner resolves `--rskill <name>` to its `RSkillManifest`, looks up the
configured `RobotDescription`, and verifies the manifest's `embodiment_tags`
and `sensors_required` intersect the robot's capabilities and sensor catalogue
**before** policy load. The check fires inside
`openral_sim.runner._check_rskill_compatibility` and raises `ROSConfigError`
(missing manifest / unregistered robot) or `ROSCapabilityMismatch`
(incompatible) — there is no warn-and-proceed path.

See [ADR-0002](../docs/adr/0002-eval-and-sim-environments.md) for the original
design and [ADR-0041](../docs/adr/0041-scene-three-tier-hierarchy.md) for the
three-tier split that owns the loader strictness.

## Live MuJoCo viewer

`openral sim run` and `openral benchmark scene` open a passive `mujoco.viewer`
window by default and stream the rollout in real time. Toggle with
`--view / --no-view`. The viewer is auto-disabled (single WARNING line, no
error) when:

- `MUJOCO_GL=egl` is set, or
- on Linux with `DISPLAY` unset, or
- the scene's adapter doesn't expose `mujoco_handles()` (currently `pusht` —
  gym-pusht is 2D and not MuJoCo-backed).

Pass `--view` explicitly to fail loud instead of warn-and-continue. The window
survives episode resets within a run (re-opens against the post-reset
`MjModel` / `MjData`; LIBERO and other robosuite-backed envs allocate fresh
handles on each reset).

Per-adapter viewer support:

| Scene           | `mujoco_handles()` | Notes                                                                                                  |
|-----------------|--------------------|--------------------------------------------------------------------------------------------------------|
| `libero_*`      | ✅                 | Reach-through `lerobot.LiberoEnv._env.sim.{model,data}._{model,data}`.                                 |
| `metaworld`     | ✅                 | Direct `env.model` / `env.data` on the gymnasium env.                                                  |
| `aloha_*`       | ✅                 | dm_control `physics.{model,data}.ptr`.                                                                 |
| `openarm_*`     | ✅                 | Direct MJCF compile.                                                                                   |
| `so101_box`     | ✅                 | Direct MJCF compile.                                                                                   |
| `tabletop_push` | ✅                 | Direct MJCF compile.                                                                                   |
| `robocasa/*`    | ✅                 | robosuite physics handles.                                                                             |
| `pusht`         | ❌                 | gym-pusht is 2D; no MuJoCo. Runs offscreen even with default `--view`.                                 |
| `maniskill3`    | ❌                 | SAPIEN backend; use `--view` with SAPIEN's own GUI window (separate from `mujoco.viewer`).             |
| `simpler_env`   | ❌                 | Same — SAPIEN-backed.                                                                                  |

## Performance knobs (rSkill `policy_extras`)

Lerobot-style families (`smolvla`, `act`, `pi05`) honour two `policy_extras`
fields on their rSkill manifest that gate inference speed:

| Key                                | Default                                                                                  | What it does |
|------------------------------------|------------------------------------------------------------------------------------------|--------------|
| `n_action_steps`                   | Per-family paper default: SmolVLA / π0.5 = `chunk_size` (synchronous mode); ACT = `1` (per-step re-inference, pair with `temporal_ensemble_coeff`); Diffusion = `8` pinned in the adapter | Number of actions consumed from each predicted chunk before the policy re-infers. The shipped lerobot checkpoints set this to **1**, which for non-ACT families throws the chunk away and pays a full forward every env step. SmolVLA / π0.5 papers document `inference_mode: synchronous` (drain the full chunk); ACT's paper protocol is temporal ensembling (per-step re-inference + weighted average); Diffusion Policy fixes `n_action_steps=8` of a 16-step horizon. The adapters install the per-family paper default so `openral benchmark run` reproduces published numbers. Implemented in `openral_rskill._vla_core.apply_chunk_replay` and per-adapter `_build_*` factories. |
| `temporal_ensemble_coeff` (ACT)    | `0.01` (paper value, Zhao et al. §V-B)                                                   | Engages ACT's temporal-ensembling buffer. Set to `null` to disable (falls back to plain chunked execution). On `gym_aloha/AlohaTransferCube-v0` the published `lerobot/act_aloha_sim_transfer_cube_human` checkpoint runs ~0.46 with TE disabled and approaches the paper's 0.95 with TE on. Implemented in `openral_sim.policies.act._apply_temporal_ensemble`. |
| `compile`                          | `false`                                                                                  | Opt-in `torch.compile` of the heavy chunk forward (`policy._get_action_chunk`). Skipped on CPU. Best-effort: setup or runtime backend errors degrade to eager and log `vla_compile_setup_failed` / `vla_compile_runtime_fallback` (see `openral_rskill._vla_core.maybe_compile_chunk_forward`). **Not exposed for `pi05`** — that adapter forces `compile_model = False` to keep the nf4 quantization path stable. |
| `compile_mode`                     | `"default"`                                                                              | Torch compile mode: `default`, `reduce-overhead` (CUDA graphs, recommended), `max-autotune` (longest warmup). |

### When to enable `compile`

It is **off by default** because it requires:

- A working system C compiler on `$PATH` for Triton (compile fails if Triton
  picks up an incompatible compiler — common with conda envs that shadow `cc`).
  Workaround: prefix with `CC=/usr/bin/gcc openral sim run …`.
- Roughly **3 GB of free VRAM** above the policy weights for Inductor's
  allocations. SmolVLA on an 8 GB laptop GPU OOMs if other GPU processes are
  resident.
- A budget for a **~30 s warmup on the first chunk inference**. The compiled
  module survives `policy.reset()`, so the warmup is paid **once per process**.

Measured on an RTX 4070 Laptop, `scenes/sim/libero_spatial.yaml`,
`--max-steps 200 --n-episodes 3`:

| Config                                                                         | Mean step latency | Notes                                                                                                                                                |
|--------------------------------------------------------------------------------|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| Shipped lerobot default (`n_action_steps=1`, no compile)                       | 324 ms            | Full SmolVLA forward every env step.                                                                                                                 |
| SmolVLA / π0.5 adapter default (`n_action_steps=50`, no compile)               | **13 ms**         | ~25× speedup; one heavy chunk forward per 50 steps. Paper-faithful "synchronous" mode — what `openral benchmark run` uses for SmolVLA / π0.5.        |
| Explicit `n_action_steps: 25` (rSkill manifest)                                | **25 ms**         | ~13× speedup; one heavy chunk forward per 25 steps. Two re-plans per chunk — trades fidelity for closed-loop reactivity.                             |
| `+ compile: true` (steady state, ep ≥ 1)                                       | **~8 ms**         | ~40× total speedup. ep0 mean is dominated by the warmup.                                                                                             |
| ACT adapter default (`n_action_steps=1`, `temporal_ensemble_coeff=0.01`)       | ~16 ms (warm)     | Per-step re-inference + TE buffer; paper protocol for ACT. Slower per step than the chunked SmolVLA modes but ACT is tiny (52 M params).             |

For one-shot debug rollouts, leave `compile: false` — the warmup eats the
win. For evaluations (`--n-episodes >= 2`, or long episodes) it is pure win.

## Discovering paste-able `--rskill` strings

`openral sim list` prints scenes, paste-able `--rskill` strings, and robots.
The listing's `rskills:` line is generated from `rskills/<dir>/rskill.yaml`
files in the repo; copy any token straight into `--rskill` (e.g.
`--rskill pi05-libero-nf4`).
