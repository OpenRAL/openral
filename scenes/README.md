# Scene YAMLs (`scenes/`)

Three scene tiers, each a Pydantic schema in `openral_core`. The directory matches the schema, the CLI command matches the directory, and every YAML loads through `load_scene_strict(path, expected=<Tier>)`, which rejects a wrong-tier YAML at parse time and names the tier the file fits.

```
DeployScene  ⊆  SimScene  ⊆  BenchmarkScene
scenes/deploy/   scenes/sim/   scenes/benchmark/
```

The per-file catalogue, the benchmark suite table and the Isaac, placement-grading and deploy-scene rules are in [`docs/reference/sim-environments.md`](../docs/reference/sim-environments.md). Authoring a new scene, robot or adapter: [Create a sim environment](../docs/tutorials/sim/create-a-sim-environment.md).

## Layout

| Path | Contents |
| --- | --- |
| `scenes/deploy/` | `DeployScene` YAMLs, plus `calibration/`, `drivers/` and `place_zones/` (paste-in lists) |
| `scenes/sim/` | `SimScene` YAMLs, plus `assets/` for scenes that ship their own files |
| `scenes/benchmark/` | `BenchmarkScene` YAMLs, one paper-protocol scene each |
| [`benchmarks/`](../benchmarks/) | Suite YAMLs: a bare `list[BenchmarkScene]` aggregating scenes under uniform invariants |

`deployments/` is retired; real deploys use `scenes/deploy/*.yaml`.

## The three tiers

| Tier | Pins | Command | Output |
| --- | --- | --- | --- |
| `DeployScene` | Workcell context: scene, robot, safety tightening, additive collision pairs. No task. | `openral deploy sim` or `openral deploy run --config scenes/deploy/...` | A live ROS graph (HAL, reasoner, kernel); the reasoner picks the rSkill |
| `SimScene` | Scene + one `task:`. The policy comes from `--rskill`. | `openral sim run --config scenes/sim/...` | `EpisodeResult`s; `summary.json` with `--save-dir` |
| `BenchmarkScene` | A `SimScene` + `metadata` (paper, `honest_scope`), `seed` and `n_episodes` | `openral benchmark scene --config scenes/benchmark/...` | `rskills/<dir>/eval/scene_<scene_id>.json` |
| Suite | N `BenchmarkScene`s | `openral benchmark run --suite <id>` | `rskills/<dir>/eval/<suite_id>.json` |

A scene can exist at several tiers, such as `scenes/benchmark/libero_spatial.yaml` (paper protocol) and `scenes/sim/libero_spatial.yaml` (ad-hoc smoke). The sim-tier file is not valid for paper claims; the loader prevents tier confusion.

## Run any of them

```bash
# DeployScene: env-only playground; the reasoner picks the rSkill.
uv run openral deploy sim --config scenes/deploy/openarm_tabletop.yaml

# The same robot on real hardware. Bringup MOVES BOTH ARMS (openarm_bringup
# returns to zero on activate): read the scene's header and stay attended.
uv run openral deploy run --config scenes/deploy/openarm_bench.yaml

# BEHAVIOR-1K R1 Pro through the OmniGibson evaluator.
uv run openral deploy sim --config scenes/deploy/behavior_r1pro.yaml \
  --initial-task "turn on the radio"

# SimScene: one rollout with an explicit rSkill.
MUJOCO_GL=egl uv run --all-packages --group libero \
  openral sim run --config scenes/sim/libero_spatial.yaml \
  --rskill smolvla-libero --task libero_spatial/3

# BenchmarkScene: writes rskills/<vla>/eval/scene_<scene_id>.json.
MUJOCO_GL=egl uv run --all-packages --group libero \
  openral benchmark scene --config scenes/benchmark/libero_spatial.yaml \
  --rskill smolvla-libero

# Suite.
MUJOCO_GL=egl uv run --all-packages --group libero \
  openral benchmark run --suite libero_spatial --rskill smolvla-libero
```

Use `--all-packages` with `--group libero`; without it uv evicts the `openral` script. The `just sim-libero` recipe does this for you. `uv run openral sim list` prints every scene path for `--config`, and `uv run openral rskill list` prints the `--rskill` names (`--json` for machine-readable).

An `openral sim run` also accepts a suite file with `--task`: `--config benchmarks/libero_10.yaml --task libero_10/3`.

### Override flags

The YAML pins defaults; a flag wins.

| Command | Overrides |
| --- | --- |
| `sim run` | `--task`, `--instruction`, `--max-steps`, `--n-episodes`, `--seed`, `--device`, `--n-action-steps`, and `--robot` on free-axis scenes |
| `benchmark scene` | `--n-episodes` only |
| `benchmark run` | `--task` and `--n-episodes` |

A lowered `--n-episodes` is for iteration, not paper numbers. Recipes: `just sim-libero`, `sim-xvla-libero`, `sim-pi05-libero`, `sim-act-libero`, `sim-eval <config>`, `sim-metaworld`, `sim-maniskill3`, `sim-simpler-widowx`, `sim-act-aloha`, `sim-diffusion-pusht`, `sim-custom` and `sim-audit`. The table is in the reference page.

## Schema fields

| Tier | Fields |
| --- | --- |
| `DeployScene` | `scene` plus optional `robot_id` or `robots` (two or more named robots in a shared scene frame), `robot_unit`, `base_pose`, `hal`, `sensors`, `drivers`, `runtime`, `safety`, `composition`, `extra_allowed_collision_pairs`, `grasp_declaration`, `place_declaration`, `place_zones`, `nav_goals`, `success_conditions`, `object_catalog`, `memory_dir`, `seed` |
| `SimScene` | A `DeployScene` with a required `task:` and optional `n_episodes`, `record_video`, `save_dir`, `metadata`. It runs one robot: `robots:` is refused. |
| `BenchmarkScene` | A `SimScene` with `metadata: BenchmarkMetadata` (`paper`, `honest_scope`), `seed` and `n_episodes` required |

Semantics of `success_conditions`, `place_zones`, `nav_goals` and `composition` are in the [reference page](../docs/reference/sim-environments.md#deploy-scene-fields). A YAML that carries a `vla:` block raises `ROSConfigError`: the policy always comes from `--rskill`.

Relative paths in a scene (`scene.assets_uri`, `backend_options.objects[].usd`, `runtime.static_geometry_file`, `runtime.vision_attachment.segmenter_manifest`) resolve beside the scene YAML first, then under the repo root, then the working directory. `scene.backend_options` is opaque to the tier schemas: each backend validates it with its own `options_model` (`SCENES.validate_options`), so a bad key fails at load naming the scene id and the field.

**Scene files name code, so load them from trusted sources only.** A `composition.composer` is a `"module.path:function"` the HAL node imports. Only `openral_*` modules (and OpenRAL Pro's `openral_pro_*`) are admitted, checked at load and again before the import. `OPENRAL_ALLOW_EXTERNAL_PLUGINS=1` admits another with a warning on every use.

## Scene ids (`scene.id`)

| Backend | Scene ids | Adapter (`python/sim/src/openral_sim/backends/`) |
| --- | --- | --- |
| LIBERO | `libero_spatial`, `libero_object`, `libero_goal`, `libero_10` | `libero.py` |
| MetaWorld | `metaworld` (passes `<env_id>` to `gym.make`) | `metaworld.py` |
| gym-aloha | `aloha_transfer_cube`, `aloha_insertion` | `aloha.py` |
| gym-pusht | `pusht` (2-D pymunk) | `pusht.py` |
| RLBench (CoppeliaSim sidecar) | `rlbench` (task from `backend_options.rlbench_task`) | `rlbench.py` |
| BEHAVIOR-1K (OmniGibson sidecar) | `behavior` (task and instance from `backend_options`) | `behavior.py` |
| RoboCasa | `robocasa` (procedural), `robocasa/<Task>` (kitchen), `robocasa/gr1/<Task>` (GR1 tabletop); a typo fails at build with `ROSConfigError` | `robocasa.py` |
| ManiSkill3 (SAPIEN) | `maniskill3` (passes `<env_id>` to `gym.make`) | `maniskill3.py` |
| SimplerEnv (SAPIEN) | `simpler_env` (Bridge V2: 4 WidowX tasks) | `simpler_env.py` |
| RoboTwin 2.0 (SAPIEN sidecar) | `robotwin` (task via `backend_options`, e.g. `lift_pot`) | `robotwin.py` |
| VLABench | `vlabench` (task via `backend_options`, e.g. `select_fruit`) | `vlabench.py` |
| Custom OpenArm | `openarm_tabletop_pnp` | `openarm_robosuite/env.py` |
| Custom SO-101 | `so101_box` (box arena, overhead and wrist cameras, tube insertion; `BoxSceneOptions`) | `so101_box/` |
| Custom MakerMods Metal | `metal_pick_place`, `metal_pen_holder` (sim twins of the real rigs) | `metal_pick_place/`, `metal_pen_holder/` |
| Custom tabletop | `tabletop_push` (robot-agnostic cube push; free-axis) | `tabletop_push/` |
| Isaac Sim (sidecar) | `isaac_sim` (free-axis; any manifest robot inside any environment USD) | `isaac_sim.py` |
| Mock | `mock` (no physics; any action size) | `policies/mock.py` |

## Scene-fixed robots

Some scenes hard-wire their robot with `@SCENES.register(..., fixed_robot=...)`:

| Scene | Fixed robot |
| --- | --- |
| `libero_spatial`, `libero_object`, `libero_goal`, `libero_10` | `franka_panda` |
| `metaworld` | `sawyer` |
| `pusht` | `pusht_2d` |
| `aloha_transfer_cube`, `aloha_insertion` | `aloha_bimanual` |
| `rlbench`, `vlabench`, `maniskill3` | `franka_panda` |
| `so101_box` | `so101_follower` |
| `metal_pick_place` | `makermods_metal` |
| `metal_pen_holder` | `makermods_metal_station` |
| `robocasa`, `robocasa/*` (kitchen) | `panda_mobile`, `panda_mobile_vslam` |
| `robocasa/gr1/*` | `gr1` |
| `behavior` | `r1pro` |
| `robotwin` | `aloha_agilex` |
| `simpler_env` | `widowx` |
| `openarm_tabletop_pnp` | `openarm` |

One rule, `SCENES.resolve_robot`, binds the robot in `sim run`, `benchmark`, `deploy sim` and the sim HAL. A `--robot` or `robot_id:` outside the scene's set raises `ROSConfigError` listing the allowed robots; a matching one is accepted; an omitted one takes the default. Free-axis scenes (`tabletop_push`, `isaac_sim`, `mock`) require a robot.

## Placing robots with `base_pose:`

Scenes registered with `base_pose=True` (`tabletop_push`, `isaac_sim`, `openarm_tabletop_pnp`) accept a `base_pose:` block anchoring the robot in the scene's world frame. Adapters write the `world → base_frame` transform into the MJCF at load:

```yaml
robot_id: so100_follower
scene: {id: tabletop_push, backend: mujoco}
task: {id: tabletop_push/0, scene_id: tabletop_push, instruction: ""}
base_pose:
  xyz: [0.0, 0.0, 0.0]               # world-frame position (m)
  quat_xyzw: [0.0, 0.0, 0.0, 1.0]    # world-frame orientation
  frame_id: world
```

`base_pose:` on any other scene is a `ROSConfigError`: those scenes ship their own MJCF, so the field has no meaning there.

## rSkill compatibility check

Before the policy loads, the runner resolves `--rskill` to its manifest and checks that the manifest's `embodiment_tags` and `sensors_required` fit the robot's capabilities and sensors (`_check_rskill_compatibility`). A missing manifest or an unregistered robot raises `ROSConfigError`; an incompatible pairing raises `ROSCapabilityMismatch`. There is no warn-and-proceed path.

## Live MuJoCo viewer

`openral sim run` opens a passive `mujoco.viewer` window by default when a display exists. `openral benchmark scene` stays headless unless you pass `--view`. Toggle with `--view / --no-view`. The default turns itself off with one warning when `MUJOCO_GL=egl` is set, `DISPLAY` is unset on Linux, or the scene exposes no `mujoco_handles()`. An explicit `--view` fails loudly instead. The window survives episode resets.

| Scene | Viewer |
| --- | --- |
| `libero_*`, `metaworld`, `aloha_*`, `openarm_*`, `so101_box`, `metal_*`, `tabletop_push`, `robocasa/*` | Supported |
| `pusht` | No (2-D; runs offscreen) |
| `rlbench` | No (sidecar; `render()` returns the latest RGB frame for video) |
| `maniskill3`, `simpler_env` | No `mujoco.viewer`; `--view` opens SAPIEN's own window |

## Performance knobs (rSkill `policy_extras`)

Lerobot-style families (`smolvla`, `act`, `pi05`) read these from the rSkill manifest's `policy_extras`:

| Key | Default | What it does |
| --- | --- | --- |
| `n_action_steps` | Per family: SmolVLA and π0.5 = `chunk_size`; ACT = `1`; Diffusion = `8` | Actions consumed from each chunk before the policy re-infers. The shipped lerobot checkpoints set `1`, which for non-ACT families discards the chunk and pays a full forward every step. Adapters install the paper default so `benchmark run` reproduces published numbers. |
| `temporal_ensemble_coeff` (ACT) | `0.01` (paper value) | Turns on ACT's temporal-ensembling buffer; `null` disables it. |
| `compile` | `false` | Opt-in `torch.compile` of the chunk forward; skipped on CPU; falls back to eager and logs `vla_compile_setup_failed` or `vla_compile_runtime_fallback`. Not exposed for `pi05`. |
| `compile_mode` | `"default"` | `default`, `reduce-overhead` (CUDA graphs, recommended) or `max-autotune` |

`compile` needs a working system C compiler for Triton (a conda `cc` shadow breaks it; prefix the command with `CC=/usr/bin/gcc`), roughly 3 GB of free VRAM above the weights, and a warm-up of about 30 s on the first chunk, paid once per process. Leave it off for one-shot debug rollouts and turn it on for evaluations.
