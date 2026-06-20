# Sim Environments

Scene YAMLs under [`scenes/`](https://github.com/OpenRAL/openral/tree/master/scenes)
follow the three-tier hierarchy introduced by
[ADR-0041](../adr/0041-scene-three-tier-hierarchy.md): `DeployScene ⊆ SimScene
⊆ BenchmarkScene`. Each tier has its own directory, its own loader-strictness
gate, and its own CLI consumer. The conceptual overview, decision matrix,
authoring guide, and per-backend `scene.id` catalogue all live in the in-tree
[`scenes/README.md`](https://github.com/OpenRAL/openral/tree/master/scenes/README.md);
this page is the **per-file catalogue** — one row per YAML.

Scene dependencies are auto-installed on first use. Bypass the install prompt
in CI with `OPENRAL_AUTO_INSTALL_DEPS=1`.

## Quick CLI

```bash
# DeployScene — env-only playground (reasoner picks the rSkill at runtime).
openral deploy sim --config scenes/deploy/openarm_tabletop.yaml

# SimScene — single rollout; supply the policy at the CLI.
openral sim run --config scenes/sim/libero_spatial.yaml --rskill smolvla-libero

# BenchmarkScene — paper-comparable single-scene eval; writes
# rskills/<vla>/eval/<scene_id>.json with reproduced_locally=true.
openral benchmark scene --config scenes/benchmark/libero_spatial.yaml \
                        --rskill smolvla-libero

# Benchmark suite — multi-scene aggregate (lives in benchmarks/, not scenes/).
openral benchmark run --suite libero_spatial --rskill smolvla-libero
```

Override flags (`--task`, `--instruction`, `--max-steps`, `--n-episodes`,
`--robot` for free-axis scenes) work on every tier except `benchmark run`,
which intentionally rejects them to guarantee suite reproducibility. See
[`scenes/README.md`](https://github.com/OpenRAL/openral/tree/master/scenes/README.md#swap-any-axis)
for the full override matrix.

## DeployScene catalogue (`scenes/deploy/`)

Env-only "robot + scene" pins. No `task:` block, no eval; the runtime
reasoner picks the rSkill. Consumed by `openral deploy sim`.

| Config | Fixed / declared robot | `scene.id` | Backend | Use |
|---|---|---|---|---|
| [`libero_pnp.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/deploy/libero_pnp.yaml) | `franka_panda` *(scene-fixed)* | `libero_spatial` | LIBERO (robosuite + MuJoCo) | Boot LIBERO in deploy mode so a reasoner can issue arbitrary pick-and-place commands |
| [`openarm_tabletop.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/deploy/openarm_tabletop.yaml) | `openarm` *(free-axis)* | `openarm_tabletop_pnp` | Custom MJCF | OpenArm bimanual tabletop sandbox; default top camera matches the mddoai dataset POV |
| [`robocasa_pnp.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/deploy/robocasa_pnp.yaml) | `panda_mobile` *(scene-fixed)* | `robocasa/PickPlaceCounterToCabinet` | RoboCasa (MuJoCo) | Mobile-base kitchen pick-and-place sandbox; reasoner-driven |
| [`so101_box.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/deploy/so101_box.yaml) | `so101_follower` *(scene-fixed)* | `so101_box` | Custom MJCF | 100×61.5×75 cm box arena + OAK-D Pro overhead + wrist camera; deploy sandbox |

## SimScene catalogue (`scenes/sim/`)

`DeployScene` + a single `task:` block. One CLI invocation, one or more
`EpisodeResult`s; sized for ad-hoc development and smoke tests. The policy is
supplied at the CLI via `--rskill <name>` — scene YAMLs no longer pin a VLA.
Consumed by `openral sim run`.

| Config | Fixed / declared robot | `scene.id` | `task.id` | Notes |
|---|---|---|---|---|
| [`libero_spatial.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/sim/libero_spatial.yaml) | `franka_panda` *(scene-fixed)* | `libero_spatial` | `libero_spatial/0` | LIBERO-Spatial smoke; ad-hoc sibling of `scenes/benchmark/libero_spatial.yaml` |
| [`openarm_tabletop.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/sim/openarm_tabletop.yaml) | `openarm` *(free-axis)* | `openarm_tabletop_pnp` | `openarm/pnp_cube_to_drawer` | Bimanual cube-to-drawer; mirrors the mddoai dataset POV |
| [`robocasa_gr1_pnp_cup_to_drawer.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/sim/robocasa_gr1_pnp_cup_to_drawer.yaml) | `gr1` *(scene-fixed)* | `robocasa/gr1/PnPCupToDrawerClose` | `robocasa/gr1/PnPCupToDrawerClose/0` | RoboCasa GR1 humanoid tabletop pnp |
| [`robocasa_panda_mobile_kitchen.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/sim/robocasa_panda_mobile_kitchen.yaml) | `panda_mobile` *(scene-fixed)* | `robocasa/NavigateKitchen` | `robocasa/NavigateKitchen/0` | Mobile-base kitchen navigation; `deploy sim` Nav2 graph compatible |
| [`robocasa_pnp.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/sim/robocasa_pnp.yaml) | `panda_mobile` *(scene-fixed)* | `robocasa/PickPlaceCounterToCabinet` | `robocasa/PickPlaceCounterToCabinet/0` | RoboCasa kitchen pnp smoke |
| [`so101_tube_insertion.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/sim/so101_tube_insertion.yaml) | `so101_follower` *(scene-fixed)* | `so101_box` | `so101_box/tube_insertion` | Box-arena tube-insertion smoke; geometry/sensors/spawn ranges configurable via `BoxSceneOptions` |
| [`tabletop_cube_push.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/sim/tabletop_cube_push.yaml) | `so101_follower` *(free-axis default; pass `--robot` to override)* | `tabletop_push` | `tabletop_push/push_to_goal` | Robot-agnostic cube push-to-goal ([ADR-0033](../adr/0033-robot-parameterized-native-scenes.md)) |
| [`widowx_carrot_on_plate.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/sim/widowx_carrot_on_plate.yaml) | `widowx` *(scene-fixed)* | `simpler_env` | `simpler_env/widowx_carrot_on_plate` | SimScene sibling of the SimplerEnv WidowX carrot benchmark; used by the OpenVLA-OFT issue #55 reproduction path |

## BenchmarkScene catalogue (`scenes/benchmark/`)

`SimScene` + required `metadata: BenchmarkMetadata` (paper URL +
`honest_scope`) + non-`None` `seed` and `n_episodes`. The shipped values
match the canonical paper protocol; running `openral benchmark scene` against
one of these writes `rskills/<vla>/eval/<scene_id>.json` with
`reproduced_locally=true`. Consumed by `openral benchmark scene`. Most are
also aggregated into a multi-scene suite (bare `list[BenchmarkScene]` per
ADR-0042) under
[`benchmarks/`](https://github.com/OpenRAL/openral/tree/master/benchmarks).

| Config | Fixed / declared robot | `scene.id` | `task.id` | `n_episodes` | Paper |
|---|---|---|---|---|---|
| [`aloha_insertion.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/benchmark/aloha_insertion.yaml) | `aloha_bimanual` *(scene-fixed)* | `aloha_insertion` | `aloha_insertion/0` | 200 | [ALOHA / ACT](https://arxiv.org/abs/2304.13705) |
| [`aloha_transfer_cube.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/benchmark/aloha_transfer_cube.yaml) | `aloha_bimanual` *(scene-fixed)* | `aloha_transfer_cube` | `aloha_transfer_cube/0` | 200 | [ALOHA / ACT](https://arxiv.org/abs/2304.13705) |
| [`libero_spatial.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/benchmark/libero_spatial.yaml) | `franka_panda` *(scene-fixed)* | `libero_spatial` | `libero_spatial/0` | 500 | [LIBERO](https://arxiv.org/abs/2309.11500) |
| [`maniskill_pick_cube.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/benchmark/maniskill_pick_cube.yaml) | `franka_panda` *(free-axis)* | `maniskill3` | `maniskill3/PickCube-v1` | 500 | [ManiSkill3](https://arxiv.org/abs/2410.00425) |
| [`metaworld_push.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/benchmark/metaworld_push.yaml) | `sawyer` *(scene-fixed)* | `metaworld` | `metaworld/push` | 200 | [MetaWorld MT50](https://arxiv.org/abs/1910.10897) |
| [`pusht.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/benchmark/pusht.yaml) | `pusht_2d` *(scene-fixed; 2-D pymunk)* | `pusht` | `pusht/0` | 200 | [Diffusion Policy](https://arxiv.org/abs/2303.04137) |
| [`rlbench_open_drawer.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/benchmark/rlbench_open_drawer.yaml) | `franka_panda` *(scene-fixed)* | `rlbench` | `rlbench/open_drawer` | 25 | [RLBench](https://arxiv.org/abs/1909.12271) / [3D Diffuser Actor](https://arxiv.org/abs/2402.10885) |
| [`rlbench_meat_off_grill.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/benchmark/rlbench_meat_off_grill.yaml) | `franka_panda` *(scene-fixed)* | `rlbench` | `rlbench/meat_off_grill` | 25 | [RLBench](https://arxiv.org/abs/1909.12271) / [3D Diffuser Actor](https://arxiv.org/abs/2402.10885) |
| [`rlbench_close_jar.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/benchmark/rlbench_close_jar.yaml) | `franka_panda` *(scene-fixed)* | `rlbench` | `rlbench/close_jar` | 25 | [RLBench](https://arxiv.org/abs/1909.12271) / [3D Diffuser Actor](https://arxiv.org/abs/2402.10885) |
| [`widowx_carrot_on_plate.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/benchmark/widowx_carrot_on_plate.yaml) | `widowx` *(scene-fixed)* | `simpler_env` | `simpler_env/widowx_carrot_on_plate` | 24 | [SimplerEnv](https://arxiv.org/abs/2405.05941) |

The `n_episodes` and `seed` columns ship in the file at the paper-canonical
value. Overriding `--n-episodes` on `openral benchmark scene` is allowed
(useful for cheap smoke runs that don't claim paper-reproduction); the
resulting `RSkillEvalResult` records the lowered count.

Multi-scene aggregations (e.g. all 10 LIBERO-Spatial tasks, all 50 MetaWorld
tasks, all 4 SimplerEnv WidowX tasks) live in
[`benchmarks/`](https://github.com/OpenRAL/openral/tree/master/benchmarks).
A suite YAML is a bare `list[BenchmarkScene]` at the YAML root (ADR-0042);
suite-level invariants (uniform `robot_id`, `seed`, `n_episodes`, and full
`metadata` block) are enforced by `openral_core.raise_on_invalid_suite`.

## Justfile shortcuts

The repo's [`Justfile`](https://github.com/OpenRAL/openral/blob/master/Justfile)
groups `sim-*` recipes by which CLI they drive:

```bash
# SimScene-tier — `openral sim run --save-video` (debug smoke; no eval JSON).
just sim-libero                     # SmolVLA × LIBERO        (GPU + MUJOCO_GL)
just sim-xvla-libero                # xVLA × LIBERO           (Florence-2)
just sim-pi05-libero                # π0.5 × LIBERO           (≥8 GB VRAM)
just sim-act-libero                 # ACT × LIBERO            (paper protocol)
just sim-pi05-robocasa              # π0.5 × RoboCasa kitchen (≥8 GB VRAM)

# BenchmarkScene-tier — `openral benchmark scene --no-update-manifest \
#     --n-episodes 1 --save-dir` (paper protocol, single rollout for smoke).
just sim-metaworld --task metaworld/reach-v3
just sim-maniskill3                 # SAPIEN-backed PickCube-v1
just sim-simpler-widowx             # RLDX-1 × WidowX carrot-on-plate
just sim-act-aloha                  # ACT × gym-aloha bimanual cube-transfer
just sim-diffusion-pusht            # Diffusion Policy × gym-pusht (CPU)
just sim-custom                     # ACT × gym-aloha insertion (rskills/act-aloha-insertion)
```

`just sim-audit` runs
[`tools/audit_sim_configs.py`](https://github.com/OpenRAL/openral/blob/master/tools/audit_sim_configs.py)
over the per-tier catalogue and reports row-by-row latency + success
metrics. `just sim-eval` runs the full benchmark suites end-to-end.

## See also

- [`scenes/README.md`](https://github.com/OpenRAL/openral/tree/master/scenes/README.md)
  — conceptual hierarchy, decision matrix, override flags, scene-id /
  fixed-robot tables, `base_pose` for free-axis scenes, rSkill compatibility,
  live MuJoCo viewer, `policy_extras` performance knobs.
- [Tutorial — Create a sim environment](../tutorials/sim/create-a-sim-environment.md)
  — long-form YAML authoring guide (new scene adapter, new robot manifest,
  custom policy).
- [ADR-0002](../adr/0002-eval-and-sim-environments.md) — original
  scene/eval design.
- [ADR-0041](../adr/0041-scene-three-tier-hierarchy.md) — three-tier
  hierarchy (`DeployScene ⊆ SimScene ⊆ BenchmarkScene`) + loader strictness.
- [ADR-0009](../adr/0009-separate-sim-and-benchmarking.md) — separation
  of `sim run` (debug) and `benchmark *` (paper-comparable eval).
