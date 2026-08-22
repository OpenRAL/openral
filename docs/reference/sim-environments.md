# Sim Environments

Scene YAMLs under [`scenes/`](https://github.com/OpenRAL/openral/tree/master/scenes)
follow a three-tier hierarchy:
`DeployScene ⊆ SimScene
⊆ BenchmarkScene`. Each tier has its own directory, its own loader-strictness
gate, and its own CLI consumer. The conceptual overview, decision matrix,
authoring guide, and per-backend `scene.id` catalogue all live in the in-tree
[`scenes/README.md`](https://github.com/OpenRAL/openral/tree/master/scenes/README.md);
this page is the **per-file catalogue** — one row per YAML.

Scene dependencies are auto-installed on first use (auto-install is on by
default; set `OPENRAL_AUTO_INSTALL_DEPS=0` to be prompted instead, e.g. when
*not* in CI). Most scenes need an opt-in dependency group first — sync it with
`just sync --group <name>` (`sim` / `libero` / `robocasa` / `metaworld` /
`maniskill3`), **never** a bare `uv sync`. RoboCasa is a special case: `just
sync --group robocasa` provides only robosuite + deps, while the RoboCasa fork
itself is git-cloned + installed editable at runtime by the HAL
(`ensure_backend_deps('robocasa_kitchen')`). LIBERO and RoboCasa pin
conflicting robosuite versions, so swap groups per task. Full recipe →
[Managing the Python environment & dependency
groups](../contributing/toolchain.md#managing-the-python-environment-dependency-groups).

An install step's output streams to your terminal as it runs, and the tail of
it is quoted back inside the `ROSConfigError` if the step fails — so when a
backend or sidecar won't provision, read the quoted lines rather than the exit
code. That matters most when the step ran *inside a ROS node*, where the raw
output is scattered through the launch log: the reasoner reports the failure as
`goal_rejected`, and the quoted tail is the only place the cause travels with
it. A sidecar pinned to wheels that don't exist for your platform (aarch64 is
the common case) shows up there as a `uv` "no source distribution or wheel for
the current platform" line naming the offending package — see [aarch64 CUDA
hosts](aarch64-support.md) for the per-sidecar status on GB10 / DGX Spark and
Jetson Thor.

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
| [`behavior_r1pro.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/deploy/behavior_r1pro.yaml) | `r1pro` *(scene-fixed)* | `behavior` | BEHAVIOR-1K / OmniGibson (Isaac Sim sidecar) | Full deploy graph on public `turning_on_radio` instance 0 |
| [`so101_box.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/deploy/so101_box.yaml) | `so101_follower` *(scene-fixed)* | `so101_box` | Custom MJCF | 100×61.5×75 cm box arena + OAK-D Pro overhead + wrist camera; deploy sandbox |
| [`so101_bench.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/deploy/so101_bench.yaml) | `so101_follower` *(scene-fixed)* | `so101_bench` | Custom MJCF | SO-101 bench-arena deploy sandbox |
| [`libero_object.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/deploy/libero_object.yaml) | `franka_panda` *(scene-fixed)* | `libero_object` | LIBERO (robosuite + MuJoCo) | Boot LIBERO-Object in deploy mode; reasoner-driven pick-and-place |
| [`robocasa_baguette.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/deploy/robocasa_baguette.yaml) | `panda_mobile` *(scene-fixed)* | `robocasa/PickPlaceCounterToCabinet` | RoboCasa (MuJoCo) | Baguette pick-and-place kitchen sandbox; declares its ADR-0097 place target (`sim:cab_1_left_group_main`, read off the rounds 5/6 artifacts) |
| [`robocasa_sink_cup.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/deploy/robocasa_sink_cup.yaml) | `panda_mobile` *(scene-fixed)* | `robocasa/PickPlaceCounterToSink` | RoboCasa (MuJoCo) | Cup counter→sink pick-and-place; generalization sibling of `robocasa_baguette`. Declares its ADR-0097 place target — best-evidenced, pending live validation |
| [`robocasa_drawer_utensil.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/deploy/robocasa_drawer_utensil.yaml) | `panda_mobile` *(scene-fixed)* | `robocasa/PickPlaceCounterToDrawer` | RoboCasa (MuJoCo) | Utensil counter→drawer pick-and-place; reach-down place target. No ADR-0097 declaration: its target is one of many generated `stack` levels, so a guessed name would resolve to a *different* real receptacle instead of failing closed |
| [`robocasa_fridge_drawer.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/deploy/robocasa_fridge_drawer.yaml) | `panda_mobile` *(scene-fixed)* | `robocasa/PickPlaceFridgeShelfToDrawer` | RoboCasa (MuJoCo) | Vegetable fridge-shelf→fridge-drawer pick-and-place; enclosed workspace. Declares its ADR-0097 place target (the fridge, whose subtree carries the drawer) — best-evidenced, pending live validation |
| [`robocasa_navigate.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/deploy/robocasa_navigate.yaml) | `panda_mobile` *(scene-fixed)* | `robocasa/NavigateKitchen` | RoboCasa (MuJoCo) | Mobile-base kitchen navigation sandbox (Nav2 graph compatible) |
| [`isaac_franka.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/deploy/isaac_franka.yaml) | `franka_panda` *(scene-fixed)* | `isaac_sim` | Isaac Sim | Franka tabletop sandbox on the Isaac Sim backend (requires Isaac Sim license) |
| [`isaac_franka_bowl.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/deploy/isaac_franka_bowl.yaml) | `franka_panda` *(scene-fixed)* | `isaac_sim` | Isaac Sim | Franka bowl-manipulation sandbox (Isaac Sim) |
| [`isaac_franka_urdf.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/deploy/isaac_franka_urdf.yaml) | `franka_panda` *(scene-fixed)* | `isaac_sim` | Isaac Sim | Franka sandbox loaded from URDF (Isaac Sim) |
| [`isaac_panda_mobile_urdf.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/deploy/isaac_panda_mobile_urdf.yaml) | `panda_mobile` *(scene-fixed)* | `isaac_sim` | Isaac Sim | Mobile-base Panda loaded from URDF (Isaac Sim) |

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
| [`tabletop_cube_push.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/sim/tabletop_cube_push.yaml) | `so101_follower` *(free-axis default; pass `--robot` to override)* | `tabletop_push` | `tabletop_push/push_to_goal` | Robot-agnostic cube push-to-goal |
| [`widowx_carrot_on_plate.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/sim/widowx_carrot_on_plate.yaml) | `widowx` *(scene-fixed)* | `simpler_env` | `simpler_env/widowx_carrot_on_plate` | SimScene sibling of the SimplerEnv WidowX carrot benchmark; used by the OpenVLA-OFT issue #55 reproduction path |
| [`aloha_transfer_cube.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/sim/aloha_transfer_cube.yaml) | `aloha_bimanual` *(scene-fixed)* | `aloha_transfer_cube` | `aloha_transfer_cube/0` | gym-aloha bimanual cube-transfer smoke |
| [`franka_tabletop_push.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/sim/franka_tabletop_push.yaml) | `franka_panda` *(scene-fixed)* | `tabletop_push` | `tabletop_push/push_to_goal` | Franka variant of the robot-agnostic cube push-to-goal |
| [`pusht.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/sim/pusht.yaml) | `pusht_2d` *(scene-fixed; 2-D pymunk)* | `pusht` | `pusht/0` | gym-pusht 2-D push smoke |
| [`isaac_franka_bowl_plate.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/sim/isaac_franka_bowl_plate.yaml) | `franka_panda` *(scene-fixed)* | `isaac_sim` | `isaac_sim/put_the_bowl_on_the_plate` | Isaac Sim bowl-on-plate; two-camera layout used by `gr00t-n17-libero` |
| [`behavior_turning_on_radio.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/sim/behavior_turning_on_radio.yaml) | `r1pro` *(scene-fixed)* | `behavior` | `behavior/turning_on_radio` | Official public-test instance 0 through the OmniGibson evaluator sidecar |

## BenchmarkScene catalogue (`scenes/benchmark/`)

`SimScene` + required `metadata: BenchmarkMetadata` (paper URL +
`honest_scope`) + non-`None` `seed` and `n_episodes`. The shipped values
match the canonical paper protocol; running `openral benchmark scene` against
one of these writes `rskills/<vla>/eval/<scene_id>.json` with
`reproduced_locally=true`. Consumed by `openral benchmark scene`. Most are
also aggregated into a multi-scene suite (bare `list[BenchmarkScene]` at the
YAML root) under
[`benchmarks/`](https://github.com/OpenRAL/openral/tree/master/benchmarks).

| Config | Fixed / declared robot | `scene.id` | `task.id` | `n_episodes` | Paper |
|---|---|---|---|---|---|
| [`aloha_insertion.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/benchmark/aloha_insertion.yaml) | `aloha_bimanual` *(scene-fixed)* | `aloha_insertion` | `aloha_insertion/0` | 200 | [ALOHA / ACT](https://arxiv.org/abs/2304.13705) |
| [`aloha_transfer_cube.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/benchmark/aloha_transfer_cube.yaml) | `aloha_bimanual` *(scene-fixed)* | `aloha_transfer_cube` | `aloha_transfer_cube/0` | 200 | [ALOHA / ACT](https://arxiv.org/abs/2304.13705) |
| [`libero_spatial.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/benchmark/libero_spatial.yaml) | `franka_panda` *(scene-fixed)* | `libero_spatial` | `libero_spatial/0` | 500 | [LIBERO](https://arxiv.org/abs/2309.11500) |
| [`maniskill_pick_cube.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/benchmark/maniskill_pick_cube.yaml) | `franka_panda` *(free-axis)* | `maniskill3` | `maniskill3/PickCube-v1` | 500 | [ManiSkill3](https://arxiv.org/abs/2410.00425) |
| [`metaworld_push.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/benchmark/metaworld_push.yaml) | `sawyer` *(scene-fixed)* | `metaworld` | `metaworld/push-v3` | 50 | [MetaWorld MT10/MT50](https://arxiv.org/abs/1910.10897) |
| [`metaworld_pick_place.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/benchmark/metaworld_pick_place.yaml) | `sawyer` *(scene-fixed)* | `metaworld` | `metaworld/pick-place-v3` | 50 | [MetaWorld MT10/MT50](https://arxiv.org/abs/1910.10897) |
| [`metaworld_button_press.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/benchmark/metaworld_button_press.yaml) | `sawyer` *(scene-fixed)* | `metaworld` | `metaworld/button-press-v3` | 50 | [MetaWorld MT10/MT50](https://arxiv.org/abs/1910.10897) |
| [`behavior_turning_on_radio.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/benchmark/behavior_turning_on_radio.yaml) | `r1pro` *(scene-fixed)* | `behavior` | `behavior/turning_on_radio` | 1 | [BEHAVIOR-1K](https://arxiv.org/abs/2403.09227) |
| [`metaworld_door_open.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/benchmark/metaworld_door_open.yaml) | `sawyer` *(scene-fixed)* | `metaworld` | `metaworld/door-open-v3` | 50 | [MetaWorld MT10/MT50](https://arxiv.org/abs/1910.10897) |
| [`metaworld_drawer_open.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/benchmark/metaworld_drawer_open.yaml) | `sawyer` *(scene-fixed)* | `metaworld` | `metaworld/drawer-open-v3` | 50 | [MetaWorld MT10/MT50](https://arxiv.org/abs/1910.10897) |
| [`pusht.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/benchmark/pusht.yaml) | `pusht_2d` *(scene-fixed; 2-D pymunk)* | `pusht` | `pusht/0` | 200 | [Diffusion Policy](https://arxiv.org/abs/2303.04137) |
| [`rlbench_open_drawer.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/benchmark/rlbench_open_drawer.yaml) | `franka_panda` *(scene-fixed)* | `rlbench` | `rlbench/open_drawer` | 25 | [RLBench](https://arxiv.org/abs/1909.12271) / [3D Diffuser Actor](https://arxiv.org/abs/2402.10885) |
| [`rlbench_meat_off_grill.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/benchmark/rlbench_meat_off_grill.yaml) | `franka_panda` *(scene-fixed)* | `rlbench` | `rlbench/meat_off_grill` | 25 | [RLBench](https://arxiv.org/abs/1909.12271) / [3D Diffuser Actor](https://arxiv.org/abs/2402.10885) |
| [`rlbench_close_jar.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/benchmark/rlbench_close_jar.yaml) | `franka_panda` *(scene-fixed)* | `rlbench` | `rlbench/close_jar` | 25 | [RLBench](https://arxiv.org/abs/1909.12271) / [3D Diffuser Actor](https://arxiv.org/abs/2402.10885) |
| [`widowx_carrot_on_plate.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/benchmark/widowx_carrot_on_plate.yaml) | `widowx` *(scene-fixed)* | `simpler_env` | `simpler_env/widowx_carrot_on_plate` | 24 | [SimplerEnv](https://arxiv.org/abs/2405.05941) |
| [`libero_object.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/benchmark/libero_object.yaml) | `franka_panda` *(scene-fixed)* | `libero_object` | `libero_object/0` | 50 | [LIBERO](https://arxiv.org/abs/2306.03310) |
| [`libero_goal.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/benchmark/libero_goal.yaml) | `franka_panda` *(scene-fixed)* | `libero_goal` | `libero_goal/0` | 50 | [LIBERO](https://arxiv.org/abs/2306.03310) |
| [`libero_10.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/benchmark/libero_10.yaml) | `franka_panda` *(scene-fixed)* | `libero_10` | `libero_10/0` | 50 | [LIBERO](https://arxiv.org/abs/2306.03310) (= LIBERO-Long) |
| [`robocasa_pnp.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/benchmark/robocasa_pnp.yaml) | `panda_mobile` *(scene-fixed)* | `robocasa/PickPlaceCounterToCabinet` | `robocasa/PickPlaceCounterToCabinet/0` | 50 | [RoboCasa](https://arxiv.org/abs/2406.02523) |
| [`robotwin_beat_block_hammer.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/benchmark/robotwin_beat_block_hammer.yaml) | `aloha_agilex` *(scene-fixed)* | `robotwin` | `robotwin/beat_block_hammer` | 100 | [RoboTwin 2.0](https://arxiv.org/abs/2506.18088) |
| [`robotwin_handover_block.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/benchmark/robotwin_handover_block.yaml) | `aloha_agilex` *(scene-fixed)* | `robotwin` | `robotwin/handover_block` | 100 | [RoboTwin 2.0](https://arxiv.org/abs/2506.18088) |
| [`robotwin_lift_pot.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/benchmark/robotwin_lift_pot.yaml) | `aloha_agilex` *(scene-fixed)* | `robotwin` | `robotwin/lift_pot` | 100 | [RoboTwin 2.0](https://arxiv.org/abs/2506.18088) |
| [`robotwin_place_empty_cup.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/benchmark/robotwin_place_empty_cup.yaml) | `aloha_agilex` *(scene-fixed)* | `robotwin` | `robotwin/place_empty_cup` | 100 | [RoboTwin 2.0](https://arxiv.org/abs/2506.18088) |
| [`robotwin_stack_blocks_two.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/benchmark/robotwin_stack_blocks_two.yaml) | `aloha_agilex` *(scene-fixed)* | `robotwin` | `robotwin/stack_blocks_two` | 100 | [RoboTwin 2.0](https://arxiv.org/abs/2506.18088) |
| [`vlabench_select_fruit.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/benchmark/vlabench_select_fruit.yaml) | `franka_panda` *(scene-fixed)* | `vlabench` | `vlabench/select_fruit` | 50 | [VLABench](https://arxiv.org/abs/2412.18194) |

The `n_episodes` and `seed` columns ship in the file at the paper-canonical
value. Overriding `--n-episodes` on `openral benchmark scene` is allowed
(useful for cheap smoke runs that don't claim paper-reproduction); the
resulting `RSkillEvalResult` records the lowered count.

Multi-scene aggregations (e.g. all 10 LIBERO-Spatial tasks, the MetaWorld MT10
and MT50 task sets — `benchmarks/metaworld_mt10.yaml` (10 tasks) and
`benchmarks/metaworld_mt50.yaml` (50 tasks) — all 4 SimplerEnv WidowX tasks)
live in
[`benchmarks/`](https://github.com/OpenRAL/openral/tree/master/benchmarks).
A suite YAML is a bare `list[BenchmarkScene]` at the YAML root;
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

# RoboCasa has no dedicated recipe — drive it through the generic `sim-eval`
# with one of the in-tree RoboCasa SimScenes (XR-1 is the maintained pairing;
# there is no π0.5 RoboCasa rSkill).
just sim-eval scenes/sim/xr1_robocasa_pnp.yaml --rskill rskill://rskills/xr1-robocasa

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
- The original scene/eval design established the base `sim run` +
  eval-layer split.
- A later decision introduced the three-tier
  hierarchy (`DeployScene ⊆ SimScene ⊆ BenchmarkScene`) + loader strictness.
- Another decision separated
  `sim run` (debug) from `benchmark *` (paper-comparable eval).
