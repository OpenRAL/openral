# Sim Environments

Scene YAMLs under [`scenes/`](https://github.com/OpenRAL/openral/tree/master/scenes)
follow a three-tier hierarchy,
`DeployScene ⊆ SimScene ⊆ BenchmarkScene`. Each tier has its own directory, its own loader-strictness
gate, and its own CLI consumer. The conceptual overview, decision matrix,
authoring guide, and per-backend `scene.id` catalogue all live in the in-tree
[`scenes/README.md`](https://github.com/OpenRAL/openral/tree/master/scenes/README.md);
this page is the per-file catalogue, one row per YAML.

Scene dependencies are auto-installed on first use (auto-install is on by
default; set `OPENRAL_AUTO_INSTALL_DEPS=0` to be prompted instead, e.g. when
not in CI). Most scenes need an opt-in dependency group first: `just sync --group
<name>` (`sim` / `libero` / `robocasa` / `metaworld` / `maniskill3`), never a bare
`uv sync`.

RoboCasa is a special case: `just sync --group robocasa` provides only
robosuite + deps, while the RoboCasa fork itself is git-cloned + installed
editable at runtime by the HAL (`ensure_backend_deps('robocasa_kitchen')`).
LIBERO and RoboCasa pin conflicting robosuite versions, so swap groups per
task. Full recipe → [Managing the Python environment & dependency
groups](../contributing/toolchain.md#managing-the-python-environment-dependency-groups).

An install step's output streams to your terminal, and its tail is quoted in the
`ROSConfigError` if the step fails: when a backend or sidecar won't provision,
read the quoted lines, not the exit code. Inside a ROS node the raw output is
scattered through the launch log and the reasoner reports only `goal_rejected`, so
the quoted tail is the only place the cause travels.

A sidecar pinned to wheels that don't exist for your platform (usually aarch64)
shows up there as a `uv` "no source distribution or wheel for the current
platform" line naming the package; see [aarch64 CUDA hosts](aarch64-support.md)
for the per-sidecar status on GB10 / DGX Spark and Jetson Thor.

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
reasoner picks the rSkill. Consumed by `openral deploy sim`, or by
`openral deploy run` for the real-hardware scenes.

| Config | Fixed / declared robot | `scene.id` | Backend | Use |
|---|---|---|---|---|
| [`libero_pnp.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/deploy/libero_pnp.yaml) | `franka_panda` *(scene-fixed)* | `libero_spatial` | LIBERO (robosuite + MuJoCo) | Boot LIBERO in deploy mode so a reasoner can issue arbitrary pick-and-place commands |
| [`openarm_tabletop.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/deploy/openarm_tabletop.yaml) | `openarm` *(scene-fixed)* | `openarm_tabletop_pnp` | Custom MJCF | OpenArm bimanual tabletop sandbox; default top camera matches the mddoai dataset POV |
| [`robocasa_pnp.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/deploy/robocasa_pnp.yaml) | `panda_mobile` *(scene-fixed)* | `robocasa/PickPlaceCounterToCabinet` | RoboCasa (MuJoCo) | Mobile-base kitchen pick-and-place sandbox; reasoner-driven |
| [`behavior_r1pro.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/deploy/behavior_r1pro.yaml) | `r1pro` *(scene-fixed)* | `behavior` | BEHAVIOR-1K / OmniGibson (Isaac Sim sidecar) | Full deploy graph on public `turning_on_radio` instance 0 |
| [`so101_box.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/deploy/so101_box.yaml) | `so101_follower` *(scene-fixed)* | `so101_box` | Custom MJCF | 100×61.5×75 cm box arena + OAK-D Pro overhead + wrist camera; deploy sandbox |
| [`so101_bench.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/deploy/so101_bench.yaml) | `so101_follower` *(scene-fixed)* | `so101_bench` | Custom MJCF | SO-101 bench-arena deploy sandbox |
| [`libero_object.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/deploy/libero_object.yaml) | `franka_panda` *(scene-fixed)* | `libero_object` | LIBERO (robosuite + MuJoCo) | Boot LIBERO-Object in deploy mode; reasoner-driven pick-and-place |
| [`robocasa_baguette.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/deploy/robocasa_baguette.yaml) | `panda_mobile` *(scene-fixed)* | `robocasa/PickPlaceCounterToCabinet` | RoboCasa (MuJoCo) | Baguette pick-and-place kitchen sandbox; declares its ADR-0097 place target (`sim:cab_1_left_group_main`, read off the rounds 5/6 artifacts and confirmed live at seed 1 — region measured, allowance published). Carries the canonical recipe for reading a place target off a live MuJoCo body table |
| [`robocasa_sink_cup.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/deploy/robocasa_sink_cup.yaml) | `panda_mobile` *(scene-fixed)* | `robocasa/PickPlaceCounterToSink` | RoboCasa (MuJoCo) | Cup counter→sink pick-and-place; generalization sibling of `robocasa_baguette`. Declares its ADR-0097 place target (`sim:sink_island_group_main`, body 319 — corrected against a live seed-1 run after the HAL refused #142's guess fail-closed) |
| [`robocasa_drawer_utensil.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/deploy/robocasa_drawer_utensil.yaml) | `panda_mobile` *(scene-fixed)* | `robocasa/PickPlaceCounterToDrawer` | RoboCasa (MuJoCo) | Utensil counter→drawer pick-and-place; reach-down place target. No ADR-0097 declaration: its target is one of many generated `stack` levels, so a guessed name would resolve to a *different* real receptacle instead of failing closed. Confirmed live at seed 1 — the only bodies matching `drawer` are the *fridge's* freezer drawers, while the counter drawer is `stack_2_left_group_3_*` |
| [`robocasa_fridge_drawer.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/deploy/robocasa_fridge_drawer.yaml) | `panda_mobile` *(scene-fixed)* | `robocasa/PickPlaceFridgeShelfToDrawer` | RoboCasa (MuJoCo) | Vegetable fridge-shelf→fridge-drawer pick-and-place; enclosed workspace. Declares its ADR-0097 place target (`sim:fridge_main_group_fridge_drawer0`, body 98 — corrected against a live seed-1 run after the HAL refused #142's guess fail-closed; the drawer, not the fridge root, whose subtree would also exempt both doors) |
| [`robocasa_navigate.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/deploy/robocasa_navigate.yaml) | `panda_mobile` *(scene-fixed)* | `robocasa/NavigateKitchen` | RoboCasa (MuJoCo) | Mobile-base kitchen navigation sandbox (Nav2 graph compatible) |
| [`robocasa_deliver_straw.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/deploy/robocasa_deliver_straw.yaml) | `panda_mobile` *(scene-fixed)* | `robocasa/DeliverStraw` | RoboCasa (MuJoCo) | The **carry-while-navigating** scene for issue #108: the base drives 3.795 m at the pinned seed 3 while holding the straw, which is what exercises `openral_nav2_bringup` mid-carry (the scan payload filter; the dynamic payload footprint was removed in PR #186). In `target50`, so XR-1 stays in distribution. Drawer is closed at reset — open → grasp → carry → place. See [the carry survey](robocasa-carry-survey.md) |
| [`robocasa_vslam.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/deploy/robocasa_vslam.yaml) | `panda_mobile_vslam` *(declared)* | `robocasa/PickPlaceCounterToCabinet` | RoboCasa (MuJoCo) | Lidar-less twin of `robocasa_pnp`: the base localises from two base-mounted RGB cameras via in-process PyCuVSLAM, not a `LaserScan` |
| [`robocasa_vslam_mono.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/deploy/robocasa_vslam_mono.yaml) | `panda_mobile_vslam` *(declared)* | `robocasa/PickPlaceCounterToCabinet` | RoboCasa (MuJoCo) | Single-camera variant: one RGB camera + DepthAnything-3 metric depth into cuVSLAM RGB-D odometry and nvblox |
| [`g1_vln.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/deploy/g1_vln.yaml) | `g1` *(declared)* | `g1_vln_bench` | Bare MuJoCo twin | Unitree G1 VLN bench: head camera, ADR-0089 pinned walking controller, `BODY_TWIST` navigation skills |
| [`openarm_zed_octomap.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/deploy/openarm_zed_octomap.yaml) | `openarm` *(declared)* | `openarm_tabletop_pnp` | Custom MJCF HAL + **real** ZED-M | Real ZED depth through the octomap leg into the world map, with the simulated OpenArm HAL |
| [`openarm_bench.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/deploy/openarm_bench.yaml) | `openarm` *(declared)* | `openarm_bench` | **Real hardware** (`deploy run`) | OpenArm v2 real bench cell: real arms and real cameras |
| [`openarm_real_world_voxels.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/deploy/openarm_real_world_voxels.yaml) | `openarm` *(declared)* | `openarm_real_world_voxels` | **Real hardware** (`deploy run`) | Real cell with the safety kernel's world-voxel check on; follow [the runbook](../tutorials/deploy/openarm-real-world-voxel-check.md) |
| [`openarm_real_autonomous.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/deploy/openarm_real_autonomous.yaml) | `openarm` *(declared)* | `openarm_real_autonomous` | **Real hardware** (`deploy run`) | `openarm_real_world_voxels` plus the reasoner, open-vocab detector and spatial-memory ingest: an LLM chooses the motion |
| [`galaxea_a1_bench.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/deploy/galaxea_a1_bench.yaml) | `galaxea_a1` *(declared)* | `galaxea_a1_bench` | **Real hardware** (`deploy run`) | Galaxea A1 deployment; start the ROS 1 sidecar, camera bridge, LingBot policy server and A1 runtime policy gateway first |
| [`isaac_franka_urdf.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/deploy/isaac_franka_urdf.yaml) | `franka_panda` *(scene-fixed)* | `isaac_sim` | Isaac Sim | Franka sandbox loaded from URDF (Isaac Sim) |
| [`isaac_panda_mobile_urdf.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/deploy/isaac_panda_mobile_urdf.yaml) | `panda_mobile` *(scene-fixed)* | `isaac_sim` | Isaac Sim | Mobile-base Panda loaded from URDF (Isaac Sim) |
| [`isaac_openarm_warehouse.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/deploy/isaac_openarm_warehouse.yaml) | `openarm` *(declared)* | `isaac_sim` | Isaac Sim | Bimanual OpenArm imported from its URDF, standing at a pallet of three pickable YCB props in the `Simple_Warehouse`; its meshes come from Enactic's public `openarm_description`, fetched on first use |
| [`isaac_panda_mobile_warehouse.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/deploy/isaac_panda_mobile_warehouse.yaml) | `panda_mobile` *(declared)* | `isaac_sim` | Isaac Sim | Mobile-base Panda spawned in an aisle of NVIDIA's `Simple_Warehouse` (three shelf racks, lidar/SLAM/Nav2), three pickable YCB props on a pallet 5 m down the aisle. The generic template: `scene.assets_uri` (environment USD) + `robot_id` + `base_pose` (spawn) — see [`scenes/README.md`](https://github.com/OpenRAL/openral/tree/master/scenes/README.md#isaac-sim-any-stage-any-robot) |

Deploy scenes are timing-faithful: one scene step is one control period of the
robot (`action_spec.control_freq_hz`), and `SimAttachedHAL.connect` refuses a
scene whose tick is anything else (`openral_core.check_time_base`, 1 %).

- **Isaac** derives its physics substeps at boot (non-integer ratios refused); see
  [Isaac Sim: any stage, any robot](https://github.com/OpenRAL/openral/tree/master/scenes/README.md#isaac-sim-any-stage-any-robot).
- **RoboCasa and LIBERO** deploy scenes pin `backend_options.control_freq_hz` to the
  robot's rate (robosuite's physics time step is shrunk to divide the period);
  without the pin both keep robosuite's native 20 Hz, the benchmark contract the
  eval tier and every published LIBERO number run at.
- **Bare MuJoCo twins** (`MujocoArmHAL`: OpenArm, SO-100/101, UR, Franka, ...)
  derive their physics steps per action from the same rate at connect (issue #358).

Isaac camera images trail the physics by 2–4 steps (the RTX pipeline), so each
observation carries `image_time_ns` and `deploy sim` stamps `/openral/cameras/*`
with the time each image shows, not the publish time (issue #361).

Every Isaac step also carries `info["robot_contacts"]` (that step's PhysX
robot-vs-world contacts), and `openral sim run` folds them into each episode's
`contact_stats` in `summary.json`: which links touched what, for how many steps,
how deep (`max_penetration_m`) and how hard (`max_impulse_ns`). That is the
evidence for which contacts a successful episode needs, to compare against the
safety kernel's stops.

Isaac safety-kernel stops carry ground truth too: each `sim.estop_ground_truth_snapshot`
line (`backend: "isaacsim"`) lists the PhysX robot-vs-world contact pairs of the
stop's physics step (link, other prim, `min_separation_m` (negative = penetration),
contact count, impulse). PhysX has no shape-to-shape distance, so unlike MuJoCo there
are no near-miss probes beyond the contact-offset band (`contacts_caveat`).

## SimScene catalogue (`scenes/sim/`)

`DeployScene` + a single `task:` block. One CLI invocation, one or more
`EpisodeResult`s; sized for ad-hoc development and smoke tests. The policy comes
from `--rskill <name>`; scene YAMLs do not pin a VLA.
Consumed by `openral sim run`.

Isaac SimScenes can grade placement prompts: `backend_options.placement_success`
(`IsaacPlacementSuccess`) maps a prompt, the scene's `task.instruction` verbatim,
to the scene objects it asks to place and named world-frame `regions` (target and
neighbours, e.g. shelf slots). Once an object rests for `settle_s` (neither
moving nor turning), it is graded by the scene's `grades` rubric
(`IsaacPlacementGrade`: name, score, `target` / `adjacent` region, whether it
must be upright; first match wins, else `fail` 0). The default rubric is
`complete` (in its target, upright: 1.0), `partial` (in its target, fallen: 0.5)
and `semi` (in a neighbouring region: 0.25); a scene declares its own levels to
grade differently. The episode takes its worst object's grade. Reaching the first
grade terminates the episode with `success`; otherwise the episode's
`total_reward` is its final grade's score, and every step's
`info["placement_grade"]` reports the grade so far. Without the block an Isaac
scene has no task reward. A region with `relative_to: <object>` is an offset box
around that scene object's centre as it is at each graded step (a plate's
surface), and a graded object may be one the environment USD already holds
(`backend_options.stage_objects`, see [`scenes/README.md`](https://github.com/OpenRAL/openral/tree/master/scenes/README.md#isaac-sim-any-stage-any-robot)).

Under `openral deploy sim` the scene is task-less (`<scene>/_hal_deploy_noop`, empty
instruction), so no rubric is active at boot and every prompt's rubric ships to the
sidecar. The HAL follows `/openral/reward/active_task`, where the rSkill runner
publishes each `ExecuteRskill` goal's prompt on accept and `""` at its end: the
goal's prompt, verbatim, selects `tasks[prompt]` (sidecar `set_task`); a prompt with
no entry is not graded (one `sim.placement_grade_unavailable` line). Each grade
change logs `sim.placement_grade` and the goal's end `sim.placement_grade_final`
(closing and best grade); the rubric keeps grading the settle after the goal ends,
until the next prompt replaces it. A deploy run never terminates on the grade -- it
is only reported (see [telemetry](telemetry.md)). `openral sim run` still resolves
the scene's own `task.instruction` at build and refuses one without an entry.

```yaml
task:
  instruction: "put the cracker box on the shelf"
scene:
  backend_options:
    placement_success:
      regions:
        shelf_1: {min_xyz: [-0.70, 4.75, 0.20], max_xyz: [-0.40, 4.95, 0.50]}
        shelf_2: {min_xyz: [-0.70, 4.95, 0.20], max_xyz: [-0.40, 5.25, 0.50]}
      tasks:
        "put the cracker box on the shelf":
          - {object: cracker_box, target: shelf_1, adjacent: [shelf_2]}
```

| Config | Fixed / declared robot | `scene.id` | `task.id` | Notes |
|---|---|---|---|---|
| [`libero_spatial.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/sim/libero_spatial.yaml) | `franka_panda` *(scene-fixed)* | `libero_spatial` | `libero_spatial/0` | LIBERO-Spatial smoke; ad-hoc sibling of `scenes/benchmark/libero_spatial.yaml` |
| [`openarm_tabletop.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/sim/openarm_tabletop.yaml) | `openarm` *(scene-fixed)* | `openarm_tabletop_pnp` | `openarm/pnp_cube_to_drawer` | Bimanual cube-to-drawer; mirrors the mddoai dataset POV |
| [`robocasa_gr1_pnp_cup_to_drawer.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/sim/robocasa_gr1_pnp_cup_to_drawer.yaml) | `gr1` *(scene-fixed)* | `robocasa/gr1/PnPCupToDrawerClose` | `robocasa/gr1/PnPCupToDrawerClose/0` | RoboCasa GR1 humanoid tabletop pnp |
| [`robocasa_panda_mobile_kitchen.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/sim/robocasa_panda_mobile_kitchen.yaml) | `panda_mobile` *(scene-fixed)* | `robocasa/NavigateKitchen` | `robocasa/NavigateKitchen/0` | Mobile-base kitchen navigation; `deploy sim` Nav2 graph compatible |
| [`robocasa_pnp.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/sim/robocasa_pnp.yaml) | `panda_mobile` *(scene-fixed)* | `robocasa/PickPlaceCounterToCabinet` | `robocasa/PickPlaceCounterToCabinet/0` | RoboCasa kitchen pnp smoke |
| [`so101_tube_insertion.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/sim/so101_tube_insertion.yaml) | `so101_follower` *(scene-fixed)* | `so101_box` | `so101_box/tube_insertion` | Box-arena tube-insertion smoke; geometry/sensors/spawn ranges configurable via `BoxSceneOptions` |
| [`tabletop_cube_push.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/sim/tabletop_cube_push.yaml) | `so101_follower` *(free-axis default; pass `--robot` to override)* | `tabletop_push` | `tabletop_push/push_to_goal` | Robot-agnostic cube push-to-goal |
| [`widowx_carrot_on_plate.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/sim/widowx_carrot_on_plate.yaml) | `widowx` *(scene-fixed)* | `simpler_env` | `simpler_env/widowx_carrot_on_plate` | SimScene sibling of the SimplerEnv WidowX carrot benchmark; used by the OpenVLA-OFT issue #55 reproduction path |
| [`aloha_transfer_cube.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/sim/aloha_transfer_cube.yaml) | `aloha_bimanual` *(scene-fixed)* | `aloha_transfer_cube` | `aloha_transfer_cube/0` | gym-aloha bimanual cube-transfer smoke |
| [`franka_tabletop_push.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/sim/franka_tabletop_push.yaml) | `franka_panda` *(scene-fixed)* | `tabletop_push` | `tabletop_push/push_to_goal` | Franka variant of the robot-agnostic cube push-to-goal |
| [`pusht.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/sim/pusht.yaml) | `pusht_2d` *(scene-fixed; 2-D pymunk)* | `pusht` | `pusht/0` | gym-pusht 2-D push smoke |
| [`behavior_turning_on_radio.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/sim/behavior_turning_on_radio.yaml) | `r1pro` *(scene-fixed)* | `behavior` | `behavior/turning_on_radio` | Official public-test instance 0 through the OmniGibson evaluator sidecar |
| [`xr1_robocasa_pnp.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/sim/xr1_robocasa_pnp.yaml) | `panda_mobile` *(scene-fixed)* | `robocasa/PickPlaceCounterToCabinet` | `robocasa/PickPlaceCounterToCabinet/0` | XR-1 observation/action integration check; not Xiaomi's RoboCasa v0.2 benchmark |
| [`xr1_robocasa365_open_drawer.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/sim/xr1_robocasa365_open_drawer.yaml) | `panda_mobile` *(scene-fixed)* | `robocasa/OpenDrawer` | `robocasa/OpenDrawer/0` | 200-step XR-1 RoboCasa365 review rollout, not the target50 evaluation |
| [`xr1_robocasa365_close_blender_lid.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/sim/xr1_robocasa365_close_blender_lid.yaml) | `panda_mobile` *(scene-fixed)* | `robocasa/CloseBlenderLid` | `robocasa/CloseBlenderLid/0` | 150-step XR-1 RoboCasa365 approval rollout, not the target50 evaluation |
| [`isaac_leisaac_pick_orange.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/sim/isaac_leisaac_pick_orange.yaml) | `so101_follower` *(declared)* | `isaac_sim` | `isaac_sim/leisaac_pick_orange` | LeIsaac v0.1.0 PickOrange kitchen (Isaac 4.5–5.1 stage) on the Isaac 6.x backend: pinned stage archive, in-stage oranges + plate (`stage_objects`), success = every orange within the plate's ±0.10 m x/y, ±0.07 m z (LeIsaac's arm-at-rest term is not graded); LeIsaac's front/wrist cameras. 60 Hz control (LeIsaac's one action per physics step), LeIsaac's drive gains; pairs with `rskills/smolvla-so101-leisaac-pick-orange` (its `joint_calibration` is applied by `SimRunner`, `isaac_sim` being `robot_unit_io`) |
| [`xr1_vlabench_select_fruit.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/sim/xr1_vlabench_select_fruit.yaml) | `franka_panda` *(scene-fixed)* | `vlabench` | `vlabench/select_fruit` | Short XR-1 wiring check (front/base/wrist camera order, NF4 rSkill); not a benchmark score |

## BenchmarkScene catalogue (`scenes/benchmark/`)

`SimScene` + required `metadata: BenchmarkMetadata` (paper URL +
`honest_scope`) + non-`None` `seed` and `n_episodes`. The shipped values
match the canonical paper protocol; `openral benchmark scene` against one of
these writes `rskills/<vla>/eval/<scene_id>.json` with `reproduced_locally=true`.
Most are also aggregated into a multi-scene suite under `benchmarks/` (below).

| Config | Fixed / declared robot | `scene.id` | `task.id` | `n_episodes` | Paper |
|---|---|---|---|---|---|
| [`aloha_insertion.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/benchmark/aloha_insertion.yaml) | `aloha_bimanual` *(scene-fixed)* | `aloha_insertion` | `aloha_insertion/0` | 200 | [ALOHA / ACT](https://arxiv.org/abs/2304.13705) |
| [`aloha_transfer_cube.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/benchmark/aloha_transfer_cube.yaml) | `aloha_bimanual` *(scene-fixed)* | `aloha_transfer_cube` | `aloha_transfer_cube/0` | 200 | [ALOHA / ACT](https://arxiv.org/abs/2304.13705) |
| [`libero_spatial.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/benchmark/libero_spatial.yaml) | `franka_panda` *(scene-fixed)* | `libero_spatial` | `libero_spatial/0` | 500 | [LIBERO](https://arxiv.org/abs/2309.11500) |
| [`maniskill_pick_cube.yaml`](https://github.com/OpenRAL/openral/blob/master/scenes/benchmark/maniskill_pick_cube.yaml) | `franka_panda` *(scene-fixed)* | `maniskill3` | `maniskill3/PickCube-v1` | 500 | [ManiSkill3](https://arxiv.org/abs/2410.00425) |
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

`n_episodes` and `seed` ship at the paper-canonical value. Overriding `--n-episodes` on `openral benchmark scene` is allowed
(useful for cheap smoke runs that don't claim paper-reproduction); the
resulting `RSkillEvalResult` records the lowered count.

Multi-scene aggregations (e.g. all 10 LIBERO-Spatial tasks,
`benchmarks/metaworld_mt10.yaml` (10 tasks) and `benchmarks/metaworld_mt50.yaml`
(50 tasks), all 4 SimplerEnv WidowX tasks) live in
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
over the per-tier catalogue (one real rollout per scene × rSkill row) and
classifies each row's outcome into a JSON report + Markdown table. `just sim-eval <config>` runs one scene config through
`openral sim run`; suites run through `openral benchmark run --suite <id>`.

## See also

- [`scenes/README.md`](https://github.com/OpenRAL/openral/tree/master/scenes/README.md)
  — conceptual hierarchy, decision matrix, override flags, scene-id /
  fixed-robot tables, `base_pose` for free-axis scenes, rSkill compatibility,
  live MuJoCo viewer, `policy_extras` performance knobs.
- [Tutorial — Create a sim environment](../tutorials/sim/create-a-sim-environment.md)
  — long-form YAML authoring guide (new scene adapter, new robot manifest,
  custom policy).
