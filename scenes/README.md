# Scene YAMLs (`scenes/`)

Three scene tiers, each a Pydantic schema in `openral_core`. Directory layout
matches the schema, the CLI matches the directory, and every YAML loads through
the strict per-tier loader `load_scene_strict(path, expected=<Tier>)`, which
rejects a wrong-tier YAML at parse time.

```
DeployScene  ⊆  SimScene  ⊆  BenchmarkScene
   scenes/deploy/   scenes/sim/   scenes/benchmark/
```

| Tier              | What it pins                                                                 | CLI consumer                                   | Output                                  |
|-------------------|------------------------------------------------------------------------------|------------------------------------------------|-----------------------------------------|
| `DeployScene`     | workcell/deploy context: scene, robot, safety tightening, additive ACM       | `openral deploy sim/run --config scenes/deploy/…` | Live ROS graph (HAL + reasoner + kernel) |
| `SimScene`        | scene + task (single rollout; policy supplied via `--rskill <name>`)         | `openral sim run --config scenes/sim/…`        | One or more `EpisodeResult`s            |
| `BenchmarkScene`  | scene + task + paper metadata + `n_episodes` + `seed` (paper-comparable)     | `openral benchmark scene --config scenes/benchmark/…` | `RSkillEvalResult` JSON         |

Sibling resources:

- [`benchmarks/`](../benchmarks/) — suite YAMLs: a bare `list[BenchmarkScene]`
  at the root, aggregating scenes under uniform invariants. Consumed by
  `openral benchmark run --suite <id> --rskill <name>`.
- [`deployments/`](../deployments/) — retired; real deploys use
  `scenes/deploy/*.yaml` through `DeployScene`.

## Choosing a tier

| Question                                                                  | Tier              | Notes                                              |
|---------------------------------------------------------------------------|-------------------|----------------------------------------------------|
| "Boot the full stack so the reasoner can pick its own rSkill"             | `DeployScene`     | Env-only; no task; no eval.                        |
| "Run one rollout with a specific rSkill" / "save a debug video"           | `SimScene`        | Single CLI invocation; sized for ad-hoc / smoke.   |
| "Reproduce a paper number for this rSkill" (one scene)                    | `BenchmarkScene`  | Writes a citable `RSkillEvalResult` JSON.          |
| "Reproduce a paper number for this rSkill" (suite of N scenes)            | `list[BenchmarkScene]` | Lives in [`benchmarks/`](../benchmarks/). |

A scene can have a YAML at several tiers, e.g. `scenes/benchmark/libero_spatial.yaml`
(paper protocol) and `scenes/sim/libero_spatial.yaml` (ad-hoc smoke). The sim-tier
one is **not** valid for paper claims; `load_scene_strict` prevents tier confusion.

## Run any of them

```bash
# DeployScene — env-only playground (reasoner picks the rSkill at runtime).
openral deploy sim --config scenes/deploy/openarm_tabletop.yaml

# The same robot on real hardware. `deploy run`, not `deploy sim`: it binds the
# cell's real cameras and the real CAN/ros2_control HAL. Bringup MOVES BOTH ARMS
# (openarm_bringup returns to zero on activate) — read the scene's header first.
openral deploy run --config scenes/deploy/openarm_bench.yaml

# BEHAVIOR-1K R1 Pro — official OmniGibson evaluator environment.
openral deploy sim --config scenes/deploy/behavior_r1pro.yaml \
  --initial-task "turn on the radio"

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

Every tier accepts override flags; the YAML pins defaults and the flag wins:

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

`openral benchmark scene` takes only `--n-episodes` of these (the YAML pins
the rest); `openral benchmark run` takes `--task` (one task of the suite) and
`--n-episodes`. A lowered `--n-episodes` is for iteration, not paper numbers.

## Justfile shortcuts

```bash
# SimScene-tier — `openral sim run --save-video`.
just sim-libero                     # SmolVLA × LIBERO        (GPU + MUJOCO_GL)
just sim-xvla-libero                # xVLA × LIBERO           (Florence-2)
just sim-pi05-libero                # π0.5 × LIBERO           (≥8 GB VRAM)
just sim-act-libero                 # ACT × LIBERO            (per-policy comparison)
# XR-1 uses direct commands and an isolated NF4 sidecar:
# OPENRAL_ALLOW_REMOTE_CODE=1 openral sim run --config scenes/sim/xr1_robocasa_pnp.yaml --rskill rskills/xr1-robocasa
# OPENRAL_ALLOW_REMOTE_CODE=1 openral sim run --config scenes/sim/xr1_robocasa365_close_blender_lid.yaml --rskill rskills/xr1-robocasa365

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

A real-hardware `DeployScene` with a StereoLabs ZED (driver block, RGB + depth
bindings, octomap topic, TF and USB-hub pitfalls) is worked through in
[the deploy tutorial](../docs/tutorials/deploy/deploy-run-and-dashboard.md#cameras-whose-stream-only-exists-as-a-ros-topic-ros2_image);
`scenes/deploy/openarm_bench.yaml` is the verified instance.

[Create a sim environment](../docs/tutorials/sim/create-a-sim-environment.md)
covers YAML authoring, adding a robot manifest, and custom scene / policy
adapters. Quick reference:

- A `DeployScene` is a `scene:` block plus optional deploy context:
  `robot_id`, `robot_unit`, `base_pose`, `hal`, `sensors`, `drivers`,
  `runtime`, `safety`, `composition`, `extra_allowed_collision_pairs`,
  `grasp_declaration` / `place_declaration`, `memory_dir`, `seed`.
- A `SimScene` is a `DeployScene` + required `task:` block (+ optional
  `n_episodes`, `record_video`, `save_dir`, `metadata`).
- A `BenchmarkScene` is a `SimScene` with `metadata: BenchmarkMetadata`
  (paper URL + honest_scope string), `seed` and `n_episodes` required.

On a wrong-tier YAML the loader names the tier the file actually fits. A YAML
that still carries a `vla:` block raises `ROSConfigError`; the policy is always
supplied via `--rskill <name>`.

`scene.backend_options` is an opaque dict to the tier schemas; each backend
owns its model (`options_model=` on `@SCENES.register`: RoboCasa's
`RoboCasaBackendOptions`, `tabletop_push`'s `TabletopOptions`, `so101_box`'s
`BoxSceneOptions`). `SCENES.validate_options` runs it at load time in
`openral sim run`, `openral benchmark scene`, `openral deploy validate` and
`make_env`, so a misspelled or out-of-range key fails with a
`ROSConfigError` naming the scene id and the field.

## Available scene IDs (`scene.id`)

| Backend             | Built-in scene IDs                                                                                                                                                                                                                                                                          | Adapter file                              |
|---------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------|
| LIBERO              | `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`                                                                                                                                                                                                                                      | `python/sim/.../backends/libero*.py`      |
| MetaWorld           | `metaworld` (passes `<env_id>` through to `gym.make`)                                                                                                                                                                                                                                       | `python/sim/.../backends/metaworld.py`    |
| gym-aloha           | `aloha_transfer_cube`, `aloha_insertion`                                                                                                                                                                                                                                                    | `python/sim/.../backends/aloha.py`        |
| gym-pusht           | `pusht` (2-D pymunk)                                                                                                                                                                                                                                                                        | `python/sim/.../backends/pusht.py`        |
| RLBench (CoppeliaSim/PyRep) | `rlbench` (scene-fixed Franka Panda; task selected by `backend_options.rlbench_task`) | `python/sim/.../backends/rlbench.py` |
| BEHAVIOR-1K (OmniGibson / Isaac Sim) | `behavior` (scene-fixed R1 Pro; task/instance selected by `backend_options`) | `python/sim/.../backends/behavior.py` |
| RoboCasa (MuJoCo)   | `robocasa` (procedural) + any `robocasa/<Task>` kitchen task + any `robocasa/gr1/<Task>` GR1 tabletop task (resolved by id prefix; a typo fails at build with `ROSConfigError`) | `python/sim/.../backends/robocasa.py`     |
| ManiSkill3 (SAPIEN) | `maniskill3` (scene-fixed Franka Panda; passes `<env_id>` to `gym.make`)                                                                                                                                                                                                                    | `python/sim/.../backends/maniskill3.py`   |
| SimplerEnv (SAPIEN) | `simpler_env` (Bridge V2 digital twin: 4 WidowX tasks on MS3 v3.0.x)                                                                                                                                                                                                                        | `python/sim/.../backends/simpler_env.py`  |
| Custom OpenArm      | `openarm_tabletop_pnp` (bimanual; default top camera matches the mddoai dataset POV)                                                                                                                                                                                                        | `python/sim/.../backends/openarm_*/env.py`|
| Custom SO-101       | `so101_box` (100 × 61.5 × 75 cm box arena + OAK-D Pro overhead RGB-D + wrist camera + tube-insertion task — geometry/sensors/spawn ranges configurable via `BoxSceneOptions`)                                                                                                               | `python/sim/.../backends/so101_box/env.py`|
| Custom tabletop     | `tabletop_push` (robot-agnostic cube push-to-goal; free-axis — pass `--robot`; SO-101 sim YAML pins pi0.5-style degree reset pose + top/front/wrist camera routing)                                                                                                                        | `python/sim/.../backends/tabletop_push/env.py` |
| RoboTwin 2.0 (SAPIEN, sidecar) | `robotwin` (scene-fixed AgileX ALOHA; task selected via `backend_options`, e.g. `lift_pot`, `beat_block_hammer`, `handover_block`, `place_empty_cup`, `stack_blocks_two`)                                                                                                            | `python/sim/.../backends/robotwin.py`     |
| VLABench (lerobot envs) | `vlabench` (scene-fixed Franka Panda; task selected via `backend_options`, e.g. `select_fruit`)                                                                                                                                                                                        | `python/sim/.../backends/vlabench.py`     |
| Isaac Sim (sidecar) | `isaac_sim` (free-axis; any manifest robot imported from its URDF with `layout: manifest`, inside any environment USD named by `scene.assets_uri` — see [below](#isaac-sim-any-stage-any-robot)) | `python/sim/.../backends/isaac_sim.py` |

`openral sim list` prints every scene YAML path here, ready for `--config`;
`openral rskill list` gives the `--rskill` half (see
[below](#discovering-paste-able---rskill-strings)).

## Scene-fixed robots

Some scenes hard-wire the physics robot(s) via `@SCENES.register(..., fixed_robot=...)`
(one id or a set):

| Scene                                            | Fixed robot         |
|--------------------------------------------------|---------------------|
| `libero_spatial` / `libero_object` / `libero_goal` / `libero_10` | `franka_panda`    |
| `metaworld`                                      | `sawyer`            |
| `pusht`                                          | `pusht_2d`          |
| `aloha_transfer_cube` / `aloha_insertion`        | `aloha_bimanual`    |
| `rlbench`                                       | `franka_panda`      |
| `so101_box`                                      | `so101_follower`    |
| `robocasa`, `robocasa/*` (kitchen)               | `panda_mobile`, `panda_mobile_vslam` |
| `robocasa/gr1/*` (humanoid tabletop)             | `gr1`               |
| `behavior`                                       | `r1pro`              |
| `robotwin`                                       | `aloha_agilex`       |
| `vlabench`                                       | `franka_panda`       |
| `maniskill3`                                     | `franka_panda`       |
| `simpler_env`                                    | `widowx`             |
| `openarm_tabletop_pnp`                           | `openarm`            |

One rule, `SCENES.resolve_robot`, binds the robot in `sim run`, `benchmark`,
`deploy sim` and the sim HAL: a `--robot` / `robot_id:` outside the scene's set
raises `ROSConfigError` at config-build time (the message lists the allowed
robots); a matching one is accepted; an omitted one takes the scene's default.
Free-axis scenes (`tabletop_push`, `isaac_sim`, `mock`) require a robot.

## Placing robots with `base_pose:`

Scenes registered with `base_pose=True` (free-axis `tabletop_push` and
`isaac_sim`; scene-fixed `openarm_tabletop_pnp`) accept an optional `base_pose:`
block anchoring the robot in the scene's world frame. Adapters write the
`world → base_frame` transform (`RobotDescription.base_frame`) into the scene's
MJCF at load:

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

Setting `base_pose:` on a scene not registered with `base_pose=True` is a
`ROSConfigError`: those scenes ship their own MJCF, so the field has no
physical meaning there (rationale: the mandatory-mounting-pose design note).

### Isaac Sim: any stage, any robot

The `isaac_sim` scene loads an external **environment USD** around any manifest
robot. A deploy scene needs exactly three fields
([`deploy/isaac_panda_mobile_warehouse.yaml`](deploy/isaac_panda_mobile_warehouse.yaml)):

```yaml
robot_id: panda_mobile              # imported from robots/<id>/ (its URDF)
base_pose:                          # spawn in the stage's world frame
  xyz: [-4.8, 0.0, 0.0]
  quat_xyzw: [0.0, 0.0, 0.7071068, 0.7071068]   # yaw only — robots stand upright
  frame_id: world
scene:
  id: isaac_sim
  backend: isaacsim
  assets_uri: "isaac:Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd"
```

`assets_uri` takes a local path (`file://` optional, relative to the working
directory), an `http(s)://` / `omniverse://` URL, or `isaac:<path>` — a path
under the sidecar's own Isaac asset root, so the asset matches the installed
Isaac release. Either field implies `backend_options.layout: manifest`. The
environment replaces the bring-up ground plane, so it must carry its own floor
collider; a mobile base's odometry starts at the spawn. Assets authored in
Blender or elsewhere load the same way once exported to USD. NVIDIA's Isaac
assets are licensed for use inside Isaac Sim, where they are referenced at run
time — never converted or vendored.

**Time base.** One scene step is one control period of the robot
(`action_spec.control_freq_hz`, which the manifest must declare): the scene
runs physics at the smallest multiple of the control rate at or above Isaac's
default 60 Hz (60 Hz for a 30 Hz robot, 100 Hz for 50 Hz, 75 Hz for 25 Hz) and
`physics_substeps = physics_hz / control_freq_hz` PhysX steps per action
(2 for a 30 Hz robot), rendering only the last so cameras stay at the control
rate; the kinematic base integrates the same tick. The robot's rate is never
changed to suit the simulator; a scene that must run at another rate (a checkpoint
trained in a simulator that stepped faster) says so with
`backend_options.control_freq_hz`, and then `deploy sim` runs the whole graph at it:
`openral_core.sim_scene_description` re-times the robot once, and the CLI, the launch
(the runner's `rate_hz`, the kernel's control period, the recorder's fps) and the
scene-attached HAL (`build_hal`) all apply it, so `SimAttachedHAL.connect` holds the
scene to that rate and the runner still refuses a skill trained at any other one.
A bare MuJoCo twin derives its time base from the manifest and refuses the key. The
sidecar reports the resulting `sim_dt_per_tick_s` in its `ping`, and
`SimAttachedHAL.connect` refuses the scene if it is not one control period, so
a `deploy sim` rehearsal on Isaac traverses a trajectory at the commanded speed
in simulation time and the kernel's velocity / tracking checks read the real
clock (issue #355). The HAL logs the time base once per connect as
`sim.time_base`. A scene class may pin `physics_substeps` explicitly; the pin
wins and is reported as an override.

The MuJoCo-backed deploy scenes follow the same rule (issue #358). RoboCasa
and LIBERO build robosuite at its native 20 Hz, which under a 30 Hz robot would
rehearse at 1.5x speed; every `scenes/deploy/robocasa_*.yaml` / `libero_*.yaml`
therefore pins `backend_options.control_freq_hz` to the manifest rate, the
factory shrinks robosuite's physics time step so the period is a whole number
of steps (1/510 s for 30 Hz; `pin_robosuite_time_base`), the env reports what
its step loop really integrates as `sim_dt_per_tick_s`, and the HAL guard
refuses a scene that forgot the pin. Benchmark scenes leave it unset and keep
the 20 Hz protocol their published numbers were measured at. The bare MuJoCo
twins (`MujocoArmHAL`) need no scene knob: with no `settle_steps` pin they
derive the physics steps per action from `action_spec.control_freq_hz` at
connect (17 steps at 1/510 s for a 0.002 s MJCF at 30 Hz) and stop the
wall-time idle stepper from pacing the clock while a skill is active.

**Camera images lag the state, and say so.** Isaac's RTX pipeline delivers a
camera frame 2–4 steps after the physics state it shows (133 ms at 30 Hz in the
tote scene), and not at a constant lag. The scene does not render extra frames
to hide it: every observation carries `image_time_ns`, the sim time each camera's
frame shows, and `deploy sim` stamps `/openral/cameras/*` (RGB, depth, colour,
clouds) with it, so consumers pair images with joint states and TF at the right
time, as with a real camera's latency (issue #361). `sim run` hands the policy
the latest frame with the current state, as a real robot would; it does not align
the state to the image time.

**Frames are rendered on demand.** Rendering is most of an Isaac step's cost (25 of
47 ms per step on the tote scene, GB10), so neither caller renders every tick:
`sim run` renders the policy's observation only on the step before it infers (a
chunked policy reads images once per chunk) and the video at 10 Hz, and `deploy
sim` renders once per camera/depth publish period (`camera_publish_rate_hz`,
`depth_publish_rate_hz`). A frame rendered for the policy or a publish is
*fresh*: the scene keeps rendering until every camera's frame shows that step's
state, so it carries no pipeline lag; steps in between reuse the last frames,
which keep the `image_time_ns` of the state they show. The sidecar boots Kit's zero-delay experience when the install ships it, so a frame trails the physics by ~2 renders instead of ~4 and a fresh frame needs half the catch-up. Depth cameras ship their registered RGB-D frame only; the HAL deprojects it into the published point cloud, each point coloured by its own pixel. A recorder or
`OPENRAL_DUMP_OBS_TICK` brings back a render every step. Render the cameras at the
size the policy consumes (`camera_render_height`): a policy that resizes to 224 px
gains nothing from a 720p raster.

**Keep Isaac open between runs.** Kit's boot costs 10–15 s per `sim run`. With
`--keep-sim-open` (`backend_options.keep_alive`) the sidecar stays up when the run
ends, and the next run of the same scene adopts it: same launch arguments and the
same sidecar code (the `launch_id` its ping reports), never a different scene or
checkout. The first run without the flag ends it. For several seeds of one scene,
`--n-episodes N` in one run is simpler still.

**Pickable objects** go in `backend_options.objects` (validated by
`IsaacSimOptions`, so a typo fails at load):

```yaml
  backend_options:
    objects:
      - name: cracker_box          # prim name; also the key in step info
        usd: "isaac:Isaac/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd"
        xyz: [-0.45, 4.80, 0.40]   # world frame; dropped a few cm, it settles
        yaw: 0.0                   # rad, optional
        dynamic: true              # graspable rigid body (default) vs static prop
```

A dynamic object without physics gets a rigid body and convex-hull colliders;
`world.reset()` returns every object to its declared pose. Step `info` carries
`object_positions` and `robot_position` (simulator ground truth, not a policy
observation).

**Objects the stage already holds** (rigid bodies authored inside the
environment USD) are named, not spawned, in `backend_options.stage_objects`:
`{name, prim}`, the prim's path below the USD's default prim (the stage is
referenced at `/World/environment`). The prim must carry a rigid body (the boot
fails otherwise); each reset returns it to its authored pose, and it is a scene
object for `placement_success` and step `info` like `objects`. A placement region
may follow one: `relative_to: plate` makes `min_xyz` / `max_xyz` offsets (world
axes) from the plate's centre at each graded step.

**Stages from older Isaac releases or other tools** may need two opt-in load
fixes, both logged at boot: `strip_xform_colliders: true` removes
`UsdPhysics.CollisionAPI` from `Xform` prims of the environment (Isaac 6.1 PhysX
rejects a collider that is no geometry: "Supplied PxGeometry is not valid"), and
`dome_light: {intensity, color}` lights a stage that ships no lights. A stage
published only as a zip comes from `environment_archive: {url, sha256, usd}`
instead of `scene.assets_uri`: downloaded once into
`~/.cache/openral/isaac-environments/` (`$OPENRAL_CACHE_DIR`), checked against the
pinned sha256, extracted; never committed. Shipped:
[`sim/isaac_leisaac_pick_orange.yaml`](sim/isaac_leisaac_pick_orange.yaml), LeIsaac's
(LightwheelAI, Apache-2.0) PickOrange kitchen built for Isaac 4.5–5.1, with the SO-101
at LeIsaac's spawn, its oranges and plate graded in place, and LeIsaac's front and
wrist cameras as `camera_mounts` (Isaac Lab ROS-convention offsets, `axes: ros`; the
field of view `hfov_deg = 2·atan(horizontal_aperture / (2·focal_length))` of its
`PinholeCameraCfg`), LeIsaac's STS3215 drive gains (`joint_drive_gains`) and its
time base: LeIsaac records and evaluates one action per 1/60 s physics step, so the
scene sets `control_freq_hz: 60` (an `IsaacSimOptions` override of the robot's
`action_spec.control_freq_hz`, logged at build). Run it with
[`rskills/smolvla-so101-leisaac-pick-orange`](../rskills/smolvla-so101-leisaac-pick-orange/)
and `--instruction "Grab orange and place into plate"` (the dataset's string; the
rubric grades it too). `isaac_sim` is a `robot_unit_io` scene: its state is the
manifest joint vector, so `sim run` applies the rSkill's `PolicyIOCodec` (here
LeIsaac's per-joint motor-unit `joint_calibration`) exactly as the deploy runner
does. LeIsaac's dynamic gripper effort limit (nearest object's mass / 0.15) is held
at the oranges' 1.0 N·m (`joint_effort_limits`). The finger colliders use LeIsaac's
`convexDecomposition` (`collision_approximation`; the URDF converter's single hulls are
2.3–2.6× the finger meshes and pinch an orange off-centre) and the PhysX scene LeIsaac's
friction correlation distance and bounce threshold (`physx_scene`). Not reproduced: LeIsaac's per-episode
object and camera randomization, and the arm-at-rest term of its success check.
Its deploy twin, [`deploy/isaac_leisaac_pick_orange.yaml`](deploy/isaac_leisaac_pick_orange.yaml),
runs the same kitchen under `openral deploy sim` at 60 Hz with the safety kernel's
self-collision and world-voxel checks on. The SO-101's cameras are RGB, so the scene
adds a simulated depth camera at LeIsaac's front view through
`backend_options.sim_sensors`: depth `SensorSpec`s with intrinsics and no
`deploy_binding`, under names the manifest does not use, appended to the robot's
sensors for this scene only (`sim_scene_description`), mounted by `camera_mounts`.
The HAL publishes its cloud, octomap maps it, and the kernel checks every chunk
against the voxels; the robot manifest, and so a real deploy, is unchanged.
Both Isaac deploy twins commit `runtime.object_tracking: tracking` (ADR-0121 step 3) with
`grounding_required: true` and the grounding wait the successful Spark runs used
(pick-orange 8 s, R372; banana-in-bowl 15 s, R365, 2026-10-23). The #418 object tracker
grounds the target query, owns the attachment set (the held payload is the tracker's box,
cleared from the voxel map by tracked identity) and measures the boxes the kernel's open box
rules take: the grasp region (the target track's box until the hand holds it), the place
region (the tracked destination; the runner arms its goal-scope place declaration) and the
workspace region. The HAL's grasp- and place-target legs stand down. `on` +
`exemption_provider:` (OpenRAL Pro, ADR-0121) adds the tracker-keyed rules those runs used.
History: Spark r6 (2026-10-25) under the removed `shadow` (HAL legs measuring) went 0/5 on
both twins; the HAL leg fit the 5.7 cm orange as ~10 x 10 x 18 cm and never measured the
banana. `tracking` has not been run in sim yet: an open-only tier-3 run is owed.
Their target query has no seed, so both also commit
`object_detector_locators: [omdet-turbo-locator]` to ground it; that needs the `omdet`
dependency group (`just sync --group omdet`). SIM-ONLY, pending Safety-WG review.

[`sim/isaac_arena_droid_banana_in_bowl.yaml`](sim/isaac_arena_droid_banana_in_bowl.yaml)
is the second Isaac task: Isaac Lab-Arena's DROID / RoboLab `banana_in_bowl`
(Apache-2.0) with `robots/droid_franka_robotiq` (Franka + Robotiq 2F-85, composed
URDF) on Arena's stand at its maple table, every asset referenced in place under the
Isaac 6.1 asset root (`isaac:Isaac/IsaacLab/Arena/...`). The banana and bowl are
`objects` at RoboLab's fixed layout (Arena re-samples one per reset); the success
region follows the bowl (`relative_to`). The Robotiq's leader `finger_joint` closes
toward +0.8 rad, the opposite sense of OpenRAL's `[0 = closed, 1 = open]` jaw, so
`gripper_joint_curve` maps it onto Arena's 0 .. pi/4 travel. Run it with
[`rskills/pi05-droid-jointpos-arena`](../rskills/pi05-droid-jointpos-arena/); its
deploy twin is [`deploy/isaac_arena_droid_banana_in_bowl.yaml`](deploy/isaac_arena_droid_banana_in_bowl.yaml), which maps the world from a simulated
`exterior_depth` camera (`sim_sensors`) for the kernel's world-voxel check.

**Cameras**: every manifest RGB/depth sensor becomes an Isaac camera with a
1 cm near clip, rendered at its manifest intrinsics' raster (a depth camera at
that aspect, no wider than `observation_width`) with their FOV. Each is mounted
from the manifest, a selected robot unit's calibration included
(`$OPENRAL_ROBOT_UNIT`, which `deploy` sets from the scene's `robot_unit`): a
sensor that `shares_mount_with` another rides that one's mount, a sensor with
`parent_frame` + `static_transform_xyz_rpy` rides that link, and one the robot's
MJCF declares a camera for rides the MJCF body at that camera's pose (`axes:
usd`, the MJCF's FOV unless the unit calibrated the intrinsics). A camera none of
these places gets a generic base-relative viewpoint (not its `frame_id`'s view —
the load prints every robot-framed camera left that way; one can render
all-black inside the stage). `backend_options.camera_mounts` overrides a mount:

```yaml
    camera_mounts:
      head_zed:                    # manifest sensor name
        link: openarm_base         # URDF link, or the manifest base_frame; omit = robot root
        xyz: [0.0, 0.0, 0.2]       # in that link's frame
        quat_wxyz: [0.830285, -0.005106, 0.557306, 0.003427]  # or look_at: [x, y, z]
        axes: world                # quat convention: world (x fwd, z up) | usd | ros
        hfov_deg: 90.0             # optional; default from the manifest intrinsics
```

Depth clouds are published in the manifest `base_frame` with misses dropped,
and since the camera renders the robot, `deploy sim` runs the robot self-filter
in front of octomap for an `isaacsim` scene, as on real hardware.

For calibrated RGB cameras, Isaac applies the complete `fx`, `fy`, `cx`, `cy`
and Brown (`plumb_bob`, 5), OpenCV rational (`rational_polynomial`, 8) or fisheye
(`equidistant`, 4) coefficients through its native lens schema. A radial model that
folds inside the image is refused at boot: Isaac inverts the lens per output pixel, and
past the fold it has no ray, so it would render swirls along the edges. A strong Brown fit
can fold (the Oct-2 ZED-M fit turns over at normalised radius 0.858, its corners sit at
0.995); extend it with `rational_polynomial` k4-k6 fitted to stay monotonic. An explicit mount `hfov_deg` selects an ideal pinhole instead.
Depth keeps its existing pinhole/deprojection path. The camera schema is read back
and logged at boot; Isaac versions without this API fail rather than drop calibration.

`backend_options.camera_intrinsics` overrides **RGB** image models for a simulation
without changing the physical robot's stream bindings. For example, reproducing a
raw training camera alongside a rectified depth camera:

```yaml
    camera_intrinsics:
      top:
        width: 672
        height: 376
        fx: 388.508267
        fy: 388.527128
        cx: 336.707405
        cy: 190.106848
        # The Oct-2 Brown fit extended with monotonic rational terms (equal to it within
        # 0.007 px inside its calibrated radius; the 5-term fit folds inside the frame).
        distortion_model: rational_polynomial
        distortion_coeffs: [-1.441958182, 0.728991661, -0.000311155, 0.000069646,
                            0.195321664, -1.082444798, 0.136938664, 0.568911907]
    translucent_materials: true
    exposure_ev: 0.0
    image_blur_sigma_px: 0.9
    robot_material_colors:
      "*matte_black*": [0.03, 0.03, 0.03]
```

These top-camera numbers describe the **unrectified ZED-M left lens** measured on
2 October 2026; they do not calibrate an SDK-rectified image. Use the profile that
matches the recording pipeline. Native rasters are preserved before policy preprocessing.
`translucent_materials` enables RTX refraction and indirect light, including the
Real-Time 2.0/path-tracing limits (eight total bounces, twelve specular/transmission
bounces), legacy refraction limit (eight), and the DLSS Quality profile for
small camera rasters. Settings are applied after stage
initialization; USD files alone do not transfer those process settings. Material
compatibility still requires a rendered check; enabling this option alone does not
guarantee transparent plastic. `exposure_ev` selects fixed
manual exposure (ISO 100, 20 ms, f/5 at zero; +1 doubles exposure), preventing automatic
exposure changes between poses. `image_blur_sigma_px` blurs every RGB frame by that
Gaussian sigma: RTX renders are sharper than a real camera's compressed stream (the
pick-and-place training frames match about 0.9 px). `robot_material_colors` recolours the imported
robot's visual materials by name (`fnmatch` on the material prim name, diffuse RGB): the
OpenArm meshes' "matte_black" ships at diffuse 0.247, which renders mid-grey next to the
real near-black plastic. All are opt-in; omitted values retain
runtime defaults.
The complete resolved robot spec is hashed into the sidecar handshake, so changing
calibration, initial state, or renderer settings cannot reuse a stale sidecar silently.

**Any manifest robot** with an `assets.urdf` imports — manifest joints are
matched to URDF joints by the URDF's own structure, each gripper's mimic finger
follows its leader, and `package://` meshes resolve via `AMENT_PREFIX_PATH`
(a sourced workspace), an ancestor directory of the URDF, or — for known public
packages such as Enactic's `openarm_description` — a pinned clone fetched into
the openral cache. Shipped:
[`isaac_panda_mobile_warehouse.yaml`](deploy/isaac_panda_mobile_warehouse.yaml)
(navigate the aisle to a pallet of YCB props) and
[`isaac_openarm_warehouse.yaml`](deploy/isaac_openarm_warehouse.yaml) (bimanual
OpenArm at the same pallet).

**Driving it**: the scene takes absolute joint targets — the meaning the HAL
sends — and packs typed commands by name: JOINT_POSITION by `joint_names` (or a
whole-vector row), GRIPPER_POSITION by `ee_name`, BODY_TWIST into the base; an
untouched joint holds. A bimanual slot tick (left arm, left gripper, right arm,
right gripper) commits as one simulator step.

**Isaac install**: a host-wide binary install (`/opt/isaac-sim`, `~/isaacsim`)
is picked up automatically after the pip venv; `OPENRAL_ISAAC_SIDECAR_PYTHON`
overrides. Verified on Isaac Sim 6.1. Isaac 6 manifest imports use the world
fixed joint as the articulation root, so a fixed arm stays at its declared spawn
rather than settling against a floating-base constraint. Scene initial joint
positions are seeded before physics initialization and retained on reset.

## rSkill compatibility check

The runner resolves `--rskill <name>` to its `RSkillManifest`, looks up the
configured `RobotDescription`, and verifies the manifest's `embodiment_tags`
and `sensors_required` intersect the robot's capabilities and sensor catalogue
**before** policy load. The check fires inside
`openral_sim.sim_runner._check_rskill_compatibility` and raises `ROSConfigError`
(missing manifest / unregistered robot) or `ROSCapabilityMismatch`
(incompatible); there is no warn-and-proceed path. Background: the original
scene-schema design and the later three-tier-split design (loader strictness).

## Live MuJoCo viewer

`openral sim run` and `openral benchmark scene` open a passive `mujoco.viewer`
window by default and stream the rollout in real time. Toggle with
`--view / --no-view`. It is auto-disabled (one WARNING line, no error) when:

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
| `rlbench`       | ❌                 | CoppeliaSim/PyRep sidecar; `render()` returns the latest RGB frame for video capture, not MuJoCo handles. |
| `pusht`         | ❌                 | gym-pusht is 2D; no MuJoCo. Runs offscreen even with default `--view`.                                 |
| `maniskill3`    | ❌                 | SAPIEN backend; use `--view` with SAPIEN's own GUI window (separate from `mujoco.viewer`).             |
| `simpler_env`   | ❌                 | Same — SAPIEN-backed.                                                                                  |

## Performance knobs (rSkill `policy_extras`)

Lerobot-style families (`smolvla`, `act`, `pi05`) honour these rSkill-manifest
`policy_extras` fields, which gate inference speed:

| Key                                | Default                                                                                  | What it does |
|------------------------------------|------------------------------------------------------------------------------------------|--------------|
| `n_action_steps`                   | Per-family paper default: SmolVLA / π0.5 = `chunk_size` (synchronous mode); ACT = `1` (per-step re-inference, pair with `temporal_ensemble_coeff`); Diffusion = `8` pinned in the adapter | Number of actions consumed from each predicted chunk before the policy re-infers. The shipped lerobot checkpoints set this to **1**, which for non-ACT families throws the chunk away and pays a full forward every env step. SmolVLA / π0.5 papers document `inference_mode: synchronous` (drain the full chunk); ACT's paper protocol is temporal ensembling (per-step re-inference + weighted average); Diffusion Policy fixes `n_action_steps=8` of a 16-step horizon. The adapters install the per-family paper default so `openral benchmark run` reproduces published numbers. Implemented in `openral_rskill._vla_core.apply_chunk_replay` and per-adapter `_build_*` factories. |
| `temporal_ensemble_coeff` (ACT)    | `0.01` (paper value, Zhao et al. §V-B)                                                   | Engages ACT's temporal-ensembling buffer. Set to `null` to disable (falls back to plain chunked execution). On `gym_aloha/AlohaTransferCube-v0` the published `lerobot/act_aloha_sim_transfer_cube_human` checkpoint runs ~0.46 with TE disabled and approaches the paper's 0.95 with TE on. Implemented in `openral_sim.policies.act._apply_temporal_ensemble`. |
| `compile`                          | `false`                                                                                  | Opt-in `torch.compile` of the heavy chunk forward (`policy._get_action_chunk`). Skipped on CPU. Best-effort: setup or runtime backend errors degrade to eager and log `vla_compile_setup_failed` / `vla_compile_runtime_fallback` (see `openral_rskill._vla_core.maybe_compile_chunk_forward`). **Not exposed for `pi05`** — that adapter forces `compile_model = False` to keep the nf4 quantization path stable. |
| `compile_mode`                     | `"default"`                                                                              | Torch compile mode: `default`, `reduce-overhead` (CUDA graphs, recommended), `max-autotune` (longest warmup). |

### When to enable `compile`

Off by default because it requires:

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

`openral rskill list` prints every rSkill, in-tree (`rskills/<dir>/rskill.yaml`)
and HF-Hub-installed, with a `source` column showing which are paste-able now.
Copy any name into `--rskill` (e.g. `--rskill pi05-libero-int8`); `--json` gives
a machine-readable form.
