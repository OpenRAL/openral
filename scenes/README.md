# Scene YAMLs (`scenes/`)

This directory holds the three scene tiers.
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
| `DeployScene`     | workcell/deploy context: scene, robot, safety tightening, additive ACM       | `openral deploy sim/run --config scenes/deploy/…` | Live ROS graph (HAL + reasoner + kernel) |
| `SimScene`        | scene + task (single rollout; policy supplied via `--rskill <name>`)         | `openral sim run --config scenes/sim/…`        | One or more `EpisodeResult`s            |
| `BenchmarkScene`  | scene + task + paper metadata + `n_episodes` + `seed` (paper-comparable)     | `openral benchmark scene --config scenes/benchmark/…` | `RSkillEvalResult` JSON         |

Sibling resources:

- [`benchmarks/`](../benchmarks/) — suite-style benchmark YAMLs (bare
  `list[BenchmarkScene]` at the YAML root) that aggregate multiple
  `BenchmarkScene`s under uniform invariants.
  Consumed by `openral benchmark run --suite <id> --rskill <name>`.
- [`deployments/`](../deployments/) — retired; real deploys use
  `scenes/deploy/*.yaml` through `DeployScene`.

## Choosing a tier

| Question                                                                  | Tier              | Notes                                              |
|---------------------------------------------------------------------------|-------------------|----------------------------------------------------|
| "Boot the full stack so the reasoner can pick its own rSkill"             | `DeployScene`     | Env-only; no task; no eval.                        |
| "Run one rollout with a specific rSkill" / "save a debug video"           | `SimScene`        | Single CLI invocation; sized for ad-hoc / smoke.   |
| "Reproduce a paper number for this rSkill" (one scene)                    | `BenchmarkScene`  | Writes a citable `RSkillEvalResult` JSON.          |
| "Reproduce a paper number for this rSkill" (suite of N scenes)            | `list[BenchmarkScene]` | Lives in [`benchmarks/`](../benchmarks/). |

A scene can have a sibling YAML at multiple tiers — e.g. `scenes/benchmark/libero_spatial.yaml`
(paper protocol) and `scenes/sim/libero_spatial.yaml` (ad-hoc smoke). The sim-tier
sibling is **not** valid for paper claims; the loader-strictness gate
(`load_scene_strict`) prevents accidental tier confusion.

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

`openral sim list` walks every subdirectory here and prints each scene YAML
path — paste one straight into `--config`. The `--rskill` half comes from
`openral rskill list`.

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

Scenes registered with `base_pose=True` (free-axis `tabletop_push` and `isaac_sim`; scene-fixed `openarm_tabletop_pnp`) accept an optional `base_pose:` block that anchors the robot
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

Setting `base_pose:` on a scene not registered with `base_pose=True` is a
`ROSConfigError` — those scenes ship their own MJCF and the field has no
physical meaning there. See
the mandatory-mounting-pose design note for the rationale.

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
and Brown (`plumb_bob`) or fisheye (`equidistant`) coefficients through its native
lens schema. An explicit mount `hfov_deg` selects an ideal pinhole instead.
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
        distortion_model: plumb_bob
        distortion_coeffs: [-0.358038981, 0.187982349, -0.000311155, 0.000069646, -0.056267709]
    translucent_materials: true
    exposure_ev: 0.0
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
exposure changes between poses. Both are opt-in; omitted values retain runtime defaults.
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
overrides. Verified on Isaac Sim 6.1.

## rSkill compatibility check

The runner resolves `--rskill <name>` to its `RSkillManifest`, looks up the
configured `RobotDescription`, and verifies the manifest's `embodiment_tags`
and `sensors_required` intersect the robot's capabilities and sensor catalogue
**before** policy load. The check fires inside
`openral_sim.runner._check_rskill_compatibility` and raises `ROSConfigError`
(missing manifest / unregistered robot) or `ROSCapabilityMismatch`
(incompatible) — there is no warn-and-proceed path.

See the original scene-schema design and the later
three-tier-split design that owns the loader strictness for further detail.

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
| `rlbench`       | ❌                 | CoppeliaSim/PyRep sidecar; `render()` returns the latest RGB frame for video capture, not MuJoCo handles. |
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

`openral rskill list` prints every rSkill — in-tree (`rskills/<dir>/rskill.yaml`)
and HF-Hub-installed — with a `source` column so you can tell which are
paste-able right now. Copy any name straight into `--rskill` (e.g.
`--rskill pi05-libero-int8`). Add `--json` for a machine-readable form.

(`openral sim list` is the other half: it prints scene config paths for
`--config`, not rSkills.)

Isaac 6 manifest imports use the world fixed joint as the articulation root, so a fixed arm stays at its declared spawn rather than settling against a floating-base constraint. Scene initial joint positions are seeded before physics initialization and retained on reset.
