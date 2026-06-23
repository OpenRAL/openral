# ADR-0045: NVIDIA Isaac Sim as an optional sim backend

- Status: **Proposed**
- Date: 2026-06-10
- Related: [ADR-0002](0002-eval-and-sim-environments.md) (eval & sim environments);
  [ADR-0031](0031-sim-real-hal-separation.md) (`build_hal` sim/real seam);
  [ADR-0033](0033-robot-parameterized-native-scenes.md) (robot-parameterised native scenes);
  [ADR-0034](0034-deploy-sim-scene-attach-for-arms.md) (deploy-sim scene-attach);
  [ADR-0012](0012-open-core-licensing.md) (open-core licensing posture).
- ADR number note: `0043` is an unfilled gap in `docs/adr/` (renumber-in-flight). This ADR
  claims `0045` (next after the highest present, `0044`) to avoid colliding with whatever
  lands at `0043`.

## Context

OpenRAL drives VLA-policy rollouts through a single minimal scene seam. Every simulator —
native MuJoCo (`tabletop_push`, `openarm_robosuite`), LIBERO, MetaWorld, RoboCasa, ALOHA,
PushT, ManiSkill3 (SAPIEN), SimplerEnv — is a `SimRollout` factory registered under a
`scene_id` in `openral_sim.registry.SCENES`. `openral sim run --config <scene>.yaml`
resolves the factory and runs the episode loop; `openral deploy sim` wraps the same
`SimRollout` in a ROS lifecycle node via `SimAttachedHAL` (ADR-0034).

There is recurring interest in **NVIDIA Isaac Sim** (PhysX + RTX rendering + USD, on the
Omniverse Kit platform) for photoreal rendering, GPU-parallel rollouts, and USD asset
pipelines. The scaffolding already anticipates it: `PhysicsBackend.ISAACSIM = "isaacsim"`
exists in `openral_core.schemas` (tagged "Future"). This ADR records **how** Isaac Sim
would integrate and the two judgment calls that must be settled **before** code is written,
because both cross a layer boundary (a new sim backend) and pull in a closed dependency
(CLAUDE.md §3, §1.9).

### What the seam already gives us (the easy half)

The integration surface is small and well-precedented (ManiSkill3 and SimplerEnv are both
non-MuJoCo, free-axis backends following the same path):

- **`SimRollout` Protocol** (`python/sim/src/openral_sim/rollout.py`) — four methods:
  ```python
  @runtime_checkable
  class SimRollout(Protocol):
      scene: SceneSpec
      task: TaskSpec
      def reset(self, seed: int | None = ...) -> Observation
      def step(self, action: NDArray[np.float32]) -> StepResult
      def render(self) -> NDArray[np.uint8] | None
      def close(self) -> None
  ```
  Optional duck-typed extensions (`mujoco_handles`, `viewer_render`,
  `enable_intrinsic_viewer`) are **not** part of the Protocol; Isaac Sim implements none of
  them (it has no MuJoCo handles and a self-managed viewport).
- **One registration** — `@SCENES.register("isaac_sim", fixed_robot=None)` in a new
  `python/sim/src/openral_sim/backends/isaac_sim.py`, plus an import line in
  `backends/__init__._register_backends()`.
- **Schema** — `PhysicsBackend.ISAACSIM` already exists; no schema change.
- **Dependency isolation** — a new `isaacsim` group in `pyproject.toml`, mirroring the
  per-backend `libero` / `robocasa` / `maniskill3` groups.
- **Control** — Isaac Lab ships first-class OSC (`OperationalSpaceControllerActionCfg`) and
  differential-IK (`DifferentialIKControllerActionCfg`) end-effector action terms, matching
  the robosuite-OSC convention OpenRAL already mandates for new arm scenes.

The hard half is **runtime**, not plumbing, and is captured in the two decisions below.

## Decision

Integrate Isaac Sim as an **optional, externally-provisioned, free-axis `SimRollout`
backend**, subject to the two constraints below. **Isaac Lab** (not raw Isaac Sim) is the
integration target: its `ManagerBasedRLEnv` subclasses `gymnasium.Env` and provides the
reset/step/observation/reward/termination managers we adapt to `SimRollout`.

### Decision 1 — Process model: out-of-process Python 3.11 sidecar

Isaac Sim wheels are hard-pinned per interpreter (4.x→py3.10, 5.x→py3.11, **6.x→py3.12**).
OpenRAL pins `>=3.12,<3.13`, so an **in-process** backend can only use Isaac Sim **6.x** —
the newest, least-proven line — and must share one venv with the VLA torch/CUDA stack
(known `LD_PRELOAD`/libgomp OpenMP clash) under the rigid `SimulationApp`-before-`omni.*`
import ordering.

**Decision: adopt an out-of-process sidecar.** A long-lived Isaac Sim process in its own
py3.11 environment running Isaac Lab 5.1 (the mature line), fronted by an in-process
`IsaacSimEnv` (`SimRollout`) that marshals `reset`/`step`/`render` over an IPC channel —
the same out-of-process isolation pattern OpenRAL already uses for the LocateAnything
detector rSkill. This decouples Isaac's interpreter and torch/CUDA stack from OpenRAL's,
sidesteps the import-order and OpenMP constraints, lets us pin the mature 5.1 line
independent of the repo's 3.12 interpreter, and amortises the tens-of-seconds Omniverse Kit
startup across an episode instead of paying it per test.

The in-process 6.x-on-3.12 single-env path is **explicitly deferred** as a documented
future fallback, to be revisited only if 6.x matures and the IPC overhead proves limiting;
it is not implemented under this ADR.

### Decision 2 — Licensing: externally provisioned, never vendored

Isaac Lab is BSD-3 and Isaac Sim **source** is Apache-2.0 (both compatible with open-core),
but the **bundled Omniverse Kit SDK, models, and textures** ship under the proprietary
"NVIDIA Isaac Sim Additional Software and Materials License," which restricts
redistribution. Per CLAUDE.md §1.9 / §3 ("Closed-SDK code is not bundled in open-core"),
Isaac Sim is treated like GR00T weights and other closed components:

- **Never vendored.** No Omniverse Kit binaries, USD assets, or wheels checked into the
  repo. The `isaacsim` dependency group documents the NVIDIA index + license requirement;
  it does not bundle the artifacts.
- **Install-time guard.** The factory lazy-imports the SDK and raises a typed
  `ROSConfigError` with a provisioning hint when absent — failing closed, never silently
  degrading. No commercial-deployment guard beyond NVIDIA's own terms is added here; the
  posture is "external dependency the user provisions," matching ADR-0012.

### Backend shape

```python
# python/sim/src/openral_sim/backends/isaac_sim.py
@SCENES.register("isaac_sim", fixed_robot=None)   # free-axis: any manifest robot
def _build_isaac_sim(env_cfg: SimEnvironment) -> SimRollout:
    try:
        from openral_sim.backends._isaac_bridge import IsaacSidecarClient  # lazy
    except ImportError as exc:
        raise ROSConfigError(
            "Isaac Sim backend unavailable. Provision NVIDIA Isaac Sim / Isaac Lab "
            "(separate license, RTX GPU) and `uv sync --group isaacsim`."
        ) from exc
    return IsaacSimEnv(env_cfg, client=IsaacSidecarClient.launch(env_cfg))
```

Observation contract is unchanged: `dict` with `images` (HWC uint8 RGB per camera),
`state` (1-D float32 proprioception), `task` (instruction str). The sidecar marshals Isaac
Lab's GPU tensors to numpy and unbatches the `num_envs=1` leading dim before returning.

## Safety posture

No safety-kernel, E-stop, or velocity-limit code is touched. Isaac Sim is a simulation
backend behind the existing `SimRollout`/`SimAttachedHAL` seam (ADR-0034); the
real-hardware exclusion in `build_hal` (`mode="real"` + scene attach → `ROSConfigError`)
already prevents a sim scene from ever attaching to physical actuators. No new flag disables
or bypasses any safety check.

## Alternatives considered

1. **In-process Isaac Sim 6.x on py3.12.** Rejected as the default: forces the newest/least
   proven line, shares one venv with the VLA torch stack (OpenMP/`LD_PRELOAD` clash), and
   imposes `SimulationApp`-first import ordering on the whole process. Kept as a documented
   future fallback once 6.x matures.
2. **Relax the repo's `>=3.12,<3.13` pin to allow py3.11.** Rejected: a repo-wide
   interpreter change to accommodate one optional backend is disproportionate and crosses
   far more than the sim layer.
3. **Raw Isaac Sim (no Isaac Lab).** Rejected: we would hand-roll the env/MDP managers that
   Isaac Lab already provides as a `gymnasium.Env`; more code, less standard.
4. **Don't integrate; stay on MuJoCo/SAPIEN.** The status-quo option. Valid until a concrete
   need for RTX photoreal rendering, USD pipelines, or GPU-parallel rollouts materialises —
   this ADR makes the path ready without committing implementation effort prematurely.

## Consequences

- **Positive:** RTX photoreal rendering, USD assets, and GPU-parallel rollouts become
  available behind the existing scene seam; the OSC/diff-IK control surface maps cleanly to
  the robosuite-OSC convention; out-of-process isolation keeps Isaac's heavy stack from
  contaminating the core env.
- **Negative / costs:** RTX-only GPUs (datacenter A100/H100 unsupported), ~50 GB install,
  tens-of-seconds Kit startup (mitigated by the long-lived sidecar), an IPC marshalling
  layer to build and test, and a closed dependency that can never be vendored. Sim tests
  for this backend will not fit the `<10 min` budget on a cold CI runner without a
  GPU-equipped self-hosted runner; absent one, they take the legitimate `pytest.skip` path
  (CLAUDE.md §1.12).
- **Follow-up work (separate PRs, gated on this ADR):** (a) `_isaac_bridge` sidecar + IPC;
  (b) `IsaacSimEnv` `SimRollout` adapter + registration; (c) `isaacsim` dependency group;
  (d) an example `scenes/sim/isaac_sim_*.yaml`; (e) `tests/sim/` coverage on a self-hosted
  RTX runner; (f) `docs/METHODS.md` + repo-state-map updates.

## Implementation note — PoC built & verified 2026-06-10

A working proof-of-concept landed on branch `feat/isaac-sim` and was run for real
on an RTX 4070 Laptop (8 GB):

- **Sidecar venv:** py3.11 with `isaacsim[all]==5.1.0.0` + `isaaclab==2.3.2` +
  pyzmq + msgpack (188 packages). Install gotchas worth recording: `flatdict`
  (an `isaaclab` dep) needs `--no-build-isolation-package flatdict` + a
  pre-installed `setuptools<80`; the multi-GB Isaac wheels need a raised
  `UV_HTTP_TIMEOUT`; cross-index resolution needs `--index-strategy
  unsafe-best-match` (isaacsim on `pypi.nvidia.com`, isaaclab on PyPI).
- **Isaac Sim core, not Isaac Lab, for the env.** The PyPI `isaaclab` wheel does
  **not** ship the `isaaclab.sim` / `isaaclab.envs` task machinery (`import
  isaaclab.sim` → `ModuleNotFoundError`); those need the git-source install
  (`./isaaclab.sh`) plus the PyPI-absent `isaaclab_assets` / `isaaclab_tasks`.
  The Isaac Sim **core** API (`isaacsim.core.api.World`,
  `isaacsim.robot.manipulators.examples.franka.Franka`,
  `isaacsim.sensors.camera.Camera`) is fully present and is what the PoC scene
  (`tools/isaac_scene.py`) uses. **Wiring the full Isaac Lab manager-based env
  (OSC/diff-IK action terms, task MDP) is deferred** to a follow-up that
  provisions the source install — it does not change the sidecar architecture.
- **Verified end-to-end:** openral (py3.12) auto-spawns the sidecar, which boots
  a headless Omniverse Kit app (~10 s warm; first boot ~6 min to fill the
  extension cache), builds a Franka + cube + RTX camera scene, and answers
  `reset`/`step`/`render` over ZMQ. `reset` returns a `(128,128,3)` RGB frame +
  12-D state; `step` advances real PhysX and returns `cube_z` reward; `render`
  returns a non-trivial RTX frame (99.9 % non-zero pixels). Covered by
  `tests/sim/test_franka_random_isaac.py` (real GPU, skips without the sidecar
  venv) and `tests/unit/test_isaac_sim_sidecar_wire.py` (wire codec against a
  real ZMQ boundary, no GPU).
- **8 GB-GPU gotcha:** constructing the core `World(device="cuda:0")` forces the
  GPU PhysX pipeline, whose first `reset`/`step` warmup hangs for minutes on an
  8 GB laptop GPU. The default-device `World(stage_units_in_meters=1.0)` renders
  the same scene in ~15 s. The PoC uses the default; GPU PhysX is a follow-up
  tuning knob. Also: never run two Kit apps concurrently — a second app disables
  the shared kvdb and starves the first (observed as a stuck boot).
- **EULA:** the sidecar sets `OMNI_KIT_ACCEPT_EULA=YES` (user's acceptance of the
  proprietary NVIDIA Omniverse license) — reinforces §Decision-2's "externally
  provisioned, never vendored" posture.

## Follow-up — Isaac-Lab-free control + LIBERO-shaped scene + real VLA (2026-06-10)

A second iteration confirmed Isaac Lab is **not** needed even for end-effector
control, and ran a real rSkill through `openral sim run`:

- **Moving the robot / OSC needs neither Isaac Lab nor its OSC term.** Arm motion
  is `ArticulationController.apply_action` on the core `Franka`. End-effector
  control uses the **core** `isaacsim.robot_motion.motion_generation` Lula
  kinematics solver (`LulaKinematicsSolver` + `ArticulationKinematicsSolver` on
  the `right_gripper` frame): position-delta IK returns joint targets and the EE
  tracks them (verified: ee z 0.39→0.34→0.29 for −0.05/−0.10 deltas). Isaac Lab's
  `OperationalSpaceController` is only a convenience; it is not the sole OSC path.
- **`tools/isaac_bowl_plate_scene.py`** — a table + YCB `024_bowl` USD + plate +
  Franka scene mirroring the LIBERO contract (camera1/camera2 + 8-D
  `[eef_pos‖axisangle‖gripper_qpos]` state, 7-D OSC-pose-delta action via Lula
  IK). Selected by the sidecar `--layout bowl_plate`
  (`scenes/sim/isaac_franka_bowl_plate.yaml`). Assets confirmed reachable on the
  Isaac S3 nucleus (table ✓, `024_bowl` ✓; no plate USD ships → primitive).
- **Real VLA verified:** `openral sim run --config
  scenes/sim/isaac_franka_bowl_plate.yaml --rskill rskills/act-libero` ran the
  full pipeline — rSkill compat check → ACT weights from HF → Isaac sidecar →
  ACT consumes camera1/camera2 + 8-D state → 7-D actions → Lula IK drives the
  Franka — for 200 steps at ~15 ms/step. `success=False` is expected (the
  bowl/plate task is out-of-distribution for a LIBERO policy); the check is that
  the pipeline runs and the arm is driven, not task success. ACT is preferred on
  an 8 GB GPU (tiny next to the ~2 GB Isaac sidecar).
- **2026-06-19 SimScene cull:** the `isaac_franka_lift` task YAML was removed
  from `scenes/sim/` after live evaluation showed the arm does not attempt the
  lift task with any in-tree rSkill. The layout remains available for
  deploy/wire bring-up (`scenes/deploy/isaac_franka.yaml`, `tools/isaac_scene.py`)
  because it still proves sidecar boot, RTX render, and 8-D joint-position HAL
  plumbing. Online policy search found no downloadable, license-compatible
  Franka Isaac lift policy that emits the required 8-D joint-delta action.

## Follow-up — `openral deploy sim` minimal bring-up (2026-06-10)

`deploy sim` wraps the scene in `SimAttachedHAL` and runs the ROS lifecycle
stack. An audit found the path is mostly backend-agnostic; the **one hard
blocker** was that `SimAttachedHAL._probe_env_action_dim` reads `env.action_dim`
and `_IsaacSimSidecar` didn't expose it → `ROSConfigError` at `connect()`.

**Minimal bring-up (this PR):** `_IsaacSimSidecar` gains an `action_dim` property
that reads the value the sidecar's `ping` already returns (8 for `lift_cube`,
7 for `bowl_plate`) — no sidecar change. With it, `SimAttachedHAL` connects,
`read_images()` flows the RTX frames to the sensor bridge, and `send_action()`
steps the env. The deploy scene is `scenes/deploy/isaac_franka.yaml` (env-only
`DeployScene`, `lift_cube` layout: its 8-D action matches the franka_panda
manifest's 8 joints + `JOINT_POSITION` mode, so a HAL action packs to
`env_action_dim=8` with no n_dof gap — cf. ADR-0036). Covered in-process by
`tests/sim/test_franka_isaac_deploy_hal.py` (real sidecar; no ROS launch needed).

**Backend-agnostic joint-state + idle-step (RESOLVED — ADR-0034 amendment
2026-06-10):** `SimAttachedHAL` no longer needs a MuJoCo handle to be useful to a
non-MuJoCo backend:
- `read_state()` sources real joint angles from `obs["joint_positions"]` (the
  Isaac scenes emit it via `IsaacSceneBase._joint_positions()` — the Franka's
  9 DOF mapped to the manifest's 8 joints); `/joint_states` carries live values,
  not zeros. Falls back to zeros only when a backend provides none. This helps
  ManiSkill3 / SimplerEnv too.
- The idle stepper drops the MuJoCo-handle gate, so Isaac cameras stay live when
  idle. The method-only-on-`SimAttachedHAL` exclusion remains the safety
  guarantee. Verified by `tests/unit/test_sim_attached_non_mujoco.py` (no GPU)
  + the live deploy HAL test (real non-zero joint values).

**Full `openral deploy sim` run verified (2026-06-10).** The complete ROS graph
was brought up against the Isaac scene
(`openral deploy sim --config scenes/deploy/isaac_franka.yaml --no-enable-octomap
--hal viewer_enabled=false`): safety kernel + reasoner + prompt-router + the
`openral_hal_franka` lifecycle node all reached `active`; the HAL connected to
the auto-spawned Isaac sidecar, `SimSensorBridge` published the cameras, the
`sim-only idle stepper @ 10 Hz` started (proving the de-gated idle path runs for
a non-MuJoCo backend), and `/joint_states` carried the **real Franka rest pose**
(`panda_joint2≈−0.52, joint4≈−2.86, joint6≈3.04, joint7≈0.74`), not zeros.

Bug surfaced + fixed by that run: `SidecarClient._spawn` inherited the parent's
`PYTHONPATH`. `openral deploy sim` injects this py3.12 venv's site-packages onto
`PYTHONPATH`, which then shadowed the py3.11 Isaac venv's own numpy in the
spawned sidecar → `No module named 'numpy._core._multiarray_umath'` at boot. Fix:
strip `PYTHONPATH` / `VIRTUAL_ENV` from the sidecar child env (it is
self-contained). Operational notes: the reasoner needs an LLM env
(`OPENRAL_REASONER_LLM_*`; a local ollama openai-compatible endpoint suffices to
activate); run with `OPENRAL_AUTO_INSTALL_DEPS=0` and pyzmq/msgpack present on the
openral venv so the `isaac_client` probe doesn't trigger a `uv sync`.

**Previously-deferred items, now RESOLVED (2026-06-10):**
- *Joint velocities* — the Isaac scenes now also emit `obs["joint_velocities"]`
  (`franka_joint_velocities()` maps the 9 DOF → 8); `read_state` populates
  `JointState.velocity` from it for non-MuJoCo backends (zeros fallback retained).
- *2-camera deploy + camera-VLA action contract* — there was no real gap:
  `SimAttachedHAL` (the scene-attach path) packs purely on `action.control_mode`
  and does **not** gate on the manifest's `supported_control_modes` (only
  `_mujoco_arm.py` does). So a LIBERO franka rSkill dispatching as
  `CARTESIAN_DELTA` (env_action_dim=7, ADR-0036) drives the `bowl_plate` Isaac
  scene exactly as it drives the LIBERO MuJoCo scene — the 7-D vector's pos-delta
  lands at slots 0–2 and the gripper at slot 6, which is what the scene reads.
  `scenes/deploy/isaac_franka_bowl.yaml` (bowl_plate, two cameras) is the deploy
  scene for it — no missing-camera warning. **Verified live:** the full deploy-sim
  graph on that scene + `ros2 action send_goal /openral/execute_rskill` for
  `OpenRAL/rskill-act-libero` ran to a SUCCEEDED result — the VLA loaded, consumed
  the live Isaac `state_to_policy`, emitted 7-D actions the HAL dispatched as
  `mode=cartesian_delta env_dim=7` (+ `gripper_position` for the gripper slot),
  and the bowl_plate scene drove the Franka via Lula IK under the safety kernel's
  active self-collision check — no E-stop.

## Amendment — Robot-agnostic, URDF-driven scenes (2026-06-11)

**Problem found.** Every Isaac scene built so far (`lift_cube` in `tools/isaac_scene.py`,
`bowl_plate` in `tools/isaac_bowl_plate_scene.py`) **hardcodes Isaac's built-in Franka
example USD asset**:

```python
from isaacsim.robot.manipulators.examples.franka import Franka
self._franka = self._world.scene.add(Franka(prim_path="/World/Franka", name="franka"))
```

The sidecar already accepts `--robot` and `isaac_sim._build_isaac_sim_scene` forwards
`env_cfg.robot_id`, but the geometry **ignores it** — the scene is a Franka regardless of the
manifest, and the DOF↔manifest mapping (`_franka_dof_to_manifest` in `tools/_isaac_scene_base.py`)
is Franka-specific. This contradicts the `DeployScene` contract, under which a scene is
**environment + backend** and the robot is **pluggable from its `RobotDescription`** — exactly
how MuJoCo/robosuite native scenes already work (ADR-0033 robot-parameterised native scenes;
ADR-0034 deploy-sim scene-attach). The base ADR even registers the backend `fixed_robot=None`
("free-axis: any manifest robot") — the intent was always robot-agnostic; the PoC just took the
Isaac-example-asset shortcut to get PhysX + RTX up.

**Goal.** Bring up **any manifest robot** in Isaac — starting with `panda_mobile` — and have it
emit the correct ROS topics/controllers for rSkills (`/joint_states`, per-camera RGB, depth
`PointCloud2`, `/scan`, `/odom`, control-mode dispatch), so a user defining their own
`RobotDescription` gets a working Isaac scene **without bespoke per-robot Isaac code** and
**without fabricating sensors the robot does not declare**.

### Design — `IsaacManifestScene` built from a marshaled robot spec

The sidecar runs py3.11 and **cannot import `openral_core`**, so the robot-agnostic path cannot
pass a `RobotDescription` object across the boundary. Instead:

1. **Marshal the manifest to a plain-JSON "isaac robot spec."** The openral-side backend
   (`isaac_sim.py`, py3.12) resolves the `RobotDescription` and serializes only what the scene
   needs to a temp JSON file, passed via a new `--robot-spec <path>` CLI arg:
   - `urdf_path` — the manifest's `urdf_path` **resolved to an on-disk file** on the py3.12 side
     (the `python:robot_descriptions.<pkg>:URDF_PATH` form resolves where `robot_descriptions`
     lives — the sidecar venv need not carry it);
   - `joints` — ordered `[{name, role, sim_joint_name?}]` so the scene maps Isaac articulation
     DOF ↔ manifest joint order **generically**, retiring `_franka_dof_to_manifest`;
   - `base_joints` — `[forward, side, yaw]` when the embodiment has a planar base, else absent;
   - `sensors` — each `SensorSpec` (`name`, `modality` ∈ rgb|depth|lidar_2d, `frame_id`,
     `parent_frame`, `intrinsics`, `range_min_m`/`range_max_m`, `vla_feature_key`);
   - `action` — `{dim, control_mode}` from the action contract.

2. **Import the URDF → USD articulation.** Replace `world.scene.add(Franka(...))` with Isaac
   Sim's URDF importer extension (`isaacsim.asset.importer.urdf`, the 5.1-line successor to
   `omni.importer.urdf`) converting `urdf_path` into a USD articulation prim. *Exact importer
   class/config is settled against the provisioned 5.1 install at implementation time* — the base
   ADR set the precedent of verifying Isaac APIs by running, not guessing (truth-over-plausibility,
   CLAUDE.md §1.2).

3. **Generic articulation controller.** Drive the imported articulation through the core
   `ArticulationController.apply_action` (already used by `IsaacLiftScene`), but index targets by
   the spec's joint **order + role** (arm/gripper/base) rather than the Franka 7+2 layout. The
   action→joint mapping keys on `action.control_mode` — JOINT_POSITION direct, CARTESIAN_DELTA via
   the core Lula IK already proven in the base ADR, BODY_TWIST for the base — mirroring
   `SimAttachedHAL`'s existing control-mode dispatch.

4. **Generic sensors from the spec — one attachment per declared `SensorSpec`, nothing more:**
   - **rgb** → `isaacsim.sensors.camera.Camera` + `get_rgba()` (today's path), keyed by the
     sensor's `vla_feature_key` into `images`;
   - **depth** → the same `Camera` with the `distance_to_image_plane` annotator → a depth array
     fed to the **existing** `openral_sim.backends.depth_camera.synthesize_depth_pointcloud`
     (the bridge is already backend-agnostic; only the array *source* changes from MuJoCo ray-cast
     to the Isaac annotator) → the `/…/points` `PointCloud2` octomap consumes;
   - **lidar_2d** → an Isaac RTX lidar (or a rotating ray-cast at base height) → the synthetic
     `/scan` the panda_mobile HAL already expects for Nav2 + slam_toolbox.
   A modality the manifest does **not** declare is **not created** — `franka_panda` (no depth, no
   lidar) gets neither; only `panda_mobile` (which genuinely declares `front_depth` + `base_scan`)
   gets them. This is the explicit constraint: never add a sensor a robot does not have.

5. **Mobile base — kinematic (decided 2026-06-11).** `panda_mobile`'s `urdf_path` is the Panda
   **arm only** (`panda_description`); its 3-DOF holonomic base is **manifest-defined**
   (`base_joints`), not in the URDF, and exists nowhere as an Isaac-importable asset (robosuite
   composes base+arm at the MJCF layer; the MJCF importer would have to ingest a runtime-composed,
   controller-coupled model). Two options were weighed — a PhysX-articulated base (real
   prismatic-x / prismatic-y / revolute-yaw joints on the imported arm root) vs a **kinematic
   base** — and we chose **kinematic**: import the arm `fix_base=True` and teleport the whole
   articulation root each step from a base-frame-twist-integrated `(x, y, yaw)` pose, surfacing a
   `base_pose` for `/odom`. Rationale: robust (no fragile USD articulation surgery), delivers
   exactly what the goal needs — correct `/joint_states` (base joints filled from the pose), base
   motion, and `/odom` for rSkills — and base obstacle avoidance is Nav2's 2-D costmap job, not
   PhysX's. The base joints are NOT URDF DOFs; their `/joint_states` values come from the
   kinematic pose. Data-driven by the spec's `base_joints` — **not** a hardcoded robot name.

### Incremental milestones (separate commits, gated on this amendment)

- **M1 (DONE, verified live)** — `IsaacManifestScene` imports `franka_panda` from its **URDF**
  (fixed arm, single RGB camera, JOINT_POSITION controller), replacing the hardcoded `Franka`
  example asset. `tests/sim/test_franka_urdf_isaac.py`: `/joint_states` carries the imported arm's
  live pose and a JOINT_POSITION action drives it.
- **M3-base (DONE, verified live)** — `panda_mobile` on the same path: kinematic holonomic base
  (11-D action = 7 arm + 1 gripper + 3 base-twist), 11-joint `/joint_states` (3 base + 7 arm + 1
  gripper) with base joints filled from the kinematic pose, and a forward `base_twist` that moves
  the base. `tests/sim/test_panda_mobile_isaac.py`.
- **HAL base generalization (DONE, verified live)** — `SimAttachedHAL` (the deploy-sim HAL) was
  MuJoCo-coupled for the mobile base; generalized **in place** (no Mujoco/Isaac subclass split — that
  stays the last resort): `base_pose` reads `obs["base_pose"]` without a MuJoCo handle, and a
  `BODY_TWIST` (the `/cmd_vel` bridge's output) routes through `_apply_body_twist_via_env_step` →
  `env.step` instead of raising. `tests/sim/test_panda_mobile_isaac_hal.py`: through the same HAL, a
  `BODY_TWIST` moves the Isaac base, `base_pose`/`base_twist` feed `/odom`, `/joint_states` tracks it.
- **Full deploy-sim ROS graph (DONE, verified live 2026-06-11)** — `openral deploy sim --config
  scenes/deploy/isaac_panda_mobile_urdf.yaml` brought up the complete graph on the Isaac
  `panda_mobile` scene (no MuJoCo anywhere): the `openral_hal_panda_mobile` lifecycle node activated
  @30 Hz wrapping the Isaac `SimAttachedHAL`; the C++ `safety_kernel` armed (`n_dof=11`, 12-link
  self-collision, ADR-0040 velocity+cartesian); the `reasoner` activated with a 3-skill palette
  matching the robot; `runtime_node` (WorldState) ran. Live topics confirmed: `/joint_states` (11
  joints, real imported-Panda arm pose), `/odom` + TF `odom→base_link`, three `/openral/cameras/*`
  publishers (`camera1` @10 Hz), `/scan` @10 Hz. **End-to-end actuation:** a `/cmd_vel`
  `linear.x=0.3` moved `/odom` x `0.0 → 0.315 m` (and `/joint_states` `base_x` tracked it) — the full
  chain `/cmd_vel → MobileBaseBridge → BODY_TWIST → _apply_body_twist_via_env_step → Isaac kinematic
  base → obs["base_pose"] → /odom`, under the active safety kernel, no E-stop. Run with
  `--no-enable-slam/nav2/octomap` (the perception-richness leg below) but the HAL/control/safety/odom
  graph the robot-agnostic goal targets is proven.
- **Multi-RGB + depth → octomap (DONE, verified live)** — the scene now renders one base-relative
  camera per manifest `SensorSpec`: `camera1/2/3` (256×256 RTX frames) + a forward-facing `front_depth`.
  Depth uses Isaac's `Camera.get_pointcloud(world_frame=True)` (Isaac owns the camera convention — no
  optical-frame guess) transformed world→base_link by the kinematic base pose, surfaced as
  `obs["depth_points"]`; `SimAttachedHAL.read_depth_clouds` + `SimSensorBridge._publish_depth_clouds_from_obs`
  publish it as a `base_link` `PointCloud2`. **Verified live:** `--enable-octomap` ran the full chain —
  `/openral/cameras/front_depth/points` (62 k pts, `base_link`, ~5 Hz, geometrically in front of the
  base near the ground) → `octomap_server` (`/octomap_binary`) → `/openral/world_voxels` (the kernel's
  world-collision input). The manual `deproject_depth_image` helper was superseded by Isaac's
  convention-correct `get_pointcloud` and removed.
- **2-D lidar → real `/scan` (DONE, verified)** — the scene casts a PhysX `raycast_closest` fan
  (`_scan_ranges`, `n_channels` beams `-π`→`+π` in `base_link`, rotated to world by the base yaw) over
  a few static obstacles it seeds (`_add_obstacles`, since a bare ground plane returns no hits). Each
  ray starts `range_min_m` *beyond* the base so it clears the robot's own chassis/arm (a centre-origin
  ray hits `panda_link1` at distance 0); robot hits (prim under `/panda`) are ignored. Surfaced as
  `obs["scan"]` → `SimAttachedHAL.read_scan` → `SimSensorBridge._compute_scan_ranges`. Verified at the
  `SimRollout` level (`tests/sim/test_panda_mobile_isaac.py`: 360 beams, a meaningful fraction hit the
  obstacles, none NaN). Live in the graph `SimSensorBridge` publishes `/scan` @10 Hz and Nav2's full
  stack activates (`/navigate_to_pose`, planner/controller/costmaps/bt_navigator).
- **Full slam-map + obstacle-aware Nav2 (DONE, verified live 2026-06-11)** — the slam loop needs a
  common clock (slam + Nav2 run `use_sim_time:true`); the deploy-sim `/clock` publisher (ADR-0048,
  #309) on master closes it. With this branch rebased onto that master,
  `openral deploy sim --config scenes/deploy/isaac_panda_mobile_urdf.yaml --enable-slam
  --enable-nav2` ran the **complete autonomous-navigation loop** on Isaac panda_mobile:
  the HAL published `/clock` from sim time, `slam_toolbox` **registered the Isaac lidar** and built a
  388×480 @ 0.05 m `/map` from the `/scan`, Nav2's planner+controller+costmaps came up, and a
  `NavigateToPose` goal to `(1.6, 0)` in `map` returned **SUCCEEDED** with `/odom` advancing
  `(0,0) → (1.38, 0.01)` — Nav2 planned a path, drove the base via `/cmd_vel → MobileBaseBridge →
  BODY_TWIST → _apply_body_twist_via_env_step → Isaac kinematic base`, under the active safety kernel.
  End-to-end: a manifest-driven, URDF-imported robot navigating autonomously around obstacles in
  Isaac Sim, with every rSkill ROS topic/controller (`/joint_states`, cameras, depth `PointCloud2`,
  `/scan`, `/odom`, `/map`, `/cmd_vel`) live.

### Backward compatibility & safety

The hardcoded `lift_cube` / `bowl_plate` layouts stay (selected by `--layout`); `--robot-spec`
selects the new generic path. `lift_cube` is retained as deploy/wire bring-up, not
as a task-level SimScene. **No schema change** —
`PhysicsBackend.ISAACSIM` already exists; the robot spec is an IPC transport detail, never an
on-disk schema. **No safety-kernel change** — Isaac stays behind `SimRollout`/`SimAttachedHAL`;
the real-HW exclusion in `build_hal` (`mode="real"` + scene attach → `ROSConfigError`) still
prevents a sim scene from attaching to physical actuators. No new flag bypasses any safety check.

## Open questions

- **IPC transport.** Reuse the RLDX-1 `pyzmq`/`msgpack` sidecar pattern (`rldx` group) vs a
  bespoke channel — decide at implementation time. Leaning `pyzmq`/`msgpack` for consistency.
- **Sidecar env provisioning.** How the py3.11 Isaac Lab env is created and discovered by the
  in-process backend (env var pointing at the sidecar interpreter vs a `uv`-managed sibling
  env) — settle when the first PoC lands.
