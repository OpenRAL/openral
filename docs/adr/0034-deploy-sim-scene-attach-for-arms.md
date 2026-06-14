# ADR-0034: Deploy-sim scene-attach and sim-sensor bridge for manifest-driven arms

- Status: **Accepted**
- Date: 2026-06-03
- Related: [ADR-0029](0029-unified-hal-lifecycle-node.md) (unified HAL lifecycle node);
  [ADR-0031](0031-sim-real-hal-separation.md) (`build_hal` sim/real seam);
  [ADR-0032](0032-deploy-run-ros-graph.md) (deploy-run ROS graph);
  [ADR-0033](0033-robot-parameterized-native-scenes.md) (robot-parameterised native scenes).
  Supersedes the ADR-0033 §Decision-4 parenthetical (corrects an overstatement).

## Context

`openral deploy sim --config <scene>.yaml` for manifest-driven arms (franka_panda, ur5e, ur10e,
aloha, g1, h1, rizon4) builds a **headless bare-arm digital twin** and publishes only
`/joint_states`. No MuJoCo window opens and no cameras render, so:

- Camera-conditioned rSkills (MolmoAct2, π0.5, …) abort with
  `ROSConfigError: got no camera frames; expected ('agentview','wrist'), saw []`.
- The two `deploy sim` paths — pure `sim run` (which drives a `SimRollout` directly) and
  `deploy sim` (which wraps a `SimRollout` in a ROS lifecycle node) — produce different
  sensor output for the same scene YAML.

ADR-0033 §Decision-4 contained an overstated parenthetical —
*"`deploy sim` already wraps scenes behind `SimAttachedHAL` for the ROS path — that stays"*
— which was only true for the bespoke `panda_mobile` package. Every other manifest-driven
arm used the generic `_ManifestHALLifecycleNode` (ADR-0029), which called
`build_hal(mode="sim")` and received the bare twin. The ADR-0033 verification (`tabletop_push`
scenes for SO-101/Franka/UR5e) ran through `sim run`, not `deploy sim`.

## Decision

Generalise scene-attach and sensor publishing to **all** manifest-driven arms via four
coordinated changes, preserving the ADR-0031 `build_hal` seam:

### 1. `build_hal` gains a `sim_env_yaml` parameter — `openral_hal/resolver.py`

```python
def build_hal(
    description: RobotDescription,
    *,
    mode: Literal["sim", "real"],
    transport: dict[str, object] | None = None,
    sim_env_yaml: str | None = None,
) -> HAL
```

`mode="sim"` + `sim_env_yaml` set → calls `build_sim_env_from_yaml(sim_env_yaml,
robot_id_fallback=description.name)`, wraps the result in
`SimAttachedHAL(env, description, env_reset_seed=seed)`, and returns it. The bare-twin /
`hal.sim` class is bypassed entirely — the scene owns physics and pixels. `mode="real"` +
`sim_env_yaml` → `ROSConfigError` (a real-hardware HAL never attaches a sim scene). The
HAL type is still decided in one place (ADR-0031 seam preserved).

### 2. Shared `SimSensorBridge` — `openral_hal/sim_sensor_bridge.py`

A stateful helper (rclpy imported lazily) that owns all manifest-gated sim sensor publishers
and the viewer:

```python
class SimSensorBridge:
    def __init__(
        self,
        node: Any,
        hal: Any,
        description: RobotDescription,
        *,
        viewer_enabled: bool = True,
        camera_rate_hz: float = 10.0,
        viewer_sync_rate_hz: float = 30.0,
    ) -> None: ...
    def setup(self) -> None     # called from on_activate_post_subs
    def teardown(self) -> None  # called from on_deactivate / on_cleanup
```

`setup()` wires two streams, each gated on manifest + HAL capability:

| Stream | Topic | Gate | Producer |
|---|---|---|---|
| RGB | `/openral/cameras/<n>/image` | `hasattr(hal,"read_images")` + RGB `SensorSpec` in manifest | `_publish_images` |
| Viewer | — | `viewer_enabled` + `mujoco_handles()` not None | `mujoco.viewer.launch_passive`; headless → warn + continue |

Phase 2 (ADR-0034 §Safety posture) adds `/scan` + depth `PointCloud2` streams.

### 3. `_ManifestHALLifecycleNode` adopts the bridge — `openral_hal/lifecycle.py`

The node declares `sim_env_yaml` (default `""`), `viewer_enabled` (default `True`), and
`camera_publish_rate_hz` (default `10.0`) ROS parameters. `_create_hal` passes `sim_env_yaml`
to `build_hal`. `on_activate_post_subs` calls `SimSensorBridge.setup()`; the deactivate and
cleanup hooks call `teardown()`. Every manifest-driven arm gains scene + cameras + viewer
under `deploy sim` with zero per-package wiring (honoring ADR-0029).

### 4. CLI injects the scene path — `openral_cli/deploy_sim.py`

In `resolve_launch_invocation`, the `manifest_driven` branch, when `hal_mode == "sim"` and a
`--config` path is present: `hal_params.setdefault("sim_env_yaml", str(config.resolve()))`.
The scene path is already resolved to derive `robot_id`; this step forwards it to the node.

### 5. Consumer-side camera-slot realignment — `openral_rskill_ros/rskill_runner_node.py`

The bridge publishes each frame on `/openral/cameras/<sensor.name>/image`, so the WorldState
node keys `WorldState.image_frames` by the manifest **sensor name** (`agentview`, `wrist`).
VLA adapters, however, look up `obs["images"]` by the **VLA slot** (`camera1`, `camera2`, …)
— the LIBERO convention `openral sim run` and the rldx adapter already use, and the key the
checkpoint's `cam_alias` maps (`camera1 → image`). A manifest whose RGB sensors are
descriptively named (franka) therefore handed pi0.5 `obs["images"]["agentview"]` while it
looked up `camera1` → no frames → `got no camera frames` abort.

`rskill_runner_node` realigns the two namespaces from a single source of truth — the manifest's
`vla_feature_key` suffix, mirroring the bridge's `_obs_key_for_sensor` (§2):

- `_sensor_name_to_vla_slot(description)` maps each RGB `SensorSpec.name` → its slot
  (`observation.images.camera1` → `camera1`); sensors with no `vla_feature_key` fall back to
  their own name (robocasa real-name keys).
- `_build_runtime_skill_from_manifest` derives the adapter's `scene_cameras` from
  `_vla_camera_slots(description)` (the slots, in manifest order) instead of the sensor-name
  `camera_names` `runtime_node` forwards — so `resolve_camera_keys → _camera_keys` lands on the
  slots. Falls back to the passed `scene_cameras` when the manifest declares no RGB sensors.
- `_PolicyAdapterSkill._step_impl` rekeys `obs["images"]` through `_decode_image_frames`
  (sensor name → slot) so the decoded frames match the adapter's `_camera_keys`.

The realignment is keyed off `description.sensors` `vla_feature_key` everywhere, so the obs
keys and the adapter's camera keys agree by construction. It only affects the `deploy sim`
(ROS) path; `openral sim run` already receives `camera1`/`camera2` directly from the LIBERO env.
A Layer-3 skill package must not import the Layer-0 HAL (CLAUDE.md §3), so the slot-resolution
rule is duplicated (3 lines) rather than shared with `sim_sensor_bridge`.

### Joint-name unification — `openral_hal/sim_attached.py` §3.6

`SimAttachedHAL.read_state` resolves joint names from the scene's MJCF by name
(`mj_name2id`), unlike the bare `MujocoArmHAL` which is index-based and name-agnostic.
MJCF joint naming is heterogeneous across backends:

- **Native MJCF** (franka `panda.xml`): joints are `joint1..7` + `finger_joint1`.
- **Robosuite** scenes: the same arm's joints are prefixed `robot0_joint1..7`.
- **Canonical manifest `name`**: `panda_joint1..7` (the safety-envelope / real-HAL contract).

Resolution (implemented in `normalized_joint_index` + manifest `sim_joint_name`):

1. `sim_joint_name` in the manifest carries the robot's **native MJCF joint name** wherever
   it differs from the canonical `name` (franka: `joint1..7` + `finger_joint1`; so100/so101:
   `Rotation…Jaw`). Robots whose canonical names already match the MJCF (ur5e, ur10e, rizon4,
   g1, h1) need none. The canonical `name` is never changed.
2. `normalized_joint_index` builds a lookup that maps both the exact MJCF joint name **and**
   a robosuite-prefix-stripped form (`^[a-z]+[0-9]+_` strip: `robot0_joint1` → `joint1`) to
   the MuJoCo joint index. Exact names always win; stripped names are added only when they
   neither shadow an exact name nor produce an ambiguous collision (bimanual
   `robot0_`/`robot1_` → keep un-normalized, require explicit `sim_joint_name`).
3. `read_state` tries `sim_joint_name or name` first; the normalized fallback catches
   robosuite prefixes. One manifest entry serves both native MjSpec and robosuite scenes.
   `robot0_` never appears in a manifest.

## Safety posture (Phase 2)

Phase 2 of the bridge adds `/scan` lidar and depth `PointCloud2` streams. The
`PointCloud2` feeds octomap_server → the C++ safety kernel's capsule-vs-voxel check
(ADR-0030). `panda_mobile` now delegates to the shared bridge, lifting
`synthesize_depth_pointcloud` / `robot_self_body_ids` / `camera_optical_tf_to_base`
**unchanged** (the refactor is at-least-as-conservative — the ray-cast, self-body exclusion,
and point filtering are byte-identical to the pre-refactor in-node code).

Evidence + remaining gates:

- **Regression test** `packages/openral_hal_panda_mobile/test/test_sensor_bridge_regression.py`
  brings up the refactored `panda_mobile` against `robocasa/NavigateKitchen` and asserts
  `/scan` (finite ranges), `/openral/cameras/<depth>/points` (**non-empty** — the octomap input
  must not collapse), and `/odom` all still publish. It **fails loudly** if the cloud is empty.
- It is **env-gated**: robocasa needs `robosuite>=1.5.2`; the workspace venv currently has
  `1.5.1`, so the test skips with that reason until `just sync --group robocasa`. The live run
  on a robocasa-provisioned host/CI is a **required pre-merge gate**.
- **Safety-WG reviewer approval** + a **hazard-log entry** referencing this ADR are required
  before the Phase-2 commits merge (CLAUDE.md §3). NOT yet obtained — Phase 2 must not merge
  without them.

## Alternatives considered

- **Per-robot lifecycle node subclasses** — each arm gets its own node that wires scene-attach.
  Rejected: that is exactly the ADR-0029 anti-pattern (per-package boilerplate).
- **Scene-attach only on `sim run`, not `deploy sim`** — rejected: both paths must produce
  identical sensor output for the same scene YAML (the spec's stated goal).
- **`build_hal` returns the bare twin; lifecycle node post-processes it** — rejected: the
  HAL *type* must be decided in one place (ADR-0031). Lifting the `SimAttachedHAL`
  construction into the node would scatter the seam.

## Consequences

- All scenes run identically on both `sim run` and `deploy sim` paths.
- Manifest-driven arms (franka, ur5e, ur10e, aloha, g1, h1, rizon4) now receive scene +
  camera publishing + MuJoCo viewer under `deploy sim` with zero per-package changes.
- `panda_mobile` will dedup onto the shared bridge in Phase 2; until then it retains its own
  per-robot sensor wiring.
- ADR-0033 §Decision-4 parenthetical is corrected: scene-attach under `deploy sim` was only
  true for `panda_mobile`; it is now true for all manifest-driven arms.
- `build_hal`'s signature gains one keyword-only parameter (`sim_env_yaml`). All callers that
  do not pass it are unaffected (`None` default, same behavior as before).
- The `schema_version` stays `"0.1"` (no migrators; CLAUDE.md §6).

## Amendment 2026-06-04 — sim-only free-running idle stepper

### Problem

Under `deploy sim` the MuJoCo env lives only in the HAL node via `SimAttachedHAL`,
and `env.step()` is called only from `SimAttachedHAL.send_action` — reached only on
`/openral/safe_action` receipt, which only flows **while a skill is executing**. When
the deploy-sim graph is idle (no skill running), the env never steps: physics is frozen,
the rendered camera frames cached in `_last_obs` go stale, and the ADR-0035
perception / object-detector bus sees a dead scene (the detector runs over a single
frozen frame forever, motion/occupancy never updates). The cameras only "came alive"
once a skill happened to start stepping the env.

### Decision

Add a **sim-only free-running idle stepper** that advances the env one tick with a
zero/HOLD action whenever the scene is idle, so cameras keep rendering and the
perception bus sees a live scene with no skill running.

- **`SimAttachedHAL.idle_step() -> bool`** steps the wrapped `SimRollout` with
  `np.zeros(env_action_dim, dtype=np.float32)` — exactly the proven zero-action idiom
  `robocasa.refresh_obs` already uses — re-caching `_last_obs` from the `StepResult`.
  It mirrors `send_action`'s ADR-0036 deferred-reset branch (a terminated episodic
  backend is reset before stepping; robocasa's `ignore_done` never latches, so that
  branch is dead there) and re-latches `_episode_done`. It does **not** touch the
  commanded-slot merge state (`_last_env_action`) or the latched base twist
  (`_last_body_twist`) — an idle HOLD is orthogonal to whatever a skill last commanded.
- **`SimSensorBridge`** creates an idle-step timer at the existing `camera_rate_hz`
  (default 10 Hz — step-then-publish stays matched to the camera publisher, no new rate
  param) with a quiet window `idle_hold_ms` (default 200 ms). Its callback calls
  `idle_step()` only when `should_idle_step(monotonic_ns(), hal.last_action_ns, idle_hold_ns)`
  is `True` — i.e. no real action arrived within the hold window. `send_action` stamps
  `last_action_ns` at its top (the single choke point both `_on_safe_action` and
  `_on_cmd_vel` reach), so an active skill always wins the env: the idle tick yields.
  The single-threaded rclpy executor guarantees the idle timer and `_on_safe_action`
  never run concurrently, so the timestamp check alone is a sufficient hand-off — no lock.

### Real-hardware exclusion (3 layers; safety-critical)

A zero action is a **HOLD** for the sim's velocity / OSC-delta / robosuite composite
controllers, but on a real **absolute-position** arm (Franka FCI, lerobot follower) a
zero joint-target vector commands "drive every joint to 0 rad" — a violent motion.
"Zero is harmless" is therefore **false** on real hardware and is explicitly NOT the
guarantee. The guarantee is structural, in three layers:

**Honest caveat — zero is not even a literal HOLD on every *sim* backend.** It is a true
hold only for velocity / OSC-delta / robosuite composite controllers. A
**position-controlled native backend** (e.g. `so101_box`, whose `step` consumes
joint-position targets) reads a zero vector as "go toward 0 rad", not "stay put". That is
acceptable for the idle stepper — the goal is to keep the scene physically live so cameras
render, not to freeze the arm — but the idle pose on such backends is the zero-rad pose,
not the last commanded one. (Separately: those native backends now each expose
`action_dim` so `_probe_env_action_dim` resolves their true `step` width — see
"Probe-gap fix" below.)

1. **Primary — method-only-on-`SimAttachedHAL`.** `idle_step` is defined *only* on
   `SimAttachedHAL`. Real HALs (`FrankaPandaRealHAL`, ros2_control bridges, lerobot
   followers) do not define it. `SimSensorBridge` gates the idle timer on
   `callable(getattr(hal, "idle_step", None))`, so against a real HAL the timer is
   **never created**. This is the real guarantee.
2. **Secondary — `hal_mode`.** `SimAttachedHAL` is only ever constructed under
   `build_hal(..., mode="sim", sim_env_yaml=...)`. `build_hal` raises `ROSConfigError`
   when `sim_env_yaml` is supplied with `mode="real"`, so a sim scene can never attach
   to a real-hardware HAL.
3. **Tertiary — live MuJoCo handles.** `idle_step` returns `False` (and the bridge also
   gates) when `mujoco_handles()` is `None` — a non-MJCF backend is never stepped this way.

`idle_step` also honors the estop contract: it returns `False` while `_estop_latched`
is set (an estopped HAL freezes — that is the correct, safe behaviour).

### Contention / hand-off rule

The idle stepper is a *fallback ticker*, never a co-driver. It steps **only** during the
quiet window between real actions (`now - last_action_ns >= idle_hold_ms`). The moment a
skill streams actions, `send_action` updates `last_action_ns` every tick and the idle
stepper yields for the whole burst. There is exactly one writer to `env.step` at a time.

### Probe-gap fix (resolved)

Originally `SimAttachedHAL._probe_env_action_dim` fell back to **11** (the robosuite BASIC
composite width) for backends that didn't expose `action_dim`. A native backend whose
`step` required a different width (`so101_box` → 6, `tabletop_push` → actuator count,
`openarm_tabletop_pnp` → `state_dim`) then raised a width mismatch on the first `env.step`.
That gap hit `send_action` too, but `idle_step` made it fire **autonomously** on the bridge
timer (with `last_action_ns == 0` an idle scene begins stepping immediately, before any
skill runs), gating only on `mujoco_handles()` (which those backends *do* expose).

**Fix — single source of truth.** Every native MuJoCo rollout now exposes an `action_dim`
property reporting its true `step` width (mirroring the robosuite/robocasa/LIBERO backends
that carry it natively), so `_probe_env_action_dim` resolves the authoritative width for
**all** backends. As a safety net, the probe no longer guesses: when it genuinely cannot
introspect a width *and* no `env_action_dim` override was supplied, it raises
`ROSConfigError` naming the backend at `connect` time — a loud boot-time failure beats a
wrong-width mid-run E-stop. `SimSensorBridge._idle_step_tick` keeps its catch-once-and-
disable guard around `idle_step()` as defence in depth (e.g. an override that disagrees
with the backend), but the probe gap itself can no longer turn into a per-tick crash-loop.

### Tests

`python/hal/tests/test_sim_attached_idle_step.py` — (a) idle → `idle_step()` advances the
rendered frame (the frozen-scene regression), proven against the native-MuJoCo `so101_box`
backend (and the LIBERO twin where installed); (b) a terminated episode → `idle_step` does
reset-then-zero-step with the latch cleared (LIBERO); (c) estop latched → `idle_step()`
returns `False` and the frame is unchanged; (d) **safety**: a real HAL has no `idle_step`,
and `build_hal(..., mode="real", sim_env_yaml=...)` raises `ROSConfigError`; (e) the pure
`should_idle_step` predicate yields within the hold and engages after.

### Consequences

- `deploy sim` cameras + the ADR-0035 perception bus stay live when idle; no behaviour
  change while a skill is active (the idle tick yields).
- `SimSensorBridge.__init__` gains one keyword-only parameter (`idle_hold_ms`, default
  200 ms). No new ROS param is introduced — the default is used and the step rate reuses
  `camera_rate_hz`.
- No real-hardware path is touched; the `schema_version` stays `"0.1"`.

## Amendment 2026-06-10 — backend-agnostic joint-state + idle-step (non-MuJoCo sims)

### Problem

`SimAttachedHAL` was MuJoCo-coupled in two places that left a **non-MuJoCo**
`SimRollout` (the Isaac Sim sidecar of ADR-0045; also ManiSkill3 / SimplerEnv,
SAPIEN-backed) half-functional under `openral deploy sim`:

1. `read_state()` reads joint angles from the env's MJCF `qpos` via
   `mujoco_handles()`. With no handle it returned **all-zeros** — `/joint_states`
   (and the world-state aggregator, dashboard, and the geometric collision
   checker that reads it) saw a frozen-at-zero arm.
2. `idle_step()` and `SimSensorBridge._setup_idle_stepper` were gated on
   `mujoco_handles() is not None`, so cameras only refreshed while a skill was
   stepping; an idle non-MuJoCo scene went stale.

Neither is Isaac-specific — every non-MuJoCo backend hit both.

### Decision

Generalize both to source from the `SimRollout`, touching **only** the
non-MuJoCo path (the MJCF path is byte-for-byte unchanged):

- **`read_state()`** — when `mujoco_handles()` is `None`, build the `JointState`
  from `obs["joint_positions"]` (a 1-D vector in `description.joints` order) when
  the backend provides it; pad/truncate to the manifest joint count; fall back to
  the prior all-zeros only when absent. The Isaac sidecar scenes emit it via a
  new `IsaacSceneBase._joint_positions()` hook (the Franka's 9 DOF mapped to the
  manifest's 8 joints — 7 arm + a mean-finger gripper).
- **`idle_step()` + `_setup_idle_stepper`** — drop the `mujoco_handles()` gate.
  Idle-stepping is valid for any wrapped `SimRollout`; the **method-only-on-
  `SimAttachedHAL`** exclusion (real HALs never define `idle_step`) remains the
  real safety guarantee, and `_idle_step_tick`'s catch-once-and-disable guard
  still contains a per-tick fault. A zero action is a HOLD for the sim's
  velocity / OSC-delta controllers.

### Safety posture

No real-hardware path is touched and no safety check is weakened — the change
*improves* safety (real joint angles instead of zeros feed the collision
checker), and the idle stepper stays gated on the sim-only `idle_step` method.
`schema_version` stays `"0.1"`.

### Tests

`tests/unit/test_sim_attached_non_mujoco.py` — against a fake non-MuJoCo
`SimRollout` (+ the real franka_panda manifest, no GPU/ROS): `read_state` uses
`obs["joint_positions"]`; falls back to zeros without it; tolerates a
length-mismatched vector; `idle_step()` advances the env with no MuJoCo handle.
`tests/sim/test_franka_isaac_deploy_hal.py` asserts real (non-zero) joint values
live against the Isaac sidecar.

### Deferred

Joint *velocities* still report zero for non-MuJoCo backends (positions only).
Driving a LIBERO camera-VLA via deploy needs action-contract alignment
(`bowl_plate` 7-D EE-delta vs the franka manifest's `JOINT_POSITION`; ADR-0036) —
the `lift_cube` deploy scene sidesteps it.
