# ADR-0029: Unify per-robot HAL lifecycle nodes into one robot.yaml-driven node

- Status: **Accepted, complete** (Phases 1–3 landed under issue #191; every robot is manifest-driven)
- Date: 2026-05-29 (amended 2026-06-04)
- Related: [ADR-0023](0023-data-driven-mujoco-hal.md) (`MujocoArmHAL.from_description`);
  [ADR-0025](0025-reasoner-managed-background-services.md) (panda_mobile lifecycle node);
  [ADR-0031](0031-sim-real-hal-separation.md) / [ADR-0032](0032-deploy-run-ros-graph.md) (`build_hal` + `make_lifecycle_main_from_manifest`);
  [ADR-0034](0034-deploy-sim-scene-attach-for-arms.md) (`SimSensorBridge`);
  CLAUDE.md §3 (HAL is layer 0); §4.2.5 (smallest viable PR).

## Context

Every robot ships a ROS 2 lifecycle node under `packages/openral_hal_<robot>/`.
The shared base class `HALLifecycleNodeBase`
(`python/hal/src/openral_hal/lifecycle.py`) already factors out ~90 % of the
wiring (`/joint_states` + `~/joint_states` publishers, `/openral/safe_action`
+ `/openral/estop` subscribers, the heartbeat, the OTel `hal.read_state` /
`hal.send_action` spans, the e-stop latch). The remaining per-robot code is
mostly a thin `_create_hal()` factory:

| Robot | Lifecycle LOC | Shape |
|-------|--------------:|-------|
| ur5e / ur10e / franka_panda / aloha_bimanual / g1 / h1 / rizon4 | ~25 each | zero-param stub via `make_lifecycle_main()` |
| so100_follower | 87 | one serial-`port` param + heartbeat field |
| openarm | 462 | MJCF scene composition + cameras + viewer + `ResetToPose` service |
| panda_mobile | 932 | mobile base (`/odom` + `/scan` + TF) + `/cmd_vel`→BODY_TWIST bridge + cameras + viewer |

So 8 of 10 robots are already trivial; only `panda_mobile` and `openarm`
carry real per-robot logic. A single `robot.yaml`-driven
`OpenralHalLifecycleNode` could let "add a robot" mean "add a `robot.yaml`
+ a HAL class + a registry entry" — no new ROS package or node class.

## Decision

Defer. The unification is **feasible** but is a ~10–12 day effort with
medium risk and its own design surface — it does not belong inside the
already-large ADR-0024/0025 cleanup PR (CLAUDE.md §4.2.5). This ADR records
the verdict and the blockers so the work can be picked up standalone.

### Required `robot.yaml` schema additions
- `hal_construction` — factory module/class (or `factory_fn`) + `factory_kind`.
- `hal_parameters` — per-HAL ROS parameter defaults (serial port, MJCF path,
  `sim_env_yaml`, viewer toggles, …), keyed by robot.
- `lifecycle_features` — conditional feature flags the node reads to enable
  the mobile-base block (`/odom` + `/scan` + TF + `/cmd_vel`), camera
  publishers, the MuJoCo viewer, and scene composition.

### Hard blockers (each needs a small, isolated change)
1. **Camera introspection vs. param-declare timing** — whether a HAL exposes
   `read_images()` is only known after the HAL is constructed (configure),
   but ROS params are declared in `__init__`. Resolution: pre-declare camera
   params; defer publisher creation to `on_activate_post_subs` (panda_mobile
   already does this).
2. **Per-robot control-mode allowlist** — panda_mobile overrides
   `_on_safe_action` to accept BODY_TWIST / CARTESIAN_DELTA / GRIPPER_POSITION
   / COMPOSITE_MODE, not just JOINT_POSITION. Resolution: a
   `_supported_control_modes()` hook on `HALLifecycleNodeBase` (~5 lines base
   + a per-robot override).
3. **openarm MJCF scene composition** — composes a tabletop MJCF at configure
   time. Resolution: a declarative `scene_composition` block (composer
   module/fn + params) + conditional import.
4. **openarm `ResetToPose` service** — created only when the HAL exposes
   `reset_to_pose()`. Resolution: reflect on the HAL at configure time and
   wire the service when present.

### Effort
Phase 1 (low risk, ~2 d): the generic node + migrate the 8 zero-param robots.
Phase 2 (medium, ~3–4 d): the control-mode-allowlist hook + SO-100 + scene
composition config. Phase 3 (higher, ~4–5 d): migrate panda_mobile (mobile
base + lidar + cmd_vel + cameras + viewer) and openarm.

## Consequences

Until this lands, each new robot keeps shipping a (usually 25-line) lifecycle
stub. That cost is small for arm-only robots; the duplication concentrates in
the two heavy nodes, which this ADR's Phase 3 would consolidate. Tracked as a
GitHub issue so it isn't lost.

## Amendment — 2026-06-04 (the substrate already landed; Phase 1 started)

When this ADR was written it assumed a greenfield unification. Two efforts
that landed afterwards already build most of Phase 1's substrate, so the
remaining work is smaller than the original ~10–12 day estimate (~6–8 days):

- **ADR-0032** shipped `make_lifecycle_main_from_manifest` + the generic
  `_ManifestHALLifecycleNode`, which reads `robot_yaml` + `hal_mode` and
  builds its HAL through the single `build_hal` seam. The 8 "trivial" robots
  (ur5e / ur10e / franka / aloha / g1 / h1 / rizon4) **already run on it** — they
  are no longer per-robot subclasses, just `main()` shims. This is the
  `OpenralHalLifecycleNode` the Decision section envisioned, under a different
  name.
- **ADR-0034** shipped `SimSensorBridge`, which the manifest node attaches in
  `on_activate_post_subs` — this resolves **blocker #1** (camera
  introspection vs. param-declare timing) generically.

### What this PR (issue #191, Phase 1) lands
- Promotes `_ManifestHALLifecycleNode` → public `ManifestHALLifecycleNode`
  (the supported extension point; back-compat alias retained).
- Adds the `HalParameters` schema as the `RobotDescription.hal.parameters`
  block — the ADR's `hal_parameters` requirement — and threads its
  `defaults` through `build_hal` (explicit `transport` wins; unaccepted keys
  dropped). This lets a parameterised robot (e.g. the SO-100's serial `port`)
  declare its construction kwargs in the manifest instead of a bespoke
  `_create_hal`, removing the last reason the SO-100 needs a custom subclass.

### What Phase 2 lands
- **`ResetToPose` reflection (blocker #4):** `ManifestHALLifecycleNode`
  reflects on the built HAL in `on_configure_post_hal` and opens
  `/openral/<robot>/reset_to_pose` only when it exposes `reset_to_pose` — so
  every `MujocoArmHAL` sim arm gains the starting-pose snap the openarm node
  hand-wired, and HALs without it (panda_mobile, scene-attached twins) get no
  service. Verified live on ROS 2 Jazzy (franka opens + snaps; panda_mobile
  absent).
- **SO-100 / SO-101 migration:** the bespoke `openral_hal_so100` node collapses
  into `make_lifecycle_main_from_manifest`; `port` / `calibrate_on_connect`
  move to the manifest's `hal.parameters` (Phase 1 seam). The deploy registry
  flips so100/so101 to `manifest_driven` (a `bare_twin_sim` flag preserves the
  current bare-MuJoCo-twin deploy-sim behaviour rather than scene-attaching).

### What Phase 3a lands (panda_mobile)
- **Control modes (blocker #2) — resolved by deletion, no hook.** The HAL's
  `send_action` is already the per-robot mode contract (`PandaMobileHAL` /
  `SimAttachedHAL` reject unsupported modes), and the base `_on_safe_action` +
  `decode_action_chunk` already decode every wire mode. panda_mobile's
  `_on_safe_action` override was therefore behaviour-redundant — **dropped**.
  No `_supported_control_modes()` hook and no manifest field were added (the
  manifest's coarse `capabilities.supported_control_modes` is not a safe runtime
  allowlist). Verified live: BODY_TWIST through the base path advances `/odom` by
  the exact same 0.5 m; an unsupported mode is dropped without crashing.
- **Mobile base — `MobileBaseBridge`.** New
  `python/hal/src/openral_hal/mobile_base_bridge.py` (sibling of
  `SimSensorBridge`) owns `/odom` + `odom->base_link` TF + `/cmd_vel`→BODY_TWIST.
  The generic node attaches it in `on_activate_post_subs` iff the manifest
  declares `base_joints` — any future mobile robot reuses it.
- **`/scan` — single owner.** `SimSensorBridge._setup_scan` no longer gates on
  live MuJoCo handles; it publishes the live ray-cast when bound and the
  `constant_scan_no_hit_ranges` fan for the bare digital twin. The panda_mobile
  node's separate scan publisher is gone.
- **Migration:** `packages/openral_hal_panda_mobile/.../lifecycle_node.py` →
  `make_lifecycle_main_from_manifest`; the deploy registry entry is
  `manifest_driven` (keeping `supports_sim_env_yaml` scene-attach). Verified live
  via the existing panda_mobile integration test on the generic node
  (joint_states + /odom + /scan + TF + body_twist odom advance).

### What Phase 3b lands (openarm) — completes the unification
- **Scene composition (blocker #3) — declarative.** New `SceneComposition`
  schema + `SceneDefaults.composition` field; `ManifestHALLifecycleNode._create_hal`
  calls the named composer (`compose_openarm_tabletop_mjcf`, with
  `robot_lift_z` / `robot_forward_x` / `white_background` from the manifest) and
  threads the composed MJCF in as the HAL's `mjcf_path` transport kwarg.
- **Cameras — `read_images()` on `MujocoArmHAL`.** Renders the manifest's RGB
  sensors off the live MJCF (lazy `mujoco.Renderer` on the executor thread —
  EGL-thread-safe), keyed by sensor name; `SimSensorBridge` publishes them
  unchanged. A new `SensorSpec.sim_camera_name` maps a sensor to a
  differently-named MJCF camera (openarm `base` → MJCF `top`). The 3 RGB sensors
  were added to `OPENARM_DESCRIPTION` (synced with the manifest).
- **Deletions.** openarm's bespoke `ResetToPose` is gone — the Phase-2 reflective
  service covers it (now keyed by the `robot_id` directory, matching deploy_sim,
  so the openarm `description.name` "openarm_v2" vs id "openarm" mismatch is
  handled). openarm's `lifecycle_node.py` → `make_lifecycle_main_from_manifest`;
  registry entry `manifest_driven` + `bare_twin_sim` (composes its own MJCF, no
  scene-attach).
- **Verified live** (ROS 2 Jazzy + RTX 4070): the openarm integration test runs on
  the generic node — 16-DoF joint states, a real `rgb8` 640×480 frame on
  `/openral/cameras/base/image`, safe_action, spans, estop — all headless.

**Status: Accepted, complete.** Every robot is manifest-driven; no `openral_hal_*`
package ships a lifecycle node subclass. The `hal_construction` and
`lifecycle_features` blocks proposed in the original Decision were never needed —
the manifest's `hal.sim`/`hal.real` strings cover construction routing, and the
existing `base_joints` / `scene_defaults` / `sensors` fields are the gates the
generic node keys off.
