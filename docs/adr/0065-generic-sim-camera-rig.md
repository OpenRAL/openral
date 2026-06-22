# ADR-0065 — Generic sim camera rig driven by `SensorSpec.sim_placement`

- **Status:** Accepted 2026-06-22. Closes issue #88 (so101_box / so100 deploy
  sim rendered no cameras) with a robot-agnostic mechanism rather than a
  per-robot scene composer.
- **Date:** 2026-06-22
- **ADR number:** `0065`. The integer is not load-bearing — cross-refs use
  filenames.
- **Related:**
  - issue #191 Phase 3b / ADR-0033 — `scene_defaults.composition`: the
    manifest-driven HAL node calls a named composer to build a *task* MJCF (arm
    + props + cameras) and threads it in as the HAL's `mjcf_path`. This ADR
    *narrows* that mechanism: composition is for **scene props**, never the
    robot's own cameras, and the spec moves off the robot manifest.
  - ADR-0044 — shared gaze geometry (`look_at_quat_wxyz`). This ADR promotes it
    to `openral_core.geometry` so the HAL (layer 0) can compute camera
    orientations without importing world-state (layer 2).
  - ADR-0034 — deploy-sim scene-attach (`SimAttachedHAL`). Scene-attached HALs
    render their cameras from the scene's own MJCF; the rig is idempotent and
    skips cameras that already exist, so it is a no-op there.

## Context

`openral deploy sim` builds a bare MuJoCo digital twin for a manifest-driven arm
(`hal.sim: null` → `MujocoArmHAL.from_description`). The upstream arm MJCFs
(`so_arm100`, `so_arm101`, `panda`, `ur5e`, …) declare **zero `<camera>`
elements**, so the HAL rendered nothing and every manifest camera came back
absent (`hal.read_images.camera_absent`) — the dashboard, WorldState and
detectors got no frames (issue #88).

The first fix (issue #88, this branch's earlier commits) gave so101 — then
so100 — cameras by pointing the manifest's `scene_defaults.composition` at the
`so101_box` **task** composer, which splices a whole tube-insertion arena (walls
+ slotted block + tube) around the arm purely to carry two cameras. That works
but is wrong-shaped:

- It couples the **robot manifest** to a specific **scene** (`scene_defaults`
  describing a tube-insertion box belongs to a scene, not the robot).
- It reuses a *task* composer for a *deploy* twin that never runs the task, so
  the tube/block are dead scenery.
- It does not generalise: every new bare-twin arm (franka, ur5e, …) would need
  its own composer + manifest hook + body-name special-casing.

A camera is a property of the robot's **sensor suite** — the wrist camera is
bolted to the gripper; the overhead camera is a declared `SensorSpec`. The
manifest already describes each camera's intrinsics, frame and VLA key. The only
thing missing to *place* it in sim is its pose.

## Decision

1. **Camera placement lives on the `SensorSpec`, via a new optional
   `sim_placement: CameraSimPlacement`.** Fields: `parent_body: str | None`
   (the MJCF body the camera is rigidly mounted to; `None` = world-fixed),
   `pos: (x,y,z)` and `target: (x,y,z)` (look-at point, in the parent body's
   frame or world), and `fovy_deg: float | None` (override; otherwise derived
   from the sensor's pinhole `intrinsics` as `2·atan(h / 2·fy)`). The camera's
   MJCF name is `sim_camera_name or name` (unchanged). This is additive and
   backward-compatible: a sensor with no `sim_placement` is not rigged.

2. **A generic, robot-agnostic camera rig in the HAL splices the manifest's
   cameras into whatever MJCF it loads.** `MujocoArmHAL.connect` runs the rig
   before compiling the model: for each RGB sensor with a `sim_placement` whose
   camera is **absent** from the MJCF, it splices a `<camera>` (at the look-at
   orientation) into the named parent body (or `<worldbody>`), plus a one-time
   ambient fill light. It is **idempotent** — cameras already present (a
   scene-attached or composed-props MJCF) are left untouched — so it is a no-op
   for `SimAttachedHAL` and composes cleanly with scene props.

3. **`scene_defaults.composition` is for scene PROPS only.** so100 / so101 need
   no composition at all — their cameras come from the rig, and this ADR removes
   their `scene_defaults` blocks. openarm's tabletop arena (table + 3 cubes +
   drawer) is genuine *task* scene config and its wrist cameras come from the
   upstream OpenArm MJCF (`camera_wrist_*`); migrating it to cameras-via-rig +
   arena-via-`DeployScene` is a larger, separable change that touches a working
   eval/deploy path (pi05-openarm). **It is deferred to a focused follow-up
   PR**, tracked below; until then openarm keeps its existing
   `scene_defaults.composition` (issue #191 Phase 3b), which is unchanged by
   this ADR. The new camera rig already composes with it: the rig is idempotent
   and openarm's composed MJCF already carries its cameras, so the rig is a
   no-op there.

   **Follow-up (openarm migration):** add a `DeployScene.composition` field;
   `deploy sim` threads it to the node so the tabletop arena lives in
   `scenes/deploy/openarm_tabletop.yaml`, not `robot.yaml`; give openarm's
   sensors `sim_placement` (or `sim_camera_name` mapping the upstream
   `camera_wrist_*`) so cameras come from the rig; remove openarm's
   `scene_defaults.composition`. Gated on the openarm sim/eval tests proving the
   `top` / `left_wrist` / `right_wrist` cameras are unchanged.

4. **`look_at_quat_wxyz` + `rotation_to_quat_wxyz` move to
   `openral_core.geometry`**, re-exported from `openral_world_state.geometry`
   for back-compat, so the rig (HAL, layer 0) computes orientations without a
   backward layer-2 dependency.

## Consequences

**Positive**
- One mechanism for every bare-twin arm; adding a camera to any robot is a
  manifest edit (a `sim_placement` block), no code.
- Robot manifest describes the robot; scene files describe scenes. No
  scene-specific data on `robot.yaml`.
- Deploy twins are minimal (arm + its cameras + light); task props no longer
  leak into deploy.
- The rig is idempotent, so scene-attach (ADR-0034) and prop-composition
  (openarm) compose with it rather than fighting it.

**Negative / costs**
- Additive change to the core `SensorSpec` schema (new optional submodel) — JSON
  Schema export + repo-state map + fuzz coverage updated; `schema_version`
  unchanged (backward-compatible addition, CLAUDE.md §1.6).
- Two camera-staging mechanisms coexist until the openarm follow-up lands: the
  generic rig (so100/so101 and every future bare twin) and openarm's
  `scene_defaults.composition` (its task arena). They do not conflict (the rig
  is idempotent), but the duplication is intentional and temporary.

## Alternatives considered

- **Keep the per-robot `so101_box`-style composer (status quo).** Rejected:
  couples robot↔scene, doesn't generalise, dead props in deploy.
- **Inject cameras inside each task composer.** Rejected: every robot needs a
  composer; cameras (a robot property) end up defined per-scene.
- **Inline a look-at helper in the HAL instead of moving the shared one.**
  Rejected: duplicates ADR-0044 geometry (CLAUDE.md §1.13).
