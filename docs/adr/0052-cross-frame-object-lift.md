# ADR-0052: Cross-frame object-lift (RGB-camera optical TF + octomap/kernel decoupling)

- Status: **Proposed**
- Date: 2026-06-12
- Related: [ADR-0030](0030-geometric-safety-collision-checking.md) (octomap world-voxel leg + kernel
  capsule-vs-voxel check); [ADR-0035](0035-perception-spatial-memory-object-lift.md) (2-D detection → 3-D scene graph);
  [ADR-0037](0037-gstreamer-perception-bus-object-detection.md) (detector rSkill); [ADR-0043](0043-locate-in-view-reasoner-tool.md);
  [ADR-0050](0050-single-resident-skill-vram-eviction.md).

## Context

In deploy-sim robocasa the autonomous grab never fired: `recall_object` always returned empty,
because `world_state.detected_objects` stayed empty even though the detector published detections.
Three compounding causes:

1. **No world voxel map.** `ObjectLifter` projects the occupied world voxels
   (`/openral/world_voxels`) into each detection camera and intersects with the 2-D box. The demo
   ran `--no-enable-octomap` (the ADR-0030 kernel capsule-vs-voxel check false-positives in the
   dense kitchen — arm starts ~3 mm inside a counter voxel → E-stop at step 1), so there were **no
   voxels** to lift against.
2. **The RGB camera had no optical frame.** The agentview cameras declared `frame_id: world` (the
   global origin, not the camera pose) and no `*_optical_frame` TF was broadcast — only **depth**
   cameras got a live `base_link → <camera>_optical_frame` TF. So even with voxels, the lifter
   could not place the box's camera.
3. **The detection was mis-stamped.** The launch hard-coded the detector's `sensor_id=front_depth`
   while it ran on `agentview_left` RGB — so the lifter resolved the wrong camera's
   intrinsics/extrinsics.

The voxel-grid lift is the **real-hardware-correct** model (one body-mounted depth sensor builds
the map; RGB detections from separately-mounted cameras lift against it across frames). The fix is
to make it actually work, generically.

## Decision

1. **Broadcast a generic RGB-camera optical-frame TF.** `SimSensorBridge` broadcasts
   `base_frame → <camera>_optical_frame` for **every** RGB camera whose `frame_id` is a dedicated
   `*_optical_frame`, from live MuJoCo poses (reusing `camera_optical_tf_to_base` +
   `mjcf_camera_name`, the same mechanism depth cameras already use). Generic over all robots and
   camera names — the MJCF camera is read from each `SensorSpec.metadata.mjcf_camera`. Cameras whose
   `frame_id` is a robot link (e.g. an eye-in-hand at `panda_hand`) are skipped — they already have
   TF from `robot_state_publisher` and must not be clobbered.
2. **Per-robot camera config.** Each liftable RGB camera in `robot.yaml` gets a dedicated
   `*_optical_frame` `frame_id` + `metadata.mjcf_camera` (done for panda_mobile's agentview L/R).
3. **Stamp the detection with its real camera.** The launch derives the detection camera from the
   robot's first liftable RGB camera (an `*_optical_frame` RGB sensor) and sets **both**
   `image_topic` and `sensor_id` from it — generic, no hard-coded `front_depth`/`agentview_left`.
4. **Decouple octomap perception from the kernel check.** New
   `--no-enable-octomap-kernel-check` (launch `enable_octomap_kernel_check`, default True). With
   `--enable-octomap --no-enable-octomap-kernel-check`, `/openral/world_voxels` is published for
   the object-lift while the kernel's capsule-vs-voxel check stays **off**.

The lift itself (`ObjectLifter`) is unchanged — it already does the cross-frame projection.

## Safety (CLAUDE.md §3)

Decision 4 touches the kernel voxel-check **gating**, so it carries a safety-WG note + hazard-log
entry. It **never weakens** the kernel below the existing `--no-enable-octomap` baseline: when the
new flag is False the kernel posture is identical to `--no-enable-octomap` (envelope +
self-collision checks on, capsule-vs-voxel off) — it only **adds** a perception topic. Default True
preserves bundled ADR-0030 behaviour. No code path lets the flag re-enable a less-conservative
kernel.

## Consequences

- **Positive:** the autonomous detect → recall → navigate → grab loop works on robocasa;
  generic over robots / camera names / depth names; reuses existing lift + TF mechanisms; real-HW
  correct (separate depth sensor + RGB cameras).
- **Negative / costs:** publishing world_voxels adds octomap_server + bridge load; the per-robot
  `robot.yaml` cameras need an `*_optical_frame` + `metadata.mjcf_camera` to be liftable.

## Testing

- Unit: `SimSensorBridge` broadcasts `base → <cam>_optical_frame` only for RGB cameras with an
  `*_optical_frame` (a link-framed eye-in-hand camera is skipped); generic over a fake 2-camera
  description.
- Integration/sim: with `--enable-octomap --no-enable-octomap-kernel-check`, the agentview optical
  TF resolves, `/openral/world_voxels` publishes, and a detected object appears in
  `world_state.detected_objects` so `recall_object` resolves (the deploy-sim robocasa repro).
- Safety: a test pinning that `enable_octomap_kernel_check=False` leaves the kernel's
  `world_voxel_enabled` False (no less conservative than `--no-enable-octomap`).
