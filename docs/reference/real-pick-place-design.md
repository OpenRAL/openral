# Pick and place on the real OpenArm cell with the world-voxel check on

Status: **implemented on the branch, default-off, pending Safety-WG review**, 2026-10-03.
Branch `feat/real-pick-place-attachment` (draft PR #332). Every step of §3 except 5 (the attended
effort measurement) and 9 (enabling the Thor scene) is implemented and committed off; the
kernel exemption, the release window and the place witness are each proven against the real
`safety_kernel_node` (the tests named in §3). The §4 items are still decisions, not code.
Facts in §1 cite code at `fe3c8944` unless marked *since*; §2 is now the design as built.

Goal: on the real OpenArm cell (Thor: ZED-M head camera, RGB-only wrist Arducams,
`openral deploy run` with the safety kernel's world-voxel check at the real 20 mm margin on
20 mm cells), pick an object (the restock box) and place it (on a shelf) without a kernel stop,
using SAM 2.1 to see the object.

## 1. Why it cannot work today (facts)

1. **The approach stops first.** The robot-link-vs-voxel check
   (`cpp/openral_safety_kernel/src/collision.cpp` `check_voxel_collision`, L1240-1380) has no
   exemption of any kind. Every exemption the kernel has — ADR-0092 D6 support witness,
   attach residue, ADR-0097/0098 place allowance — applies to an *attached payload* only. On the
   real margin the finger hull trips on the target's own cells before gripper effort can rise, so
   no grasp event, no attachment, no payload path. Sim hides this behind a 0 mm margin
   (`deploy_e2e.launch.py` L178-186).
2. **A margin reduction cannot fix it on the OpenArm.** `openarm_<side>_finger_pair` is one convex
   hull of both jaws swept over the stroke (`docs/reference/collision-geometry-review.md`
   L90-92, L349-351); a grasped box sits centimetres inside it, and 20 mm cube quantisation plus
   the 20 mm margin already exceeds the ADR-0097 cap of min(1.5 voxel, 40 mm) = 30 mm.
3. **The real attachment leg cannot construct for a bimanual robot.** `GripperEffortTrigger` and
   `VisionAttachmentEvidenceProducer` require exactly one `role: gripper` joint / one parent link
   (`_grasp_trigger.py` L164-185, `_vision_attachment_evidence.py` L476-483); OpenArm has two.
   It has no heartbeat (`vision_attachment_bridge.py` L642-660) and its revision restarts at 0 on
   every activate (L270), so with the kernel's attached check on every chunk drops as
   `DROP_ATTACHED_OVERFLOW`/`_UNAVAILABLE`. Nothing launches the segmenter; `DeployRuntime` has no
   field for it; `_attached_collision_enabled(hal_mode)` is sim-only (`deploy_e2e.launch.py`
   L452-454).
   *Since (§3 rows 1-2, 4):* the bridge builds one trigger/producer leg per `role: gripper`
   joint (`object_id` suffixed by the joint), heartbeats its set at 5 Hz only while every leg has
   live effort evidence, seeds its revision from the node clock, and the real-mode sim bridge no
   longer heartbeats "nothing attached". `DeployRuntime.vision_attachment` (off by default)
   launches the segmenter and couples the leg to the kernel's attached check.
4. **Three latent geometry bugs on the head camera.** `head_zed.frame_id` is `zed_camera_link`
   (ZED *body* frame) while the segmenter, the object lift and the attachment bridge all treat
   `SensorSpec.frame_id` as an optical frame; the manifest's ZED intrinsics are nominal; the
   attachment leg requires mask, depth and intrinsics at exactly one resolution
   (`_vision_attachment_evidence.py` L275-283) while the ZED publishes RGB at native resolution
   and depth at `pub_resolution`. Any geometry taken from `head_zed` today is rotated or refused.
   *Partly fixed since:* the segmenter now projects in the image header's frame through the
   driver's `CameraInfo` (`camera_infos` parameter; measured K fx = fy = 1498.18,
   cx 936.11, cy 541.81 vs the manifest's nominal 960/960/540), and
   `openral_hal.depth_cloud.intrinsics_from_camera_info` / `resample_mask_nearest` exist. The
   attachment bridge now consumes them: it looks tf2 up for the depth image's `header.frame_id`
   (`zed_left_camera_frame_optical` on Thor, via the driver's `/tf_static` chain under the
   manifest's `openarm_base -> zed_camera_link`), sends the TCP/jaw-tip prompts already in that
   optical frame (empty `SegmentInView.frame_id`; precondition: the segmenter's RGB is registered
   to the depth stream), projects only through the depth `CameraInfo`
   (`vision_attachment_camera_info_topic`), and resamples a mask onto the depth raster when only
   the resolution differs. A missing frame, a missing/mismatched `CameraInfo` or a different crop
   is a logged `GRIPPER_FORCE` fallback, never the manifest's nominal K. Still open: the
   world-state object lift (`ObjectsMetadata` carries no image frame).
5. **Two more HAL bugs.** Both effort read paths zero-fill a missing effort channel
   (`ros_control_transport.py` L617-627, `openarm_real.py` L380-383), so the trigger can never
   notice the channel is gone. The attach link (`openarm_*_link7`) and finger frames do not exist
   in the vendored URDF's TF tree (it describes a different assembly, `robot.yaml` L650-660); the
   vendor bringup's `robot_description` is what the real cell publishes and is unverified.
   *Since (§3 rows 0-1):* a missing effort channel reads as `effort == []` and is counted by the
   trigger; the TCP comes from the gripper joint's origin, not TF; and
   `VisionAttachmentConfig.tf_frames` maps a manifest link to its TF frame as a proven identity
   (`openarm_*_link7` is the MJCF/URDF body `openarm_*_ee_base_link`, the one the vendor
   `openarm_ros2` description publishes). The octomap bridge clears a held payload by looking
   its attach link up on TF too, so it gets the same renames (`attach_link_tf_frames`).
6. **Place has no real producer for any of its inputs.** Region (sim: MuJoCo subtree), support
   witness (sim: `mj_geomDistance`), release (sim: contact loss + 10 mm rigid-follow tolerance),
   and nothing subscribes `/openral/place_declaration` on real. Three real-only hazards sim never
   shows: fingers vs the shelf at the 20 mm link margin; the released object reappearing inside
   the finger margin on retreat (worse on real: DETACH fires when the jaws open, before the
   fingers move); ~1 cm joint play against zero-clearance place geometry.
7. **Already true on real, and dangerous once anything publishes attachments:** the octomap
   bridge's payload clearing (`attached_clear_enabled` default true) and the self-filter both
   consume `/openral/world_state_fast` regardless of the kernel flag. A vision leg turned on
   without the kernel's attached check would clear the payload from the map while nothing checks
   it — an invisible payload.

## 2. The design (proposal)

One discipline, copied from ADR-0097: **dispatch names, a producer measures, the kernel bounds,
everything dies with the goal.** One goal-scoped declaration carries both halves.

### 2.1 Grasp-target exemption (kernel; Safety-WG)

- New `GraspDeclaration` (msgs + `openral_core`), field-for-field a mirror of `PlaceDeclaration`:
  `target_id`, `object_id`, `contact_links`, `rskill_id`, `trace_id`, `timeout_s`, `stamp_ns`,
  `active`, `region_valid`, `region` (reuse `PlaceRegion`; `geometry[]` must be empty in v1).
  Rides the `AttachmentState` → `WorldStateStamped` envelope beside the place declaration so
  the kernel applies region and attachment set from one snapshot. Additive optional fields: no
  `schema_version` bump.
- Dispatch (`rskill_runner_node.py`, cloned from `_arm_place_declaration` L2020-2063) names the
  target and **strips any region**; retracts on every exit incl. E-stop (L977, L2102, L2228).
- Kernel: while live, cells whose centre lies in the producer-measured oriented box are exempt
  **for the declared gripper's finger link only** (intersection of a launch-derived manifest
  allowlist — the `role: gripper` joints' `child_link`s — and the declaration's `contact_links`).
  Every other link, every cell outside the region, self-collision, attached checks and the force
  gate are unchanged. An exempt pair never supplies the reported identity (the attached-path
  contract) and reaches `sweep_min` at its own depth; the velocity band clamps an untripped
  check's slack to 0, so it reads as the band's slowest rate rather than a negative slack that
  would discard the non-exempt pairs' graded slowdown. It
  skips the stage-2 narrow phase, leaving the call's shared refinement budget to the pairs that
  can trip. Fail-closed on: retraction, timeout, future stamp, stale
  world state, frame mismatch, oversize/degenerate region, non-empty geometry, non-allowlisted
  link, grid-frame change, rejected attachment set. Feature parameter default **off**.
- Handover: on the attachment edge that adds the declared object the exemption stays alive only
  while the payload origin (FK of the measured configuration) is inside the region, then retires
  permanently; a detach retires it. Only a payload attached on the declaring gripper's own chain
  (a declared contact link or a non-root ancestor) counts — never the other hand's payload or a
  release record frozen on the base — and an undeclared object attached there retires the
  declaration (`handover_object_mismatch`, WARNed once per declaration as
  `grasp_region_rejected` with the declared object, attached label, target, rskill and trace). The region is latched at handover: later snapshots of the
  same declaration cannot move it, so a producer re-measuring the carried payload cannot extend
  the exemption. From then on the existing path applies (bridge clears the payload cells; kernel
  checks it as attached geometry).
- Rejected: masking target cells out of the map upstream (object invisible to every consumer,
  decision outside the kernel, breaks the bridge's "an occupied cell is an obstacle" invariant);
  lowering the global real margin (HZ-0095-2 class; it also derives the extrinsic gate).
- Caps are WG placeholders: half-extent ≤ 0.20 m, volume ≤ 0.03 m³, `timeout_s` ≤ 120 s, a region
  max age. Monotonicity property to pin in gtest: `trips_with ⊆ trips_without`; the difference is
  {(finger link, cell in region)} plus only those non-exempt pairs the exact stage-2 distance clears
  once exempt pairs stop spending the shared refinement budget; and the band slack of a sweep that
  passes without the region is never raised by it (it may only slow). The region may LOWER
  `sweep_min` — an exempt hull link reports stage 1's bound, not the refined distance — which errs
  slower, never faster.

### 2.2 Target perception before the grasp

- **Prompt (built): the reasoner names, perception grounds, the producer measures.** What to
  pick is task knowledge, so it lives in the reasoner, not in a scene. `ExecuteRskillTool` carries
  `grasp_target: GraspTargetRef{label, object_id?, contact_links}` and
  `place_target: PlaceTargetRef{fixture_id | place_node_id}`; the system prompt lists the robot
  unit's fixtures as the place choices. At dispatch the reasoner (`openral_reasoner.grounding`,
  called from `ReasonerNode._dispatch_execute_rskill`) grounds them from perception it already
  holds: `object_id` → the recalled spatial-memory node's 3D box; else the ONE live detection on
  `/openral/world_state_slow` carrying the label — the world-state lift's (`VoxelFrustumLifter`)
  axis-aligned box, which `WorldStateStamped` now carries (`detected_object_bbox_*`). That box,
  padded by one voxel + the extrinsic accuracy bound on every side (downward too: the lifted
  bottom is the lowest occupied centre in the detection's frustum, which can sit above or below
  the true support, so the pad keeps the support layer inside the box) and gravity-aligned in
  `openarm_base`,
  becomes `GraspDeclaration.search_box`; the fixture id becomes `PlaceDeclaration.target_id`. No
  match, more than one match without `object_id`, a box outside the base frame, an unknown
  fixture, no `contact_links` on a robot with more than one hand (the bimanual OpenArm:
  defaulting to every gripper would exempt the idle hand too), or `contact_links` that are not
  gripper child links of ONE hand (a hand = the `role: gripper` joints hanging off one arm, so
  R1 Pro's two finger joints per arm are one hand) refuses the dispatch (no goal sent) and tells the LLM to disambiguate. Both
  declarations ride the `ExecuteRskill` goal; the runner stamps them and strips any region, and
  the producers below measure. The search box only *seeds* perception: the exemption region is
  always the measured one, never named (HZ-0097-2/4 precedent), so the kernel's trust boundary
  is unchanged — policy attention and LLM-named *regions* stay rejected as safety inputs.
  *Direct dispatch (reasoner off):* `DeployScene.grasp_declaration` (with an optional
  `search_box`) remains for an attended run; the committed cell scene carries none.
  *Limits:* `locate_in_view`'s one-shot 2D answer is not lifted, so the label must be one the
  continuous detector publishes; the lift must run in the base frame (`object_lift_map_frame`)
  on a fixed-base cell without a `map` frame; a recalled free-space place (`place_node_id`) is
  refused until a free-space place producer exists.
- **Measurement:** the support plane is *measured*, never read off the search box (whose
  bottom is a lifted detection bbox min-z and can sit below the real table top, HZ-01xx-6): in a
  column under the box (reaching 0.15 m below its bottom), scanned top-down, the first layer
  whose top-surface cells ring the footprint of the target standing above it (between one cell
  and 0.05 m out, which must be at least two cells — a grid too coarse for it is a
  `probe_margin_under_two_cells` lost view, never silently widened — a surface extends past what stands on it, the target's own dense top does
  not), whose top face is the support — no such layer is a typed `no_support` refusal and no
  region; the target's lowest cell must sit within one voxel (+ one of tolerance) of it, else
  `not_on_support` (a bench below the shelf board the target stands on) and no region. Then occupied voxels inside the
  search box → cluster above the measured support plane →
  cluster top-centre projected into the ZED left image as SAM 2.1's positive point → mask (eroded
  2-3 px) → masked ZED depth → base-frame cloud → robust PCA OBB (reuse `_pca_basis` /
  `clustered_obb_primitives`), extruded down to the support plane, padded by ≥ √3·10 mm plus
  extrinsic error. Cross-check: the OBB must contain enough occupied cells or it is refused.
  *Pure-geometry core landed:* `openral_hal._grasp_target` (seed, prompt projection, region, map
  cross-check, tracking gate; tests in `tests/unit/test_grasp_target.py`). *ROS wiring landed,
  default off:* `openral_hal._grasp_target_leg`, owned by `VisionAttachmentBridge`
  (`vision_attachment_grasp_target_enabled`, scene `runtime.vision_attachment.grasp_target_enabled`;
  `deploy run` refuses `grasp_allowance_enabled` without it, judged on the effective HAL params
  after `--hal`, and `deploy_e2e.launch.py` refuses it again on any non-sim launch whose
  `hal_params_file` does not turn the leg on, so a bare `ros2 launch` cannot skip it): the search box is `GraspDeclaration.search_box`
  (optional, grounded by the reasoner or supplied by a direct-dispatch scene, passed through by
  the runner; a search hint only — the support layer is measured from the voxel map, never read
  off the box's bottom face),
  re-measured at `grasp_target_rate_hz` (3 Hz), the region filled onto every attachment
  publication; contradicting evidence retracts at once, a lost view freezes the last accepted region
  (a map-covered re-fit that shrinks or shifts *inside* the held region grown by one voxel while a
  declared contact link's tf origin is within one voxel + `occluder_margin_m` (0.05 m) of it —
  the robot's own hand occluding part of the target — is a lost view, `occluded_refit`; the same
  shrink with no contact link near is `unoccluded_refit`, and a re-fit the map does not cover is
  `map_disagrees`, both retracting at once, as does one reaching outside the held region)
  for `grasp_target_freeze_s` (default twice `grid_max_age_s`, the deploy's kernel voxel deadline —
  2 s on the real cell) from its depth stamp; a grid older than `grid_max_age_s` is not used. A mask whose capture stamp is more than
  `mask_depth_max_skew_s` (0.1 s) from the depth frame it would be back-projected through is
  refused (`mask_depth_skew`, a lost view here; a `GRIPPER_FORCE` fallback in the attachment path). Tests: `tests/unit/test_grasp_target_leg.py`,
  live `tests/integration/test_grasp_target_leg_live.py`. The occlusion freeze is bounded by the TTL and
  requires the declared contact link near the held region (tf origin, not a swept-hull test).
- **Representation:** an oriented box in `openarm_base` (reuse `PlaceRegion`): ~150 B, grid-instance
  independent, exact point-in-OBB already in the kernel.
- **Tracking:** re-prompt from geometry at 2-5 Hz (project the previous centroid, re-fit, gate on
  ≤ 1 voxel centroid shift), stateless and replayable; no SAM 2 video memory. Under gripper
  occlusion (expected: the hand covers the box in the head view) freeze the last validated box
  under a short TTL (≈ 2 s); the TTL ends at attach, goal end, cancel or E-stop. Target lost →
  exemption retracted → kernel holds at the normal margin.
- **Handover:** at the ATTACH event reuse the pre-grasp box as the payload (gated by containment),
  with `object_id = target_id` and a new `AttachmentEvidenceKind`; the head view is occluded
  exactly then and the wrist cams have no depth, so re-segmenting at the TCP is the worse option
  on this robot.
- Must be fixed first (all three are silent): `head_zed` optical frame, intrinsics from the
  driver's `camera_info`, explicit mask/depth resampling.

### 2.3 Place

- Reuse `PlaceDeclaration`/`PlaceRegion` on the wire; **no kernel change for the payload path.**
- Region source: a unit-surveyed shelf fixture (`robots/openarm/units/<unit>.yaml`, cell-specific
  like the ZED mount, fixed-base robots only) that the producer **verifies live against the voxel
  map** before it arms (face occupied within ±1 voxel, free volume above it empty); on failure the
  declaration goes out region-less (margins unchanged = fail-closed). ADR-0097 amendment: a
  unit-surveyed + map-verified fixture counts as producer-measured on a fixed base.
- Witness substitute: a proximity-based attestation (`DECLARED_FIXTURE` evidence kind, labelled as
  not sensed contact), once per declaration, when the payload's lowest primitive is within
  max(1 voxel, survey uncertainty) of the verified plane and the gripper is still loaded.
- Release: keep gripper-effort DETACH as "jaws opened", but keep the object as an attached record
  (frozen at the DETACH-stamp FK pose, still checked against arm and world) until every finger
  hull is > margin + 1 voxel from it, a timeout, a new ATTACH or goal end; only then publish `[]`.
  The bridge keeps clearing the frozen primitives at zero padding until then (extends
  `AttachSweepLedger`).
  *Implemented* in the vision leg (`VisionAttachmentBridge`, default behaviour; the leg itself
  stays off): the frozen record is attached to the collision model's root link
  (`base_frame` = `openarm_base`, identity FK, no primitives), not to the hand — the kernel
  places an attached object at `link_world[attach_link]` for every predicted configuration, so
  a hand-attached record would ride the retreating hand through a whole chunk while the real
  object stays put. `touch_links` = the held record's hand + finger links (no link the held
  record did not already exempt); the separation test is a separating-axis lower bound between
  the manifest's hand/finger boxes (posed by tf2 + the jaw angle) and the payload, against
  `release_clear_m` (the deploy's kernel world-voxel margin + one voxel; 0.04 m on the real cell,
  HAL param `vision_attachment_release_clear_m`); `release_timeout_s` (3.0 s,
  `vision_attachment_release_timeout_s`) bounds it, since the bridge sees no goal end. On a
  deploy, `deploy_e2e.launch.py` derives `vision_attachment_release_clear_m` (the kernel's world
  margin + one octree cell) and `vision_attachment_grid_max_age_s` (the kernel's voxel deadline;
  the HAL node has no default for either and refuses to activate the leg without them)
  from its own single sources, and refuses to launch when an override (`--hal`, a hand-written
  params file) sets a grid age above the kernel's deadline or a clearance below its margin plus
  one cell; `release_timeout_s` is the scene's
  `runtime.vision_attachment.release_timeout_s`. No octomap-bridge change was needed: payload clearing clears every attached object
  on `/openral/world_state_fast`, and the frozen record does not move, so its
  `AttachSweepLedger` window behaves as a held payload's. Proven on the real kernel by
  `tests/integration/test_vision_attachment_release_window_live.py`.
  A place witness armed while held carries onto the frozen record (same object and stamp, so
  the kernel's latch key does not change) until the window closes, keeping the support patch
  exempt in the kernel and partitioned by the octomap bridge — proven against the real kernel
  by `tests/integration/test_place_fixture_release_live.py`.
- Open: fingers vs shelf at the 20 mm link margin. Either accept and measure finger-shelf
  clearance in the attended runs first, or extend the map-verified region's allowance to the
  finger link for cells inside the region and below the plane + 1 voxel — the same exemption class
  as §2.1, WG.

### 2.4 Shared declaration

Extend `PlaceDeclaration` additively or add a `TaskDeclaration`: `grasp_target` (id + source
fixture), `place_target` (id + fixture), `object_id`, `rskill_id`, `trace_id`, `stamp_ns`,
`timeout_s`, `active`, monotonic `revision`. One runner arm/retract lifecycle, one producer (the
single `/openral/attachment_state` authority), per-arm gripper legs, one heartbeat. A grasp
exemption never applies to the place target and vice versa: scope each by its own `target_id`
and region.

## 3. Order of work

Everything is off by default until the last step; nothing before it can actuate.

| # | PR | Layer | Notes |
|---|---|---|---|
| 0 | `fix(hal)`: report an absent effort channel instead of zero-filling it | HAL | pre-existing bug (§1.5), own commit |
| 1 | `feat(hal)`: one grasp trigger + evidence producer per gripper; TCP from the gripper joint origin, not TF | HAL | B1, B7 |
| 2 | `fix(hal)`: 5 Hz attachment heartbeat gated on live effort evidence; time-seeded monotonic revision; real-mode `SimSensorBridge` stops claiming "nothing attached" | HAL | B2, B3 |
| 3 | `fix(hal,perception)`: back-project in the depth header's optical frame; intrinsics from live `camera_info`; explicit mask resampling; `top`/`head_zed` optical `frame_id` | HAL, perception | B4, B5 |
| 4 | `feat(deploy)`: `DeployRuntime.vision_attachment`, segmenter lifecycle node in the launch, **vision leg on real always turns the kernel attached check on** (1000 ms deadline) | CLI, launch | **Safety-WG + hazard log**. *Implemented, committed off* (`enabled: false` in `scenes/deploy/openarm_real_world_voxels.yaml`) pending WG review and the hazard-log entry |
| 5 | `test(hil)`: attended OpenArm gripper-effort readback (gripper-only motion, user at the E-stop) | HIL | decides whether effort is a grasp signal at all |
| 6 | `feat(kernel)`: `GraspDeclaration` across IDL/core/world-state/runner/HAL/launch/kernel + conservativeness tests | all | **ADR + hazard log; split (>800 lines)**. *Wire landed* (IDL, `openral_core.GraspDeclaration`, World State relay, runner arm/retract on `/openral/grasp_declaration`, CLI/launch `grasp_declaration_json`, committed target in `scenes/deploy/openarm_real_world_voxels.yaml`); the sim producer landed (`SimAttachmentEvidenceTracker.set_grasp_declaration` / `grasp_declaration`: the target subtree's box, base frame, no geometry, on every `AttachmentState` envelope), the launch flag `DeployRuntime.grasp_allowance_enabled` (default off) with the always-passed manifest-derived `grasp_contact_links`, and the kernel consumer landed (`ingest_grasp_declaration`, default off; per-candidate scoping, handover retirement against the region latched at handover, proven on the twin control pair `tests/sim/test_gripper_twin_hal_mujoco_grasp_pair.py`); the real producer pending |
| 7 | `feat(perception)`: pre-grasp target producer (search box → SAM 2.1 → OBB → region), tracking, handover | HAL/perception | develop on the twin pass (real ZED, twin HAL). *Producer leg implemented, default off* (`_grasp_target_leg`); the OpenArm scene's committed declaration carries no `search_box` yet, and the thresholds are uncalibrated |
| 8 | `feat(hal)`: place on real — unit fixture + map verification + proximity witness + frozen release | HAL, bridge | **ADR-0097/0092 amendments**. *Schema landed*: `openral_core.UnitFixture` on `RobotUnit.fixtures` (checked by `fixture_problems` in `load_robot_unit`) and `AttachmentEvidenceKind.DECLARED_FIXTURE`; no unit carries a fixture yet and no producer reads one Frozen release window implemented* in the vision leg (§2.3 "Release"); fixture, map verification and witness pending . *Producer leg implemented, default off* (`_place_fixture_leg`, `vision_attachment_place_fixture_enabled`): resolves the declaration's `target_id` to a unit fixture, verifies its top face and the free volume above it against the live voxel map (unverified → region-less, reason logged), ships the fixture box as the region (no geometry), and attests the `DECLARED_FIXTURE` proximity witness once per declaration; the frozen release and the finger allowance are not part of it, and the thresholds are uncalibrated |
| 9 | `feat(scenes)`: enable on the Thor scene with measured thresholds | scenes | only after 5's verdict |

Steps 0-3 are plain bug fixes on code that exists and can start now. Step 4 is where safety
posture first changes on real hardware. Steps 6 and 8 are the semantics the Safety-WG has to
decide; 7 and 8 need the attended cell for calibration.

## 4. Safety-WG items (private `OpenRAL/management`)

1. ADR: declaration-scoped grasp-target exemption for gripper finger links (amends ADR-0097's
   "arm-vs-world unchanged" for those links only); caps; measure-once vs re-measured region;
   handover retirement rule; whether `link7` is a contact link; the producer's prompt source.
2. Hazard HZ-01xx: exemption misapplied (non-target body inside the region; wrong object; stale
   declaration; target moved while frozen; leak to other links/arms; fingers into the support).
3. Turning `attached_collision_enabled` on for real, with the deadline, and trusting vision
   geometry for map clearing (undersized box clears a real obstacle; phantom fallback box on a
   closed-on-nothing gripper; dead effort channel → kernel drop window).
4. ADR-0097 amendment (unit-surveyed + map-verified fixture = measured region on fixed bases);
   ADR-0092 D6 amendment (proximity witness); the ADR-0098 joint-play offset if adopted;
   the frozen-release window; finger allowance inside the place region.
5. Bimanual attachment (two attach links) — shared by both halves.

## 5. Measure before deciding (attended, cell)

- `/joint_states` effort presence and raw gripper values; close-on-nothing vs close-on-foam
  percentiles → are the 0.30/0.10 × 333 thresholds meaningful at all.
- *Measured 2026-10-02, no motion:* depth and RGB images both carry
  `header.frame_id = zed_left_camera_frame_optical` at 1920x1080; `camera_info` K = fx = fy =
  1498.18, cx = 936.11, cy = 541.81 (the manifest's nominal fx = 960 is 56 % short);
  `/tf_static` carries `zed_camera_link -> zed_camera_center -> zed_left_camera_frame ->
  zed_left_camera_frame_optical` from the driver. The vendor `openarm_ros2` description names the
  hand link `openarm_*_ee_base_link`; the manifest's `openarm_*_link7` is the same body
  (`tf_frames` in the Thor scene). Still to read with the real bringup up: `/joint_states`
  effort presence and `tf2_echo openarm_left_ee_base_link zed_left_camera_frame_optical`.
- *Measured 2026-10-02:* SAM 2.1 hiera-small, bf16, `transformers` 5.5.4 on Thor, one live
  1920x1080 ZED left frame, point prompt: warm median 53 ms, p95 57 ms, cold 635 ms, peak
  279 MiB allocated — the same as the RTX 4070 Laptop figures, with π0.5 not loaded. The
  attach barrier (~100 ms) holds it; measure again beside the running policy.
- The restock box dimensions (for the caps) and finger-shelf clearance during a real place.

Sources: four read-only investigations of this branch (kernel allowance, pre-grasp perception,
HAL vision leg, place path), 2026-10-02.
