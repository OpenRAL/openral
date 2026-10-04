# Pick and place on the real OpenArm cell with the world-voxel check on

Status: **implemented on the branch, default-off, pending Safety-WG review**, 2026-10-03.
Branch `feat/real-pick-place-attachment` (draft PR #332). Every step of §3 except 5 (the attended
position-stall measurement) and 9 (enabling the Thor scene) is implemented and committed off; the
kernel exemption, the release window and the place witness are each proven against the real
`safety_kernel_node` (the tests named in §3). Place needs no surveyed cell geometry and no named
target: the producer measures the surface under the carried payload (§2.3). The §4 items are still decisions, not code.
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
   live jaw-position evidence (it was effort evidence until the position-stall trigger, §1.6),
   seeds its revision from the node clock, and the real-mode sim bridge no
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
   is a logged `GRIPPER_CLOSURE` fallback, never the manifest's nominal K. The world-state object
   lift too, since: the detector stamps each `ObjectsMetadata` with its driver `CameraInfo`'s
   frame and K (`camera_frame_id` / `camera_intrinsics`, `camera_infos` parameter), and the lift
   and its eviction FOV project through them. `deploy_e2e` points the detector at the driver's
   `CameraInfo` beside the unit overlay's `ros2_image` binding (Thor `top` →
   `/zed/zed_node/rgb/color/rect/camera_info`, `zed_left_camera_frame_optical`, fx 1498.18),
   never the sensor leg's manifest-built one; only a camera with no driver `CameraInfo` falls
   back to the `SensorSpec` (the stand-in `world` / fx 640 for `top`), logged once.
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
6. **The real OpenArm gripper reports no effort at all.** Vendor `openarm_ros2` (4e837e1,
   `openarm_simple_hardware.cpp` L268-279) hard-codes effort and velocity to `0.0` every tick, so
   `joint_state_broadcaster` publishes a full list of zeros: the empty-effort guard of row 0 never
   fires and an effort trigger silently never ATTACHes. The gripper is a DM4310 in MIT position
   control (kp 5, kd 0.1) behind a `JointTrajectoryController` position command; the manifest
   declares `has_torque_sensor: false`, and its `effort_limit: 333` is not physical. Real 30 fps
   teleop (`qualiadev/openarm-canonical-and-fabians-vr`): closing on an object stalls
   0.18-0.29 rad short of a 0.0 command and stays flat to ~1e-3 rad; closing on nothing reaches
   <= 0.02 rad (rest offsets ~0.0086 left / 0.0116 right); free-motion steady-state error
   0.006-0.025 rad; position LSB 3.815e-4 rad.
   *Since (2026-10-03, `feat/position-stall-trigger`):* the trigger is
   `_grasp_trigger.PositionStallTrigger` — the jaw settling short of its **commanded** target
   (fed from every applied safe action, `VisionAttachmentBridge.observe_command`) by more than
   `closed_rest_offset + stall_gap`, thresholds per joint in the manifest
   (`JointSpec.closure_calibration`, from the numbers above), refusing an uncalibrated gripper.
   The effort trigger, its `effort_limit` fractions and the `attach_effort` / `release_effort`
   scene knobs are removed (no in-tree gripper declares a torque sensor). The heartbeat gates on
   jaw-position liveness; an unconfirmed stall's jaw box is stamped `GRIPPER_CLOSURE` (the
   `GRIPPER_FORCE` value stays for force-sensing grippers and old records). The twin reproduces
   the stall: `tests/sim/test_openarm_hal_mujoco_position_stall.py` (a free 5 cm box stalls the
   MuJoCo jaw at ~0.25 rad, settled to ~1e-4 rad → ATTACH; a box *fixed* to the world chatters
   ±0.02 rad against the finger meshes and never settles — a twin artefact).
   *Since (2026-10-03, `fix/r4-bridge`):* the debounce and settle windows are seconds of
   sample time (`PositionStallConfig.consecutive_s=0.06`, `settle_s=0.13` — the spans the old
   3- and 5-tick counts covered at 30 Hz), so a 500 Hz joint-state stream cannot call a slowly
   closing jaw settled; a sample whose `stamp_ns` is within `min_sample_interval_s` (0.5 ms) of
   the last is a repeated cached read and is dropped by the trigger and by the heartbeat's
   evidence alike (a stamp that moves *backwards* is dropped the same way, so a wall-clock step
   back stalls the trigger and ages the heartbeat out — fail-closed). *Since
   (2026-10-03, `fix/r5-trigger`):* time alone is not enough — every window also needs a
   minimum sample count (`consecutive_samples=3`, `settle_samples=5`, the old tick counts), and
   an accepted sample more than `max_gap_s` (0.1 s, 3 periods at 30 Hz) after the last restarts
   the settle history, the debounce and the heartbeat's evidence run: a late-tick stream used to
   confirm ATTACH on 4 samples, and a sparse stream catching a chattering jaw at the same phase
   on 3. On the real OpenArm the
   jaw commands are ADR-0102 `GRIPPER_POSITION` slots: the HAL lifecycle node hands
   `observe_command` the HAL's own applied command (`last_applied_action`, the composed
   full-dof action, once the HAL committed that `(session, tick)`), so the trigger's reference
   is exactly the target the gripper controller was sent. *Since (2026-10-03,
   `fix/r5-trigger`):* the bridge no longer stages slots itself — its copy diverged on the HAL's
   error path (a slot whose send raised was never observed, so the next complete tick was held
   at 3 of 4 slots and its close command never reached the trigger); padded
   `JOINT_POSITION` rows are read at manifest indices (a right slot never reads the left jaw's
   zero pad). Every grasp event supersedes the leg's in-flight `SegmentInView` request
   (generation-tagged; a late reply for an older event is dropped, a DETACH in flight resolves
   to no attachment), and the pre-grasp region is the payload only on a leg's first ATTACH of a
   declaration — a REGRASP, or a re-pick under the same declaration, segments.
   *Since (2026-10-03, `fix/r7-twin`):* the twin-only stretch of the jaw-evidence timeout over
   the idle stepper's hold (`twin_jaw_evidence_timeout_s`) is gated on `hal_mode == "sim"` and a
   HAL with a callable `idle_step` — it used to key on the node carrying a `SimSensorBridge`,
   which every mode builds, so the real OpenArm's 0.5 s timeout became ~2.1 s, past the kernel's
   1000 ms real attached deadline (`_attached_collision_deadline_ms`) (a dead jaw channel kept a fresh heartbeat). On the same gate only,
   the evidence run spans sample gaps shorter than that timeout
   (`VisionAttachmentConfig.evidence_run_spans_gaps`), so a re-inference pause no longer
   withholds the heartbeat as motion resumes; real hardware still restarts the run on any gap
   over `max_gap_s`. A twin slot group that does not compose into one joint command (a base
   twist, uncommanded joints) now records its gripper targets as a compact row
   (`sim_attached.gripper_targets_action`) instead of `None`, which had cleared the jaw command
   every tick on a mobile manipulator.
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
  can trip. Fail-closed on: retraction, timeout, future stamp, a region measurement older than
  `grasp_region_max_age_s` / `place_region_max_age_s` or stamped in the future (`region_stale`;
  default 2 × the kernel voxel deadline, 2 s on the real cell, ≤ 4 s; checked at ingest and per
  candidate; not applied to a grasp box latched at handover), stale
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
  the exemption. The region-age bound stops at the latch: the fingers occlude the target, the box
  is frozen anyway, and the payload-in-box rule, stream deadline and `timeout_s` bound the window;
  the latched box must itself have been fresh at the handover edge. From then on the existing path applies (bridge clears the payload cells; kernel
  checks it as attached geometry).
- Rejected: masking target cells out of the map upstream (object invisible to every consumer,
  decision outside the kernel, breaks the bridge's "an occupied cell is an obstacle" invariant);
  lowering the global real margin (HZ-0095-2 class; it also derives the extrinsic gate).
- Caps are WG placeholders: half-extent ≤ 0.20 m, volume ≤ 0.03 m³, `timeout_s` ≤ 120 s, region
  max age 2 × voxel deadline (implemented; it equals the vision leg's `grasp_target_freeze_s`
  default, so a held region and the kernel's trust in it expire together). Monotonicity property to pin in gtest: `trips_with ⊆ trips_without`; the difference is
  {(finger link, cell in region)} plus only those non-exempt pairs the exact stage-2 distance clears
  once exempt pairs stop spending the shared refinement budget; and the band slack of a sweep that
  passes without the region is never raised by it (it may only slow). The region may LOWER
  `sweep_min` — an exempt hull link reports stage 1's bound, not the refined distance — which errs
  slower, never faster.

### 2.2 Target perception before the grasp

- **Prompt (built): the reasoner names, perception grounds, the producer measures.** What to
  pick is task knowledge, so it lives in the reasoner, not in a scene. `ExecuteRskillTool` carries
  `grasp_target: GraspTargetRef{label, object_id?, contact_links}` and an optional
  `place_target: PlaceTargetRef{label, object_id? | place_node_id?}` (normally unset: where to
  place is the policy's job, §2.3). At dispatch the reasoner (`openral_reasoner.grounding`,
  called from `ReasonerNode._dispatch_execute_rskill`) grounds them from perception it already
  holds: `object_id` → the recalled spatial-memory node's 3D box; else the ONE live detection on
  `/openral/world_state_slow` carrying the label — the world-state lift's (`VoxelFrustumLifter`)
  axis-aligned box, which `WorldStateStamped` now carries (`detected_object_bbox_*`). That box,
  padded by one voxel + the extrinsic accuracy bound on every side (downward too: the lifted
  bottom is the lowest occupied centre in the detection's frustum, which can sit above or below
  the true support, so the pad keeps the support layer inside the box) and gravity-aligned in
  `openarm_base`,
  becomes `GraspDeclaration.search_box`; a named place surface grounds the same way into
  `PlaceDeclaration.search_box` (a hint, `target_id = surface:<label or node id>`). No
  match, more than one match without `object_id`, a box outside the base frame, no `contact_links` on a robot with more than one hand (the bimanual OpenArm:
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
  on a fixed-base cell without a `map` frame.
- **Measurement:** the support plane is *measured*, never read off the search box (whose
  bottom is a lifted detection bbox min-z and can sit below the real table top, HZ-0115-6): in a
  column under the box (reaching 0.15 m below its bottom), scanned top-down, the first layer
  whose top-surface cells ring the footprint of the target standing above it (the target
  anchored at every layer on the top of the occupied column nearest the box centre, so a taller
  neighbour never makes the target's own top its support; between one cell
  and 0.05 m out, counted on the column grown by that margin + one cell since the ring lies
  outside a tightly padded box; the margin must be at least two cells — a grid too coarse for it is a
  `probe_margin_under_two_cells` lost view, never silently widened — a surface extends past what stands on it, the target's own dense top does
  not), whose top face is the support — no such layer is a typed `no_support` refusal and no
  region; the target's lowest cell must sit within one voxel (+ one of tolerance) of it, else
  `not_on_support` (a bench below the shelf board the target stands on) and no region; the
  masked depth cloud's lowest point must also sit within two voxels of the support, else
  `not_on_support` (a target on a same-footprint box or a hidden riser clusters with it in the
  voxels, but the mask names the target alone; a view of its top face only is refused too). Then occupied voxels inside the
  search box → the anchored cluster above the measured support plane (another cluster at least
  half its size is `ambiguous`) → cluster top-centre projected into the ZED left image as SAM 2.1's positive point → mask (eroded
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
  declared contact link's hand point (the bridge's TCP for that jaw link's leg: attach link via
  tf2 + the gripper joint's `origin_xyz`, since OpenArm's `finger_pair` is no tf frame) is within
  one voxel + `occluder_margin_m` (0.05 m) of it —
  the robot's own hand occluding part of the target — is a lost view, `occluded_refit`; the same
  shrink with no contact link near is `unoccluded_refit`, and a re-fit the map does not cover is
  `map_disagrees`, both retracting at once, as does one reaching outside the held region)
  for `grasp_target_freeze_s` (default twice `grid_max_age_s`, the deploy's kernel voxel deadline —
  2 s on the real cell; refused above twice it — the kernel ages a grasp region out at
  `grasp_region_max_age_s` = 2 × its voxel deadline, so a longer freeze would publish a region the
  kernel already refuses — as are a support search deeper than 0.5 m and an occluder margin over
  0.10 m) from the region's stamp: the older of its depth frame and the grid's `source_stamp`.
  A grid is not used when received more than `grid_max_age_s` ago (`grid_stale`), when it carries
  no `source_stamp` (`grid_source_unknown`) or when its `source_stamp` — the world data's capture
  time, not `header.stamp`, which the octomap bridge refreshes on every republish of a stalled
  octree — is older than the freeze (`grid_source_stale`); all lost views. A stalled octomap
  therefore ages the region out instead of vouching for it. A mask whose capture stamp is more than
  `mask_depth_max_skew_s` (0.1 s) from the depth frame it would be back-projected through is
  refused (`mask_depth_skew`, a lost view here; a `GRIPPER_CLOSURE` fallback in the attachment path). Tests: `tests/unit/test_grasp_target_leg.py`,
  live `tests/integration/test_grasp_target_leg_live.py`. The occlusion freeze is bounded by the TTL and
  requires the declared contact link near the held region (its TCP point, not a swept-hull test).
- **Approach-armed target (built, default off): the policy picks, the hand's approach arms.**
  The VLA policy decides which object it grasps; nothing may require the reasoner to name it
  (`ExecuteRskillTool.grasp_target` is optional and the system prompt says so: naming one only
  *pins* the target). With `vision_attachment.grasp_target_approach_m` set (HAL param
  `vision_attachment_grasp_target_approach_m`, needs `grasp_target_enabled`; 0.10 m is the
  calibration starting point, capped at `GraspDeclaration.MAX_HALF_EXTENT_M`), the rSkill
  runner arms a **goal-scope declaration** for every goal that brings none of its own
  (`grasp_approach_enabled`, set by `deploy_e2e` only when `grasp_allowance_enabled` is on too):
  `target_id="approach"`, every hand's contact links, no search box, the goal's
  `rskill_id`/`trace_id`/stamp, `timeout_s` = the goal deadline (≤ 120 s), retracted on every
  runner exit like any declaration. It exempts nothing on its own. The HAL's grasp-target leg
  narrows it: each tick, for every hand (`openral_core.gripper_hands`) the declaration names, the
  hand's TCP points (`VisionAttachmentBridge.jaw_point`, which locates the OpenArm's
  manifest-only `finger_pair` through the attach link) span a gravity-aligned box grown by the
  approach distance (`approach_box`); a hand whose box holds ≥ `grasp_target_min_cells` occupied
  cells is *approaching*. **Exactly one** approaching hand arms a one-hand declaration
  (`target_id="approach:<first link>:<n>"`, one per pick; `contact_links` = that hand only, the box as its
  `search_box`, everything else — `stamp_ns` included, so the kernel's `timeout_s` backstop
  still runs from the goal — the goal's); two at once arm none. The box follows the TCP and
  feeds the measurement above **unchanged** (measured support, anchored seed at the column
  nearest the TCP, `not_on_support`, `ambiguous`, SAM point prompt, mask fit, map cover,
  tracking, freeze TTL); the hand leaving the approach distance retracts the region at once
  (`approach_ended`), and it re-arms only from scratch. The count ignores the box's lowest
  occupied layer (`approach_cell_count`): a hand low over an empty table holds only the table and
  is not approaching (ceiling: a surface two layers thick still counts its upper layer). A hand
  whose legs hold a payload, a release window or a pending segmentation is never approaching
  (what is near its TCP is what it carries) — but the *armed* hand in that state keeps its arming
  and box unchanged (no `approach_ended`, no backoff): its ATTACH went to segmentation and only
  `_finish` → `on_attach` resolves it, so retracting there would hand nothing over
  (`attach_unarmed`). Only the hand holding nothing again ends it, on a later tick. After any retraction of an arming, that hand
  re-arms only once its box centre moved more than one voxel or one freeze window elapsed — the
  same map at the same pose gives the same verdict, so re-arming every tick would only flood the
  log and the segmenter; a backed-off hand still counts toward "two at once".
  **Handover is per hand.** Only an ATTACH on the *armed* hand (or on a link of a named
  `search_box` declaration) hands over and stops re-measurement. The goal-scope declaration names
  every hand before any approach arms, so an ATTACH there — a false positive, or a grasp of
  something never measured — hands nothing over and does not stop the other hand's approach.
  A grasp-leg measurement in flight at the handover (or at any change of what is measured) is
  discarded: requests carry the tracker's `generation`, and a reply or deadline under an older
  one is dropped — its refusal never retracts the handed-over arming, its region never
  overwrites the frozen one; a handed-over tracker also ignores `accept` / `refuse` outright,
  and nothing un-hands it before the goal ends.
  *Handover only on what was measured:* the ATTACH that resolves an arming — its `confirm` test
  evaluated before any tracker state changes, an exception in it counting as off-target —
  hands its region over
  only when the payload is the region (`_region_payload`: the jaw at it; and the tracker still
  holds exactly that region) or a segmented payload every primitive centre of which lies in the
  region grown by one voxel with the jaw within `occluder_margin_m` of it (also the test for a
  region first accepted mid-close). Any other ATTACH — the jaws closed on a neighbour, the
  region refused for this grasp and the grasp segmented — ends the arming as
  `attach_off_target`: handed over with **no** region (re-measurement stops; the hand holds
  something else), never the measured region for a payload it was not measured for — and a
  region payload whose region the tracker no longer holds at all (dropped since) is
  `attach_off_target` too, never "absent".
  *Threading (2026-10-03):* on a sim HAL the bridge's grasp trigger (`observe_joint_state`,
  hence every ATTACH/DETACH) runs on the HAL's proprio publisher thread while the heartbeat,
  release poll, segmentation replies and both producer legs' timers run on the executor. Three
  review rounds found races between them, so the bridge is **serialized by one re-entrant
  lock** rather than patched per field: every entry point that reads or mutates leg, tracker
  or attachment state takes it — the trigger/command entry points, the heartbeat and release
  poll, every publish, the segmentation reply/deadline commits, teardown, and the producer
  legs' ticks, replies, deadlines and declaration callbacks (the legs alias the bridge's lock).
  Lock order is bridge lock → the tracker's own lock (kept as an inner lock), never the
  reverse: nothing run under the tracker lock — its log sink, `on_attach`'s `confirm` — takes
  the bridge lock (unit test with order-recording locks). Slow pure work stays outside it, on
  a snapshot, committed under the lock only while the generation it was asked under is
  unchanged: a `SegmentInView` reply's mask decode and depth back-projection (bridge
  `_finish`, leg `_measure` → `accept(generation=)`, given the grid, the contact links' hand
  points and the held region read under the lock), and the grasp leg's support/seed scan of
  the grid column (`_seed`): the tick snapshots target, generation and grid under the lock,
  scans outside, then commits the request — depth, tf2, projection, the `SegmentInView` call,
  the in-flight record and its deadline timer — under the lock only while the generation is
  unchanged and the leg is not torn down (a flag `teardown` sets under the lock, also checked
  by the deadline, so no timer outlives a torn-down leg). The bridge owns the one
  `/openral/world_voxels` subscription for both producer legs: each grid is decoded once and
  shared, its whole-grid occupied-cell scan deferred to first use and cached on the lattice (a
  grid nothing is armed for costs one decode). A malformed `SegmentInView` reply resolves like
  a missed deadline (`GRIPPER_CLOSURE`, barrier released), never as an exception out of the
  executor. The HAL node re-checks barrier readiness right after storing a deferred tick, so a
  release on the proprio thread between its check and the store is not a lost wakeup. tf2
  lookups at the latest time pass no timeout (a non-blocking buffer read) and run under the
  lock. On real hardware every caller runs on the executor's one thread: the lock is
  uncontended and behaviour is unchanged.
  *Multi-pick per goal (2026-10-03, default off with the approach-armed target):* a VLA may
  pick and place several objects within one goal, or one object per goal; both run on the same
  path. Each pick arms under its own identity, `approach:<link>:<n>`: `n` is the tracker's pick
  counter, **seeded from the wall clock at tracker construction** (like the bridge's
  attachment revision) and advanced at every completed pick, so a HAL re-activated mid-goal
  never re-mints an identity the kernel retired; `stamp_ns` stays the goal's, so the kernel's
  `timeout_s` backstop still counts from the goal. While a hand is handed over nothing arms.
  The pick ends on the bridge's **DETACH event**, never by polling: the bridge hands the leg the
  release window's frozen record (`GraspTargetLeg.on_detach`), the tracker drops that hand's
  region at once while keeping the handover so nothing re-measures or re-latches it
  (`on_release`), and once the hand's legs hold nothing — no attachment, no release window, no
  pending segmentation, no trigger reading the jaws loaded; re-checked on every DETACH — also
  one that finds nothing latched (a regrasp superseded before its reply) — on every barrier
  release and when the window closes (`on_hand_settled`; the release poll and a re-ATTACH
  cannot interleave under the bridge lock) — the pick is complete (`on_pick_complete`): the counter advances and
  the hand may re-arm behind the same backoff as a refusal. **Not on what it just released
  (HZ-0115-11):** until the column the hand's next measurement would search (its approach
  box reaching `support_search_below_m` below it, `search_column`) clears the released
  payload's frozen pose by more than one voxel — the separating-axis lower bound between that
  column and each primitive's oriented bounding box, so an elongated payload is not inflated
  to a sphere — the
  hand does not arm, and is not a candidate at all: it does not count toward "two hands at
  once", so a hand guarded for the goal never blocks the other hand (one hand per declaration
  is the kernel's own rule, HZ-0115-12); a release that left no record (no tf2 at the DETACH) keeps it
  from arming for the rest of the goal. **The kernel bounds it, not the producer:** it retires
  each pick's identity at its release — when the payload attached on the declaring chain
  disappears (`reason=released`; the other hand may still hold, and the released payload may
  sit frozen on the base through its window), when the whole attachment set empties at a new
  revision (`detached` — before the handover too, whichever hand let go: the kernel never
  guesses "elsewhere" from the last snapshot it saw), or when a handed-over declaration loses
  its region or is retracted (`no_region`, `retracted`) — and keeps
  every retired identity, up to 16 per activation (oldest evicted, logged once; an evicted one
  stays bounded by its goal's `timeout_s`), so no pick's identity re-arms (HZ-0115-3).
  **Liveness is the producer's, which owns the attachment set:** every publish at a new
  revision whose set lost or replaced an object (a DETACH, a release window closing, a
  re-ATTACH) or is empty first calls `GraspTargetLeg.on_attachment_changed`, before that
  snapshot's envelope is filled. An arming not handed over (the other hand's) drops its region
  at once and advances to a fresh identity — the counter advances too when the arming has
  retracted since its identity was armed (the kernel may still hold that identity and retires
  it at this edge) — except a hand **mid-grasp** (an ATTACH being resolved: a segmentation in
  flight, or jaws read loaded with nothing latched yet — not a carried payload or a release
  window) across a change that leaves the set non-empty, which the kernel does not retire on:
  its identity is kept, so a bimanual pick survives the other hand's regrasp or release, but
  its region is dropped like any other (it was measured before the change), so that ATTACH is
  handed over with **no** region (fail closed: no exemption from a pre-change measurement); on
  an emptying change it is refreshed regardless (the kernel retires it, so that grasp hands
  nothing over — fail closed). From then on no region is accepted unless its
  `stamp_ns` — the older of its depth frame and its grid's `source_stamp` — is later than the
  change: the other hand re-arms only from a measurement of the post-detach scene, and the
  region it measured before is never re-used (live kernel row:
  `test_another_hands_release_retires_a_pre_handover_arming_and_the_producer_re_arms`). The
  hand that let go follows the pick-complete path above. A
  pre-handover retraction (`approach_ended`, a refusal) re-arms behind the backoff under the
  same identity, so an arming the kernel already retired for a fault (`grid_frame_changed`,
  `attachment_rejected`; the producer cannot see either) stays refused — until the hand
  approaches afresh: an arming that ended by leaving the approach distance, with the hand then
  seen away on every sample for a continuous freeze window — located, its approach box off the
  column over its last target — re-arms under a fresh identity (ADR draft judgement call 4:
  wiggling, coming straight back, a tf2 gap or a `min_cells` dip over the target keeps the
  retired one, and an unlocated or holding sample restarts the window — as does every tick
  that cannot sample at all: no fresh grid, a request in flight, a stale verdict; unknown is
  not away). A named
  `search_box` declaration stays handed over after its pick: a new target needs a new
  declaration from dispatch. A dispatch/reasoner declaration with a
  `search_box` wins (no approach runs); one naming a hand but no box narrows the approach to
  that hand. The kernel reads it from the envelope like any producer-measured declaration —
  it never required the dispatch relay — and now also refuses a region-carrying declaration
  whose links span two hands (`reason=links_span_hands`). *Arming rule chosen: TCP proximity
  alone, not "proximity AND the gripper command closing".* The exemption is needed **before**
  the close command: the open jaws straddle the target inside the 20 mm margin while the
  policy is still approaching (the finger hull contains the target, §1), so an intent gate
  would stop every grasp short of the close and never arm; "proximity OR closing" would arm
  on a close in free air. The conservatism comes instead from where the box is (around the
  TCP only), what fills it (a measured, unambiguous object standing on a measured support),
  one hand, the region caps, the short region-age bound (kernel `grasp_region_max_age_s`,
  2 × the voxel deadline) and attended operation. Residual (HZ-0115-11): the hand passing
  within the approach distance of a neighbour arms on the neighbour for as long as it stays
  there; the exemption then covers that neighbour's cells for that hand's links only.
  Tests: `tests/unit/test_grasp_target_leg.py` (approach rows), live
  `tests/integration/test_grasp_target_leg_live.py::test_an_approaching_hand_arms_the_target_with_no_named_target`,
  the real kernel in
  `tests/integration/test_safety_kernel_grasp_target_band.py::test_an_approach_armed_declaration_exempts_one_hand_and_nothing_beside_the_target`
  and `::test_two_approach_armed_picks_in_one_goal_on_the_real_kernel` (multi-pick: both picks
  exempt, the release retiring pick 1 while the other hand holds, pick 1's identity refused
  afterwards), the kernel gtests `…ASecondPickInTheGoalArmsUnderItsOwnIdentity`,
  `…EveryRetiredPickIdentityStaysRefusedEvenFromARestartedProducer`,
  `…AReleaseOnTheDeclaringChainRetiresWhileTheOtherHandHolds`,
  `…AHandedOverDeclarationThatLosesItsRegionRetires`,
  the runner in `packages/openral_rskill_ros/test/test_grasp_declaration_lifecycle.py`.
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
  *Implemented* (`VisionAttachmentBridge._region_payload` / `region_attachment`): on a stall
  ATTACH, when the grasp-target envelope (TTL and expiry applied) holds a region for a declaration
  naming this leg's jaw link and the leg's TCP lies within `grasp_target_occluder_margin_m` of it,
  the region box is the payload — `object_id` = the declaration's `object_id` (what the kernel's
  handover matches; `target_id` when empty), `AttachmentEvidenceKind.GRASP_TARGET_REGION`, no
  `SegmentInView` call; otherwise the attach segments as before. A leg takes each *measured*
  region once (`region_spent` = `(target_id, region.stamp_ns)`): an ATTACH offered the same region
  again segments (a handed-over goal re-measures nothing). Live:
  `tests/integration/test_grasp_target_leg_live.py` step 6.
- Must be fixed first (all three are silent): `head_zed` optical frame, intrinsics from the
  driver's `camera_info`, explicit mask/depth resampling.

### 2.3 Place

- Reuse `PlaceDeclaration`/`PlaceRegion` on the wire; **no kernel change for the payload path.**
- **No predefined knowledge.** "Pick up the boxes and put them on the shelf" must work on a
  cell nobody surveyed: no furniture in the unit overlay, no tape-measured planes, no fixture
  ids in the prompt. Where to set the payload down is the **policy's** job (learned); whether
  the spot under the payload is a surface is the **map's**. So the place producer
  (`openral_hal._place_target_leg`, owned by `VisionAttachmentBridge`, HAL param
  `vision_attachment_place_target_enabled`, scene `runtime.vision_attachment.place_target_enabled`,
  default **off**) needs no declaration from the reasoner — but it never arms outside a goal:
  - *Column.* While a goal's place declaration is live and a payload is held and loaded, at
    `place_target_rate_hz` (2 Hz) on the newest `/openral/world_voxels` (received, or its
    `source_stamp` data, older than `grid_max_age_s` = unusable; an unset or future
    `source_stamp` too — the kernel's own voxel data-age rule): the column under the
    payload's measured footprint (its primitives' AABB in the lattice axes) grown by one voxel +
    the extrinsic accuracy bound (`place_target_extrinsic_error_m`, default
    `MAX_PLANAR_ERR_M` = 15 mm) on every side, from the payload's bottom down
    `place_target_search_depth_m` (0.20 m). Cells within one voxel of any published payload are
    ignored (the octomap bridge clears exactly those).
  - *Surface.* The first occupied layer down the column is what the payload would land on. It is a
    support only if **every** footprint cell of that layer is occupied (else `partial_support`: an
    edge, clutter, a hole) and the layers above it are free up to the payload's measured height +
    one voxel + the extrinsic bound (else `no_headroom` — a shelf gap too low for the payload);
    nothing within the depth is `no_surface`. The region is that patch as a **one-voxel slab** —
    the support cells only — in the grid frame, no `geometry`, under `PlaceRegion`'s caps.
  - *Latch and freeze.* The patch is latched with the payload's identity and re-verified (never
    re-chosen) while the payload's centre stays over it: new occupancy in its free volume retracts it
    at once; support cells missing (the payload and hand occlude the board from the head camera,
    payload clearing removes the cells under it) or a stale/missing grid is a lost view that holds
    the latched region for at most `place_target_freeze_s` (default **and ceiling** 2 ×
    `grid_max_age_s` — the kernel's `place_region_max_age_s`) from the grid stamp it was last
    verified on; the region's `stamp_ns` is that grid's `source_stamp` (the world data's capture
    time — never `header.stamp`, which a bridge republishing a stalled octree keeps fresh), so the
    published region is never older than the kernel accepts, and a stalled map ages it out. The payload moving off the patch re-measures; a different payload
    drops the latch. Consequence: a set-down has to complete within the freeze after the board
    leaves view.
  - *Declaration.* The kernel applies a region only inside a `PlaceDeclaration`, and the leg never
    declares on its own: no goal, no place allowance. Mirroring the grasp side's approach
    declaration, the runner arms a goal-scope one per goal (`place_approach_enabled`, turned on by
    `deploy_e2e` exactly when the HAL runs this leg): `target_id = surface`, no object, no search
    box, stamped at goal start, `timeout_s` = the goal deadline (≤ 120 s), retracted on every goal
    exit incl. cancel and E-stop. A goal's own declaration (the optional `place_target`) or a
    direct-dispatch scene's wins over it. The leg attaches the measured region to whichever is
    live, its `object_id` narrowed to the measured payload when dispatch names none (the kernel
    scopes the region to it); a `search_box` narrows the place: a patch whose centre lies outside
    it is refused (`outside_hint`). A retraction or expiry retracts the patch and the witness at
    once (`no_declaration`) and nothing re-arms until a new declaration, however long the payload
    stays held over the surface; a new declaration drops a latched patch (`redeclared`), so it is
    re-measured under the new hint. Dispatch still never supplies a region (the runner strips
    it). A place location is never required, but a supplied hint is binding: a `place_target`
    the reasoner cannot ground (unknown label, several matches and no node id, an unknown
    memory node, no detector or no memory at all, a box outside the base frame) refuses the
    goal with `ROSReasonerInvalidPlan` — no goal is sent, and the refusal tells the LLM to omit
    the hint — and a direct-dispatch declaration that does not validate refuses the goal at the
    runner (aborted, no skill tick). Neither ever falls back to "place on whatever surface is
    measured".
  - *Witness substitute.* A proximity attestation (`MAP_SUPPORT_PROXIMITY` evidence kind, labelled
    as **not sensed contact and not a proven support**), once per declaration, when the payload's
    lowest primitive is within max(1 voxel, extrinsic bound) of the latched plane, its centre is
    over the patch and the gripper is still loaded (`leg.trigger.attached`); `support_id` = the
    declaration's `target_id`, `max_penetration_m = min(extrinsic bound, 10 mm)`. It dies with the
    region — contradicting evidence, the freeze TTL, a goal retraction or expiry, or a payload
    change — so
    no part of the place allowance outlives the region age the kernel accepts
    (`place_region_max_age_s`, 2 × the voxel deadline; the kernel independently drops an older
    region as `region_stale`). Set-down and release therefore have to finish within the freeze.
  - Tests: `tests/unit/test_place_target_leg.py` (table, shelf boards, headroom, edge, clutter,
    occlusion freeze bounded by the kernel's region age, hint, release), live
    `tests/integration/test_place_target_leg_live.py` and
    `tests/integration/test_place_target_release_live.py` against the real `safety_kernel_node`.
- Release: keep the trigger's DETACH (position stall: the jaw opened past its hold) as "jaws
  opened", but keep the object as an attached record
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
  by `tests/integration/test_place_target_release_live.py`.
- Open: fingers vs shelf at the 20 mm link margin. Either accept and measure finger-shelf
  clearance in the attended runs first, or extend the measured region's allowance to the
  finger link for cells inside the region and below the plane + 1 voxel — the same exemption class
  as §2.1, WG.

### 2.4 Shared declaration

Extend `PlaceDeclaration` additively or add a `TaskDeclaration`: `grasp_target` (id + search
box), `place_target` (optional hint), `object_id`, `rskill_id`, `trace_id`, `stamp_ns`,
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
| 2 | `fix(hal)`: 5 Hz attachment heartbeat gated on live jaw-position evidence (was effort, §1.6); time-seeded monotonic revision; real-mode `SimSensorBridge` stops claiming "nothing attached" | HAL | B2, B3 |
| 3 | `fix(hal,perception)`: back-project in the depth header's optical frame; intrinsics from live `camera_info`; explicit mask resampling; `top`/`head_zed` optical `frame_id` | HAL, perception | B4, B5 |
| 4 | `feat(deploy)`: `DeployRuntime.vision_attachment`, segmenter lifecycle node in the launch, **vision leg on real always turns the kernel attached check on** (1000 ms deadline) | CLI, launch | **Safety-WG + hazard log**. *Implemented, committed off* (`enabled: false` in `scenes/deploy/openarm_real_world_voxels.yaml`) pending WG review and the hazard-log entry |
| 5 | — (no calibration session) | — | effort is settled — the driver hard-codes it to 0 (§1.6). The position-stall thresholds (`closure_calibration`: rest offset, stall gap, settle tolerance) are robot-type constants of the gripper mechanism in `robots/openarm/robot.yaml`, derived from teleop data; not per cell, no unit overrides them. Verify once by commanding an empty close on the rig and reading where the jaw rests — no session. The thin-object false negative stays a Safety-WG item (§4) |
| 6 | `feat(kernel)`: `GraspDeclaration` across IDL/core/world-state/runner/HAL/launch/kernel + conservativeness tests | all | **ADR + hazard log; split (>800 lines)**. *Wire landed* (IDL, `openral_core.GraspDeclaration`, World State relay, runner arm/retract on `/openral/grasp_declaration`, CLI/launch `grasp_declaration_json`, committed target in `scenes/deploy/openarm_real_world_voxels.yaml`); the sim producer landed (`SimAttachmentEvidenceTracker.set_grasp_declaration` / `grasp_declaration`: the target subtree's box, base frame, no geometry, on every `AttachmentState` envelope), the launch flag `DeployRuntime.grasp_allowance_enabled` (default off) with the always-passed manifest-derived `grasp_contact_links`, and the kernel consumer landed (`ingest_grasp_declaration`, default off; per-candidate scoping, handover retirement against the region latched at handover, proven on the twin control pair `tests/sim/test_gripper_twin_hal_mujoco_grasp_pair.py`); the real producer pending |
| 7 | `feat(perception)`: pre-grasp target producer (search box → SAM 2.1 → OBB → region), tracking, handover | HAL/perception | develop on the twin pass (real ZED, twin HAL). *Producer leg implemented, default off* (`_grasp_target_leg`); the OpenArm scene's committed declaration carries no `search_box` yet, and the thresholds are uncalibrated. *Approach-armed target implemented, default off* (`grasp_target_approach_m`; §2.2): no named target needed, one hand at a time, kernel refuses two-hand declarations |
| 8 | `feat(hal)`: place on real — surface measured under the carried payload + proximity witness + frozen release | HAL, bridge | **ADR-0097/0092 amendments**. *Frozen release window implemented* in the vision leg (§2.3 "Release"). *Producer leg implemented, default off* (`_place_target_leg`, `vision_attachment_place_target_enabled`): no surveyed fixture and no named target — it measures the support patch directly under the held payload in the live voxel map, latches it (re-verified, frozen under the kernel's region age bound on a lost view), attaches it as the region of dispatch's goal-scope declaration (`target_id="surface"`, the runner's `place_approach_enabled`; it never declares on its own, and an optional reasoner-grounded `PlaceDeclaration.search_box` only narrows it), and attests the `MAP_SUPPORT_PROXIMITY` witness. `UnitFixture` / `RobotUnit.fixtures` / `DECLARED_FIXTURE` were removed (never released, no committed unit carried one). The finger allowance is not part of it, and the thresholds are uncalibrated |
| 9 | `feat(scenes)`: enable on the Thor scene | scenes | only after 5's empty-close check |

Steps 0-3 are plain bug fixes on code that exists and can start now. Step 4 is where safety
posture first changes on real hardware. Steps 6 and 8 are the semantics the Safety-WG has to
decide; 7 and 8 need the attended cell for calibration.

## 4. Safety-WG items (private `OpenRAL/management`)

1. ADR: declaration-scoped grasp-target exemption for gripper finger links (amends ADR-0097's
   "arm-vs-world unchanged" for those links only); caps; measure-once vs re-measured region;
   handover retirement rule; whether `link7` is a contact link; the producer's prompt source;
   the approach-armed target (§2.2): arming without a named target, the approach distance,
   proximity-only arming vs a closing-intent gate, one hand at a time (and refusing two
   approaching hands at once), and whether a goal may re-arm after an attach.
2. Hazard HZ-0115: exemption misapplied (non-target body inside the region; wrong object; stale
   declaration; target moved while frozen; leak to other links/arms; fingers into the support;
   the exemption arming on an unintended object near the hand, HZ-0115-11; a producer declaring
   two hands, HZ-0115-12).
3. Turning `attached_collision_enabled` on for real, with the deadline, and trusting vision
   geometry for map clearing (undersized box clears a real obstacle; phantom fallback box on a
   closed-on-nothing gripper; dead jaw-position channel → kernel drop window).
6. Position-stall trigger hazards (`_grasp_trigger.PositionStallTrigger`): **false positive** — a
   jaw obstructed by something other than the target (table, shelf lip, the other hand, the
   object's edge before it is seated) stalls exactly like a grasp and attaches a jaw box (with
   the grasp-target region payload, also hands the exemption over); **false negative** — an
   object thinner than the stall gap in jaw angle (a card, a cable) never reads as held, so it is
   carried invisible to the attached check and its cells stay in the map; a commanded target
   that is not the jaw's real reference (trajectory interpolation, a controller that clamps)
   shifts the gap. Mitigations to judge: the AND with vision at ATTACH (region containment or a
   gated mask, else the `GRIPPER_CLOSURE` box), the release window, the calibration's per-side
   rest offset.
7. Vision confirmation semantics: a stall vision cannot confirm still attaches the conservative
   `GRIPPER_CLOSURE` box (collision-conservative, but a phantom box can clear map cells); detach
   is trigger-only (vision is never asked to keep a payload the jaws released) — whether a
   refused confirmation should instead withhold the attachment is a WG decision.
4. ADR-0097 amendment (the producer may declare, region and all, the surface it measured under
   the carried payload — no dispatch naming); ADR-0092 D6 amendment (proximity to a map-measured
   plane); hazard "a measured surface is not a support"; the ADR-0098 joint-play offset if
   adopted; the frozen-release window; finger allowance inside the place region.
5. Bimanual attachment (two attach links) — shared by both halves.
8. A **frozen jaw position inside a live `/joint_states` stream** is indistinguishable from a
   stalled jaw, and no guard is added for it (2026-10-03, `fix/r5-trigger`, investigated
   read-only on Thor). Vendor path: `openarm_simple_hardware.cpp` (`openarm_ros2` 4e837e1)
   `read()` calls `refresh_all()` + `recv_all()` and copies `Motor::get_position()` into the
   state interface, returning `OK` unconditionally; in `openarm_can` (b001148)
   `recv_all()` drains whatever frames arrived within its select timeout, and
   `DMCANDevice::callback` → `Motor::update_state` is the only writer of `state_q_`. A motor
   that does not answer simply keeps its last `state_q_` while the other joints update — there
   is no per-motor stamp, counter or error flag, and the gripper's velocity and effort are
   hard-coded `0`. `joint_state_broadcaster` republishes the frozen value with a fresh header
   stamp every cycle, and `RosControlHAL.read_state` stamps the newest arrival, so the
   trigger's repeat filter and `_JawEvidence` both see new samples of a perfectly flat jaw: a
   frozen jaw commanded closed from > `rest + stall_gap` away attaches on nothing, and the
   heartbeat keeps claiming live evidence. A "require ≥ 1 LSB flicker over a window" guard was
   considered and **rejected on evidence**: in the 89 s stationary real-cell bag
   (`/tmp/qua1985_staging_real/openarm-real/qua1985-rig-check-real/2026-09-22/…/bag`, 66 176
   `/joint_states` at 1.35 ms), every channel quantises at one LSB (3.815e-4 rad) and several
   live, healthy channels never flicker at all — `left_joint3`, `left_joint4`, `right_joint4`
   bit-identical for all 89 s, `left_finger_joint1` for 72 s — while others flicker every few
   ms. Flatness is therefore normal for a stationary DM motor and a flicker requirement would
   refuse real stalls (or, tuned loose, guard nothing). Options for the WG: a vendor-side
   per-motor reply counter / last-reply stamp exported as a state interface (the only sound
   signal; not verified whether the DM reply frame carries a usable status field), or
   accepting the hazard with the vision AND at ATTACH as the mitigation.

## 5. Measure before deciding (attended, cell)

- *Settled 2026-10-03 (vendor source + teleop data, §1.6):* effort is hard-coded 0 — not a
  signal. The position-stall thresholds (`closure_calibration`: rest offset, `stall_gap` 0.08,
  `settle_tolerance` 1e-3) are manifest constants of the gripper mechanism, derived from teleop
  data; not per cell. Verify once by reading an empty close on the rig — no calibration session.
- *Measured 2026-10-02, no motion:* depth and RGB images both carry
  `header.frame_id = zed_left_camera_frame_optical` at 1920x1080; `camera_info` K = fx = fy =
  1498.18, cx = 936.11, cy = 541.81 (the manifest's nominal fx = 960 is 56 % short);
  `/tf_static` carries `zed_camera_link -> zed_camera_center -> zed_left_camera_frame ->
  zed_left_camera_frame_optical` from the driver. The vendor `openarm_ros2` description names the
  hand link `openarm_*_ee_base_link`; the manifest's `openarm_*_link7` is the same body
  (`tf_frames` in the Thor scene). Still to read with the real bringup up:
  `tf2_echo openarm_left_ee_base_link zed_left_camera_frame_optical`.
- *Measured 2026-10-02:* SAM 2.1 hiera-small, bf16, `transformers` 5.5.4 on Thor, one live
  1920x1080 ZED left frame, point prompt: warm median 53 ms, p95 57 ms, cold 635 ms, peak
  279 MiB allocated — the same as the RTX 4070 Laptop figures, with π0.5 not loaded. The
  attach barrier (~100 ms) holds it; measure again beside the running policy.
- The restock box dimensions (for the caps) and finger-shelf clearance during a real place.

Sources: four read-only investigations of this branch (kernel allowance, pre-grasp perception,
HAL vision leg, place path), 2026-10-02.
