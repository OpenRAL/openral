# `openral_msgs`

ROS 2 IDL (`.msg`, `.srv`, `.action`) for OpenRAL: the normative schema for
everything that crosses the ROS 2 boundary (CLAUDE.md §1.3). Pydantic
counterparts live in `openral_core.schemas`; `tools/schema_export.py` checks the
two for drift (CI runs it with `--check`; JSON Schema lands in
`docs/reference/schemas/*.json`).

```bash
source /opt/ros/jazzy/setup.bash
just ros2-build                                        # builds openral_msgs with the ROS graph
colcon build --merge-install --packages-select openral_msgs   # msgs alone
source install/setup.bash
ros2 interface list --only-msgs | grep openral_msgs
ros2 interface show openral_msgs/msg/ActionChunk
```

QoS per data class is in CLAUDE.md §2; the table below names a topic's QoS only
where it departs from that.

## Messages (`msg/`)

| File | Role |
| --- | --- |
| `ActionChunk` | Bounded action chunk (`horizon` x `n_dof`, row-major `flat`) from an S1 skill; mirrors `openral_core.Action`. `cartesian_delta_scale` (empty = identity) converts normalised deltas to physical m/rad for predictive safety without touching the raw bytes. `joint_names` names the joints a slot-dispatched `JOINT_*` chunk owns (ADR-0102): such a chunk is zero-padded to full dof, so without names `0.0` is ambiguous; empty = whole vector in `RobotDescription.joints` order. `tick_index` / `tick_group_size` group the slot chunks of one tick. `runner_session_id` is the runner process's random id: `tick_index` restarts per process, so the HAL keys its replay watermark on `(runner_session_id, tick_index)` and refuses a superseded session; `0` = unknown (legacy, tick-only heuristic). |
| `PromptStamped` | Natural-language prompt plus Pydantic metadata as JSON. `/openral/prompt` carries operator prompts (`SkillPrompt`); `/openral/perception/{motion,objects,ocr,scene_change}` carry perception events from `PerceptionEventPublisher` (`openral_core.PerceptionEventMetadata`, discriminated by `kind`). |
| `FailureTrigger` | Typed failure event on `/openral/failure/{hal,sensor,rskill,safety,wam,critic,world_state}`. `kind` / `severity` are `uint8` constants; `evidence_json` is a serialized `openral_core.FailureEvidence`. Published by `openral_observability.FailureBusPublisher` with per-(kind, severity) token-bucket limiting and a 1 Hz `KIND_SUPPRESSED_SUMMARY` roll-up. `world_state` carries `KIND_OBJECT_QUERY`; `hal` carries `KIND_GRASP_OUTCOME`. |
| `WorldStateStamped` | WorldState wire format on `/openral/world_state_fast` (30 Hz) and `/openral/world_state_slow` (5 Hz): joint state, base pose/twist, parallel arrays for EE poses / image refs / diagnostics / staleness, battery, tf2 `frame_ids[]` (consumers read `/tf` themselves), `DIAG_OK|WARN|STALE|ERROR`. Also relays the attachment set, the live `place_declaration` / `grasp_declaration` (each behind a `*_valid` flag), `tracked_objects` (`tracked_objects_valid` false = no tracker) and the lifted `detected_object_*` arrays including optional axis-aligned 3D boxes (`detected_object_bbox_valid[]` / `_min[]` / `_max[]`; empty = no boxes). |
| `SafetyStatus` | Safety state on the **latched** `/openral/safety_status` (ADR-0096): `latched`, `drop_reason`, `detail`, `rskill_id`, `trace_id`. QoS `RELIABLE + TRANSIENT_LOCAL + KEEP_LAST=1` (the description/static profile, not the e-stop one) so a late joiner sees current state. Published by the C++ safety kernel and `SafetyPassthroughNode` on every latch / fail-closed drop / clear, on lifecycle activation and as a 1 Hz refresh; a consumer must treat a stale `header.stamp` as unknown, not safe. `/openral/estop` and `/openral/failure/safety` are unchanged. |
| `PlaceDeclaration` | Dispatch's statement that a place phase is active for a payload and into which target (ADR-0097). Published on `/openral/place_declaration` by the `ExecuteRskill` server; `invocation_id` is the 32-hex goal UUID, kept through runner restamping. It exempts nothing on its own: it only lets support contact *on the declared target* be attested as a place-phase `SupportContactWitness`, under the same bounds and fail-closed rules as the pick witness. Goal-scoped: retracted (`active=false`) on goal end, cancel and E-stop, and expired by each consumer on `timeout_s`. `region_valid` / `region` is filled only by the measuring producer, never dispatch (`DeployScene` refuses a scene-supplied region and the runner strips one). `search_box_valid` / `search_box` is a grounded hint naming the surface, never an exemption. |
| `GraspDeclaration` | Grasp-phase mirror of `PlaceDeclaration`: `target_id`, `object_id`, `contact_links[]`, `rskill_id`, `trace_id`, `invocation_id`, `timeout_s`, `stamp_ns`, `active`, `region_valid` / `region` (`PlaceRegion`, `geometry[]` empty), `search_box_valid` / `search_box`. Published on `/openral/grasp_declaration` (`RELIABLE + TRANSIENT_LOCAL`, depth 1) by the `ExecuteRskill` server and carried on `AttachmentState` / `WorldStateStamped`. Exempts nothing on its own; same goal scoping and expiry as `PlaceDeclaration`. |
| `PlaceRegion` | Producer-measured oriented box (`frame_id` + `pose` + `half_extents`; `geometry[]` carries optional shape detail) of a declared place or grasp target. `frame_id` must be the robot base frame (the `OccupancyVoxels` frame); the kernel refuses another frame, a half-extent above 1.5 m or a volume above 8 m^3. Inside it the declared payload's world-collision margin is reduced by `min(1.5 x voxel, 40 mm)` so the payload can reach the support contact (ADR-0097, second amendment). Absent = no allowance. |
| `AttachmentState` | Atomic attached-object snapshot (`header`, `revision`, `AttachedCollisionObject[] objects`) on `/openral/attachment_state` (`TRANSIENT_LOCAL`), also carrying the resolved place / grasp declarations. World State swaps its set only after the whole message validates. |
| `AttachedCollisionObject` | One payload rigidly attached to a robot link: `object_id`, `attach_link`, `touch_links[]`, `pose_in_link`, `AttachedCollisionPrimitive[]`, optional `SupportContactWitness` / `ContactForceWitness`. The octomap bridge clears the cells it explains. |
| `AttachedCollisionPrimitive` | One convex primitive (`SHAPE_SPHERE` / `SHAPE_CAPSULE` / `SHAPE_BOX`) of an attached object, posed in the object frame. |
| `SupportContactWitness` | World State's bounded attestation that an attached payload rests on a named surface (ADR-0092 D6). No witness, no exemption. |
| `ContactForceWitness` | Measured contact force of an attached payload on its declared place target (ADR-0100). The gate it feeds only adds refusals. |
| `OccupancyVoxels` | Dense, fixed-capacity, oriented 3-D occupancy grid on `/openral/world_voxels` in the robot base frame for the kernel's world-collision check; a grid above the kernel's capacity fails closed. |
| `TrackedObject` | One object the Layer-2 tracker grounds, tracks, binds and releases; wire form of `core.TrackedObject`. `object_id` (`trk:<n>`), `epoch`, `label`, `role`, `status` (`tentative|confirmed|occluded|split|bound|released|lost`), a `PlaceRegion box` (grasp caps: half-extent <= 0.20 m, volume <= 0.03 m^3), `tight` hulls (at most 4, 64 vertices each), `fit_kind`, `support_z` / `support_ref` / `floor_z`, cover counts, stamps, bound-hand fields, `identity_inferred`, `provisional`. Python proposes the geometry; the kernel re-verifies cover and decides every exemption. |
| `TrackedObjectSet` | The tracker's whole object set (`schema_version "1"`, `frame_id` = voxel grid frame, `stamp_ns`, strictly increasing `revision`, at most 32 objects) on `/openral/tracked_objects` (`RELIABLE, VOLATILE, KEEP_LAST 1`, >= 2 Hz heartbeat) and inside `WorldStateStamped`. The kernel refuses the whole set on a frame mismatch, cap breach, non-monotone revision or future stamp. |
| `ObjectQuery` | A class of objects a prompt names (`label`, `count` `one|all`, `role`, `hand`, `min_instances`, `synonyms`, optional `seed`); wire form of `core.ObjectQuery`. The tracker holds it as a standing query for the goal's life. Never kernel geometry. |
| `GraspEvent` | What the HAL gripper trigger computed, on `/openral/grasp_events` (`RELIABLE, VOLATILE, KEEP_LAST 10`): `KIND_ATTACH` / `KIND_REGRASP` / `KIND_DETACH`, gripper joint, hand link, jaw measured / commanded / hold position, stamp. The tracker binds and releases objects from it. IDL-only (no Pydantic counterpart). |
| `InstanceCells` | Experimental, sim-only: per-track voxel cells labelled by the instance-mask propagator on `/openral/instance_cells`, in the lattice of the `/openral/world_voxels` grid (`origin`, `resolution`, `object_ids[]`, `cell_counts[]`, flat `cells` index triples). The tracker fits boxes to them and refuses another lattice. |
| `SegmentMasks` | **Diagnostic only**: mirror of the latest `SegmentInView` reply on `/openral/perception/masks`, published by `openral_perception_ros.segmenter_node` only when `publish_debug_masks` is set (default false), sensor-class QoS. The dashboard draws it on camera tiles. Nothing in the safety or attachment path reads it. |
| `CriticScore` | Per-frame critic / reward score on `/openral/critic/score`, keyed by `critic_id`; a stalled critic emits a Tier-C `FailureTrigger`. |
| `ResultOnlyGoal` | Runner to deadman watchdog on `/openral/result_only_goal` (`TRANSIENT_LOCAL`): an `ExecuteRskill` goal that publishes no chunks (a wrapped result-only skill such as Nav2): `goal_uuid`, `wall_start`, `rskill_id`, `wall_bound_s`. The watchdog bounds it by `wall_start` + bound + `result_only_grace_s` (capped by `result_only_max_s`) instead of the first-chunk window (ADR-0127). |
| `SuccessCondition` | What success means for a policy run; wire form of `core.SuccessCondition`: `object_label`, a `PlaceRegion` in the tracker's grid frame (`geometry[]` empty), `min_count`, `settle_s`, `max_evidence_gap_s` (a zero — unset — reads the schema default), `end_on_success`. Judged by the runner over the tracked objects; advisory, never a kernel input. |
| `OperatorAction` | An operator acting through a channel, announced so the compliance recorder can attribute it: `action` (`estop_reset`, `param_set`, `estop`, `prompt`, `skill_execute`), `channel_label`, pseudonymous `actor_id` (HMAC-SHA256 under `$OPENRAL_ACTOR_KEY`, or empty), `detail`. On `/openral/operator_action` (`RELIABLE, VOLATILE`, depth 10). Published just before the action, except `estop`, which is announced after it fires. The recorder observes the effect itself; this message only says who. |
| `Episode` | Episode-boundary marker on `/openral/episode` (`PHASE_START` / `PHASE_END`, `task_string`, `success`); the dataset converter segments bags on it. |
| `Tick` | Per-tick metadata from `DeployRunner` on `/openral/tick` (episode/step index, reward, termination flags, `trace_id` / `span_id`); mirrors `openral_core.TickResult`. |

## Services (`srv/`)

Instantaneous queries; none actuates.

| File | Role |
| --- | --- |
| `ResetToPose` | Explicit maintenance/test command that snaps a HAL-managed simulator or invokes a HAL-owned pose reset. The skill runner does not use it for startup motion. |
| `LocateInView` | Reasoner's `locate_in_view`: ask an on-demand open-vocabulary locator (LocateAnything / OmDet-Turbo) whether `query` is in the current frame of `camera`. `source_stamp_ns=0` = latest frame; nonzero requires that exact image and the reply echoes the stamp used. Returns `ObjectsMetadata` as `metadata_json`. Empty `failure_reason` means the check completed (including a valid no-match); nonempty means visibility is unknown, so `found=false` is then not evidence of absence. Optional `roi_norm_xyxy` crops before detection (boxes return in full-frame pixels); optional `min_score` lowers the floor for one call. |
| `QueryScene` | Reasoner's `query_scene`: ask a scene VLM an open-ended question about the current view; returns free text. |
| `QueryTaskProgress` | Reasoner's `query_task_progress`: ask the reward monitor for windowed progress/success over the co-active VLA's recent frames. Advisory. |
| `SegmentInView` | HAL attachment-evidence producer asks a promptable segmenter (SAM 2.1) which wrist-frame pixels are the grasped payload. Geometric prompt (a 3-D TCP point) in; raw `mono8` `sensor_msgs/Image[]` candidate masks plus parallel `mask_scores_advisory` out. Plural because only the HAL, which holds the depth frame, can choose between nested hypotheses; scores are trace-only. |

## Action (`action/`)

`ExecuteRskill`, served by `openral_rskill_ros`.

- **Goal:** `rskill_id`, `revision` (empty = latest pinned), `prompt`, `prompt_metadata_json`, `goal_params_json`, `deadline_s` (0 = manifest default; capped at the manifest's `max_execution_s`), plus optional, default-off pairs `place_declaration_valid` / `place_declaration`, `grasp_declaration_valid` / `grasp_declaration` and `success_condition_valid` / `success_condition` (wins over the deploy scene's condition for the prompt).
- **Result:** `success`, `failure_reason`, `trace_id`, typed `failure_kind`, `evidence_json` (one `FailureEvidence` variant, empty if none), `outcome_summary` (bounded completion line on success), `task_outcome` (`TASK_UNKNOWN`=0 / `TASK_SUCCEEDED`=1 / `TASK_FAILED`=2, whether the run did what it was asked, apart from `success`) and `task_outcome_json` (`openral_core.TaskOutcome`). An older producer decodes as `FAILURE_NONE` / `TASK_UNKNOWN`; consumers must read both as "unknown", never "succeeded".
- **Feedback:** `progress` (0..1), `state` (`loading|warming|executing|replanning|recovering`), `chunk_index`, `chunks_total`.

## Schema lineage

| ROS 2 IDL | Pydantic counterpart | Notes |
| --- | --- | --- |
| `msg/ActionChunk` | `openral_core.Action` | `control_mode` mirrors `ControlMode`; `runner_session_id` 0 decodes as legacy. |
| `msg/PromptStamped` | `openral_core.SkillPrompt` / `PerceptionEventMetadata` | Metadata is JSON. Each perception kind owns its topic, so a new kind is a new topic, not an IDL bump. |
| `msg/FailureTrigger` | `openral_core.FailureEvidence` | `kind`: TIMEOUT=0, FORCE=1, WORKSPACE=2, PERCEPTION=3, CRITIC=4, CONTROLLER=5, SELFVERIFY=6, HUMAN=7, WAM=8, REASONER_TIMEOUT=9, COLLISION=10, OBJECT_QUERY=11, GRASP_OUTCOME=12, SUPPRESSED_SUMMARY=254. `severity`: INFO=0, WARN=1, FAIL=2, ABORT=3. Decode with `pydantic.TypeAdapter(FailureEvidence).validate_json`. Mirrored as plain ints in `openral_observability.failure_bus`; `tests/unit/test_failure_bus_idl_mirror.py` pins both directions. |
| `msg/SafetyStatus` | none (ROS-only live state) | `drop_reason`'s `KIND_*` block equals `FailureTrigger`'s (redeclared; IDL cannot import constants). `DROP_*` is a disjoint range: ENVELOPE_UNCONFIGURED=100, VOXEL_UNAVAILABLE=102, STATE_UNAVAILABLE=103, VOXEL_OVERFLOW=105, EXTERNAL_ESTOP=106, ATTACHED_UNAVAILABLE=107, ATTACHED_OVERFLOW=108, PLACE_ADVISORY=109, `DROP_NONE`=255. 101 and 104 are reserved (retired world phase, ADR-0109) and still appear in recorded bags; never reuse them. Pinned by `tests/unit/test_safety_status_msg.py` and the kernel gtest. |
| `msg/WorldStateStamped` | `openral_core.WorldState` | Translation: `openral_world_state_ros.lifecycle_node.build_world_state_stamped_msg`; dicts become sorted parallel arrays. |
| `msg/PlaceDeclaration`, `msg/GraspDeclaration` | `openral_core.PlaceDeclaration`, `GraspDeclaration` | `is_live(now_ns=...)` is the consumer liveness rule (active and inside `timeout_s`). `stamp_ns` is in the publishing graph's ROS clock (sim time under `use_sim_time`); consumers must pass `now_ns` from the same domain, or a sim-stamped declaration looks decades old and is silently dropped. Caps: `PlaceDeclaration.MAX_TIMEOUT_S` 600 s; `GraspDeclaration` `MAX_TIMEOUT_S` 120 s, `MAX_HALF_EXTENT_M` 0.20 m, `MAX_VOLUME_M3` 0.03 m^3; `contact_links` non-empty while active. |
| `msg/PlaceRegion` | `openral_core.PlaceRegion` | Same bounds in schema (`MAX_HALF_EXTENT_M` 1.5, `MAX_VOLUME_M3` 8.0) and kernel (`ingest_place_region`). A refused region means no allowance, never a dropped message. |
| `msg/TrackedObject`, `TrackedObjectSet`, `ObjectQuery`, `SuccessCondition` | `openral_core.TrackedObject`, `TrackedObjectSet`, `ObjectQuery`, `SuccessCondition` | Wire forms of the core models. |
| `action/ExecuteRskill` | `openral_core.RSkillManifest` (`rskill_id`), `FailureEvidence`, `TaskOutcome`, `exceptions` | `failure_kind`: NONE=0, CONFIG_ERROR=1, CAPABILITY_MISMATCH=2, RUNTIME_ERROR=3, SAFETY_ESTOP=4, PERCEPTION_STALE=5, PLANNING_ERROR=6, DEADLINE_MISSED=7, CANCELLED=8, OBJECT_NOT_FOUND=9, UNKNOWN=255; mirrors the CLAUDE.md §5 hierarchy. |
| `msg/SegmentMasks`, `msg/GraspEvent`, `msg/InstanceCells` | none | Live ROS data, not persisted artifacts, so there is nothing to drift against. |
| `srv/SegmentInView` | `openral_core.SegmenterContract` (bounds the request) / `AttachedCollisionObject` (built from the reply) | Masks are raw `mono8`, not run-length encoded: a one-shot call is about 307 kB per candidate, at most three, and a bespoke codec on the safety-adjacent path is not worth the bytes. |

Schema changes follow CLAUDE.md §1.6: additive fields default off and need no
`schema_version` bump; a breaking change bumps it and ships a migrator.

## See also

- `python/core/src/openral_core/schemas.py` — Pydantic source of truth.
- `tools/schema_export.py`, `docs/reference/schemas/` — drift check and JSON Schema.
