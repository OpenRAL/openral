# `openral_msgs`

ROS 2 IDL — `.msg`, `.srv` and `.action` definitions for **OpenRAL**. This
is the **normative** schema for everything that crosses the ROS 2
boundary (CLAUDE.md §1 / Operating Principle 3, §6.1). Pydantic
counterparts live in `openral_core.schemas`; the two are kept in
sync by hand and by the schema-export drift check
(`tools/schema_export.py`).

## Synopsis

```bash
source /opt/ros/jazzy/setup.bash
just ros2-build      # builds openral_msgs (alongside hal_so100, world_state)
ros2 interface show openral_msgs/msg/ActionChunk
ros2 interface show openral_msgs/action/ExecuteRskill
```

## What's in here

### Messages — `msg/`

| File | Role |
| --- | --- |
| `ActionChunk.msg` | A bounded action chunk (length × DoF) emitted by an S1 Skill. Mirrors `openral_core.Action`; optional `cartesian_delta_scale` converts controller-native deltas to physical metres/radians for predictive safety without changing raw actuator bytes. |
| `PromptStamped.msg` | A natural-language prompt with stamped header and Pydantic metadata as JSON. Topic surface: `/openral/prompt` (operator prompts, Pydantic `SkillPrompt` in `metadata_json`); `/openral/perception/{motion,objects,ocr,scene_change}` (perception events published by `openral_runner.backends.gstreamer.perception_tee.PerceptionEventPublisher`, Pydantic `openral_core.PerceptionEventMetadata` discriminated union in `metadata_json`). |
| `FailureTrigger.msg` | Typed failure event on the namespaced `/openral/failure/{hal,sensor,skill,safety,wam,critic}` bus. `kind` and `severity` are `uint8` constants (`KIND_TIMEOUT`, ..., `KIND_SUPPRESSED_SUMMARY=254`; `SEVERITY_INFO|WARN|FAIL|ABORT`); `evidence_json` is a serialized Pydantic `openral_core.FailureEvidence` discriminated union. Published via `openral_observability.FailureBusPublisher` with per-(kind, severity) token-bucket rate limiting + 1 Hz suppressed-summary roll-up. |
| `WorldStateStamped.msg` | Typed WorldState wire format on `/openral/world_state_fast` (30 Hz) and `/openral/world_state_slow` (5 Hz). Carries joint state, base pose/twist, parallel arrays for EE poses / image refs / diagnostics / staleness, battery, and tf2 `frame_ids[]` (consumers read `/tf` themselves). `DIAG_OK | DIAG_WARN | DIAG_STALE | DIAG_ERROR` uint8 constants. Also relays the attachment set and, with it, the live `place_declaration` (ADR-0097) the safety kernel reads the declared target's region off. |
| `SafetyStatus.msg` | Current safety state on the **latched** topic `/openral/safety_status` (ADR-0096) — `latched`, `drop_reason` (uint8), `detail`, `rskill_id`, `trace_id`. QoS is `RELIABLE + TRANSIENT_LOCAL + KEEP_LAST=1` (the description/static profile, **not** the safety/e-stop one) so a late-joining subscriber sees current state without having witnessed the transition. Published by the C++ safety kernel and `SafetyPassthroughNode` on every latch / fail-closed drop / clear transition, on every lifecycle activation, and as a 1 Hz liveness refresh. Adds a topic; `/openral/estop` and `/openral/failure/safety` are unchanged. |
| `PlaceDeclaration.msg` | Dispatch's typed statement that a place phase is active for a payload, and which target it is being placed into (ADR-0097). Published on `/openral/place_declaration` by the `ExecuteRskill` action server and consumed by the sim attachment-evidence producer in the HAL. It exempts **nothing** on its own — it only makes support contact *on the declared target* attestable as a place-phase `SupportContactWitness`, under the identical bounds, fail-closed rules and hysteresis the pick witness is under. Scoped to one goal: retracted (`active=false`) on goal end, cancel and E-stop, and expired by every consumer on its own `timeout_s` so a dead dispatcher cannot leave an exemption armed. Its optional `region_valid` / `region` pair (ADR-0097's 2026-08-14 amendment) is filled in by the *producer*, never by dispatch, and carries the bounded approach allowance described under `PlaceRegion.msg`. |
| `PlaceRegion.msg` | The producer-measured bounded region of a declared place target (ADR-0097's 2026-08-14 amendment): an **oriented** box (`frame_id` + `pose` + `half_extents`) inside which the declared payload's world-collision margin is reduced by `min(1.5 × voxel, 40 mm)` (ADR-0097's Second Amendment, 2026-08-15; 37.5 mm at sim's 25 mm cells, 40 mm — not 75 — at a real 50 mm grid) so it can physically reach the support contact the place witness is *earned by* — round-6 measured the predictive check stopping the payload 22-30 mm short of the declared shelf, because 25 mm cells inflate a cabinet's thin opening. `frame_id` MUST be the robot base frame (the frame `OccupancyVoxels` uses); the safety kernel refuses a region declared in any other frame, any half-extent above 1.5 m, and any box above 8 m³. Sim measures it from the declared body's MuJoCo model subtree; the real-hardware producer (perception stack) is a defined seam that is **not yet implemented**, so no allowance is applied on real hardware. Absent → no allowance, i.e. exactly the pre-amendment margins. |

### Services — `srv/`

Request/response for **instantaneous queries** (CLAUDE.md §2 / ROS 2: actions for
>100 ms or cancellable work, services for instantaneous queries, topics for
streams). Every service below is read-only with respect to actuation.

| File | Role |
| --- | --- |
| `ResetToPose.srv` | Snap a HAL-managed simulator to a manifest `starting_pose` before the skill runner's first inference tick, so a policy sees its training-distribution home pose. Called by `openral_rskill_ros.rskill_runner_node`. |
| `LocateInView.srv` | Reasoner's read-only `locate_in_view` tool: ask an on-demand open-vocabulary locator (LocateAnything / OmDet-Turbo) whether a free-text object is in the current frame. Returns `ObjectsMetadata` as `metadata_json`. |
| `QueryScene.srv` | Reasoner's read-only `query_scene` tool: ask a scene VLM (`kind: "vlm"`) an open-ended question about the current camera view; returns free text. |
| `QueryTaskProgress.srv` | Reasoner's read-only `query_task_progress` tool: ask the reward monitor (`kind: "reward"`) for windowed progress/success over the co-active VLA's recent frames. Advisory only — never actuation. |

### Actions — `action/`

| File | Role |
| --- | --- |
| `ExecuteRskill.action` | Goal/result/feedback for invoking an installed rSkill. Goal carries `rskill_id`, `revision`, `prompt`, deadline, and the optional `place_declaration_valid` / `place_declaration` pair (ADR-0097, additive and defaulted off — every pre-ADR-0097 caller keeps today's behaviour); result carries `success` + free-text `failure_reason` + `trace_id` + the typed `failure_kind` uint8 (`FAILURE_NONE`=0 … `FAILURE_CANCELLED`=8, `FAILURE_UNKNOWN`=255); feedback streams `progress` / `chunk_index` / `chunks_total` / executor `state`. |

## Schema lineage

| ROS 2 IDL | Pydantic counterpart | Notes |
| --- | --- | --- |
| `msg/ActionChunk` | `openral_core.Action` | `control_mode` enum mirrors `ControlMode`; empty `cartesian_delta_scale` means identity for backward compatibility. |
| `msg/PromptStamped` | `openral_core.SkillPrompt` (operator) / `openral_core.PerceptionEventMetadata` (perception leg) | Metadata JSON-encoded for transport. On `/openral/perception/<kind>` the discriminator is `kind` (one of `motion` / `objects` / `ocr` / `scene_change`); each kind owns its own ROS topic so new kinds = new topics, not an IDL bump. |
| `msg/FailureTrigger` | `openral_core.FailureEvidence` (discriminated union) | `kind` ∈ {`KIND_TIMEOUT`=0, `KIND_FORCE`=1, `KIND_WORKSPACE`=2, `KIND_PERCEPTION`=3, `KIND_CRITIC`=4, `KIND_CONTROLLER`=5, `KIND_SELFVERIFY`=6, `KIND_HUMAN`=7, `KIND_WAM`=8, `KIND_REASONER_TIMEOUT`=9, `KIND_SUPPRESSED_SUMMARY`=254}; `severity` ∈ {`SEVERITY_INFO`=0, `WARN`=1, `FAIL`=2, `ABORT`=3}. `evidence_json` decodes via `pydantic.TypeAdapter(FailureEvidence).validate_json(...)`. This hard-breaks the wire format (no migrator, no string-shaped fallback). |
| `msg/SafetyStatus` | none (ROS-only) | Deliberately has no Pydantic counterpart: it is live ROS graph state, not a persisted artifact, so `tools/schema_export.py` has nothing to drift against. `drop_reason`'s `KIND_*` block is numerically identical to `msg/FailureTrigger`'s (redeclared, since ROS IDL cannot import constants); the `DROP_*` block (100–108, plus `DROP_NONE`=255) is a disjoint range covering the non-latching fail-closed drops and the external-e-stop latch. The pairing is pinned by `tests/unit/test_safety_status_msg.py` and the kernel gtest. |
| `msg/WorldStateStamped` | `openral_core.WorldState` | Translation lives in `openral_world_state_ros.lifecycle_node.build_world_state_stamped_msg` — dicts are flattened to deterministic, sorted parallel arrays. This hard-breaks the wire format (no migrator, no JSON fallback). |
| `msg/PlaceDeclaration` | `openral_core.PlaceDeclaration` | `is_live(now_ns=...)` is the consumer-side liveness rule (active **and** inside `timeout_s`); `timeout_s` is capped at `PlaceDeclaration.MAX_TIMEOUT_S` (600 s — strictly above the largest in-tree `latency_budget.max_execution_s`, so no legitimate goal is cut short and nothing mission-scale is accepted). `stamp_ns` is in the **publishing graph's ROS clock** (simulator time under `use_sim_time`), and every consumer must pass a `now_ns` from that same domain — its own ROS clock (rSkill runner, sim evidence producer, safety kernel) or, where it has none, the newest stamp on the stream that carried the declaration (`WorldStateAggregator`). Passing wall `time.time_ns` puts a sim-stamped declaration ~57 years past its backstop and silently drops it. |
| `msg/PlaceRegion` | `openral_core.PlaceRegion` | Bounds are enforced in both places and identically: `PlaceRegion.MAX_HALF_EXTENT_M` (1.5) / `MAX_VOLUME_M3` (8.0) in the schema, `kMaxPlaceRegionHalfExtentM` / `kMaxPlaceRegionVolumeM3` in the kernel's `ingest_place_region`. Refusal means *no allowance*, never a dropped message — a bad region can only make the kernel more permissive, so ignoring it restores the pre-amendment margin. |
| `action/ExecuteRskill` | `openral_core.RSkillManifest` (referenced via `rskill_id`), `openral_core.PlaceDeclaration` (goal's `place_declaration`) / `openral_core.exceptions` (mirrored by `failure_kind`) | The action server lives in `openral_rskill_ros`. `Result.failure_kind` ∈ {`FAILURE_NONE`=0, `FAILURE_CONFIG_ERROR`=1, `FAILURE_CAPABILITY_MISMATCH`=2, `FAILURE_RUNTIME_ERROR`=3, `FAILURE_SAFETY_ESTOP`=4, `FAILURE_PERCEPTION_STALE`=5, `FAILURE_PLANNING_ERROR`=6, `FAILURE_DEADLINE_MISSED`=7, `FAILURE_CANCELLED`=8, `FAILURE_UNKNOWN`=255} mirrors the CLAUDE.md §5 exception hierarchy one-for-one. **Additive** (CLAUDE.md §1.6 backward-compatible evolution): no `schema_version` bump, no migrator — a consumer reading only `success` / `failure_reason` is untouched, and an older producer's Result decodes as `FAILURE_NONE`, which consumers must treat as "unknown", not "succeeded". |

JSON Schema for the Pydantic models is generated by
`tools/schema_export.py` and committed under
`docs/reference/schemas/*.json`; CI fails on drift via the `--check`
flag in `lint.yml`.

## Build

```bash
source /opt/ros/jazzy/setup.bash
just ros2-build                                                     # includes msgs
# or for msgs alone:
colcon build --merge-install --packages-select openral_msgs
source install/setup.bash
ros2 interface list --only-msgs | grep openral_msgs
```

## See also

- `openral_core.schemas` (Python source of truth for the Pydantic
  side) — `python/core/src/openral_core/schemas.py`.
- `tools/schema_export.py` — drift check.
- `docs/reference/schemas/` — JSON Schema artifacts.
- CLAUDE.md §5.3 (QoS profiles) for which message classes use which QoS
  defaults.
