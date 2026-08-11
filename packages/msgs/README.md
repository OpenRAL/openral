# `openral_msgs`

ROS 2 IDL — `.msg` and `.action` definitions for **OpenRAL**. This
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
| `WorldStateStamped.msg` | Typed WorldState wire format on `/openral/world_state_fast` (30 Hz) and `/openral/world_state_slow` (5 Hz). Carries joint state, base pose/twist, parallel arrays for EE poses / image refs / diagnostics / staleness, battery, and tf2 `frame_ids[]` (consumers read `/tf` themselves). `DIAG_OK | DIAG_WARN | DIAG_STALE | DIAG_ERROR` uint8 constants. |
| `SafetyStatus.msg` | Current safety state on the **latched** topic `/openral/safety_status` (ADR-0096) — `latched`, `drop_reason` (uint8), `detail`, `rskill_id`, `trace_id`. QoS is `RELIABLE + TRANSIENT_LOCAL + KEEP_LAST=1` (the description/static profile, **not** the safety/e-stop one) so a late-joining subscriber sees current state without having witnessed the transition. Published by the C++ safety kernel and `SafetyPassthroughNode` on every latch / fail-closed drop / clear transition, on every lifecycle activation, and as a 1 Hz liveness refresh. Adds a topic; `/openral/estop` and `/openral/failure/safety` are unchanged. |

### Actions — `action/`

| File | Role |
| --- | --- |
| `ExecuteRskill.action` | Goal/result/feedback for invoking an installed rSkill. Goal carries `rskill_id`, `revision`, `prompt`, deadline; result carries `success` + free-text `failure_reason` + `trace_id` + the typed `failure_kind` uint8 (`FAILURE_NONE`=0 … `FAILURE_CANCELLED`=8, `FAILURE_UNKNOWN`=255); feedback streams `progress` / `chunk_index` / `chunks_total` / executor `state`. |

## Schema lineage

| ROS 2 IDL | Pydantic counterpart | Notes |
| --- | --- | --- |
| `msg/ActionChunk` | `openral_core.Action` | `control_mode` enum mirrors `ControlMode`; empty `cartesian_delta_scale` means identity for backward compatibility. |
| `msg/PromptStamped` | `openral_core.SkillPrompt` (operator) / `openral_core.PerceptionEventMetadata` (perception leg) | Metadata JSON-encoded for transport. On `/openral/perception/<kind>` the discriminator is `kind` (one of `motion` / `objects` / `ocr` / `scene_change`); each kind owns its own ROS topic so new kinds = new topics, not an IDL bump. |
| `msg/FailureTrigger` | `openral_core.FailureEvidence` (discriminated union) | `kind` ∈ {`KIND_TIMEOUT`=0, `KIND_FORCE`=1, `KIND_WORKSPACE`=2, `KIND_PERCEPTION`=3, `KIND_CRITIC`=4, `KIND_CONTROLLER`=5, `KIND_SELFVERIFY`=6, `KIND_HUMAN`=7, `KIND_WAM`=8, `KIND_REASONER_TIMEOUT`=9, `KIND_SUPPRESSED_SUMMARY`=254}; `severity` ∈ {`SEVERITY_INFO`=0, `WARN`=1, `FAIL`=2, `ABORT`=3}. `evidence_json` decodes via `pydantic.TypeAdapter(FailureEvidence).validate_json(...)`. This hard-breaks the wire format (no migrator, no string-shaped fallback). |
| `msg/SafetyStatus` | none (ROS-only) | Deliberately has no Pydantic counterpart: it is live ROS graph state, not a persisted artifact, so `tools/schema_export.py` has nothing to drift against. `drop_reason`'s `KIND_*` block is numerically identical to `msg/FailureTrigger`'s (redeclared, since ROS IDL cannot import constants); the `DROP_*` block (100–108, plus `DROP_NONE`=255) is a disjoint range covering the non-latching fail-closed drops and the external-e-stop latch. The pairing is pinned by `tests/unit/test_safety_status_msg.py` and the kernel gtest. |
| `msg/WorldStateStamped` | `openral_core.WorldState` | Translation lives in `openral_world_state_ros.lifecycle_node.build_world_state_stamped_msg` — dicts are flattened to deterministic, sorted parallel arrays. This hard-breaks the wire format (no migrator, no JSON fallback). |
| `action/ExecuteRskill` | `openral_core.RSkillManifest` (referenced via `rskill_id`) / `openral_core.exceptions` (mirrored by `failure_kind`) | The action server lives in `openral_rskill_ros`. `Result.failure_kind` ∈ {`FAILURE_NONE`=0, `FAILURE_CONFIG_ERROR`=1, `FAILURE_CAPABILITY_MISMATCH`=2, `FAILURE_RUNTIME_ERROR`=3, `FAILURE_SAFETY_ESTOP`=4, `FAILURE_PERCEPTION_STALE`=5, `FAILURE_PLANNING_ERROR`=6, `FAILURE_DEADLINE_MISSED`=7, `FAILURE_CANCELLED`=8, `FAILURE_UNKNOWN`=255} mirrors the CLAUDE.md §5 exception hierarchy one-for-one. **Additive** (CLAUDE.md §1.6 backward-compatible evolution): no `schema_version` bump, no migrator — a consumer reading only `success` / `failure_reason` is untouched, and an older producer's Result decodes as `FAILURE_NONE`, which consumers must treat as "unknown", not "succeeded". |

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
