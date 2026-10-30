# openral_compliance

The compliance recorder (issue #467). `recorder_node` listens to what the graph already
publishes and emits typed `openral_core.ComplianceRecord`s as OTLP log records through
`openral_observability.compliance.ComplianceEmitter`:

| Source | Record |
|---|---|
| `FailureTrigger` on `/openral/failure/<source>` | `FailureRecord` (evidence decoded; undecodable evidence keeps the record without it) |
| `/openral/safety_status` turning `latched` | `SafetyStopRecord`, linked by `parent_record_id` to the safety failure that explains it (within 2 s) |
| node activate / deactivate | `SessionRecord` start / end |

```
/openral/failure/* ─┐
                    ├─▶ recorder_node ──OTLP logs──▶ collector (dashboard, or a hosted plane)
/openral/safety_status ┘
```

It publishes nothing and is not on the actuation path: a recorder crash loses records,
never motion control. Each record carries `openral.record.kind` / `openral.schema.version`
/ `openral.record.id` / `openral.retention.class` attributes and its JSON as the body; the
machine is named by the OTel resource (`openral.unit.serial`, `openral.org.id`,
`openral.site.id`, stamped by the deploy launch).

The on-robot hash-chained log these records also go to is #469; recording who reset or
overrode a stop is #468.

## Bringup

Spawned by `packages/openral_rskill_ros/launch/deploy_e2e.launch.py` in both
`hal_mode:=sim` and `hal_mode:=real` as `/openral_compliance_recorder`, with
`run_mode` set from `hal_mode`, and driven to ACTIVE by `tools/lifecycle_autostart.py`.

## Tests

`test/test_recorder_node.py` runs the real lifecycle node against real
`FailureTrigger` / `SafetyStatus` publishers (no mocks) and reads the records back from an
in-memory OTel log exporter.
