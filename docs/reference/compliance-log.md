# The on-robot compliance log

A robot running OpenRAL keeps its own evidence: every compliance record (failures, safety
stops, sessions, interventions, the installed software) is written to an append-only,
hash-chained log on the machine, with no network needed. This page describes the format, the
retention rules and the `openral compliance` command, in a form integrators can cite in their
instructions for use (AI Act Art. 13(3)(f): how deployers "collect, store and interpret the
logs").

It serves the EU Machinery Regulation 2023/1230, Annex III:

- §1.1.9 — "collect evidence of a legitimate or illegitimate intervention" in the software or
  its configuration, and "identify the software installed … necessary for it to operate
  safely … at all times";
- §1.2.1(f) — the tracing log of interventions and safety-software versions is "enabled for
  five years after such upload", producible on a competent authority's reasoned request.

The records themselves are the typed `openral_core.ComplianceRecord` kinds (JSON Schemas in
`docs/reference/schemas/`, producers in [Telemetry](telemetry.md#compliance-records)). The same
records also go out over OTLP; the log is the record of truth when the network is down.

## Where it is and when it is on

| | |
|---|---|
| Location | `$OPENRAL_COMPLIANCE_DIR`, else `$XDG_STATE_HOME/openral/compliance` (`~/.local/state/openral/compliance`) |
| `openral deploy run` (real hardware) | always on |
| `openral deploy sim` | only when `OPENRAL_COMPLIANCE_DIR` is set |

Put the directory on storage that survives a re-image of the robot, and back it up: the
five-year duty outlives most SD cards.

## Layout

```
<root>/
  hold                                  present → nothing is deleted (legal hold)
  ship.state                            what `ship` has sent, per segment
  <writer>/                             one per recording process: recorder, runtime, cli
    <retention_class>/
      <YYYY-MM-DD>.jsonl                one record per line, UTC day of the record
      <YYYY-MM-DD>.quarantine           a torn last line found after a power cut
      pruned                            retention has deleted older segments
```

Each line is `{"prev_hash": "<sha256 of the previous line>", "record": {...}}`. The chain
runs across days inside one writer's class; the first line's `prev_hash` is 64 zeros. Each
record carries the `stream_id` of the process that wrote it and a gap-free `seq` within that
stream. So an edited, deleted or reordered line breaks a chain, and a missing record leaves a
`seq` gap — `openral compliance verify` reports both.

Writes are one `O_APPEND` write per line. Records of the safety-decision, trace and incident
classes are `fsync`'d before the recorder moves on, so a power cut can lose at most the line
being written; that torn tail is moved to `.quarantine` on the next start and reported.

## Retention

| Class | Kept at least | Records |
|---|---|---|
| `ai_log_6m` | 184 days | failures, sessions |
| `mr_safety_decision_1y` | 366 days | safety stops, kernel and planner decisions |
| `mr_trace_5y` | 1827 days | interventions, software inventories |
| `incident_10y` | 3653 days | security incidents |
| `telemetry_raw` | 90 days (default) | — |

A segment is deleted only once its whole day is older than its class's floor. Deployer policy
may keep a class longer, never shorter. While `<root>/hold` exists nothing is deleted
(AI Act Art. 73(6): no alteration before the authorities are informed of a serious incident).
Every deletion is itself recorded as an `intervention` record (`retention_delete`).

Personal data is minimised at the source: operators appear only as a pseudonymous
`actor_id`, and no camera frames are logged.

## Interventions and the stop on a lost log

Every e-stop **reset** is recorded as an `intervention` (`estop_reset`), whatever path made it:
the recorder sees the safety latch clear. `openral estop reset --operator NAME` and the
dashboard announce who acted on `/openral/operator_action` (`openral_msgs/OperatorAction`)
just before they act; the record then carries the channel and the operator as
`HMAC-SHA256(identity, $OPENRAL_ACTOR_KEY)` — the deployer keeps the key, so only the deployer
can link the id back to a person. A reset nobody announced (a raw `ros2 service call`, say) is
still recorded, with `channel: unknown` and `legitimate: null`: the Machinery Regulation asks
for evidence of illegitimate interventions too. Dashboard parameter changes are recorded as
`param_set` interventions. An intervention is `legitimate: true` only when the operator is
identified (a pseudonymous id under the deployer's key); an anonymous announcement records
the channel with `legitimate: null`. An announcement is a claim by the channel, not an
authentication (logins are #36).

**A machine that cannot keep its safety evidence does not move.** If a record cannot be
written to the log (full disk, read-only filesystem), the recorder publishes `/openral/estop`
and a `FailureTrigger(KIND_CONTROLLER, SEVERITY_ABORT)` with
`ControllerEvidence(state="log_write_failed")`; a reset while the log still fails cannot be
recorded and latches again. The recorder warns in its log while the filesystem has under 20 %
free. On `deploy run` the recorder is required: a deploy whose recorder cannot start or exits
(cleanly or not) shuts down, and deactivating it stops the machine
(`ControllerEvidence(state="inactive")`).

## `openral compliance`

| Command | What it does |
|---|---|
| `verify [--partial]` | Walk every chain; exit 1 on a broken link or a `seq` gap (`--partial` for an exported subset) |
| `inventory [--live]` | The last recorded software inventory (or this host now); safety software marked with its digest |
| `export --out bundle.tar.gz [--class C …] [--since YYYY-MM-DD]` | Whole segments for an authority; the recipient can run `verify --partial` on the unpacked bundle |
| `hold true\|false` | Legal hold on or off |
| `prune [--dry-run]` | Delete what is past retention, recording each deletion |
| `ship [--endpoint URL --serial S --org O]` | Send not-yet-shipped records to an OTLP collector; resumes after an outage, never re-sends an accepted record |

All take `--root` (default `$OPENRAL_COMPLIANCE_DIR`). `ship` reads the standard
`OTEL_EXPORTER_OTLP_HEADERS` / `OTEL_EXPORTER_OTLP_PROTOCOL` variables, so a bearer token
for a hosted collector goes in the headers variable.

Run `prune` from a daily timer and `ship` as often as the site's network allows; `verify`
before any export.
