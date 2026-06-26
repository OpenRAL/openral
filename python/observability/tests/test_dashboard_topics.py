"""Topic-bucket tests for :mod:`openral_observability.dashboard.store`.

These tests verify the per-topic dynamic aggregation added on top of the
latest-wins span cards: hal.read_state populates `topics.robot_state`,
hal.send_action populates `topics.commands`, world_state.snapshot
populates `topics.world_state` (ee poses + diagnostics), sensors.read_latest
populates `topics.perception` (per-camera entries), and the system metric
gauges populate `topics.system`.

Real-component end-to-end per CLAUDE.md §1.11: protobuf payloads are
built directly from ``opentelemetry-proto``, no mocks.
"""

from __future__ import annotations

import time

from openral_observability.dashboard import TelemetryStore
from opentelemetry.proto.common.v1.common_pb2 import AnyValue, ArrayValue, KeyValue
from opentelemetry.proto.metrics.v1.metrics_pb2 import (
    AggregationTemporality,
    Metric,
    NumberDataPoint,
    ResourceMetrics,
    ScopeMetrics,
)
from opentelemetry.proto.metrics.v1.metrics_pb2 import Sum as SumProto
from opentelemetry.proto.resource.v1.resource_pb2 import Resource
from opentelemetry.proto.trace.v1.trace_pb2 import (
    ResourceSpans,
    ScopeSpans,
    Span,
    Status,
)


def _av(value: object) -> AnyValue:
    if isinstance(value, bool):
        return AnyValue(bool_value=value)
    if isinstance(value, int):
        return AnyValue(int_value=value)
    if isinstance(value, float):
        return AnyValue(double_value=value)
    if isinstance(value, list):
        values = []
        for v in value:
            if isinstance(v, str):
                values.append(AnyValue(string_value=v))
            elif isinstance(v, bool):
                values.append(AnyValue(bool_value=v))
            elif isinstance(v, int):
                values.append(AnyValue(int_value=v))
            elif isinstance(v, float):
                values.append(AnyValue(double_value=v))
        return AnyValue(array_value=ArrayValue(values=values))
    return AnyValue(string_value=str(value))


def _attrs(d: dict[str, object]) -> list[KeyValue]:
    return [KeyValue(key=k, value=_av(v)) for k, v in d.items()]


def _resource(d: dict[str, object]) -> Resource:
    return Resource(attributes=_attrs(d))


def _make_span(
    name: str,
    *,
    duration_ms: float = 5.0,
    attrs: dict[str, object] | None = None,
) -> Span:
    start = time.time_ns()
    return Span(
        trace_id=b"\x07" * 16,
        span_id=b"\x07" * 8,
        name=name,
        start_time_unix_nano=start,
        end_time_unix_nano=start + int(duration_ms * 1_000_000),
        attributes=_attrs(attrs or {}),
        status=Status(code=0),
    )


def _wrap(
    spans: list[Span], resource_attrs: dict[str, object] | None = None
) -> list[ResourceSpans]:
    return [
        ResourceSpans(
            resource=_resource(resource_attrs or {"service.name": "ral"}),
            scope_spans=[ScopeSpans(spans=spans)],
        )
    ]


def test_hal_read_state_populates_robot_state_topic() -> None:
    store = TelemetryStore()
    span = _make_span(
        "hal.read_state",
        attrs={
            "openral.hal.adapter": "so100",
            "openral.hal.robot.model": "so100_follower",
            "openral.hal.joint.names": ["j0", "j1", "j2"],
            "openral.hal.joint.positions": [0.0, 0.5, -0.25],
            "openral.hal.joint.velocities": [0.1, 0.0, -0.02],
            "openral.hal.joint.efforts": [0.0, 1.2, 0.0],
            "openral.hal.joint.position_limits_lo": [-1.0, -1.5, -2.0],
            "openral.hal.joint.position_limits_hi": [1.0, 1.5, 2.0],
        },
    )
    store.ingest_spans(_wrap([span]))
    snap = store.snapshot()

    rs = snap["topics"]["robot_state"]
    assert rs["names"] == ["j0", "j1", "j2"]
    assert rs["positions"] == [0.0, 0.5, -0.25]
    assert rs["limits_lo"] == [-1.0, -1.5, -2.0]
    assert rs["limits_hi"] == [1.0, 1.5, 2.0]

    # Identity card latches the slow-changing hal attrs.
    assert snap["identity"]["openral.hal.adapter"] == "so100"
    assert snap["identity"]["openral.hal.robot.model"] == "so100_follower"


def test_hal_read_state_error_span_does_not_clear_joint_state() -> None:
    """An error-path hal.read_state span must not blank previously stored joint state.

    When read_state() raises inside the lifecycle node the span closes before
    record_joint_state() runs — so it carries no joint attributes.  A plain
    dict.update() with None values would overwrite good names/positions and the
    dashboard card would revert to "waiting for hal.read_state".
    """
    store = TelemetryStore()
    # First span: successful read — populates the joint card.
    store.ingest_spans(
        _wrap(
            [
                _make_span(
                    "hal.read_state",
                    attrs={
                        "openral.hal.adapter": "so100",
                        "openral.hal.joint.names": ["j0", "j1", "j2"],
                        "openral.hal.joint.positions": [0.1, 0.2, 0.3],
                        "openral.hal.joint.velocities": [0.0, 0.0, 0.0],
                    },
                )
            ]
        )
    )
    before = store.snapshot()["topics"]["robot_state"]
    assert before["names"] == ["j0", "j1", "j2"]
    assert before["positions"] == [0.1, 0.2, 0.3]

    # Second span: error path — no joint attributes (read_state raised).
    store.ingest_spans(
        _wrap(
            [
                _make_span(
                    "hal.read_state",
                    attrs={"openral.hal.adapter": "so100"},
                )
            ]
        )
    )
    after = store.snapshot()["topics"]["robot_state"]
    # Joint data must be preserved — not overwritten with None.
    assert after["names"] == ["j0", "j1", "j2"], after
    assert after["positions"] == [0.1, 0.2, 0.3], after


def test_hal_send_action_populates_commands_topic() -> None:
    store = TelemetryStore()
    span = _make_span(
        "hal.send_action",
        attrs={
            "openral.hal.adapter": "so100",
            "openral.hal.control_mode": "joint_position",
            "openral.hal.action.next": [0.1, 0.4, -0.2],
            "openral.hal.action.dim": 3,
            "openral.hal.action.horizon": 50,
            "openral.hal.action.applied": True,
            "openral.hal.gripper.position": 0.65,
        },
    )
    store.ingest_spans(_wrap([span]))
    cmd = store.snapshot()["topics"]["commands"]
    assert cmd["next_row"] == [0.1, 0.4, -0.2]
    assert cmd["dim"] == 3
    assert cmd["horizon"] == 50
    assert cmd["applied"] is True
    assert cmd["gripper_position"] == 0.65


def test_world_state_snapshot_populates_world_state_topic() -> None:
    store = TelemetryStore()
    span = _make_span(
        "world_state.snapshot",
        attrs={
            "openral.world_state.components_stale": 0,
            "openral.world_state.has_latched_error": False,
            "openral.world_state.battery_pct": 87.5,
            "openral.world_state.diagnostics_keys": ["joint_state", "cam_top"],
            "openral.world_state.diagnostics_values": ["ok", "stale"],
            "openral.hal.ee.pose.gripper": [0.3, 0.0, 0.45, 0.0, 0.0, 0.0, 1.0],
            "openral.hal.ee.names": ["gripper"],
        },
    )
    store.ingest_spans(_wrap([span]))
    ws = store.snapshot()["topics"]["world_state"]
    assert ws["components_stale"] == 0
    assert ws["has_latched_error"] is False
    assert ws["battery_pct"] == 87.5
    assert ws["diagnostics"] == {"joint_state": "ok", "cam_top": "stale"}
    assert ws["ee_poses"]["gripper"] == [0.3, 0.0, 0.45, 0.0, 0.0, 0.0, 1.0]


def test_sensors_read_latest_per_camera_entries_and_thumb_persistence() -> None:
    store = TelemetryStore()
    span_with_thumb = _make_span(
        "sensors.read_latest",
        attrs={
            "openral.sensors.source": "cam_top",
            "openral.sensors.modality": "rgb",
            "openral.sensors.encoding": "rgb8",
            "openral.sensors.width": 128,
            "openral.sensors.height": 96,
            "openral.sensors.channels": 3,
            "openral.sensors.age_ms": 12.4,
            "openral.sensors.thumbnail_jpeg_b64": "AAA",
        },
    )
    span_without_thumb = _make_span(
        "sensors.read_latest",
        attrs={
            "openral.sensors.source": "cam_top",
            "openral.sensors.modality": "rgb",
            "openral.sensors.encoding": "rgb8",
            "openral.sensors.age_ms": 8.1,
        },
    )
    store.ingest_spans(_wrap([span_with_thumb, span_without_thumb]))
    cam = store.snapshot()["topics"]["perception"]["cameras"]["cam_top"]
    # Thumbnail persisted across the no-thumb update
    assert cam["thumbnail_jpeg_b64"] == "AAA"
    # Most-recent age wins
    assert cam["age_ms"] == 8.1
    assert cam["modality"] == "rgb"


def test_safety_check_ledger_keeps_one_entry_per_check() -> None:
    store = TelemetryStore()
    s1 = _make_span(
        "safety.check",
        attrs={"safety.check_name": "ee_speed", "safety.severity": "info"},
    )
    s2 = _make_span(
        "safety.check",
        attrs={"safety.check_name": "workspace_box", "safety.severity": "warn"},
    )
    s3 = _make_span(
        "safety.check",
        attrs={"safety.check_name": "ee_speed", "safety.severity": "warn"},
    )
    store.ingest_spans(_wrap([s1, s2, s3]))
    ledger = store.snapshot()["topics"]["safety"]["checks"]
    assert set(ledger.keys()) == {"ee_speed", "workspace_box"}
    # Latest-wins for ee_speed
    assert ledger["ee_speed"]["severity"] == "warn"
    assert ledger["workspace_box"]["severity"] == "warn"


def test_system_gauges_populate_system_topic() -> None:
    store = TelemetryStore()
    rm = ResourceMetrics(
        resource=_resource({"service.name": "ral"}),
        scope_metrics=[
            ScopeMetrics(
                metrics=[
                    Metric(
                        name="openral.system.cpu.utilization_pct",
                        unit="%",
                        sum=SumProto(
                            aggregation_temporality=AggregationTemporality.AGGREGATION_TEMPORALITY_CUMULATIVE,
                            is_monotonic=False,
                            data_points=[NumberDataPoint(as_double=42.0)],
                        ),
                    ),
                    Metric(
                        name="openral.system.gpu.utilization_pct",
                        unit="%",
                        sum=SumProto(
                            aggregation_temporality=AggregationTemporality.AGGREGATION_TEMPORALITY_CUMULATIVE,
                            is_monotonic=False,
                            data_points=[
                                NumberDataPoint(
                                    as_double=88.0,
                                    attributes=_attrs(
                                        {
                                            "openral.system.gpu.index": 0,
                                            "openral.system.gpu.name": "RTX 4090",
                                        }
                                    ),
                                )
                            ],
                        ),
                    ),
                    Metric(
                        name="openral.system.gpu.memory_used_mb",
                        unit="MBy",
                        sum=SumProto(
                            aggregation_temporality=AggregationTemporality.AGGREGATION_TEMPORALITY_CUMULATIVE,
                            is_monotonic=False,
                            data_points=[
                                NumberDataPoint(
                                    as_double=18432.0,
                                    attributes=_attrs(
                                        {
                                            "openral.system.gpu.index": 0,
                                            "openral.system.gpu.name": "RTX 4090",
                                        }
                                    ),
                                )
                            ],
                        ),
                    ),
                ]
            )
        ],
    )
    store.ingest_metrics([rm])
    sys = store.snapshot()["topics"]["system"]
    assert sys["cpu_util_pct"] == 42.0
    assert sys["gpus"][0]["util_pct"] == 88.0
    assert sys["gpus"][0]["memory_used_mb"] == 18432.0
    assert sys["gpus"][0]["name"] == "RTX 4090"


def test_trace_anchor_carries_latest_trace_id() -> None:
    store = TelemetryStore()
    store.ingest_spans(_wrap([_make_span("rskill.execute")]))
    snap = store.snapshot()
    assert snap["topics"]["trace"]["latest_trace_id"] == "07" * 16


def test_rskill_chunk_inference_populates_inference_card_and_engine_identity() -> None:
    """The Inference card and the engine·device Identity field both wire off
    ``rskill.chunk_inference`` (semconv.SPAN_RSKILL_CHUNK_INFERENCE) — the
    name the ``inference_span`` helper emits. Routing the card off the older
    ``skill.chunk_inference`` name left it stuck on "waiting…".
    """
    store = TelemetryStore()
    span = _make_span(
        "rskill.chunk_inference",
        attrs={
            "inference.kind": "foreground",
            "inference.chunk_index": 0,
            "inference.chunk_size": 50,
            "inference.engine": "torch",
            "inference.device": "cuda:0",
        },
    )
    store.ingest_spans(_wrap([span]))
    snap = store.snapshot()

    # Headline Inference card is populated (not None / "waiting").
    inf_card = snap["cards"]["inference"]
    assert inf_card["attrs"]["inference.chunk_size"] == 50
    # Topic bucket carries the engine/device for the card detail.
    inf_topic = snap["topics"]["inference"]
    assert inf_topic["engine"] == "torch"
    assert inf_topic["device"] == "cuda:0"
    # engine·device Identity field latches regardless of span family.
    assert snap["identity"]["inference.engine"] == "torch"
    assert snap["identity"]["inference.device"] == "cuda:0"


def test_action_horizon_latches_into_identity_from_send_action() -> None:
    """The Identity card's "action horizon" field latches
    ``openral.hal.action.horizon`` (semconv.HAL_ACTION_HORIZON), which rides
    in on every ``hal.send_action`` span — the value the dashboard previously
    looked for under the never-emitted ``openral.skill.action_horizon`` key.
    """
    store = TelemetryStore()
    store.ingest_spans(
        _wrap(
            [
                _make_span(
                    "hal.send_action",
                    attrs={
                        "openral.hal.control_mode": "joint_position",
                        "openral.hal.action.next": [0.1, 0.2],
                        "openral.hal.action.dim": 2,
                        "openral.hal.action.horizon": 50,
                        "openral.hal.action.applied": True,
                    },
                )
            ]
        )
    )
    assert store.snapshot()["identity"]["openral.hal.action.horizon"] == 50


def test_safety_kernel_latches_into_identity_from_safety_check() -> None:
    """The Identity card's "safety kernel" field latches ``safety.kernel``
    (semconv.SAFETY_KERNEL) off ``safety.check`` spans — emitted by the
    NullSafetyClient as "null" and by the C++ kernel as "cpp". The dashboard
    previously read the never-emitted ``openral.safety.kernel`` key.
    """
    store = TelemetryStore()
    store.ingest_spans(
        _wrap(
            [
                _make_span(
                    "safety.check",
                    attrs={
                        "safety.check_name": "null",
                        "safety.severity": "info",
                        "safety.kernel": "null",
                    },
                )
            ]
        )
    )
    assert store.snapshot()["identity"]["safety.kernel"] == "null"


def test_reasoner_tick_populates_reasoner_topic_with_mission() -> None:
    """A reasoner.tick span feeds the Reasoner/Mission card, including the
    ADR-0073 ``reasoner.mission_json`` task queue (decoded to a dict)."""
    import json

    from openral_reasoner.mission import MissionState

    mission = MissionState(["pick the bowl", "place the butter"])
    mission.record_attempt(rskill_id="smolvla-libero")
    mission.complete_active("success=0.91")  # t1 done, t2 now active

    store = TelemetryStore()
    span = _make_span(
        "reasoner.tick",
        attrs={
            "reasoner.tick.idx": 7,
            "reasoner.tool": "execute_rskill",
            "reasoner.rskill_id": "smolvla-libero",
            "reasoner.model": "claude-opus-4-8",
            "reasoner.mission_json": json.dumps(mission.to_summary()),
        },
    )
    store.ingest_spans(_wrap([span]))

    reasoner = store.snapshot()["topics"]["reasoner"]
    assert reasoner["tool"] == "execute_rskill"
    assert reasoner["tick_idx"] == 7
    mission_snap = reasoner["mission"]
    assert mission_snap["max_attempts"] == 3
    assert [t["status"] for t in mission_snap["tasks"]] == ["done", "active"]
    assert mission_snap["tasks"][0]["verdict"] == "success=0.91"


def test_reasoner_tick_without_mission_leaves_mission_none() -> None:
    """A bare-goal deploy emits no ``reasoner.mission_json``; the slot stays None
    rather than raising."""
    store = TelemetryStore()
    store.ingest_spans(_wrap([_make_span("reasoner.tick", attrs={"reasoner.tick.idx": 1})]))
    assert store.snapshot()["topics"]["reasoner"]["mission"] is None
