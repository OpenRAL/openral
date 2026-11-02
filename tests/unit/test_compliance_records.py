"""Compliance record schemas (issue #467): round-trip, discrimination, retention.

Run with:
    uv run pytest tests/unit/test_compliance_records.py -v
"""

from __future__ import annotations

import hashlib
import json
import typing
import uuid
from pathlib import Path

import jsonschema
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from openral_core import (
    RETENTION_FLOOR_DAYS,
    Actor,
    CollisionEvidence,
    ComplianceRecord,
    FailureRecord,
    HumanEvidence,
    InterventionRecord,
    PerceptionStaleEvidence,
    PlanDecisionRecord,
    RetentionClass,
    RobotDescription,
    RSkillManifest,
    SafetyDecisionRecord,
    SafetyStopRecord,
    SecurityIncidentRecord,
    SessionRecord,
    SoftwareComponent,
    SoftwareInventoryRecord,
    TimeoutEvidence,
)
from pydantic import TypeAdapter, ValidationError

_REPO = Path(__file__).resolve().parents[2]
_ADAPTER: TypeAdapter[ComplianceRecord] = TypeAdapter(ComplianceRecord)
_RECORD_TYPES = typing.get_args(typing.get_args(ComplianceRecord)[0])
_UNION_SCHEMA = _ADAPTER.json_schema()

_hex32 = st.from_regex(r"\A[0-9a-f]{32}\Z")
_hex64 = st.from_regex(r"\A[0-9a-f]{64}\Z")
_ns = st.integers(min_value=0, max_value=2**63 - 1)
_text = st.text(min_size=1, max_size=24)
_pos = st.floats(min_value=1e-3, max_value=1e6, allow_nan=False)
_envelope = {
    "record_id": _hex32,
    "stream_id": st.none() | _hex32,
    "seq": st.integers(min_value=0, max_value=2**63 - 1),
    "ts_ns": _ns,
    "clock_source": st.sampled_from(["ptp", "ntp", "monotonic_unsynced"]),
    "trace_id": st.none() | _hex32,
    "span_id": st.none() | st.from_regex(r"\A[0-9a-f]{16}\Z"),
    "parent_record_id": st.none() | _hex32,
}


def _every(model: type, field: str) -> st.SearchStrategy[object]:
    """Every value of a ``Literal`` field, so no branch of it goes unsampled."""
    annotation = model.model_fields[field].annotation  # type: ignore[attr-defined]  # reason: pydantic model
    return st.sampled_from(typing.get_args(annotation))


_evidence = st.none() | st.one_of(
    st.builds(HumanEvidence, actor=_text, reason=_text),
    st.builds(TimeoutEvidence, operation=_text, deadline_s=_pos, elapsed_s=_pos),
    st.builds(PerceptionStaleEvidence, sensor_id=_text, staleness_ms=_pos, threshold_ms=_pos),
)

_record_st = st.one_of(
    st.builds(
        FailureRecord,
        **_envelope,
        source=_every(FailureRecord, "source"),
        failure_kind=st.integers(min_value=0, max_value=255),
        severity=st.integers(min_value=0, max_value=3),
        evidence=_evidence,
    ),
    st.builds(
        SafetyStopRecord,
        **_envelope,
        trigger_source=_every(SafetyStopRecord, "trigger_source"),
        latched=st.booleans(),
        evidence=_evidence,
        stop_cmd_ts_ns=st.none() | _ns,
        halt_ts_ns=st.none() | _ns,
    ),
    st.builds(
        SafetyDecisionRecord,
        **_envelope,
        decision=_every(SafetyDecisionRecord, "decision"),
        check_name=_text,
        inputs_digest=st.none() | _hex64,
    ),
    st.builds(
        PlanDecisionRecord,
        **_envelope,
        model=_text,
        tool=_text,
        ladder_rung=_every(PlanDecisionRecord, "ladder_rung"),
        prompt_digest=st.none() | _hex64,
    ),
    st.builds(
        InterventionRecord,
        **_envelope,
        intervention_kind=_every(InterventionRecord, "intervention_kind"),
        actor=st.builds(Actor, channel_label=_text, actor_id=st.none() | _hex64),
        channel=_every(InterventionRecord, "channel"),
        legitimate=st.none() | st.booleans(),
    ),
    st.builds(
        SoftwareInventoryRecord,
        **_envelope,
        components=st.lists(
            st.builds(SoftwareComponent, name=_text, version=_text, sha256=st.none() | _hex64),
            min_size=1,
            max_size=3,
        ),
    ),
    st.builds(
        SessionRecord,
        **_envelope,
        start_ts_ns=_ns,
        end_ts_ns=st.none() | _ns,
        run_mode=_every(SessionRecord, "run_mode"),
    ),
    st.builds(
        SecurityIncidentRecord,
        **_envelope,
        actively_exploited=st.booleans(),
        malicious_suspected=st.booleans(),
        summary=_text,
    ),
)


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_record_st)
def test_record_round_trips_through_the_union_and_its_json_schema(record: object) -> None:
    """Every kind decodes back to itself through ``ComplianceRecord`` and validates."""
    line = record.model_dump_json()  # type: ignore[attr-defined]  # reason: one_of of models
    assert _ADAPTER.validate_json(line) == record
    jsonschema.validate(json.loads(line), _UNION_SCHEMA)


def test_every_kind_has_a_retention_class_with_a_floor() -> None:
    """The class follows from the kind and has a floor; MR trace classes keep 5 years."""
    for cls in _RECORD_TYPES:
        assert cls._retention in RETENTION_FLOOR_DAYS, cls.__name__
    assert InterventionRecord._retention is RetentionClass.MR_TRACE_5Y
    assert SoftwareInventoryRecord._retention is RetentionClass.MR_TRACE_5Y
    assert RETENTION_FLOOR_DAYS[RetentionClass.MR_TRACE_5Y] >= 5 * 365
    assert RETENTION_FLOOR_DAYS[RetentionClass.AI_LOG_6M] >= 183


def test_retention_class_is_not_on_the_wire() -> None:
    """Callers cannot set the class: it is not serialised and an extra key is refused."""
    record = SessionRecord(
        record_id=uuid.uuid4().hex,
        seq=0,
        ts_ns=0,
        clock_source="ntp",
        start_ts_ns=0,
        run_mode="sim",
    )
    body = json.loads(record.model_dump_json())
    assert "retention_class" not in body
    body["retention_class"] = "telemetry_raw"
    with pytest.raises(ValidationError):
        _ADAPTER.validate_python(body)


def test_safety_stop_carries_kernel_collision_evidence_for_a_real_robot() -> None:
    """A kernel collision stop on franka_panda's real joints and links round-trips."""
    robot = RobotDescription.from_yaml(str(_REPO / "robots" / "franka_panda" / "robot.yaml"))
    joints = [j.name for j in robot.joints]
    evidence = CollisionEvidence(
        collision_kind="world",
        link_a=robot.end_effectors[0].name,
        link_b_or_object="shelf",
        horizon_step=3,
        min_distance_m=-0.004,
        joint_positions_rad=[0.0] * len(joints),
        joint_names=joints,
    )
    stop = SafetyStopRecord(
        record_id=uuid.uuid4().hex,
        seq=42,
        ts_ns=1_761_700_000_000_000_000,
        clock_source="ntp",
        trigger_source="kernel",
        drop_reason=10,  # FailureTrigger.KIND_COLLISION
        evidence=evidence,
        latched=True,
    )
    decoded = _ADAPTER.validate_json(stop.model_dump_json())
    assert isinstance(decoded, SafetyStopRecord)
    assert isinstance(decoded.evidence, CollisionEvidence)
    assert decoded.retention_class is RetentionClass.MR_SAFETY_DECISION_1Y


def test_software_inventory_from_a_real_rskill_manifest() -> None:
    """MR §1.1.9 identification: a real rSkill manifest becomes an inventory component."""
    path = _REPO / "rskills" / "act-aloha" / "rskill.yaml"
    manifest = RSkillManifest.from_yaml(str(path))
    component = SoftwareComponent(
        name=manifest.name,
        version=manifest.version,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    record = SoftwareInventoryRecord(
        record_id=uuid.uuid4().hex, seq=0, ts_ns=0, clock_source="ptp", components=[component]
    )
    assert _ADAPTER.validate_json(record.model_dump_json()) == record


def test_actor_id_must_be_a_pseudonym_not_a_name() -> None:
    """``actor_id`` only accepts a 64-hex HMAC, so a raw operator name is refused."""
    with pytest.raises(ValidationError):
        Actor(channel_label="slack", actor_id="alice")


def test_wire_trace_context_forms_normalise_to_the_envelope() -> None:
    """A FailureTrigger traceparent and an unset ``""`` both decode (ISSUE-109)."""
    base = {
        "record_id": "0" * 32,
        "seq": 0,
        "ts_ns": 0,
        "clock_source": "ntp",
        "trigger_source": "kernel",
        "latched": True,
    }
    tp = SafetyStopRecord.model_validate({**base, "trace_id": f"00-{'a' * 32}-{'b' * 16}-01"})
    assert (tp.trace_id, tp.span_id) == ("a" * 32, "b" * 16)
    assert SafetyStopRecord.model_validate({**base, "trace_id": ""}).trace_id is None
    with pytest.raises(ValidationError):
        SafetyStopRecord.model_validate({**base, "trace_id": "not-a-trace"})


def test_seq_beyond_int64_is_refused() -> None:
    """OTLP int attributes are int64; a larger seq would overflow on export."""
    with pytest.raises(ValidationError):
        SessionRecord(
            record_id="0" * 32,
            seq=2**63,
            ts_ns=0,
            clock_source="ntp",
            start_ts_ns=0,
            run_mode="sim",
        )
