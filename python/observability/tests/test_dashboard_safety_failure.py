"""Safety-kernel stops (``/openral/failure/safety``) become the console trace's error row.

The store builds the row's text from the real ``openral_core.CollisionEvidence`` the
kernel serializes; the console only renders ``topics.safety_failure``. No mocks
(CLAUDE.md §1.11): the production store and the core evidence schema.
"""

from __future__ import annotations

import pytest
from openral_core import CollisionEvidence, ForceEvidence
from openral_observability.dashboard import TelemetryStore

# The stop seen in the station sim: world voxel, 25.7 mm deep, grasp exemption live.
_COLLISION = CollisionEvidence(
    collision_kind="world",
    link_a="openarm_left_ee_link1",
    link_b_or_object="voxel_431037",
    horizon_step=-1,
    min_distance_m=-0.0257,
    place_allowance_active=False,
    grasp_exemption_active=True,
    grasp_exemption_source="tracked",
    workspace_contact_active=False,
    world_cell_center_m=(0.412, -0.087, 0.853),
)


def _record(store: TelemetryStore, evidence_json: str, **overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "kind_label": "kind_collision",
        "severity_label": "severity_abort",
        "evidence_json": evidence_json,
        "rskill_id": "rskills/rskill-pick-openarm",
        "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
        "stamp_unix": 1234.5,
    }
    kwargs.update(overrides)
    store.record_safety_failure(**kwargs)  # type: ignore[arg-type]  # reason: kwargs built dynamically
    entry: dict[str, object] = store.snapshot()["topics"]["safety_failure"]
    return entry


def test_starts_empty() -> None:
    assert TelemetryStore().snapshot()["topics"]["safety_failure"] == {}


def test_collision_stop_reads_as_one_operator_line() -> None:
    entry = _record(TelemetryStore(), _COLLISION.model_dump_json())
    assert entry["text"] == (
        "world collision: openarm_left_ee_link1 hit voxel_431037 · 25.7 mm deep"
        " · cell 0.41 m forward, 0.09 m right, 0.85 m up of the robot base"
        " · allowances: place off, grasp on (tracked), workspace off"
        " · stopped rskill-pick-openarm · latched: needs an operator reset"
    )
    assert entry["kind"] == "collision"
    assert entry["severity"] == "abort"
    assert entry["latched"] is True
    assert entry["rskill_id"] == "rskills/rskill-pick-openarm"
    assert entry["stamp_unix"] == pytest.approx(1234.5)
    assert isinstance(entry["ts_unix"], float)


def test_other_kinds_and_bad_evidence_still_produce_a_row() -> None:
    store = TelemetryStore()
    force = ForceEvidence(joint_or_ee="openarm_left_joint4", measured_n=41.0, limit_n=30.0)
    text = _record(store, force.model_dump_json(), kind_label="kind_force")["text"]
    assert str(text).startswith("force: joint_or_ee=openarm_left_joint4, measured_n=41.0")
    bad = _record(store, "{not json", severity_label="severity_warn", rskill_id="")
    assert bad["text"] == "collision: evidence='{not json'"
    assert bad["latched"] is False


def test_subscriber_writes_a_real_failure_trigger() -> None:
    pytest.importorskip("openral_msgs")
    from openral_msgs.msg import FailureTrigger
    from openral_observability.dashboard.safety_status_subscriber import SafetyStatusSubscriber

    store = TelemetryStore()
    subscriber = SafetyStatusSubscriber.__new__(SafetyStatusSubscriber)
    subscriber._store = store  # reason: exercising the callback without a ROS graph
    msg = FailureTrigger()
    msg.kind = FailureTrigger.KIND_COLLISION
    msg.severity = FailureTrigger.SEVERITY_ABORT
    msg.evidence_json = _COLLISION.model_dump_json()
    msg.rskill_id = "rskills/rskill-pick-openarm"
    subscriber._on_failure(msg)  # reason: the subscription callback is the unit under test
    entry = store.snapshot()["topics"]["safety_failure"]
    assert entry["kind"] == "collision"
    assert entry["latched"] is True


def test_place_allowance_names_its_target() -> None:
    ev = _COLLISION.model_copy(update={"place_allowance_active": True, "place_target": "shelf_2"})
    assert "allowances: place on (shelf_2), grasp on" in str(
        _record(TelemetryStore(), ev.model_dump_json())["text"]
    )
