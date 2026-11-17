"""Safety-kernel stops (``/openral/failure/safety``) become the console trace's error row.

The store builds the row's text from the real ``openral_core.CollisionEvidence`` the
kernel serializes; the console only renders ``topics.safety_failure``. No mocks
(CLAUDE.md §1.11): the production store and the core evidence schema.
"""

from __future__ import annotations

import sys
import uuid

import pytest
import structlog
from openral_core import CollisionEvidence, ForceEvidence
from openral_observability.dashboard import TelemetryStore
from openral_observability.dashboard.store import _SAFETY_FAILURE_COALESCE_S

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
    workspace_declaration_live=False,
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
        " · stopped rskill-pick-openarm"
    )
    assert entry["kind"] == "collision"
    assert entry["severity"] == "abort"
    # Whether the kernel latched is the safety_status slot's to say, not a guess here.
    assert "latched" not in entry
    assert entry["repeats"] == 0
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


def test_each_stop_is_an_event_row_and_repeats_fold() -> None:
    """The slot keeps the latest stop; the event log keeps every distinct one."""
    store = TelemetryStore()
    stale = '{"kind":"perception","stale_input":"joint_states","age_ms":120.0,"limit_ms":100.0}'
    for _ in range(20):  # one per dropped chunk while the input stays stale
        _record(store, stale, kind_label="kind_perception", severity_label="severity_fail")
    events = [e for e in store.snapshot()["events"] if e["kind"] == "safety.failure"]
    assert len(events) == 1
    assert events[0]["severity"] == "error"
    assert events[0]["title"].startswith("safety stop · ")
    assert store.snapshot()["topics"]["safety_failure"]["repeats"] == 19
    # A different stop is a new row, at once.
    _record(store, _COLLISION.model_dump_json())
    events = [e for e in store.snapshot()["events"] if e["kind"] == "safety.failure"]
    assert len(events) == 2
    assert store.snapshot()["topics"]["safety_failure"]["repeats"] == 0


def test_a_repeat_after_the_window_is_a_new_row() -> None:
    store = TelemetryStore()
    _record(store, _COLLISION.model_dump_json())
    run = store._safety_failure_run  # reason: age the run past the window without sleeping
    assert run is not None
    store._safety_failure_run = (run[0], run[1] - _SAFETY_FAILURE_COALESCE_S - 0.1)
    _record(store, _COLLISION.model_dump_json())
    assert sum(e["kind"] == "safety.failure" for e in store.snapshot()["events"]) == 2


def test_workspace_reads_the_declaration_not_the_in_tree_rule() -> None:
    """Under an exemption provider the in-tree rule is off while the declaration is live."""
    ev = _COLLISION.model_copy(
        update={"workspace_contact_active": False, "workspace_declaration_live": True}
    )
    assert "workspace on" in str(_record(TelemetryStore(), ev.model_dump_json())["text"])
    # An older producer without the field falls back to the in-tree rule.
    old = _COLLISION.model_copy(
        update={"workspace_contact_active": True, "workspace_declaration_live": None}
    )
    assert "workspace on" in str(_record(TelemetryStore(), old.model_dump_json())["text"])


def test_unevaluated_and_box_bound_distances_are_not_shown_as_depth() -> None:
    nan_sentinel = _COLLISION.model_copy(update={"min_distance_m": -sys.float_info.max})
    text = str(_record(TelemetryStore(), nan_sentinel.model_dump_json())["text"])
    assert "distance not evaluated" in text
    assert "mm deep" not in text
    box = _COLLISION.model_copy(update={"depth_is_box_bound": True})
    text = str(_record(TelemetryStore(), box.model_dump_json())["text"])
    assert "≤ 25.7 mm deep (box bound)" in text


def test_the_failure_topic_is_the_failure_bus_safety_topic() -> None:
    from openral_observability.dashboard.safety_status_subscriber import SAFETY_FAILURE_TOPIC
    from openral_observability.failure_bus import FailureSource, topic_for

    assert topic_for(FailureSource.SAFETY) == SAFETY_FAILURE_TOPIC


def test_unknown_severity_keeps_its_prefix() -> None:
    pytest.importorskip("openral_msgs")
    from openral_msgs.msg import FailureTrigger
    from openral_observability.failure_bus import drop_reason_label

    assert drop_reason_label(200, FailureTrigger, ("SEVERITY_",)) == "severity_200"
    assert drop_reason_label(200, FailureTrigger) == "drop_reason_200"


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
    assert entry["severity"] == "abort"


def test_place_allowance_names_its_target() -> None:
    ev = _COLLISION.model_copy(update={"place_allowance_active": True, "place_target": "shelf_2"})
    assert "allowances: place on (shelf_2), grasp on" in str(
        _record(TelemetryStore(), ev.model_dump_json())["text"]
    )


def test_api_state_serves_the_stop() -> None:
    """The store -> ``/api/state`` contract the console's trace reads."""
    import asyncio

    import httpx
    from openral_observability.dashboard import create_app

    store = TelemetryStore()
    _record(store, _COLLISION.model_dump_json())

    async def get() -> dict[str, object]:
        transport = httpx.ASGITransport(app=create_app(store))
        async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as c:
            state: dict[str, object] = (await c.get("/api/state")).json()
            return state

    stop = asyncio.run(get())["topics"]["safety_failure"]  # type: ignore[index]  # reason: JSON
    assert stop["kind"] == "collision"
    assert stop["text"].startswith("world collision: openarm_left_ee_link1 hit voxel_431037")
    assert isinstance(stop["ts_unix"], float)


def test_a_callback_failure_is_logged_not_lost() -> None:
    """An exception in any callback would end the shared spin thread; each logs instead."""
    pytest.importorskip("openral_msgs")
    from openral_msgs.msg import FailureTrigger, SafetyStatus
    from openral_observability.dashboard.safety_status_subscriber import SafetyStatusSubscriber

    subscriber = SafetyStatusSubscriber.__new__(SafetyStatusSubscriber)
    subscriber._store = None  # type: ignore[assignment]  # reason: force the store write to raise
    msg = FailureTrigger()
    msg.kind = FailureTrigger.KIND_COLLISION
    msg.evidence_json = _COLLISION.model_dump_json()
    with structlog.testing.capture_logs() as logs:
        subscriber._on_failure(msg)  # reason: must not raise
        subscriber._on_status(SafetyStatus())  # reason: must not raise
    failed = [e["callback"] for e in logs if e["event"] == "dashboard.safety_callback_failed"]
    assert failed == ["_on_failure", "_on_status"]


def test_a_safety_violation_is_never_swallowed() -> None:
    pytest.importorskip("openral_msgs")
    from openral_core import ROSSafetyViolation
    from openral_msgs.msg import SafetyStatus
    from openral_observability.dashboard.safety_status_subscriber import SafetyStatusSubscriber

    class _Raises(TelemetryStore):
        def set_safety_status(self, **_: object) -> None:
            raise ROSSafetyViolation("kernel stop")

    subscriber = SafetyStatusSubscriber.__new__(SafetyStatusSubscriber)
    subscriber._store = _Raises()
    with pytest.raises(ROSSafetyViolation):
        subscriber._on_status(SafetyStatus())  # reason: the guard re-raises it


def test_live_pub_sub_reaches_the_store(monkeypatch: pytest.MonkeyPatch) -> None:
    """A real ``FailureTrigger`` published on the ROS graph lands in the store.

    Kernel-format evidence with the ``attached:pinch:...`` link the live stop named.
    A private, localhost-only domain where this process starts rclpy; either way the
    wait matches this test's own message (a unique skill id), not any other stop.
    """
    pytest.importorskip("openral_msgs")
    monkeypatch.setenv("ROS_DOMAIN_ID", "92")
    monkeypatch.setenv("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")
    import time

    import rclpy
    from openral_msgs.msg import FailureTrigger
    from openral_observability.dashboard.safety_status_subscriber import (
        SAFETY_FAILURE_TOPIC,
        SafetyStatusSubscriber,
    )

    store = TelemetryStore()
    rskill_id = f"OpenRAL/rskill-pi05-openarm-{uuid.uuid4().hex[:8]}"
    msg = FailureTrigger()
    msg.kind = FailureTrigger.KIND_COLLISION
    msg.severity = FailureTrigger.SEVERITY_ABORT
    msg.rskill_id = rskill_id
    msg.evidence_json = _COLLISION.model_copy(
        update={
            "link_a": "attached:pinch:left_gripper@44750002334",
            "link_b_or_object": "voxel_534370",
        }
    ).model_dump_json()
    subscriber = SafetyStatusSubscriber(store)
    node = None
    try:
        assert subscriber.available
        node = rclpy.create_node("test_safety_failure_pub")
        pub = node.create_publisher(FailureTrigger, SAFETY_FAILURE_TOPIC, 50)

        def mine() -> dict[str, object]:
            entry: dict[str, object] = store.snapshot()["topics"]["safety_failure"]
            return entry if entry.get("rskill_id") == rskill_id else {}

        deadline = time.monotonic() + 10
        while not mine() and time.monotonic() < deadline:
            pub.publish(msg)
            time.sleep(0.2)
        assert str(mine().get("text", "")).startswith(
            "world collision: attached:pinch:left_gripper@44750002334 hit voxel_534370"
        )
    finally:
        if node is not None:
            node.destroy_node()
        subscriber.close()
