"""Round-trip every ``openral_core.FailureEvidence`` variant.

Exercises the real Pydantic discriminated union — no mocks per
CLAUDE.md §1.11. Each variant is serialized with ``model_dump_json``
and validated back through ``TypeAdapter(FailureEvidence)``; the
identity must hold and the discriminator (``kind`` field) must select
the correct concrete class.

Plus token-bucket coverage for
``openral_observability.failure_bus._TokenBucket``, which is the
only piece of F3's publisher that is testable without ``rclpy``.
"""

from __future__ import annotations

import pytest
from openral_core import (
    ControllerEvidence,
    CriticEvidence,
    FailureEvidence,
    ForceEvidence,
    GraspOutcomeEvidence,
    HumanEvidence,
    ObjectQueryEvidence,
    ObjectQueryObservation,
    ObjectRole,
    PerceptionStaleEvidence,
    PlaceRegion,
    Pose6D,
    ReasonerTimeoutEvidence,
    SelfVerifyEvidence,
    SuppressedSummaryEvidence,
    TimeoutEvidence,
    WamEvidence,
    WorkspaceEvidence,
)
from openral_core.exceptions import ROSConfigError, ROSObjectNotInMemory, ROSPerceptionStale
from openral_observability.failure_bus import (
    DEFAULT_RATE_LIMIT_HZ,
    KIND_FORCE,
    KIND_SUPPRESSED_SUMMARY,
    KIND_TIMEOUT,
    SEVERITY_ABORT,
    SEVERITY_INFO,
    SEVERITY_WARN,
    FailureSource,
    _TokenBucket,
    topic_for,
)
from pydantic import TypeAdapter, ValidationError

_ADAPTER: TypeAdapter[FailureEvidence] = TypeAdapter(FailureEvidence)


# ── Round-trip per variant ─────────────────────────────────────────────────


_VARIANTS: list[FailureEvidence] = [
    TimeoutEvidence(operation="hal.read_state", deadline_s=0.05, elapsed_s=0.071),
    ForceEvidence(joint_or_ee="ee", measured_n=12.5, limit_n=10.0),
    WorkspaceEvidence(
        ee_name="ee",
        measured_xyz=(0.6, 0.0, 0.4),
        box_min=(-0.5, -0.5, 0.0),
        box_max=(0.5, 0.5, 0.8),
    ),
    PerceptionStaleEvidence(sensor_id="wrist_cam", staleness_ms=140.0, threshold_ms=100.0),
    CriticEvidence(critic_id="pick_critic", score=0.42, threshold=0.5),
    ControllerEvidence(
        controller_name="joint_trajectory_controller",
        state="error",
        detail="follow goal aborted",
    ),
    SelfVerifyEvidence(check="action_chunk.shape", expected="(8, 6)", observed="(8, 5)"),
    HumanEvidence(actor="slack:adrian", reason="testing the e-stop"),
    WamEvidence(horizon=4, discrepancy=0.27, wam_id="cosmos-1.0"),
    ReasonerTimeoutEvidence(model="claude-opus-4-7", deadline_s=1.0, elapsed_s=1.42),
    ObjectQueryEvidence(
        escalation="exhausted",
        goal_id="grasp:obj:orange_1@17",
        query_id="0",
        label="orange",
        role=ObjectRole.TARGET,
        waited_s=60.2,
        attempts=14,
        reason="workspace_mismatch",
        invocation_id="0123456789abcdef0123456789abcdef",
        workspace=PlaceRegion(
            frame_id="base",
            pose=Pose6D(xyz=(0.0, 0.0, 0.1), quat_xyzw=(0, 0, 0, 1), frame_id="base"),
            half_extents=(0.2, 0.2, 0.1),
        ),
        observations=[ObjectQueryObservation(camera="top", stamp_ns=17, reason="too_few_points")],
        last_seen=PlaceRegion(
            frame_id="base",
            pose=Pose6D(xyz=(0.07, -0.30, 0.05), quat_xyzw=(0, 0, 0, 1), frame_id="base"),
            half_extents=(0.03, 0.03, 0.03),
        ),
    ),
    GraspOutcomeEvidence(
        outcome="miss",
        gripper_joint="left_gripper",
        jaw_link="openarm_left_finger_pair",
        hand_link="openarm_left_link7",
        consecutive_misses=2,
        target_id="cell:pick_box",
        region_armed=True,
        short_of_command=0.031,
        settle_spread=0.0004,
    ),
    SuppressedSummaryEvidence(
        window_s=1.0,
        kinds=[KIND_TIMEOUT, KIND_FORCE],
        severities=[SEVERITY_WARN, SEVERITY_WARN],
        counts=[3, 12],
    ),
]


@pytest.mark.parametrize("evidence", _VARIANTS, ids=lambda e: e.kind)
def test_evidence_json_round_trip(evidence: FailureEvidence) -> None:
    """``model_dump_json`` → ``TypeAdapter.validate_json`` must round-trip."""
    raw = evidence.model_dump_json()
    back = _ADAPTER.validate_json(raw)
    assert type(back) is type(evidence)
    assert back == evidence
    assert back.kind == evidence.kind


def test_unknown_kind_is_rejected() -> None:
    """The Pydantic discriminator rejects an unknown ``kind`` value."""
    with pytest.raises(ValidationError):
        _ADAPTER.validate_json('{"kind": "definitely-not-a-real-kind"}')


def test_object_query_observations_are_bounded_and_reason_is_in_typed_failure() -> None:
    evidence = next(ev for ev in _VARIANTS if isinstance(ev, ObjectQueryEvidence))
    assert evidence.reason in str(evidence.to_exception())
    assert evidence.workspace is not None
    assert evidence.invocation_id == "0123456789abcdef0123456789abcdef"
    assert evidence.observations == [
        ObjectQueryObservation(camera="top", stamp_ns=17, reason="too_few_points")
    ]
    with pytest.raises(ValidationError):
        ObjectQueryEvidence.model_validate(
            {
                **evidence.model_dump(),
                "invocation_id": "not-an-action-uuid",
            }
        )
    with pytest.raises(ValidationError):
        ObjectQueryEvidence.model_validate(
            {
                **evidence.model_dump(),
                "observations": [
                    ObjectQueryObservation(camera="top", stamp_ns=i, reason="refused")
                    for i in range(9)
                ],
            }
        )


def test_grasp_outcome_evidence_refuses_what_no_close_could_report() -> None:
    """Issue #353: a run of zero misses, a non-finite reading, or an unknown outcome."""
    evidence = next(ev for ev in _VARIANTS if isinstance(ev, GraspOutcomeEvidence))
    for bad in (
        {"consecutive_misses": 0},
        {"short_of_command": float("nan")},
        {"settle_spread": -0.001},
        {"outcome": "regrasp"},
    ):
        with pytest.raises(ValidationError):
            GraspOutcomeEvidence.model_validate({**evidence.model_dump(), **bad})
    # An undeclared close: no target, no region, readings not yet classified.
    undeclared = GraspOutcomeEvidence(
        outcome="miss",
        gripper_joint="left_gripper",
        jaw_link="openarm_left_finger_pair",
        hand_link="openarm_left_link7",
        consecutive_misses=1,
    )
    assert (undeclared.target_id, undeclared.region_armed) == ("", False)
    assert undeclared.short_of_command is None and undeclared.settle_spread is None


def test_suppressed_summary_requires_parallel_arrays() -> None:
    """Mismatched arrays in ``SuppressedSummaryEvidence`` raise."""
    with pytest.raises(ROSConfigError):
        SuppressedSummaryEvidence(window_s=1.0, kinds=[0, 1], severities=[1], counts=[3, 7])


# ── Token bucket ───────────────────────────────────────────────────────────


def test_token_bucket_rate_limits_to_configured_hz() -> None:
    """At ``rate_hz=2``, consecutive sub-half-second calls fail."""
    clock = [0.0]
    bucket = _TokenBucket(rate_hz=2.0, clock=lambda: clock[0])
    # First call at t=0 always succeeds (bucket starts full).
    assert bucket.try_consume() is True
    # Immediately again — no regenerated tokens yet.
    assert bucket.try_consume() is False
    # 0.5 s → exactly one token regenerated.
    clock[0] = 0.5
    assert bucket.try_consume() is True
    # Another tick within the same window → fails again.
    clock[0] = 0.6
    assert bucket.try_consume() is False
    # 1.0 s later → token available.
    clock[0] = 1.0
    assert bucket.try_consume() is True


def test_token_bucket_unlimited_when_rate_is_none() -> None:
    """``rate_hz=None`` (ABORT/FAIL) bypasses rate-limiting."""
    bucket = _TokenBucket(rate_hz=None)
    assert all(bucket.try_consume() for _ in range(1000))


def test_default_rate_limit_policy_matches_adr_0018() -> None:
    """WARN/INFO rate-limited; FAIL/ABORT unlimited."""
    assert DEFAULT_RATE_LIMIT_HZ[SEVERITY_INFO] == 10.0
    assert DEFAULT_RATE_LIMIT_HZ[SEVERITY_WARN] == 10.0
    assert DEFAULT_RATE_LIMIT_HZ[SEVERITY_ABORT] is None


# ── Topic helper ────────────────────────────────────────────────────────────


def test_topic_for_namespaces_every_source() -> None:
    """Every ``FailureSource`` value lands under ``/openral/failure/<suffix>``."""
    assert {s: topic_for(s) for s in FailureSource} == {
        FailureSource.HAL: "/openral/failure/hal",
        FailureSource.SENSOR: "/openral/failure/sensor",
        FailureSource.RSKILL: "/openral/failure/rskill",
        FailureSource.SAFETY: "/openral/failure/safety",
        FailureSource.WAM: "/openral/failure/wam",
        FailureSource.CRITIC: "/openral/failure/critic",
        FailureSource.WORLD_STATE: "/openral/failure/world_state",
    }


def test_object_query_evidence_records_a_typed_object_not_in_memory_failure() -> None:
    """Issue #418: the escalation's typed failure is ``ROSObjectNotInMemory`` (§5)."""
    ev = next(v for v in _VARIANTS if isinstance(v, ObjectQueryEvidence))
    exc = ev.to_exception()
    assert isinstance(exc, ROSObjectNotInMemory) and isinstance(exc, ROSPerceptionStale)
    for part in ("orange", "target", "grasp:obj:orange_1@17/0", "60.2 s", "14 detection"):
        assert part in str(exc)
    with pytest.raises(ValidationError):
        ObjectQueryEvidence.model_validate({**ev.model_dump(), "escalation": "lost"})


def test_suppressed_summary_kind_constant() -> None:
    """Reserves uint8 254 for the summary kind."""
    assert KIND_SUPPRESSED_SUMMARY == 254
