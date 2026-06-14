"""Round-trip every :data:`openral_core.FailureEvidence` variant.

Exercises the real Pydantic discriminated union — no mocks per
CLAUDE.md §1.11. Each variant is serialized with ``model_dump_json``
and validated back through ``TypeAdapter(FailureEvidence)``; the
identity must hold and the discriminator (``kind`` field) must select
the correct concrete class.

Plus token-bucket coverage for
:class:`openral_observability.failure_bus._TokenBucket`, which is the
only piece of F3's publisher that is testable without ``rclpy``.
"""

from __future__ import annotations

import pytest
from openral_core import (
    ControllerEvidence,
    CriticEvidence,
    FailureEvidence,
    ForceEvidence,
    HumanEvidence,
    PerceptionStaleEvidence,
    ReasonerTimeoutEvidence,
    SelfVerifyEvidence,
    SuppressedSummaryEvidence,
    TimeoutEvidence,
    WamEvidence,
    WorkspaceEvidence,
)
from openral_core.exceptions import ROSConfigError
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


def test_suppressed_summary_requires_parallel_arrays() -> None:
    """Mismatched arrays in :class:`SuppressedSummaryEvidence` raise."""
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
    """ADR-0018 F3: WARN/INFO rate-limited; FAIL/ABORT unlimited."""
    assert DEFAULT_RATE_LIMIT_HZ[SEVERITY_INFO] == 10.0
    assert DEFAULT_RATE_LIMIT_HZ[SEVERITY_WARN] == 10.0
    assert DEFAULT_RATE_LIMIT_HZ[SEVERITY_ABORT] is None


# ── Topic helper ────────────────────────────────────────────────────────────


def test_topic_for_namespaces_every_source() -> None:
    """Every :class:`FailureSource` value lands under ``/openral/failure/<suffix>``."""
    assert {s: topic_for(s) for s in FailureSource} == {
        FailureSource.HAL: "/openral/failure/hal",
        FailureSource.SENSOR: "/openral/failure/sensor",
        FailureSource.RSKILL: "/openral/failure/rskill",
        FailureSource.SAFETY: "/openral/failure/safety",
        FailureSource.WAM: "/openral/failure/wam",
        FailureSource.CRITIC: "/openral/failure/critic",
    }


def test_suppressed_summary_kind_constant() -> None:
    """ADR-0018 reserves uint8 254 for the summary kind."""
    assert KIND_SUPPRESSED_SUMMARY == 254
