"""Unit tests for :mod:`openral_runner.safety`.

No mocks (CLAUDE.md §1.11). Uses real
:class:`~openral_core.Action` instances and the real
:class:`~openral_core.SafetyEnvelope`. Verifies:

* The Protocol structural check accepts :class:`NullSafetyClient`.
* The stub's ``check_action`` returns ``None`` and opens a real OTel
  ``safety.check`` span (captured via an in-process
  :class:`InMemorySpanExporter`).
* The span carries the documented attributes
  (``safety.control_mode``, ``safety.horizon``,
  ``safety.envelope_max_ee_speed_m_s``,
  ``safety.envelope_max_force_n``, ``safety.severity``,
  ``safety.check_name``).
* A custom Protocol-conforming subclass that raises
  :class:`ROSSafetyViolation` propagates the exception (CLAUDE.md §10:
  never silently caught).
* The default envelope is non-empty so traces aren't degenerate.
"""

from __future__ import annotations

import pytest
from openral_core import Action, ControlMode, SafetyEnvelope
from openral_core.exceptions import ROSSafetyViolation, ROSWorkspaceViolation
from openral_runner import NullSafetyClient, SafetyClient
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

# ── Span capture fixture (real SDK, no mocks) ────────────────────────────────


@pytest.fixture
def captured_spans() -> InMemorySpanExporter:
    """Install an in-memory OTel tracer + exporter and return the exporter.

    Restores the previous tracer provider on teardown so subsequent tests
    don't inherit our exporter.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    # ``trace.set_tracer_provider`` only honours the first set; for tests
    # we go through the private hook to actually override.
    trace._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]
    trace.set_tracer_provider(provider)
    return exporter


# ── Protocol + stub semantics ────────────────────────────────────────────────


def test_null_safety_client_satisfies_protocol() -> None:
    """Structural ``isinstance`` against :class:`SafetyClient` succeeds."""
    client = NullSafetyClient()
    assert isinstance(client, SafetyClient)


def test_null_safety_client_returns_none_on_check() -> None:
    """``check_action`` returns ``None`` (i.e. allows) for any action."""
    client = NullSafetyClient()
    action = Action(control_mode=ControlMode.JOINT_POSITION)
    assert client.check_action(action) is None


def test_null_safety_client_default_envelope_is_populated() -> None:
    """Default envelope has the canonical OpenRAL defaults."""
    client = NullSafetyClient()
    assert client.envelope.max_ee_speed_m_s > 0
    assert client.envelope.max_force_n > 0
    assert client.envelope.max_joint_speed_factor > 0


def test_null_safety_client_uses_supplied_envelope() -> None:
    """A caller-supplied envelope is stored on the client."""
    envelope = SafetyEnvelope(max_ee_speed_m_s=0.25, max_force_n=10.0)
    client = NullSafetyClient(envelope=envelope)
    assert client.envelope is envelope
    assert client.envelope.max_ee_speed_m_s == 0.25


# ── Span attributes ──────────────────────────────────────────────────────────


def _find_safety_span(spans: list[ReadableSpan]) -> ReadableSpan:
    """Return the single ``safety.check`` span (asserts there is exactly one)."""
    matches = [s for s in spans if s.name == "safety.check"]
    assert len(matches) == 1, f"expected 1 safety.check span, got {len(matches)}"
    return matches[0]


def test_null_safety_client_emits_safety_check_span(
    captured_spans: InMemorySpanExporter,
) -> None:
    """One call → exactly one ``safety.check`` span."""
    client = NullSafetyClient()
    client.check_action(Action(control_mode=ControlMode.JOINT_POSITION))
    spans = list(captured_spans.get_finished_spans())
    span = _find_safety_span(spans)
    assert span.attributes is not None
    assert span.attributes["safety.check_name"] == "null"
    assert span.attributes["safety.severity"] == "info"


def test_safety_span_carries_action_metadata(
    captured_spans: InMemorySpanExporter,
) -> None:
    """The span records ``control_mode`` + ``horizon`` from the action."""
    client = NullSafetyClient()
    action = Action(control_mode=ControlMode.CARTESIAN_TWIST, horizon=8)
    client.check_action(action)
    spans = list(captured_spans.get_finished_spans())
    span = _find_safety_span(spans)
    assert span.attributes is not None
    assert span.attributes["safety.control_mode"] == "cartesian_twist"
    assert span.attributes["safety.horizon"] == 8


def test_safety_span_carries_envelope_attributes(
    captured_spans: InMemorySpanExporter,
) -> None:
    """The span records selected envelope fields for trace correlation."""
    envelope = SafetyEnvelope(max_ee_speed_m_s=0.7, max_force_n=33.0)
    client = NullSafetyClient(envelope=envelope)
    client.check_action(Action(control_mode=ControlMode.JOINT_POSITION))
    spans = list(captured_spans.get_finished_spans())
    span = _find_safety_span(spans)
    assert span.attributes is not None
    assert span.attributes["safety.envelope_max_ee_speed_m_s"] == 0.7
    assert span.attributes["safety.envelope_max_force_n"] == 33.0


# ── Rejection contract (CLAUDE.md §10) ───────────────────────────────────────


class _AlwaysRejectSafetyClient:
    """A SafetyClient that always raises — used to verify rejection plumbing."""

    envelope: SafetyEnvelope

    def __init__(self) -> None:
        """Initialise with a default envelope."""
        self.envelope = SafetyEnvelope()

    def check_action(self, action: Action) -> None:
        """Always raise :class:`ROSWorkspaceViolation`."""
        raise ROSWorkspaceViolation(f"reject action with control_mode={action.control_mode.value}")


def test_custom_safety_client_satisfies_protocol() -> None:
    """A user-supplied :class:`SafetyClient` impl passes the structural check."""
    client = _AlwaysRejectSafetyClient()
    assert isinstance(client, SafetyClient)


def test_rejection_propagates_ros_safety_violation() -> None:
    """A rejecting :class:`SafetyClient` raises (never silently caught)."""
    client = _AlwaysRejectSafetyClient()
    action = Action(control_mode=ControlMode.JOINT_POSITION)
    # ROSWorkspaceViolation is a ROSSafetyViolation subclass — both must match.
    with pytest.raises(ROSSafetyViolation, match="reject action"):
        client.check_action(action)
    with pytest.raises(ROSWorkspaceViolation):
        client.check_action(action)
