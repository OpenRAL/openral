"""SafetyClient stub.

``SafetyClient`` is the seam the inference runner calls just before
``HAL.send_action``. The real implementation is the C++ safety kernel
(a separate certifiable process, CLAUDE.md §6 Layer 6, ``packages/safety/``
planned); this module ships only the Python-side Protocol + a no-op
default, and every tick already emits a ``safety.check`` span via
``safety_span``.

Contract: ``SafetyClient.check_action`` returns ``None`` on pass
(forwarded to the HAL) or raises
``ROSSafetyViolation`` (or a subclass —
``ROSWorkspaceViolation``, ``ROSForceLimitExceeded``,
``ROSEStopRequested``) on rejection. Per CLAUDE.md §10 this is never
silently caught; the runner catches it at its supervisor boundary
(``DeployRunner``), records into
``TickResult.safety_violations``, flips
``TickResult.action_applied`` to ``False``, and propagates for E-stop.

Does NOT implement bounds-checking against a ``SafetyEnvelope`` — that
belongs in the C++ kernel; this stub keeps the seam wired for a drop-in
replacement.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import structlog
from openral_core import Action, SafetyEnvelope
from openral_observability import safety_span, semconv
from openral_observability.tracing_lttng import TP_SAFETY_VALIDATE, lttng_tracepoint

__all__ = ["NullSafetyClient", "SafetyClient"]

log = structlog.get_logger(__name__)


@runtime_checkable
class SafetyClient(Protocol):
    """Pre-action safety check the inference runner calls before HAL dispatch.

    Implementations may inspect / clamp the proposed action and decide
    whether to allow it. Returning ``None`` lets the action through;
    raising ``ROSSafetyViolation`` (or
    a subclass) rejects it.

    Attributes:
        envelope: The ``SafetyEnvelope`` against which the action
            is validated. Real implementations consult its workspace
            box, force / velocity limits, deadman state, etc.
    """

    envelope: SafetyEnvelope

    def check_action(self, action: Action) -> None:
        """Validate ``action`` and either return ``None`` or raise.

        Args:
            action: The proposed ``Action`` the runner is about to
                forward to ``HAL.send_action``.

        Raises:
            ROSSafetyViolation: When the action is rejected. The runner's
                supervisor boundary catches this and records the
                incident; the exception is not silently swallowed
                (CLAUDE.md §10).
        """
        ...


class NullSafetyClient:
    """A no-op ``SafetyClient`` that always allows.

    For digital-twin runs and pre-hardware integration tests where the C++
    safety kernel is not wired yet. Every ``check_action`` call opens a
    ``safety.check`` span at ``severity="info"``, distinguishing "skipped"
    from "approved" in traces.

    Args:
        envelope: Optional ``SafetyEnvelope`` recorded on the span
            as ``safety.envelope_max_ee_speed_m_s`` etc. Defaults to a
            stock ``SafetyEnvelope`` so the trace shape is
            consistent.

    Example:
        >>> from openral_core import Action, ControlMode, SafetyEnvelope
        >>> client = NullSafetyClient(envelope=SafetyEnvelope())
        >>> action = Action(control_mode=ControlMode.JOINT_POSITION)
        >>> client.check_action(action)  # returns None
        >>> client.envelope.max_ee_speed_m_s > 0
        True
    """

    envelope: SafetyEnvelope

    def __init__(self, envelope: SafetyEnvelope | None = None) -> None:
        """Initialise with an optional ``SafetyEnvelope``."""
        self.envelope = envelope if envelope is not None else SafetyEnvelope()

    def check_action(self, action: Action) -> None:
        """Open a ``safety.check`` span at ``info`` severity and return ``None``.

        Records ``safety.kernel`` (``"null"`` — this is the no-op client;
        the C++ kernel will emit ``"cpp"``), ``safety.control_mode`` and
        ``safety.horizon`` on the span so the trace surface — and the
        dashboard Identity card's "safety kernel" field — mirror what a
        real kernel would emit.
        """
        with (
            safety_span(
                "safety.check",
                check_name="null",
                severity="info",
                kernel=semconv.SAFETY_KERNEL_NULL,
                control_mode=action.control_mode.value,
                horizon=action.horizon,
                envelope_max_ee_speed_m_s=self.envelope.max_ee_speed_m_s,
                envelope_max_force_n=self.envelope.max_force_n,
            ),
            # LTTng entry/exit around the safety boundary.
            # Cheap when off; on, lets babeltrace2 measure validate()
            # against kernel scheduler events.
            lttng_tracepoint(
                TP_SAFETY_VALIDATE,
                check_name="null",
                control_mode=action.control_mode.value,
            ),
        ):
            log.debug(
                "safety.null_check",
                control_mode=action.control_mode.value,
                horizon=action.horizon,
            )
