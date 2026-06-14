"""Inference runner Protocol.

The :class:`InferenceRunner` Protocol is the contract every runner shape
satisfies — today the sim path (``openral_sim``-backed shim, future PR)
and the hardware path (``openral_runner.hardware.HardwareRunner``,
PR F). The Protocol is intentionally narrow so subclasses or shims only
have to honour ``activate / tick / run / deactivate``; everything else
(rate-limited loop, OTel parent span, latency budget enforcement) lives
in :class:`openral_runner.base.InferenceRunnerBase`.

See ``docs/adr/0010-inference-runner.md`` for the full design.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from openral_core import RunResult, TickResult

__all__ = ["InferenceRunner"]


@runtime_checkable
class InferenceRunner(Protocol):
    """Structural protocol for one inference loop iteration.

    A runner ticks at :attr:`rate_hz`. Each tick records a
    :class:`~openral_core.TickResult`; the aggregated
    :class:`~openral_core.RunResult` is returned by :meth:`run`.

    Attributes:
        rate_hz: Foreground tick rate. Default is 30 Hz to match the
            :class:`~openral_world_state.WorldStateAggregator` publish rate.
    """

    rate_hz: float

    def activate(self) -> None:
        """Open all resources required for ticking (sensors, HAL, executor)."""
        ...

    def tick(self) -> TickResult:
        """Run one tick and return its record.

        Implementations call into the safety client *before*
        :meth:`HAL.send_action`; the foreground tick is sync (lerobot-style)
        and rate is enforced by :meth:`run` via
        :func:`~openral_runner.clock.sleep_until`.
        """
        ...

    def run(self, max_ticks: int | None = None) -> RunResult:
        """Run the rate-limited loop and return the aggregate :class:`RunResult`.

        Args:
            max_ticks: Stop after this many ticks. ``None`` runs until the
                runner is externally deactivated (or until task termination
                in concrete subclasses).
        """
        ...

    def deactivate(self) -> None:
        """Release the resources opened by :meth:`activate` (idempotent)."""
        ...
