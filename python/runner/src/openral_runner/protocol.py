"""Inference runner Protocol.

The ``InferenceRunner`` Protocol is the contract every runner shape
satisfies — today the sim path (``openral_sim``-backed shim, future PR)
and the hardware path (``openral_runner.deploy_runner.DeployRunner``,
PR F). The Protocol is intentionally narrow so subclasses or shims only
have to honour ``activate / tick / run / deactivate``; everything else
(rate-limited loop, OTel parent span, latency budget enforcement) lives
in ``openral_runner.base.InferenceRunnerBase``.

See the OpenRAL architecture docs for the full design.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from openral_core import RunResult, TickResult

__all__ = ["InferenceRunner"]


@runtime_checkable
class InferenceRunner(Protocol):
    """Structural protocol for one inference loop iteration.

    A runner ticks at ``rate_hz``. Each tick records a
    ``TickResult``; the aggregated
    ``RunResult`` is returned by ``run``.

    Attributes:
        rate_hz: Foreground tick rate. Default is 30 Hz to match the
            ``WorldStateAggregator`` publish rate.
    """

    rate_hz: float

    def activate(self) -> None:
        """Open all resources required for ticking (sensors, HAL, executor)."""
        ...

    def tick(self) -> TickResult:
        """Run one tick and return its record.

        Implementations call into the safety client *before*
        ``HAL.send_action``; the foreground tick is sync (lerobot-style)
        and rate is enforced by ``run`` via
        ``sleep_until``.
        """
        ...

    def run(self, max_ticks: int | None = None) -> RunResult:
        """Run the rate-limited loop and return the aggregate ``RunResult``.

        Args:
            max_ticks: Stop after this many ticks. ``None`` runs until the
                runner is externally deactivated (or until task termination
                in concrete subclasses).
        """
        ...

    def deactivate(self) -> None:
        """Release the resources opened by ``activate`` (idempotent)."""
        ...
