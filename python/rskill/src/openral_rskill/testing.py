"""Test helpers for skill latency-budget enforcement (CLAUDE.md §5.4).

CLAUDE.md §5.4 requires *"latency budgets declared per skill; CI fails if
exceeded on the reference host"*.  Sim tests already *measure* warm-step and
cached-pop latency (see ``tests/sim/test_smolvla_so100.py`` and friends),
but until now they only printed measurements — there was no helper that
asserted the manifest contract.  This module supplies that helper.

Public surface
--------------
- :func:`assert_within_budget` — assert that a measured per-step latency is
  within the manifest's :class:`openral_core.RSkillLatencyBudget`.
- :class:`LatencyBudgetExceededError` — raised on overrun.

The helper is intentionally a plain ``AssertionError`` subclass (not
``ROSRuntimeError`` / ``ROSInferenceTimeout``).  At test time we want a
crisp pytest failure with the budget delta, not the operational exception
that the safety supervisor would catch.

Example:
    >>> from openral_core import RSkillLatencyBudget
    >>> from openral_rskill.testing import assert_within_budget
    >>> budget = RSkillLatencyBudget(per_chunk_ms=100.0)
    >>> assert_within_budget(measured_ms=50.0, budget=budget)
    >>> assert_within_budget(measured_ms=200.0, budget=budget)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    openral_rskill.testing.LatencyBudgetExceededError: ...
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from openral_core import RSkillLatencyBudget

__all__ = ["LatencyBudgetExceededError", "assert_within_budget"]


_Stage = Literal["per_chunk", "warmup", "load"]


class LatencyBudgetExceededError(AssertionError):
    """Raised by :func:`assert_within_budget` when a measurement exceeds the budget.

    Inherits from :class:`AssertionError` so pytest reports it as a normal
    test failure with the diff in the failure message.

    Attributes:
        stage: Which budget stage was breached (``"per_chunk"``, ``"warmup"``,
            or ``"load"``).
        measured_ms: The actual measured latency in milliseconds.
        budget_ms: The manifest's budget for this stage in milliseconds.
        rskill_id: Optional human-readable identifier surfaced in the message.
    """

    def __init__(
        self,
        *,
        stage: _Stage,
        measured_ms: float,
        budget_ms: float,
        rskill_id: str | None = None,
    ) -> None:
        """Build the failure message and store the diagnostic fields on the instance."""
        delta = measured_ms - budget_ms
        prefix = f"[{rskill_id}] " if rskill_id else ""
        msg = (
            f"{prefix}{stage} latency {measured_ms:.2f} ms exceeds manifest budget "
            f"{budget_ms:.2f} ms (over by {delta:+.2f} ms / {delta / budget_ms:+.1%})"
        )
        super().__init__(msg)
        self.stage = stage
        self.measured_ms = measured_ms
        self.budget_ms = budget_ms
        self.rskill_id = rskill_id


def assert_within_budget(
    *,
    measured_ms: float,
    budget: RSkillLatencyBudget,
    stage: _Stage = "per_chunk",
    rskill_id: str | None = None,
    tolerance_pct: float = 0.0,
) -> None:
    """Assert ``measured_ms`` does not exceed the manifest's budget for ``stage``.

    Args:
        measured_ms: The measured latency in milliseconds.  Must be ≥ 0.
        budget: The manifest's latency budget.
        stage: Which field of ``budget`` to compare against.  Defaults to
            ``"per_chunk"`` since that's the only required field.
        rskill_id: Optional skill identifier surfaced in the failure message
            (e.g. the ``RSkillManifest.name``).
        tolerance_pct: Soft margin in percent.  ``5.0`` means a measurement
            up to 5% over the budget is still considered passing.  Defaults
            to ``0.0`` (strict).  CLAUDE.md operating principle 8 ("performance
            is a feature") argues against headroom, but we expose this so
            individual tests can opt in to noise margin where genuinely
            unavoidable (e.g. cold-start variance on shared CI hardware).

    Raises:
        LatencyBudgetExceededError: If ``measured_ms`` exceeds
            ``budget.<stage>_ms * (1 + tolerance_pct / 100)``.
        ValueError: If ``measured_ms`` is negative or ``tolerance_pct`` is
            negative.

    Example:
        >>> from openral_core import RSkillLatencyBudget
        >>> b = RSkillLatencyBudget(per_chunk_ms=100.0, warmup_ms=2000.0)
        >>> assert_within_budget(measured_ms=80.0, budget=b)
        >>> assert_within_budget(measured_ms=1500.0, budget=b, stage="warmup")
        >>> assert_within_budget(measured_ms=105.0, budget=b, tolerance_pct=10.0)
    """
    if measured_ms < 0:
        raise ValueError(f"measured_ms must be >= 0, got {measured_ms!r}")
    if tolerance_pct < 0:
        raise ValueError(f"tolerance_pct must be >= 0, got {tolerance_pct!r}")

    budget_ms: float | None
    if stage == "per_chunk":
        budget_ms = budget.per_chunk_ms
    elif stage == "warmup":
        budget_ms = budget.warmup_ms
    elif stage == "load":
        budget_ms = budget.load_ms
    else:  # pragma: no cover  # reason: Literal exhausts the enum, but stay defensive
        raise ValueError(f"unknown latency stage {stage!r}")

    if budget_ms is None:
        # Optional stages (warmup_ms / load_ms) may be unset on the manifest.
        # Treat absent as "no enforcement" — silent pass is the right default,
        # since per-skill authors opt in by declaring the field.
        return

    threshold = budget_ms * (1.0 + tolerance_pct / 100.0)
    if measured_ms > threshold:
        raise LatencyBudgetExceededError(
            stage=stage,
            measured_ms=measured_ms,
            budget_ms=budget_ms,
            rskill_id=rskill_id,
        )
