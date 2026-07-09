"""Unit tests for ``openral_rskill.testing.assert_within_budget``.

The helper is the single enforcement point for the CLAUDE.md §5.4 mandate
that *"latency budgets declared per skill; CI fails if exceeded on the
reference host"*.  Sim tests measure latency; this helper turns those
measurements into pytest assertions tied to the manifest contract.

Coverage
--------
- Strict pass: measurement at or below budget passes.
- Strict fail: measurement above budget raises ``LatencyBudgetExceededError``
  with stage / measured / budget fields populated.
- Tolerance: a measurement within ``tolerance_pct`` over budget passes.
- Tolerance: a measurement above ``tolerance_pct`` over budget fails.
- Optional stages (``warmup``, ``load``) silently pass when unset on the
  manifest.
- Optional stages enforce when set.
- Defensive: negative ``measured_ms`` or ``tolerance_pct`` raise
  ``ValueError`` (catches accidental sign flips in callers).
- Failure message includes ``rskill_id`` when supplied.
"""

from __future__ import annotations

import pytest
from openral_core import RSkillLatencyBudget
from openral_rskill.testing import LatencyBudgetExceededError, assert_within_budget

# ── Strict path ──────────────────────────────────────────────────────────────


def test_strict_pass_when_measurement_below_budget() -> None:
    budget = RSkillLatencyBudget(per_chunk_ms=100.0)
    assert_within_budget(measured_ms=50.0, budget=budget)


def test_strict_pass_at_exactly_budget() -> None:
    """A measurement exactly equal to the budget is still passing."""
    budget = RSkillLatencyBudget(per_chunk_ms=100.0)
    assert_within_budget(measured_ms=100.0, budget=budget)


def test_strict_fail_when_measurement_exceeds_budget() -> None:
    budget = RSkillLatencyBudget(per_chunk_ms=100.0)
    with pytest.raises(LatencyBudgetExceededError) as excinfo:
        assert_within_budget(measured_ms=150.0, budget=budget)
    err = excinfo.value
    assert err.stage == "per_chunk"
    assert err.measured_ms == 150.0
    assert err.budget_ms == 100.0


# ── Tolerance ────────────────────────────────────────────────────────────────


def test_tolerance_allows_small_overrun() -> None:
    budget = RSkillLatencyBudget(per_chunk_ms=100.0)
    # 5% over budget with 10% tolerance → pass.
    assert_within_budget(measured_ms=105.0, budget=budget, tolerance_pct=10.0)


def test_tolerance_does_not_allow_large_overrun() -> None:
    budget = RSkillLatencyBudget(per_chunk_ms=100.0)
    # 25% over with 10% tolerance → fail.
    with pytest.raises(LatencyBudgetExceededError):
        assert_within_budget(measured_ms=125.0, budget=budget, tolerance_pct=10.0)


# ── Optional stages ──────────────────────────────────────────────────────────


def test_warmup_stage_silently_passes_when_budget_unset() -> None:
    """Manifest authors who don't declare ``warmup_ms`` opt out of warmup enforcement."""
    budget = RSkillLatencyBudget(per_chunk_ms=10.0)  # no warmup_ms
    assert_within_budget(measured_ms=99_999.0, budget=budget, stage="warmup")


def test_warmup_stage_enforces_when_budget_set() -> None:
    budget = RSkillLatencyBudget(per_chunk_ms=10.0, warmup_ms=2000.0)
    assert_within_budget(measured_ms=1500.0, budget=budget, stage="warmup")
    with pytest.raises(LatencyBudgetExceededError) as excinfo:
        assert_within_budget(measured_ms=2500.0, budget=budget, stage="warmup")
    assert excinfo.value.stage == "warmup"


def test_load_stage_enforces_when_budget_set() -> None:
    budget = RSkillLatencyBudget(per_chunk_ms=10.0, load_ms=500.0)
    assert_within_budget(measured_ms=400.0, budget=budget, stage="load")
    with pytest.raises(LatencyBudgetExceededError) as excinfo:
        assert_within_budget(measured_ms=600.0, budget=budget, stage="load")
    assert excinfo.value.stage == "load"


def test_load_stage_silently_passes_when_budget_unset() -> None:
    budget = RSkillLatencyBudget(per_chunk_ms=10.0)
    assert_within_budget(measured_ms=99_999.0, budget=budget, stage="load")


# ── Defensive ────────────────────────────────────────────────────────────────


def test_negative_measurement_raises_valueerror() -> None:
    budget = RSkillLatencyBudget(per_chunk_ms=10.0)
    with pytest.raises(ValueError, match="measured_ms"):
        assert_within_budget(measured_ms=-1.0, budget=budget)


def test_negative_tolerance_raises_valueerror() -> None:
    budget = RSkillLatencyBudget(per_chunk_ms=10.0)
    with pytest.raises(ValueError, match="tolerance_pct"):
        assert_within_budget(measured_ms=5.0, budget=budget, tolerance_pct=-1.0)


# ── Failure message ──────────────────────────────────────────────────────────


def test_failure_message_includes_skill_id() -> None:
    budget = RSkillLatencyBudget(per_chunk_ms=10.0)
    with pytest.raises(LatencyBudgetExceededError) as excinfo:
        assert_within_budget(
            measured_ms=20.0,
            budget=budget,
            rskill_id="openral/rskill-pick-cube-so100",
        )
    msg = str(excinfo.value)
    assert "openral/rskill-pick-cube-so100" in msg
    assert "10.00" in msg  # budget
    assert "20.00" in msg  # measured


def test_failure_message_includes_delta_and_percentage() -> None:
    budget = RSkillLatencyBudget(per_chunk_ms=100.0)
    with pytest.raises(LatencyBudgetExceededError) as excinfo:
        assert_within_budget(measured_ms=150.0, budget=budget)
    msg = str(excinfo.value)
    # Expect a "+50.00 ms / +50.0%" -style fragment.
    assert "+50.00" in msg
    assert "%" in msg


def test_latency_budget_exceeded_is_assertionerror_subclass() -> None:
    """pytest reports AssertionError subclasses with the diff; structural pin."""
    assert issubclass(LatencyBudgetExceededError, AssertionError)
