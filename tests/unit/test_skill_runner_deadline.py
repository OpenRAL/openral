"""Unit tests for the rSkill runner's execution-budget enforcement.

Regression cover for a live SO-101 bench observation: a goal whose first
inference took **144.5 s** against a resolved **45 s** budget still closed
``SUCCEEDED``. ``_run_until_done_or_deadline`` exited via a bare ``return``
on *every* path — completion, cancel, and deadline alike — so the caller
could not tell them apart and fell through to ``result.success = True``.
That both misreports the run and denies the reasoner's replanning ladder
the signal it needs (CLAUDE.md §3, "deadline fallback mandatory").

These tests pin the budget predicate: whether the loop must stop, and the
elapsed time recorded for the abort reason.

**No ROS context is created.** ``_deadline_lapsed`` needs exactly two
things from its instance — a real ``rclpy`` logger and the
``_last_deadline_elapsed_s`` slot — so the real function is bound to a
minimal holder carrying both. An earlier version of this file built a real
``RskillRunnerNode``; its module-scoped ``rclpy.shutdown()`` tore the
global context out from under every later rclpy test in ``tests/unit`` and
took the suite from 2m18s to >15m. The logger below is genuine
(``rclpy.logging.get_logger``), so the warning path is still exercised —
this is a fixture, not a mock of the code under test (CLAUDE.md §1.11).
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path
from types import MethodType, ModuleType
from typing import Any

import pytest


def _rclpy_available() -> bool:
    return importlib.util.find_spec("rclpy") is not None


def _load_skill_runner_module() -> ModuleType:
    """Load ``rskill_runner_node`` bypassing the ROS2-gated package init."""
    repo_root = Path(__file__).resolve().parents[2]
    src = repo_root / "packages/openral_rskill_ros/openral_rskill_ros/rskill_runner_node.py"
    spec = importlib.util.spec_from_file_location("_test_rskill_runner_deadline", src)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pytestmark = pytest.mark.skipif(
    not _rclpy_available(),
    reason="rclpy not on PYTHONPATH; source a ROS 2 install to run runner-node tests",
)


class _BudgetHolder:
    """Minimal real carrier for the two attributes ``_deadline_lapsed`` touches."""

    def __init__(self) -> None:
        import rclpy.logging

        self._logger = rclpy.logging.get_logger("test_skill_runner_deadline")
        self._last_deadline_elapsed_s: float | None = None

    def get_logger(self) -> Any:  # reason: rclpy logger is untyped
        return self._logger


@pytest.fixture
def budget() -> Any:  # reason: dynamically bound method holder
    """The real ``_deadline_lapsed`` bound to a minimal carrier — no ROS context."""
    mod = _load_skill_runner_module()
    holder = _BudgetHolder()
    holder.deadline_lapsed = MethodType(  # type: ignore[attr-defined]
        mod.RskillRunnerNode._deadline_lapsed, holder
    )
    return holder


def test_budget_not_lapsed_returns_false(budget: Any) -> None:
    """Inside the budget the loop keeps stepping and nothing is recorded."""
    assert budget.deadline_lapsed(time.monotonic(), 45.0, 0) is False
    assert budget._last_deadline_elapsed_s is None


def test_zero_budget_disables_the_check(budget: Any) -> None:
    """``budget_s <= 0`` means "no deadline" — never report a miss.

    The resolver maps a goal's ``deadline_s=0`` to a real budget before the
    loop runs, so a zero reaching this predicate means the caller
    deliberately disabled it: an unbounded run is correct, not a miss.
    """
    assert budget.deadline_lapsed(time.monotonic() - 10_000.0, 0.0, 7) is False
    assert budget._last_deadline_elapsed_s is None


def test_lapsed_budget_reports_true_and_records_true_elapsed(budget: Any) -> None:
    """A lapsed budget stops the loop and records the REAL elapsed time.

    The recorded value must be the actual overrun (~144 s), not the budget
    (45 s) — quoting the limit back would hide exactly the number that made
    this bug visible.
    """
    start = time.monotonic() - 144.5

    assert budget.deadline_lapsed(start, 45.0, 3) is True

    recorded = budget._last_deadline_elapsed_s
    assert recorded is not None
    assert recorded == pytest.approx(144.5, abs=1.0), (
        f"expected the true elapsed (~144.5 s), got {recorded}"
    )
    assert recorded > 45.0, "recorded the budget instead of the overrun"


def test_boundary_just_inside_budget_is_not_a_miss(budget: Any) -> None:
    """Strictly-greater comparison: at the budget the loop may still step."""
    assert budget.deadline_lapsed(time.monotonic() - 1.0, 45.0, 0) is False
    assert budget._last_deadline_elapsed_s is None


# ── _pace_tick — absolute-deadline loop cadence ──────────────────────────────


def test_pace_tick_sleeps_to_absolute_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_skill_runner_module()
    import openral_runner.clock

    sleeps: list[float] = []
    monkeypatch.setattr(mod.time, "perf_counter", lambda: 10.01)
    monkeypatch.setattr(openral_runner.clock, "sleep_until", sleeps.append)

    assert mod._pace_tick(10.0, 0.02) == pytest.approx(10.02)
    assert sleeps == [pytest.approx(10.02)]


def test_pace_tick_overrun_reanchors_without_burst(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_skill_runner_module()
    import openral_runner.clock

    sleeps: list[float] = []
    monkeypatch.setattr(mod.time, "perf_counter", lambda: 11.0)
    monkeypatch.setattr(openral_runner.clock, "sleep_until", sleeps.append)

    assert mod._pace_tick(10.0, 0.02) == 11.0
    assert sleeps == []
