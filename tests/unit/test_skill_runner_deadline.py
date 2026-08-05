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


def test_pace_tick_enforces_configured_cadence() -> None:
    """N paced ticks take ~N x period, not N x (work + period).

    Regression for the goal loop's unconditional ``time.sleep(period_s)``:
    with per-tick work w the old cadence was ``w + period`` (measured live:
    ~16 Hz at a configured 30 Hz on the SO-101 eraser deploy). Deadline
    pacing absorbs the work into the period. Bounds are 2x-generous to stay
    CI-safe.
    """
    mod = _load_skill_runner_module()
    period_s = 0.02
    n_ticks = 5
    work_s = 0.01  # half the period — must be absorbed, not added

    t0 = time.perf_counter()
    deadline = t0
    for _ in range(n_ticks):
        time.sleep(work_s)  # simulated tick work
        deadline = mod._pace_tick(deadline, period_s)
    elapsed = time.perf_counter() - t0

    assert elapsed >= n_ticks * period_s * 0.9, f"paced too fast: {elapsed:.3f}s"
    # Old behaviour: n * (work + period) = 0.15 s. Require clearly better.
    assert elapsed < n_ticks * (work_s + period_s) * 0.9, (
        f"work was added to the period instead of absorbed: {elapsed:.3f}s"
    )


def test_pace_tick_overrun_reanchors_without_burst() -> None:
    """A tick that blew its deadline returns immediately, re-anchored to now.

    No sleep (the loop is already late) and no stale deadline carried
    forward (which would make every subsequent sleep a zero — a burst of
    catch-up ticks slamming the HAL).
    """
    mod = _load_skill_runner_module()
    t0 = time.perf_counter()
    stale_deadline = t0 - 1.0  # long-lapsed (e.g. chunk-boundary inference)

    new_deadline = mod._pace_tick(stale_deadline, 0.02)
    elapsed = time.perf_counter() - t0

    assert elapsed < 0.005, f"overrun tick must not sleep, waited {elapsed:.3f}s"
    assert new_deadline >= t0, "deadline must re-anchor to now, not stay stale"
