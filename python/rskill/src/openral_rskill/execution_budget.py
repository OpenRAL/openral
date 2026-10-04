"""Execution budgets measured on the graph clock, with a wall-clock backstop.

An rSkill's ``latency_budget.max_execution_s`` (and a wrapped server's result
deadline) means "how long may this skill run on the robot's own clock". Under
``openral deploy sim`` the whole graph runs with ``use_sim_time=true`` and
Nav2 / MoveIt / the controllers already time themselves in sim seconds; a
budget measured on ``time.monotonic()`` instead fires while a slow simulator
is still mid-motion (a rendering graph at 0.09x real time lost a 300 s Nav2
budget after 28 s of sim time). ``ExecutionBudget`` measures the budget on
the graph clock when one is given, and keeps two wall-clock guards so a goal
still ends when ``/clock`` stops (sim pause, HAL crash, e-stop):

* ``clock_stalled`` — the graph clock has not advanced for ``stall_s`` of
  wall time (or never started: a sim node reads ``0`` until the first
  ``/clock`` message, so the budget does not start counting until it does).
* ``deadline_exceeded`` on the wall backstop — wall elapsed exceeded
  ``wall_cap_factor x budget_s``.

Without a graph clock (``use_sim_time=false``: every real robot) the budget is
``time.monotonic()`` exactly as before — never the ROS system clock, which
NTP can step.

Example:
    >>> ticks = iter([0.0, 0.0, 100.0, 100.4])
    >>> budget = ExecutionBudget(1.0, graph_now=lambda: next(ticks), stall_s=60.0)
    >>> budget.check() is None  # graph clock not started yet: nothing lapses
    True
    >>> budget.elapsed_s()  # first non-zero reading is the start
    0.0
    >>> round(budget.elapsed_s(), 1)  # 0.4 s of graph time later
    0.4
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

__all__ = [
    "DEFAULT_CLOCK_STALL_S",
    "DEFAULT_WALL_CAP_FACTOR",
    "BudgetMiss",
    "ExecutionBudget",
    "graph_clock_fn",
]

#: Wall-clock backstop: a graph-clock budget may stretch to at most this many
#: times its value in wall seconds (a 45 s budget at 0.1x real time is 450 s).
DEFAULT_WALL_CAP_FACTOR = 10.0
#: Wall seconds the graph clock may stand still before the goal is failed.
#: Comfortably above any /clock publish period; a paused sim trips it.
DEFAULT_CLOCK_STALL_S = 10.0


@dataclass(frozen=True)
class BudgetMiss:
    """Why an ``ExecutionBudget`` lapsed; ``kind`` is the ``failure_reason`` prefix."""

    kind: Literal["deadline_exceeded", "clock_stalled"]
    #: Elapsed seconds on the budget's clock (graph or wall) when it lapsed.
    elapsed_s: float
    detail: str


class ExecutionBudget:
    """One goal's execution budget; ``check()`` says whether and why it lapsed.

    Args:
        budget_s: The budget; ``<= 0`` disables every check.
        graph_now: Zero-arg callable returning graph-clock seconds (see
            ``graph_clock_fn``), or ``None`` to measure on ``wall_now``.
        wall_cap_factor: Wall backstop multiplier; ``<= 0`` disables it.
        stall_s: Wall seconds without graph-clock progress before
            ``clock_stalled``; ``<= 0`` disables it.
        wall_now: Monotonic wall clock (injectable for tests).
    """

    def __init__(
        self,
        budget_s: float,
        *,
        graph_now: Callable[[], float] | None = None,
        wall_cap_factor: float = DEFAULT_WALL_CAP_FACTOR,
        stall_s: float = DEFAULT_CLOCK_STALL_S,
        wall_now: Callable[[], float] = time.monotonic,
    ) -> None:
        """Start the budget now (see the class docstring for the arguments)."""
        self.budget_s = float(budget_s)
        self._graph_now = graph_now
        self._wall_now = wall_now
        self._wall_cap_factor = float(wall_cap_factor)
        self._stall_s = float(stall_s)
        self._wall_start = wall_now()
        # ``None`` until the graph clock reads > 0 — a sim node sits at 0 before
        # its first /clock message, and anchoring there would make that message
        # a jump that lapses the budget instantly.
        self._graph_start: float | None = None
        self._graph_last = 0.0
        self._graph_last_advance_wall = self._wall_start
        if graph_now is not None:
            self._observe_graph()

    @property
    def on_graph_clock(self) -> bool:
        """``True`` when the budget is measured on the graph clock."""
        return self._graph_now is not None

    def _observe_graph(self) -> None:
        assert self._graph_now is not None
        now = self._graph_now()
        if now > self._graph_last:
            self._graph_last = now
            self._graph_last_advance_wall = self._wall_now()
        if self._graph_start is None and now > 0.0:
            self._graph_start = now

    def elapsed_s(self) -> float:
        """Seconds elapsed on the budget's clock (0 while a graph clock has not started)."""
        if self._graph_now is None:
            return self._wall_now() - self._wall_start
        self._observe_graph()
        return 0.0 if self._graph_start is None else self._graph_last - self._graph_start

    def progress(self) -> float:
        """``elapsed / budget`` clamped to ``[0, 1]``; ``0`` when the budget is disabled."""
        return min(self.elapsed_s() / self.budget_s, 1.0) if self.budget_s > 0.0 else 0.0

    def check(self) -> BudgetMiss | None:
        """Return the miss once the budget (or a wall backstop) has lapsed, else ``None``."""
        if self.budget_s <= 0.0:
            return None
        elapsed = self.elapsed_s()
        if elapsed > self.budget_s:
            clock = "graph clock" if self.on_graph_clock else "wall clock"
            return BudgetMiss(
                "deadline_exceeded",
                elapsed,
                f"elapsed={elapsed:.1f}s budget={self.budget_s:.1f}s ({clock})",
            )
        if self._graph_now is None:
            return None
        wall = self._wall_now()
        stalled_for = wall - self._graph_last_advance_wall
        if self._stall_s > 0.0 and stalled_for > self._stall_s:
            state = "never started" if self._graph_start is None else "has not advanced"
            return BudgetMiss(
                "clock_stalled",
                elapsed,
                f"graph clock {state} for {stalled_for:.1f}s of wall time "
                f"(graph elapsed={elapsed:.1f}s budget={self.budget_s:.1f}s)",
            )
        wall_elapsed = wall - self._wall_start
        if self._wall_cap_factor > 0.0 and wall_elapsed > self._wall_cap_factor * self.budget_s:
            return BudgetMiss(
                "deadline_exceeded",
                elapsed,
                f"wall elapsed={wall_elapsed:.1f}s exceeds {self._wall_cap_factor:g}x "
                f"budget={self.budget_s:.1f}s (graph elapsed={elapsed:.1f}s)",
            )
        return None


def graph_clock_fn(node: Any) -> Callable[[], float] | None:  # noqa: ANN401  # reason: rclpy Node is untyped and not importable without a sourced ROS 2 workspace
    """The node's clock as a seconds callable when it runs on sim time, else ``None``.

    ``None`` (``use_sim_time`` false or undeclared, or no node) tells
    ``ExecutionBudget`` to use ``time.monotonic()`` — a real robot's ROS clock is
    the NTP-stepped system clock, which must not time a budget.
    """
    if node is None or not callable(getattr(node, "get_parameter", None)):
        return None
    # rclpy is importable only under a sourced ROS 2 workspace; callers without one pass node=None.
    from rclpy.exceptions import ParameterNotDeclaredException  # noqa: PLC0415  # reason: see above

    try:
        sim = bool(node.get_parameter("use_sim_time").value)
    except ParameterNotDeclaredException:
        return None
    if not sim:
        return None
    clock = node.get_clock()
    return lambda: float(clock.now().nanoseconds) * 1e-9
