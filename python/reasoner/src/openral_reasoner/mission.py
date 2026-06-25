"""ADR-0073 — typed mission state for sequential multi-task deploy goals.

An operator goal may carry several ordered subtasks: the deploy CLI joins
``DeployScene.tasks`` with ``" | "``, and an operator may type ``"… , then …"``.
Today that whole string is handed to the LLM as one opaque prompt and **drained
pull-once** (``ContextRenderer.drain_prompts``), so after the first tick the
reasoner forgets the goal and never advances to the second subtask.

This module is the deterministic fix (ADR-0073 §1): the goal is parsed into an
ordered list of :class:`TaskState`, of which at most one is ``active`` (or
``verifying``) at a time. The reasoner advances the queue only when the active
task is verified complete (§2), so a multi-task goal is *sequenced* by
bookkeeping rather than by hoping the LLM remembers it. Splitting is intentionally
simple and deterministic; richer decomposition (the ``decompose-mission``
playbook, ADR-0072) can layer on top later without changing this contract.

The state is reasoner-internal (no rclpy, no Pydantic boundary) so it lives here
as plain dataclasses and is fully unit-testable; the ROS node drives the
transitions and the :class:`~openral_reasoner.context.ContextRenderer` renders the
``## MISSION`` ledger.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Literal

__all__ = [
    "DEFAULT_MAX_ATTEMPTS",
    "MissionState",
    "TaskState",
    "TaskStatus",
    "VerdictAction",
    "evaluate_task_verdict",
    "split_mission",
]

VerdictAction = Literal["complete", "abandon", "retry"]
"""What the reward gate decides for the active task after a skill returns
(ADR-0073 §2): ``complete`` (verified), ``abandon`` (ladder exhausted), or
``retry`` (try again — keep the task active)."""

DEFAULT_MAX_ATTEMPTS: int = 3
"""Default per-task attempt cap before the reward gate abandons + hands off."""


def evaluate_task_verdict(
    *,
    ok: bool,
    succeeded: bool,
    success_now: float,
    attempts: int,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> tuple[VerdictAction, str]:
    """Pure reward-gate decision for the active task (ADR-0073 §2).

    Decides whether a just-returned skill verified the active task. Inputs mirror
    the reward monitor's ``QueryTaskProgress`` response (``ok``, ``succeeded`` =
    ``success_now >= success_threshold`` over the window, ``success_now``) plus the
    task's attempt count. The window already smooths per-frame noise, so
    ``succeeded`` is the dwell-equivalent gate; no fake success is ever returned
    when the reward is unavailable (the caller handles ``ok=False`` upstream).

    Returns ``(action, verdict_text)`` where ``verdict_text`` is the short
    human-readable note recorded on the task.

    Example:
        >>> evaluate_task_verdict(ok=True, succeeded=True, success_now=0.91, attempts=1)
        ('complete', 'success=0.91')
        >>> evaluate_task_verdict(ok=True, succeeded=False, success_now=0.40, attempts=1)[0]
        'retry'
        >>> evaluate_task_verdict(ok=True, succeeded=False, success_now=0.40, attempts=3)[0]
        'abandon'
    """
    if ok and succeeded:
        return "complete", f"success={success_now:.2f}"
    if attempts >= max_attempts:
        return "abandon", f"unverified after {attempts} attempt(s) (success={success_now:.2f})"
    return "retry", f"not verified (success={success_now:.2f}), attempt {attempts}/{max_attempts}"

TaskStatus = Literal["pending", "active", "verifying", "done", "abandoned"]
"""Lifecycle of a single subtask. ``pending → active → verifying → done |
abandoned``. ``abandoned`` and ``done`` are terminal — a task is never silently
re-queued."""

_ACTIVE_STATES: frozenset[TaskStatus] = frozenset({"active", "verifying"})
_TERMINAL_STATES: frozenset[TaskStatus] = frozenset({"done", "abandoned"})

# Deterministic subtask separators:
#   `` | `` — what ``openral deploy sim`` joins ``DeployScene.tasks`` with.
#   ``, then`` / `` then `` — the natural-language form an operator types.
# Intentionally narrow: we do NOT split on bare ``and`` (a single action often
# reads "pick X and place it"); only explicit sequencing markers separate tasks.
_SEPARATORS = re.compile(r"\s*\|\s*|\s*,?\s+then\s+", flags=re.IGNORECASE)


def split_mission(text: str) -> list[str]:
    """Split an operator goal into ordered subtask strings.

    Splits on `` | `` and ``, then`` / `` then `` (case-insensitive), trims each
    fragment, and drops empties. A single-task goal returns a one-element list; an
    empty or whitespace-only string returns ``[]``.

    Example:
        >>> split_mission("stack the bowls in the drawer, then put the plate on the box")
        ['stack the bowls in the drawer', 'put the plate on the box']
        >>> split_mission("pick the bowl | place the butter")
        ['pick the bowl', 'place the butter']
        >>> split_mission("just one task")
        ['just one task']
        >>> split_mission("   ")
        []
    """
    return [part for raw in _SEPARATORS.split(text) if (part := raw.strip())]


@dataclasses.dataclass(slots=True)
class TaskState:
    """One ordered subtask and its lifecycle (ADR-0073 §1).

    Attributes:
        task_id: Stable id within the mission (``"t1"``, ``"t2"``, …).
        text: The subtask instruction handed to the reasoner as the active goal.
        status: Lifecycle position (:data:`TaskStatus`).
        attempts: Number of ``execute_rskill`` dispatches made for this task —
            the loop guard for the replanning ladder.
        last_rskill_id: rSkill id of the most recent attempt, or ``None``.
        last_trace_id: Trace id of the most recent attempt, or ``None``.
        last_verdict: Short human-readable verdict of the last verification
            (e.g. ``"success=0.91"``, ``"stalled@0.73"``, ``"unverified"``).
    """

    task_id: str
    text: str
    status: TaskStatus = "pending"
    attempts: int = 0
    last_rskill_id: str | None = None
    last_trace_id: str | None = None
    last_verdict: str | None = None


class MissionState:
    """Ordered task queue with at most one active task (ADR-0073 §1).

    Owns the deterministic sequencing the LLM is no longer trusted to do: the
    active task is the only goal injected each tick; the queue advances only when
    the node verifies the active task complete (or abandons it after the ladder is
    exhausted). All mutators return the newly-active :class:`TaskState` (or
    ``None`` when the mission is finished) so the caller can re-inject the next
    goal and wake the reasoner.

    Example:
        >>> m = MissionState.from_prompt("pick the bowl | place the butter")
        >>> m.active().text
        'pick the bowl'
        >>> nxt = m.complete_active("success=0.9")
        >>> nxt.text
        'place the butter'
        >>> m.complete_active("success=0.9") is None
        True
        >>> m.is_complete()
        True
    """

    def __init__(self, tasks: list[str]) -> None:
        """Build the queue from ordered subtask strings; activate the first."""
        self._tasks: list[TaskState] = [
            TaskState(task_id=f"t{i + 1}", text=text) for i, text in enumerate(tasks)
        ]
        if self._tasks:
            self._tasks[0].status = "active"

    @classmethod
    def from_prompt(cls, text: str) -> MissionState:
        """Build a mission by :func:`split_mission`-ing an operator goal string."""
        return cls(split_mission(text))

    # ── readers ──────────────────────────────────────────────────────────────

    @property
    def tasks(self) -> tuple[TaskState, ...]:
        """Snapshot of every task, in order."""
        return tuple(self._tasks)

    def __len__(self) -> int:
        """Number of tasks in the mission queue."""
        return len(self._tasks)

    def active(self) -> TaskState | None:
        """The currently ``active`` or ``verifying`` task, or ``None``."""
        return next((t for t in self._tasks if t.status in _ACTIVE_STATES), None)

    def is_empty(self) -> bool:
        """True when the mission carries no tasks (empty/whitespace prompt)."""
        return not self._tasks

    def is_complete(self) -> bool:
        """True when every task is terminal (``done``/``abandoned``).

        An empty mission is never "complete" — there was nothing to do, which is
        a different condition the caller handles explicitly.
        """
        return bool(self._tasks) and all(t.status in _TERMINAL_STATES for t in self._tasks)

    # ── mutators (each returns the new active task, or None when finished) ────

    def record_attempt(self, *, rskill_id: str | None, trace_id: str | None = None) -> None:
        """Note a dispatch against the active task (increments ``attempts``)."""
        task = self.active()
        if task is None:
            return
        task.attempts += 1
        task.last_rskill_id = rskill_id
        if trace_id is not None:
            task.last_trace_id = trace_id

    def mark_verifying(self) -> None:
        """Move the active task ``active → verifying`` (a skill returned; gating)."""
        task = self.active()
        if task is not None and task.status == "active":
            task.status = "verifying"

    def complete_active(self, verdict: str) -> TaskState | None:
        """Mark the active task ``done`` and activate the next pending task.

        Returns the newly-active task, or ``None`` when the mission is finished.
        """
        return self._terminate_active("done", verdict)

    def abandon_active(self, reason: str) -> TaskState | None:
        """Mark the active task ``abandoned`` (ladder exhausted / unverifiable).

        Returns the newly-active task, or ``None`` when the mission is finished.
        """
        return self._terminate_active("abandoned", reason)

    def _terminate_active(self, status: TaskStatus, verdict: str) -> TaskState | None:
        task = self.active()
        if task is None:
            return None
        task.status = status
        task.last_verdict = verdict
        nxt = next((t for t in self._tasks if t.status == "pending"), None)
        if nxt is not None:
            nxt.status = "active"
        return nxt

    # ── rendering ────────────────────────────────────────────────────────────

    def render(self) -> str:
        """Compact ``## MISSION`` ledger: done tasks, the active task, pending count.

        Deterministic, one line per terminal/active task plus a pending tail, so a
        multi-task goal stays visible every tick even after the operator prompt is
        drained pull-once.
        """
        if not self._tasks:
            return "(no mission)"
        lines: list[str] = []
        pending = 0
        for task in self._tasks:
            if task.status == "pending":
                pending += 1
                continue
            marker = {
                "active": "▶",
                "verifying": "?",
                "done": "✓",
                "abandoned": "✗",
            }[task.status]
            verdict = f" [{task.last_verdict}]" if task.last_verdict else ""
            attempts = f" attempts={task.attempts}" if task.attempts else ""
            lines.append(f"{marker} {task.task_id}: {task.text}{attempts}{verdict}")
        if pending:
            lines.append(f"… {pending} pending task(s)")
        return "\n".join(lines)
