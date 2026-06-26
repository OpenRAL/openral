"""ADR-0073 — typed mission state for sequential multi-task deploy goals.

An operator goal may carry several ordered subtasks supplied via ``--initial-task``
(or a live ``/openral/prompt``). Prior to the ADR-0073 amendment the deploy CLI
joined ``DeployScene.tasks`` with ``" | "`` into a single opaque prompt that was
**drained pull-once** (``ContextRenderer.drain_prompts``), so the reasoner forgot
the goal after the first tick and never advanced to subsequent subtasks (removed).

This module is the deterministic fix (ADR-0073 §1): the goal is parsed into an
ordered list of :class:`TaskState`, of which at most one is ``active`` (or
``verifying``) at a time. The reasoner advances the queue only when the active
task is verified complete (§2), so a multi-task goal is *sequenced* by
bookkeeping rather than by hoping the LLM remembers it. Splitting is intentionally
simple and deterministic; richer decomposition (the ``decompose-mission``
playbook, ADR-0072) layers on top via :meth:`MissionState.subdivide_active`.

The ADR-0073 amendment (#123) adds **hierarchical subdivision on replan**: when
the active task is blocked (reward gate ``abandon``, ladder exhausted) the
reasoner may decompose it into finer subtasks instead of only handing off. The
data model stays **flat** — :meth:`MissionState.subdivide_active` *splices* the
blocked task in place with its children (``t2 → t2.1, t2.2``), so the ``##
MISSION`` ledger and the dashboard (which rebuild from the flat task list each
tick) need no change. :attr:`TaskState.depth` bounds re-decomposition
(:data:`DEFAULT_MAX_SUBDIVIDE_DEPTH`) so a perpetually-blocked task terminates in
``human-handoff`` rather than subdividing forever.

The state is reasoner-internal (no rclpy, no Pydantic boundary) so it lives here
as plain dataclasses and is fully unit-testable; the ROS node drives the
transitions and the :class:`~openral_reasoner.context.ContextRenderer` renders the
``## MISSION`` ledger.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

__all__ = [
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_MAX_SUBDIVIDE_DEPTH",
    "MissionState",
    "TaskState",
    "TaskStatus",
    "VerdictAction",
    "evaluate_task_verdict",
]

VerdictAction = Literal["complete", "abandon", "retry"]
"""What the reward gate decides for the active task after a skill returns
(ADR-0073 §2): ``complete`` (verified), ``abandon`` (ladder exhausted), or
``retry`` (try again — keep the task active)."""

DEFAULT_MAX_ATTEMPTS: int = 3
"""Default per-task attempt cap before the reward gate abandons + hands off."""

DEFAULT_MAX_SUBDIVIDE_DEPTH: int = 2
"""Max re-decomposition depth (ADR-0073 amendment / #123).

A task at the queue root has ``depth == 0``; its children from one
:meth:`MissionState.subdivide_active` are ``depth == 1``; their children
``depth == 2``. Once a blocked task is already at this depth, subdivision is
refused (``subdivide_active`` returns ``None``) and the caller falls back to
``human-handoff`` — bounding the ladder so a perpetually-blocked task cannot
subdivide forever."""


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
        depth: Re-decomposition depth (ADR-0073 amendment / #123). A task split
            from the operator goal is ``0``; a child spliced in by
            :meth:`MissionState.subdivide_active` is ``parent.depth + 1``. Bounds
            the subdivision ladder against :data:`DEFAULT_MAX_SUBDIVIDE_DEPTH`.
    """

    task_id: str
    text: str
    status: TaskStatus = "pending"
    attempts: int = 0
    last_rskill_id: str | None = None
    last_trace_id: str | None = None
    last_verdict: str | None = None
    depth: int = 0


class MissionState:
    """Ordered task queue with at most one active task (ADR-0073 §1).

    Owns the deterministic sequencing the LLM is no longer trusted to do: the
    active task is the only goal injected each tick; the queue advances only when
    the node verifies the active task complete (or abandons it after the ladder is
    exhausted). All mutators return the newly-active :class:`TaskState` (or
    ``None`` when the mission is finished) so the caller can re-inject the next
    goal and wake the reasoner.

    Example:
        >>> m = MissionState(["pick the bowl", "place the butter"])
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
        """Seed a mission from an operator goal as a SINGLE task.

        ADR-0073 amendment — the regex ``split_mission`` floor is removed: the
        operator goal is one task and the LLM owns decomposition via
        ``decompose_mission``. A blank goal yields an empty mission.
        """
        goal = text.strip()
        return cls([goal] if goal else [])

    # ── readers ──────────────────────────────────────────────────────────────

    @property
    def tasks(self) -> tuple[TaskState, ...]:
        """Snapshot of every task, in order."""
        return tuple(self._tasks)

    def __len__(self) -> int:
        """Number of tasks in the queue (terminal, active, and pending)."""
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

    def has_started(self) -> bool:
        """True once any task has terminated or the active task has been attempted.

        Used to keep a ``decompose_mission`` *populate* (whole-queue replace, #123)
        safe: the LLM may refine the single-task seed on the first tick (nothing
        started yet), but a wholesale replace mid-mission would
        discard `done`/`abandoned` progress — so the node only honours populate
        before the mission has started.

        Example:
            >>> m = MissionState.from_prompt("put the bowl on the plate")
            >>> m.has_started()
            False
            >>> m.record_attempt(rskill_id="x")
            >>> m.has_started()
            True
        """
        return any(
            t.status in _TERMINAL_STATES or (t.status in _ACTIVE_STATES and t.attempts > 0)
            for t in self._tasks
        )

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

    def rearm_active(self) -> TaskState | None:
        """Move the active task ``verifying → active`` to re-offer a fresh decision.

        The reward gate moves the active task to ``verifying`` while it queries the
        monitor (:meth:`mark_verifying`). When the node offers subdivision on a
        blocked task (#123) instead of abandoning it, it calls this so the normal
        dispatch / ``subdivide_active`` cycle resumes from ``active``. No-op
        (returns the current active task or ``None``) when nothing is ``verifying``.

        Example:
            >>> m = MissionState.from_prompt("pick the milk")
            >>> m.mark_verifying()
            >>> m.active().status
            'verifying'
            >>> m.rearm_active().status
            'active'
        """
        task = self.active()
        if task is not None and task.status == "verifying":
            task.status = "active"
        return task

    def subdivide_active(
        self,
        subtasks: list[str],
        *,
        max_depth: int = DEFAULT_MAX_SUBDIVIDE_DEPTH,
    ) -> TaskState | None:
        """Splice the active task in place with finer child subtasks (#123).

        Hierarchical subdivision on replan (ADR-0073 amendment): when the active
        task is blocked, replace it in the queue with ``subtasks`` — flat child
        tasks ``t<n>.1, t<n>.2, …`` at ``depth + 1`` — and activate the first
        child. The data model stays flat (Option 1 "flat splice"): the parent is
        *removed*, its children take its slot, and any already-pending tail keeps
        its order, so the ledger and dashboard need no change.

        Bounded by ``max_depth`` (:data:`DEFAULT_MAX_SUBDIVIDE_DEPTH`): if the
        active task is already at that depth, subdivision is **refused** and this
        returns ``None`` so the caller hands off instead of subdividing forever.
        Also a no-op (returns ``None``) when there is no active task or
        ``subtasks`` is empty after trimming — the caller must treat ``None`` as
        "could not subdivide" and fall back to :meth:`abandon_active`.

        Returns the newly-active first child, or ``None`` when subdivision was
        refused / impossible.

        Example:
            >>> m = MissionState(["tidy the kitchen", "wipe the table"])
            >>> child = m.subdivide_active(["clear the counter", "load the dishwasher"])
            >>> child.task_id, child.text, child.depth
            ('t1.1', 'clear the counter', 1)
            >>> [t.task_id for t in m.tasks]
            ['t1.1', 't1.2', 't2']
            >>> m.subdivide_active(["rinse", "stack"]).task_id  # depth 1 → 2: allowed
            't1.1.1'
            >>> m.subdivide_active(["x"]) is None  # depth 2 >= DEFAULT_MAX_SUBDIVIDE_DEPTH
            True
        """
        task = self.active()
        if task is None or task.depth >= max_depth:
            return None
        children = [t for raw in subtasks if (t := raw.strip())]
        if not children:
            return None
        index = self._tasks.index(task)
        spliced = [
            TaskState(
                task_id=f"{task.task_id}.{i + 1}",
                text=text,
                status="active" if i == 0 else "pending",
                depth=task.depth + 1,
            )
            for i, text in enumerate(children)
        ]
        self._tasks[index : index + 1] = spliced
        return spliced[0]

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

    # ── serialization / rendering ──────────────────────────────────────────────

    def to_summary(self) -> dict[str, object]:
        """JSON-able snapshot of the queue for telemetry / the dashboard card.

        Stamped on the ``reasoner.tick`` span as ``reasoner.mission_json`` so the
        live dashboard can render the mission checklist (status, attempts, last
        verdict) instead of only the single ``## MISSION`` text ledger. Pure data;
        no rclpy, no Pydantic.

        Example:
            >>> m = MissionState(["pick the bowl", "place the butter"])
            >>> s = m.to_summary()
            >>> s["max_attempts"], len(s["tasks"]), s["tasks"][0]["status"]
            (3, 2, 'active')
        """
        return {
            "max_attempts": DEFAULT_MAX_ATTEMPTS,
            "tasks": [
                {
                    "id": t.task_id,
                    "text": t.text,
                    "status": t.status,
                    "attempts": t.attempts,
                    "verdict": t.last_verdict,
                    "rskill_id": t.last_rskill_id,
                }
                for t in self._tasks
            ],
        }

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
            # Indent subdivided children (depth>0, #123) so the flat ledger still
            # reads as a hierarchy for the LLM; the dashboard ignores leading space.
            indent = "  " * task.depth
            lines.append(f"{indent}{marker} {task.task_id}: {task.text}{attempts}{verdict}")
        if pending:
            lines.append(f"… {pending} pending task(s)")
        return "\n".join(lines)
