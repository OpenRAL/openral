"""Durable reasoner ladder state — crash-safe mission + bound resume.

The mission ledger and every replanning-ladder bound (per-task attempts,
subdivision offers, collective-decompose nudges, the per-task locate budget)
used to live only in process memory: a reasoner-node crash or lifecycle
restart mid-mission silently reset every cap and re-ran work the robot had
already done — the opposite of the bounded-ladder contract (CLAUDE.md §3) and
of replayability (§1.8). This module is the LangGraph-style fix: a single
JSON snapshot written after every ledger mutation and reloaded at configure,
so a restarted reasoner *resumes* the ladder exactly where it stopped.

Pure (no rclpy): the node owns *when* to save/load; this module owns the
format. Writes are atomic (tmp file + ``os.replace``) so a crash mid-write
can never leave a truncated snapshot behind.
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_serializer,
    field_validator,
)

from openral_reasoner.mission import MissionState

__all__ = ["ReasonerLadderState", "load_ladder_state", "save_ladder_state"]

_SCHEMA_VERSION: Literal["0.1"] = "0.1"


class ReasonerLadderState(BaseModel):
    """Everything a restarted reasoner needs to resume its ladders.

    Attributes:
        mission: The mission ledger snapshot, or ``None`` when no mission
            was active.
        subdivide_offered: Task ids already offered one subdivision
            (:class:`~openral_reasoner.mission.MissionState` #123 bound).
        collective_nudges: Per-task collective-decompose nudge counts.
        locate_task_id: Task the per-task locate budget is charging, or None.
        locate_count: Locate cycles charged against ``locate_task_id``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    schema_version: Literal["0.1"] = _SCHEMA_VERSION
    mission: MissionState | None = None
    subdivide_offered: set[str] = Field(default_factory=set)
    collective_nudges: dict[str, Annotated[int, Field(ge=0)]] = Field(default_factory=dict)
    locate_task_id: str | None = None
    locate_count: int = Field(default=0, ge=0)

    @field_validator("mission", mode="before")
    @classmethod
    def _load_mission(cls, value: object) -> MissionState | None:
        if value is None or isinstance(value, MissionState):
            return value
        if isinstance(value, dict):
            try:
                mission = MissionState.from_state_dict(value)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid mission snapshot: {exc}") from exc
            tasks = mission.tasks
            active_count = sum(task.status in ("active", "verifying") for task in tasks)
            if len({task.task_id for task in tasks}) != len(tasks):
                raise ValueError("invalid mission snapshot: duplicate task ids")
            if any(
                not task.task_id or not task.text.strip() or task.attempts < 0 or task.depth < 0
                for task in tasks
            ):
                raise ValueError("invalid mission snapshot: invalid task fields")
            if tasks and not mission.is_complete() and active_count != 1:
                raise ValueError(
                    "invalid mission snapshot: incomplete mission needs one active task"
                )
            return mission
        raise ValueError("mission must be a MissionState, object, or null")

    @field_serializer("mission")
    def _dump_mission(self, mission: MissionState | None) -> dict[str, object] | None:
        return mission.to_state_dict() if mission is not None else None

    @field_serializer("subdivide_offered")
    def _dump_subdivide_offered(self, offered: set[str]) -> list[str]:
        return sorted(offered)


def save_ladder_state(path: pathlib.Path | str, state: ReasonerLadderState) -> None:
    """Atomically persist ``state`` as JSON at ``path``.

    Atomic = write to ``<path>.tmp`` then ``os.replace`` — a crash mid-write
    leaves the previous snapshot intact, never a truncated file.

    Example:
        >>> import tempfile, pathlib
        >>> p = pathlib.Path(tempfile.mkdtemp()) / "ladder.json"
        >>> m = MissionState(["pick the bowl"])
        >>> save_ladder_state(p, ReasonerLadderState(mission=m))
        >>> restored = load_ladder_state(p)
        >>> restored.mission.active().text
        'pick the bowl'
    """
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(state.model_dump_json(indent=1), encoding="utf-8")
    os.replace(tmp, target)


def load_ladder_state(path: pathlib.Path | str) -> ReasonerLadderState | None:
    """Load a persisted snapshot, or ``None`` when absent/unreadable/invalid.

    A corrupt or version-mismatched snapshot returns ``None`` (the caller
    starts fresh and logs) rather than raising — resuming from a bad snapshot
    is worse than not resuming, and node bring-up must not be blocked by a
    stale file.
    """
    target = pathlib.Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or "schema_version" not in payload:
            return None
        return ReasonerLadderState.model_validate(payload)
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError):
        return None
