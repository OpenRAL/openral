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

import dataclasses
import json
import os
import pathlib

from openral_reasoner.mission import MissionState

__all__ = ["ReasonerLadderState", "load_ladder_state", "save_ladder_state"]

_SCHEMA_VERSION = "0.1"


@dataclasses.dataclass(slots=True)
class ReasonerLadderState:
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

    mission: MissionState | None = None
    subdivide_offered: set[str] = dataclasses.field(default_factory=set)
    collective_nudges: dict[str, int] = dataclasses.field(default_factory=dict)
    locate_task_id: str | None = None
    locate_count: int = 0


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
    payload: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "mission": state.mission.to_state_dict() if state.mission is not None else None,
        "subdivide_offered": sorted(state.subdivide_offered),
        "collective_nudges": dict(state.collective_nudges),
        "locate_task_id": state.locate_task_id,
        "locate_count": state.locate_count,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=1), encoding="utf-8")
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
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict) or raw.get("schema_version") != _SCHEMA_VERSION:
        return None
    try:
        mission_raw = raw.get("mission")
        mission = (
            MissionState.from_state_dict(mission_raw) if isinstance(mission_raw, dict) else None
        )
        offered_raw = raw.get("subdivide_offered", [])
        nudges_raw = raw.get("collective_nudges", {})
        if not isinstance(offered_raw, list) or not isinstance(nudges_raw, dict):
            return None
        locate_task_id = raw.get("locate_task_id")
        return ReasonerLadderState(
            mission=mission,
            subdivide_offered={str(x) for x in offered_raw},
            collective_nudges={str(k): int(v) for k, v in nudges_raw.items()},
            locate_task_id=str(locate_task_id) if locate_task_id is not None else None,
            locate_count=int(raw.get("locate_count", 0)),
        )
    except (KeyError, TypeError, ValueError):
        return None
