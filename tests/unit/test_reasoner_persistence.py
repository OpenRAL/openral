"""Unit tests for crash-safe ladder persistence (``openral_reasoner.persistence``).

The mission ledger + every replanning-ladder bound must survive a reasoner
restart (resume, not reset). Uses real ``MissionState`` mutations — no
synthetic dicts — and a real tmp filesystem.
"""

from __future__ import annotations

import json
import pathlib

from openral_reasoner import MissionState
from openral_reasoner.persistence import (
    ReasonerLadderState,
    load_ladder_state,
    save_ladder_state,
)


def _worked_mission() -> MissionState:
    """A mission that has visibly progressed: t1 done, t2.1 active with attempts."""
    mission = MissionState(["tidy the kitchen", "wipe the table"])
    mission.record_attempt(rskill_id="openral/rskill-a", trace_id="00-abc")
    mission.complete_active("progress=0.91")
    child = mission.subdivide_active(["clear the counter", "load the dishwasher"])
    assert child is not None and child.task_id == "t2.1"
    mission.record_attempt(rskill_id="openral/rskill-b")
    mission.record_attempt(rskill_id="openral/rskill-b")
    return mission


def test_round_trip_resumes_exactly(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "ladder.json"
    state = ReasonerLadderState(
        mission=_worked_mission(),
        subdivide_offered={"t2"},
        collective_nudges={"t2": 1},
        locate_task_id="t2.1",
        locate_count=2,
    )
    save_ladder_state(path, state)
    restored = load_ladder_state(path)

    assert restored is not None
    assert restored.subdivide_offered == {"t2"}
    assert restored.collective_nudges == {"t2": 1}
    assert restored.locate_task_id == "t2.1"
    assert restored.locate_count == 2
    assert restored.mission is not None
    tasks = restored.mission.tasks
    assert [(t.task_id, t.status, t.attempts, t.depth) for t in tasks] == [
        ("t1", "done", 1, 0),
        ("t2.1", "active", 2, 1),
        ("t2.2", "pending", 0, 1),
    ]
    assert tasks[0].last_verdict == "progress=0.91"
    assert tasks[0].last_trace_id == "00-abc"
    assert restored.mission.active() is not None
    assert restored.mission.active().task_id == "t2.1"


def test_no_mission_round_trips_as_none(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "ladder.json"
    save_ladder_state(path, ReasonerLadderState())
    restored = load_ladder_state(path)
    assert restored is not None
    assert restored.mission is None
    assert restored.subdivide_offered == set()
    assert restored.locate_count == 0


def test_missing_file_loads_as_none(tmp_path: pathlib.Path) -> None:
    assert load_ladder_state(tmp_path / "absent.json") is None


def test_corrupt_file_loads_as_none(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "ladder.json"
    path.write_text("{ not json", encoding="utf-8")
    assert load_ladder_state(path) is None


def test_version_mismatch_loads_as_none(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "ladder.json"
    path.write_text(json.dumps({"schema_version": "9.9", "mission": None}), encoding="utf-8")
    assert load_ladder_state(path) is None


def test_missing_version_loads_as_none(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "ladder.json"
    path.write_text(json.dumps({"mission": None}), encoding="utf-8")
    assert load_ladder_state(path) is None


def test_invalid_mission_loads_as_none(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "ladder.json"
    valid = {
        "schema_version": "0.1",
        "subdivide_offered": [],
        "collective_nudges": {},
        "locate_task_id": None,
        "locate_count": 0,
    }
    invalid_missions = (
        {"tasks": [{"task_id": "t1", "text": "x", "status": "bogus"}]},
        {
            "tasks": [
                {"task_id": "t1", "text": "x", "status": "active"},
                {"task_id": "t2", "text": "y", "status": "active"},
            ]
        },
        {"tasks": [{"task_id": "t1", "text": "x", "status": "active", "attempts": -1}]},
    )
    for mission in invalid_missions:
        path.write_text(json.dumps(valid | {"mission": mission}), encoding="utf-8")
        assert load_ladder_state(path) is None


def test_invalid_boundary_fields_load_as_none(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "ladder.json"
    valid = {
        "schema_version": "0.1",
        "mission": None,
        "subdivide_offered": [],
        "collective_nudges": {},
        "locate_task_id": None,
        "locate_count": 0,
    }
    for invalid in (
        {"locate_count": -1},
        {"collective_nudges": {"t1": -1}},
        {"unexpected": True},
    ):
        path.write_text(json.dumps(valid | invalid), encoding="utf-8")
        assert load_ladder_state(path) is None


def test_save_is_atomic_no_tmp_left_behind(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "nested" / "ladder.json"
    save_ladder_state(path, ReasonerLadderState(mission=MissionState(["pick the bowl"])))
    save_ladder_state(path, ReasonerLadderState(mission=MissionState(["place the butter"])))
    # os.replace semantics: only the final file remains, no .tmp residue.
    assert sorted(p.name for p in path.parent.iterdir()) == ["ladder.json"]
    restored = load_ladder_state(path)
    assert restored is not None and restored.mission is not None
    assert restored.mission.active().text == "place the butter"
