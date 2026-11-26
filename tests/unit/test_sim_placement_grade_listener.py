"""The scene's success condition reaches the rSkill runner (2026-11-26, AMR aisle t22).

The sim graded the restock ``complete`` while the runner, judging by the tracker alone,
reported "no success condition judged" and the reasoner retried a finished task. The HAL
now hands each grade of the graded goal to its listeners (the HAL node publishes it on
``/openral/sim/placement_grade``); the runner's ``scene_grade_outcome`` turns it into the
goal's ``TaskOutcome``.
"""

from __future__ import annotations

from typing import Any

import pytest
from openral_core import TaskVerdict
from openral_hal.sim_attached import SimAttachedHAL


class _Env:
    def set_task(self, prompt: str) -> dict[str, Any]:
        return {"graded": True, "grade_scores": {"fail": 0.0, "partial": 0.5, "complete": 1.0}}


def _hal() -> SimAttachedHAL:
    hal = SimAttachedHAL.__new__(SimAttachedHAL)  # the grading state only, no env boot
    hal._env = _Env()  # type: ignore[assignment]  # reason: a set_task-only stand-in
    hal._step_count = 0
    hal._active_task = hal._placement_prompt = ""
    hal._placement_scores = None
    hal._placement_grade_listeners = []
    hal._placement_grade = hal._placement_best = "fail"
    hal._set_task_failed = False
    hal.sim_time_ns = lambda: 0  # type: ignore[method-assign]
    return hal


def test_listeners_get_the_starting_grade_and_each_change() -> None:
    hal = _hal()
    seen: list[dict[str, object]] = []
    hal.add_placement_grade_listener(seen.append)
    hal.set_active_task("restock box x1 into lower slot 3")
    hal._observe_placement_grade("complete")
    hal._observe_placement_grade("complete")  # unchanged: no new grade
    assert [(g["grade"], g["score"]) for g in seen] == [("fail", 0.0), ("complete", 1.0)]
    assert {g["prompt"] for g in seen} == {"restock box x1 into lower slot 3"}


def test_a_raising_listener_never_breaks_the_step() -> None:
    hal = _hal()
    seen: list[dict[str, object]] = []

    def boom(_: dict[str, object]) -> None:
        raise RuntimeError("publisher gone")

    hal.add_placement_grade_listener(boom)
    hal.add_placement_grade_listener(seen.append)
    hal.set_active_task("p")
    hal._observe_placement_grade("partial")
    assert [g["grade"] for g in seen] == ["fail", "partial"]


@pytest.mark.parametrize(
    ("grade", "verdict"),
    [
        ("complete", TaskVerdict.SUCCEEDED),
        ("partial", TaskVerdict.FAILED),
        ("fail", TaskVerdict.FAILED),
    ],
)
def test_the_scene_grade_is_the_goal_outcome(grade: str, verdict: TaskVerdict) -> None:
    runner = pytest.importorskip("openral_rskill_ros.rskill_runner_node")
    outcome = runner.scene_grade_outcome({"prompt": "p", "grade": grade, "score": 1.0})
    assert outcome.verdict is verdict and outcome.source == "scene"
