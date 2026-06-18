"""Benchmark task-compatibility gate.

The embodiment/sensor gate (``rSkill.check_compatibility``) verifies the robot
matches; it does NOT verify the rSkill was trained for the benchmark's *task*.
That gap let a LiftCube policy run on PickCube-v1 (and a pick-place policy on an
insertion task) — sensible-looking rollouts that can never succeed.

``check_benchmark_task_compatibility`` closes it: an rSkill manifest may declare
``evaluated_tasks`` (the benchmark task ids / families it was trained or
validated for). When declared, the benchmark runner refuses a scene whose task
is not covered, with a typed ``ROSCapabilityMismatch``. When undeclared (empty),
it is permissive (legacy rSkills) — the runner only logs a warning.

Validates against a real in-tree manifest fixture (CLAUDE.md §1.11).
"""

from __future__ import annotations

import pytest
from openral_core.exceptions import ROSCapabilityMismatch
from openral_rskill.loader import load_rskill_manifest
from openral_sim.benchmark import check_benchmark_task_compatibility


def _manifest_with(evaluated_tasks: list[str]):
    base = load_rskill_manifest("rskills/smolvla-libero")
    return base.model_copy(update={"evaluated_tasks": evaluated_tasks})


def test_exact_task_id_match_is_allowed() -> None:
    m = _manifest_with(["maniskill3/PickCube-v1"])
    check_benchmark_task_compatibility(m, task_id="maniskill3/PickCube-v1", scene_id="maniskill3")


def test_family_prefix_match_is_allowed() -> None:
    # 'libero_spatial' covers every per-task id 'libero_spatial/0'..'/9'.
    m = _manifest_with(["libero_spatial"])
    check_benchmark_task_compatibility(m, task_id="libero_spatial/0", scene_id="libero_spatial")


def test_declared_mismatch_raises() -> None:
    # A LiftCube policy declared for LiftCube must be refused on PickCube.
    m = _manifest_with(["maniskill3/LiftCube"])
    with pytest.raises(ROSCapabilityMismatch):
        check_benchmark_task_compatibility(
            m, task_id="maniskill3/PickCube-v1", scene_id="maniskill3"
        )


def test_undeclared_is_permissive() -> None:
    # Empty evaluated_tasks → legacy rSkill → no raise (runner warns only).
    m = _manifest_with([])
    check_benchmark_task_compatibility(m, task_id="maniskill3/PickCube-v1", scene_id="maniskill3")
