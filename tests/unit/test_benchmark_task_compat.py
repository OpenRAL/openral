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
from openral_core import load_benchmark_suite
from openral_core.exceptions import ROSCapabilityMismatch
from openral_rskill.loader import load_rskill_manifest
from openral_sim.benchmark import check_benchmark_task_compatibility, filter_scenes_for_skill


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


# ─── Suite auto-filter (run only the rSkill's matched tasks) ──────────────────


def test_filter_keeps_all_matching_suite_tasks() -> None:
    # A spatial-declared rSkill covers every task in the spatial suite.
    scenes = load_benchmark_suite("benchmarks/libero_spatial.yaml")
    kept, skipped = filter_scenes_for_skill(scenes, _manifest_with(["libero_spatial"]))
    assert len(kept) == len(scenes)
    assert skipped == []


def test_filter_skips_all_when_suite_is_other_family() -> None:
    # A spatial-only rSkill matches none of the object suite's tasks.
    scenes = load_benchmark_suite("benchmarks/libero_object.yaml")
    kept, skipped = filter_scenes_for_skill(scenes, _manifest_with(["libero_spatial"]))
    assert kept == []
    assert len(skipped) == len(scenes)


def test_filter_partitions_a_mixed_suite() -> None:
    # Mixed suite: keep the declared family, skip the rest.
    scenes = load_benchmark_suite("benchmarks/libero_spatial.yaml") + load_benchmark_suite(
        "benchmarks/libero_object.yaml"
    )
    kept, skipped = filter_scenes_for_skill(scenes, _manifest_with(["libero_spatial"]))
    assert {s.task.id for s in kept} == {f"libero_spatial/{i}" for i in range(10)}
    assert {s.task.id for s in skipped} == {f"libero_object/{i}" for i in range(10)}


def test_filter_undeclared_keeps_everything() -> None:
    # Undeclared (or no manifest) → unfilterable → permissive (keep all).
    scenes = load_benchmark_suite("benchmarks/libero_spatial.yaml")
    assert filter_scenes_for_skill(scenes, _manifest_with([])) == (scenes, [])
    assert filter_scenes_for_skill(scenes, None) == (scenes, [])


def test_run_benchmark_raises_when_skill_matches_no_suite_task() -> None:
    # smolvla-libero declares evaluated_tasks=["libero_spatial"]; pointed at the
    # object suite it matches nothing, so run_benchmark fails fast (before any
    # rollout) rather than running every object task to a silent 0.
    from openral_core import VLASpec
    from openral_sim.benchmark import run_benchmark

    scenes = load_benchmark_suite("benchmarks/libero_object.yaml")
    vla = VLASpec(id="smolvla-libero", weights_uri="rskills/smolvla-libero")
    with pytest.raises(ROSCapabilityMismatch, match="match none"):
        run_benchmark(scenes, suite_id="libero_object", vla=vla)
