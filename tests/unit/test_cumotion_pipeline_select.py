"""Gate-driven cuMotion pipeline selection for MoveGroup goals (ADR-0065 D1).

cuMotion is a MoveIt planning-pipeline plugin selected per request via
``MotionPlanRequest.pipeline_id``. When the host clears the cuMotion GPU floor
(``ComputeSpec.supports_cumotion()``), the runner injects
``request.pipeline_id`` into the MoveGroup goal; otherwise MoveIt keeps its
default pipeline (OMPL). This is a pure, ROS-free transform so it unit-tests
without rclpy.
"""

from __future__ import annotations

from openral_core import ComputeSpec
from openral_rskill.ros_action_rskill import (
    CUMOTION_PIPELINE_ID,
    maybe_inject_cumotion_pipeline,
)


def _gpu_compute() -> ComputeSpec:
    return ComputeSpec(
        gpu_vram_gb=24.0,
        cuda_compute_capability=(8, 9),
        cuda_toolkit_version="13.2",
    )


def _cpu_compute() -> ComputeSpec:
    return ComputeSpec()


def _movegroup_goal() -> dict[str, object]:
    return {"request": {"group_name": "panda_arm"}, "planning_options": {"plan_only": True}}


def test_gpu_host_injects_cumotion_pipeline_id() -> None:
    goal = maybe_inject_cumotion_pipeline(
        _movegroup_goal(), interface_type="MoveGroup", compute=_gpu_compute()
    )
    assert goal["request"]["pipeline_id"] == CUMOTION_PIPELINE_ID


def test_cpu_host_leaves_goal_untouched() -> None:
    goal = maybe_inject_cumotion_pipeline(
        _movegroup_goal(), interface_type="MoveGroup", compute=_cpu_compute()
    )
    assert "pipeline_id" not in goal["request"]


def test_non_movegroup_interface_untouched() -> None:
    nav_goal = {"pose": {"x": 1.0}}
    goal = maybe_inject_cumotion_pipeline(
        nav_goal, interface_type="NavigateToPose", compute=_gpu_compute()
    )
    assert goal == nav_goal


def test_explicit_pipeline_id_is_respected() -> None:
    goal = {"request": {"group_name": "panda_arm", "pipeline_id": "ompl"}}
    out = maybe_inject_cumotion_pipeline(goal, interface_type="MoveGroup", compute=_gpu_compute())
    assert out["request"]["pipeline_id"] == "ompl"


def test_none_compute_untouched() -> None:
    goal = maybe_inject_cumotion_pipeline(
        _movegroup_goal(), interface_type="MoveGroup", compute=None
    )
    assert "pipeline_id" not in goal["request"]


def test_does_not_mutate_input() -> None:
    goal = _movegroup_goal()
    maybe_inject_cumotion_pipeline(goal, interface_type="MoveGroup", compute=_gpu_compute())
    assert "pipeline_id" not in goal["request"], "input goal must not be mutated"


def test_missing_request_block_is_safe() -> None:
    goal: dict[str, object] = {"planning_options": {"plan_only": True}}
    out = maybe_inject_cumotion_pipeline(goal, interface_type="MoveGroup", compute=_gpu_compute())
    assert out == goal
