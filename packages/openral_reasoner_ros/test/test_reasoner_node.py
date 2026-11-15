"""colcon-test for openral_reasoner_ros.

Thin shim consumed by ``ament_add_pytest_test`` so
``colcon test --packages-select openral_reasoner_ros`` passes a real test. The
substantive integration test lives in
``tests/integration/test_reasoner_node_end_to_end.py`` (gated on
``OPENRAL_TEST_ROS_LIVE``, run in CI via ``scripts/ros_live_tests.sh`` —
issue #46); this file duplicates the import-only smoke test, with no env
gate, so the colcon CI surface stays green.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Skip unless the ROS deps + this colcon package are importable (matches the sibling
# HAL/estop shims): under plain pytest selection without a sourced colcon overlay,
# openral_reasoner_ros isn't on the path.
pytest.importorskip("rclpy")
pytest.importorskip("openral_msgs")
pytest.importorskip("openral_reasoner_ros")


def test_import_only() -> None:
    """Smoke import — the reasoner_node module loads with rclpy + openral_msgs sourced."""
    import openral_reasoner_ros.reasoner_node as mod

    assert hasattr(mod, "ReasonerNode")
    assert hasattr(mod, "main")


# ── Declared navigation goals (ADR-0133): build_spatial_memory ────────────────
_WAREHOUSE = (
    Path(__file__).resolve().parents[3] / "scenes/deploy/isaac_openamrobot_v2_warehouse.yaml"
)


def _warehouse_goals_json() -> str:
    from openral_core import DeployScene, NavGoal, load_scene_strict
    from pydantic import TypeAdapter

    goals = load_scene_strict(str(_WAREHOUSE), expected=DeployScene).nav_goals
    return TypeAdapter(list[NavGoal]).dump_json(goals).decode()


def test_goals_alone_make_a_places_only_memory_that_ingests_nothing() -> None:
    from openral_reasoner_ros.reasoner_node import build_spatial_memory

    built = build_spatial_memory("", False, _warehouse_goals_json(), now_ns=5)
    assert built.memory is not None and built.places_only and not built.writable
    places = [n for n in built.memory.to_scene_graph().nodes if n.declared]
    assert sorted(n.node_id for n in places) == ["nav_goal:aisle_start", "nav_goal:shelf_1_props"]


def test_a_map_that_fails_to_load_keeps_the_declared_goals(tmp_path: Path) -> None:
    from openral_reasoner_ros.reasoner_node import build_spatial_memory

    broken = tmp_path / "scene_graph.json"
    broken.write_text("{not json")
    built = build_spatial_memory(str(broken), False, _warehouse_goals_json())
    assert built.memory is not None and built.places_only
    assert len(built.memory.to_scene_graph().nodes) == 2
    assert built.warnings and "failed to load" in built.warnings[0]


def test_ingest_with_goals_is_writable_and_offers_recall() -> None:
    from openral_reasoner_ros.reasoner_node import build_spatial_memory

    built = build_spatial_memory("", True, _warehouse_goals_json())
    assert built.writable and not built.places_only


def test_repeated_goal_names_are_a_wiring_fault() -> None:
    import json

    from openral_reasoner_ros.reasoner_node import build_spatial_memory

    goals = json.loads(_warehouse_goals_json())
    goals[1]["aliases"] = ["aisle_start"]
    with pytest.raises(ValueError, match="repeats names"):
        build_spatial_memory("", False, json.dumps(goals))
