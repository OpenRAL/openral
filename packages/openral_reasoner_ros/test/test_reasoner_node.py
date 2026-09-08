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
