"""colcon-test for openral_reasoner_ros (ADR-0018 F4).

This file is a thin shim consumed by ``ament_add_pytest_test`` so
``colcon test --packages-select openral_reasoner_ros`` passes a real
test instead of a missing-file error. The substantive integration test
lives in ``tests/integration/test_reasoner_node_end_to_end.py`` (gated
on ``OPENRAL_TEST_ROS_LIVE`` per the repo-wide convention); we duplicate
the import-only smoke test here so the colcon CI surface stays green
without the env gate.
"""

from __future__ import annotations

import pytest

# Skip unless the ROS deps + this colcon package are importable (matches the
# sibling HAL/estop shims). Under plain `pytest` selection without a sourced
# colcon overlay (the select-and-test CI job), `openral_reasoner_ros` isn't on
# the path; skip rather than hard-fail. The substantive coverage runs under
# `colcon test` and the gated tests/integration end-to-end test.
pytest.importorskip("rclpy")
pytest.importorskip("openral_msgs")
pytest.importorskip("openral_reasoner_ros")


def test_import_only() -> None:
    """Smoke import — the reasoner_node module loads with rclpy + openral_msgs sourced."""
    import openral_reasoner_ros.reasoner_node as mod

    assert hasattr(mod, "ReasonerNode")
    assert hasattr(mod, "main")
