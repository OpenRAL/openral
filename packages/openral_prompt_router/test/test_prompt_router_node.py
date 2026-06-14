"""colcon-test for openral_prompt_router (ADR-0018 F10).

Mirrors ``packages/openral_reasoner_ros/test/test_reasoner_node.py``
— a thin import-only smoke so ``colcon test`` has a real file to run.
The live integration test lives in
``tests/integration/test_reasoner_node_end_to_end.py``.
"""

from __future__ import annotations


def test_import_only() -> None:
    """Smoke import — the prompt_router_node module loads with rclpy + openral_msgs sourced."""
    import openral_prompt_router.prompt_router_node as mod

    assert hasattr(mod, "PromptRouterNode")
    assert hasattr(mod, "main")
