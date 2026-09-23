"""``tools/gen_ros_topic_graph.py`` against the real source tree.

Each assertion pins an edge that exists in the product code today, so a
resolver regression (a literal, a constant, a ``declare_parameter`` default or
a C++ literal no longer resolving) turns the graph silently sparse and fails here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import gen_ros_topic_graph as graph  # noqa: E402


@pytest.fixture(scope="module")
def endpoints() -> list[graph.Endpoint]:
    return graph.extract_python(graph._iter_sources((".py",))) + graph.extract_cpp(
        graph._iter_sources((".cpp", ".hpp", ".h"))
    )


def _where(endpoints: list[graph.Endpoint], kind: str, role: str, name: str) -> set[str]:
    return {e.where for e in endpoints if (e.kind, e.role, e.name) == (kind, role, name)}


def test_cpp_literal_kernel_publishes_safe_action(endpoints: list[graph.Endpoint]) -> None:
    pubs = _where(endpoints, "topic", "pub", "/openral/safe_action")
    assert "cpp/openral_safety_kernel/src/lifecycle_kernel.cpp" in pubs


def test_python_function_scope_constant_resolves(endpoints: list[graph.Endpoint]) -> None:
    # TOPIC_FAST is assigned in an enclosing function, not at module level.
    pubs = _where(endpoints, "topic", "pub", "/openral/world_state_fast")
    assert any("openral_world_state_ros/lifecycle_node.py" in w for w in pubs)


def test_declare_parameter_default_resolves_with_param_note(
    endpoints: list[graph.Endpoint],
) -> None:
    subs = _where(endpoints, "topic", "sub", "/openral/estop")
    assert any("rskill_runner_node.py" in w and "[param `estop_topic`]" in w for w in subs)


def test_cpp_declare_parameter_default_resolves(endpoints: list[graph.Endpoint]) -> None:
    pubs = _where(endpoints, "topic", "pub", "/openral/world_voxels")
    assert any("octomap_voxel_bridge_node.cpp" in w for w in pubs)


def test_service_and_action_are_joined(endpoints: list[graph.Endpoint]) -> None:
    assert _where(endpoints, "service", "server", "/openral/estop_reset")
    assert _where(endpoints, "action", "server", "/openral/execute_rskill")
    assert _where(endpoints, "action", "client", "/openral/execute_rskill")


def test_message_types_are_canonical(endpoints: list[graph.Endpoint]) -> None:
    types = {e.msg_type for e in endpoints if e.name == "/openral/safe_action"}
    assert types == {"openral_msgs/ActionChunk"}


def test_tests_are_excluded(endpoints: list[graph.Endpoint]) -> None:
    assert not any("/test/" in e.where or e.where.startswith("tests/") for e in endpoints)


def test_checked_in_page_is_fresh() -> None:
    assert graph.main(["--check"]) == 0, "run `uv run python tools/gen_ros_topic_graph.py`"
