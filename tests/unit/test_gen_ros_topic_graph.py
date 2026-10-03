"""``tools/gen_ros_topic_graph.py`` against the real source tree.

Each assertion pins an edge that exists in the product code today, so a
resolver regression (a literal, a constant, a ``declare_parameter`` default or
a C++ literal no longer resolving) turns the graph silently sparse and fails here.
"""

from __future__ import annotations

import sys
import textwrap
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
    assert any("openral_octomap_bridge/src/octomap_voxel_bridge" in w for w in pubs)


def test_service_and_action_are_joined(endpoints: list[graph.Endpoint]) -> None:
    assert _where(endpoints, "service", "server", "/openral/estop_reset")
    assert _where(endpoints, "action", "server", "/openral/execute_rskill")
    assert _where(endpoints, "action", "client", "/openral/execute_rskill")


def test_message_types_are_canonical(endpoints: list[graph.Endpoint]) -> None:
    types = {e.msg_type for e in endpoints if e.name == "/openral/safe_action"}
    assert types == {"openral_msgs/ActionChunk"}


def test_tests_are_excluded(endpoints: list[graph.Endpoint]) -> None:
    assert not any("/test/" in e.where or e.where.startswith("tests/") for e in endpoints)


def test_camera_topic_rows_join_producer_and_consumer(endpoints: list[graph.Endpoint]) -> None:
    # ADR-0108: sim bridge publishes and world state subscribes on camera_topic(<name>).
    name = "/openral/cameras/{sensor}/image"
    assert any("sim_sensor_bridge.py" in w for w in _where(endpoints, "topic", "pub", name))
    assert any("world_state" in w for w in _where(endpoints, "topic", "sub", name))
    points = "/openral/cameras/{sensor}/points"
    assert any("sim_sensor_bridge.py" in w for w in _where(endpoints, "topic", "pub", points))


def _py(tmp_path: Path, source: str, extra: tuple[Path, ...] = ()) -> list[graph.Endpoint]:
    path = tmp_path / "snippet.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return [e for e in graph.extract_python([path, *extra]) if e.where.startswith(str(path))]


def _names(endpoints: list[graph.Endpoint]) -> set[tuple[str, str, str, bool]]:
    return {(e.kind, e.role, e.name, e.resolved) for e in endpoints}


def test_distinct_placeholders_stay_distinct_and_per_instance(tmp_path: Path) -> None:
    eps = _py(
        tmp_path,
        """
        def f(node, left, right):
            node.create_publisher(Empty, f"/x/{left.name}/s", 1)
            node.create_subscription(Empty, f"/x/{right.name}/s", cb, 1)
            node.create_publisher(Empty, "/x/concrete", 1)
        """,
    )
    assert _names(eps) == {
        ("topic", "pub", "/x/{left.name}/s", True),
        ("topic", "sub", "/x/{right.name}/s", True),
        ("topic", "pub", "/x/concrete", True),
    }
    page = graph.render(eps, [])
    concrete, _, rest = page.partition("## Per-instance topics (2)")
    assert "`/x/concrete`" in concrete and "{left.name}" not in concrete
    assert "`/x/{left.name}/s`" in rest and "`/x/{right.name}/s`" in rest


def test_parameter_without_default_is_not_a_module_constant(tmp_path: Path) -> None:
    eps = _py(
        tmp_path,
        """
        TOPIC = "/module/const"

        def no_default(node, TOPIC):
            node.create_publisher(Empty, TOPIC, 1)

        def with_default(node, TOPIC="/param/default"):
            node.create_subscription(Empty, TOPIC, cb, 1)

        def unbound(node):
            node.create_client(Trigger, TOPIC)
        """,
    )
    assert _names(eps) == {
        ("topic", "pub", "TOPIC", False),
        ("topic", "sub", "/param/default", True),
        ("service", "client", "/module/const", True),
    }


def test_keyword_arguments_are_extracted(tmp_path: Path) -> None:
    eps = _py(
        tmp_path,
        """
        def f(node):
            node.create_publisher(msg_type=Empty, topic="/k/pub", qos_profile=1)
            node.create_subscription(Empty, topic="/k/sub", callback=cb, qos_profile=1)
            node.create_service(srv_type=Trigger, srv_name="/k/srv", callback=cb)
            node.create_client(srv_type=Trigger, srv_name="/k/cli")
            ActionServer(node, action_type=Nav, action_name="/k/act", execute_callback=cb)
            ActionClient(node=node, action_type=Nav, action_name="/k/act")
            node.create_publisher(qos_profile=1)  # neither form: not an rclpy call
        """,
    )
    assert _names(eps) == {
        ("topic", "pub", "/k/pub", True),
        ("topic", "sub", "/k/sub", True),
        ("service", "server", "/k/srv", True),
        ("service", "client", "/k/cli", True),
        ("action", "server", "/k/act", True),
        ("action", "client", "/k/act", True),
    }
    assert {e.msg_type for e in eps if e.kind == "action"} == {"Nav"}


def test_cpp_action_overloads_parse_balanced_arguments(tmp_path: Path) -> None:
    path = tmp_path / "node.cpp"
    path.write_text(
        """
        server_ = rclcpp_action::create_server<Act>(
          this->get_node_base_interface(), this->get_node_clock_interface(),
          this->get_node_logging_interface(), this->get_node_waitables_interface(),
          "/x/act", std::bind(&N::goal, this, _1, _2), [this](auto g) { cancel(g); },
          std::bind(&N::accepted, this, _1));
        client_ = rclcpp_action::create_client<Act>(this, "/x/node_form");
        other_ = rclcpp_action::create_client<Act>(
          get_node_base_interface(), get_node_graph_interface(),
          get_node_logging_interface(), get_node_waitables_interface(), "/x/iface_form");
        pub_ = create_publisher<std_msgs::msg::Empty>(std::string("/x/") + name(), 10);
        """,
        encoding="utf-8",
    )
    eps = graph.extract_cpp([path])
    assert _names(eps) == {
        ("action", "server", "/x/act", True),
        ("action", "client", "/x/node_form", True),
        ("action", "client", "/x/iface_form", True),
        ("topic", "pub", 'std::string("/x/") + name()', False),
    }


def test_camera_topic_resolves_literal_expression_and_kinds(tmp_path: Path) -> None:
    core = REPO_ROOT / "python/core/src/openral_core"
    eps = _py(
        tmp_path,
        """
        import openral_core
        from openral_core import CAMERA_TOPIC_PREFIX, CameraTopicKind, camera_topic

        def f(node, spec):
            node.create_publisher(Image, camera_topic("wrist"), 1)
            node.create_publisher(Pc, camera_topic("front_depth", CameraTopicKind.POINTS), 1)
            node.create_publisher(Info, camera_topic(spec.name, kind="depth/camera_info"), 1)
            node.create_subscription(Image, openral_core.camera_topic(name=spec.name), cb, 1)
            node.create_subscription(Image, f"{CAMERA_TOPIC_PREFIX}/top/image", cb, 1)
        """,
        extra=(core / "__init__.py", core / "schemas.py"),
    )
    assert _names(eps) == {
        ("topic", "pub", "/openral/cameras/wrist/image", True),
        ("topic", "pub", "/openral/cameras/front_depth/points", True),
        ("topic", "pub", "/openral/cameras/{sensor}/depth/camera_info", True),
        ("topic", "sub", "/openral/cameras/{sensor}/image", True),
        ("topic", "sub", "/openral/cameras/top/image", True),
    }


def test_checked_in_page_is_fresh() -> None:
    assert graph.main(["--check"]) == 0, "run `uv run python tools/gen_ros_topic_graph.py`"


def test_untracked_files_do_not_reach_the_page() -> None:
    """Only git-tracked sources count: a scratch script must not change the page."""
    probe = REPO_ROOT / "tools" / "_untracked_graph_probe.py"
    probe.write_text(
        'def f(node):\n    node.create_publisher(int, "/untracked/probe", 1)\n', encoding="utf-8"
    )
    try:
        assert probe not in graph._iter_sources((".py",))
    finally:
        probe.unlink()
