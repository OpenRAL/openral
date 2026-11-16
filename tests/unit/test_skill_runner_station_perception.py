"""Station mode in the rSkill runner: per-skill perception activation (mobile base).

The decision comes from the manifest's ``actuators_required`` (real in-tree manifests here),
and the transitions drive a real rclpy lifecycle node over its real ``get_state`` /
``change_state`` services on a private context (no global ``rclpy.shutdown``; see
``test_skill_runner_deadline``). The runner method under test is bound to a plain rclpy node
carrying the two parameters and the callback group it reads.
"""

from __future__ import annotations

import importlib.util
import threading
from collections.abc import Iterator
from pathlib import Path
from types import MethodType, ModuleType
from typing import Any

import pytest

_REPO = Path(__file__).resolve().parents[2]


def _load_runner() -> ModuleType:
    src = _REPO / "packages/openral_rskill_ros/openral_rskill_ros/rskill_runner_node.py"
    spec = importlib.util.spec_from_file_location("_test_rskill_runner_station", src)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("rclpy") is None,
    reason="rclpy not on PYTHONPATH; source a ROS 2 install to run runner-node tests",
)


@pytest.mark.parametrize(
    ("skill", "wanted"),
    [
        ("rskill-nav2-navigate-to-pose", False),  # body_twist only: Nav2
        ("rskill-internvla_n1-mobile_base-vln-nf4", False),  # body_twist only: a VLN policy
        ("pi05-metal-pick-place", True),  # joint_position + gripper_position: pi0.5
        ("pi05-droid-jointpos-arena", True),
        ("rskill-moveit-joints", True),  # an arm motion through MoveIt
        ("stage-for-manipulation", None),  # actuates nothing: perception left as it is
    ],
)
def test_the_manifest_decides_whether_a_skill_needs_the_station_perception(
    skill: str, wanted: bool | None
) -> None:
    from openral_rskill.loader import parse_rskill_manifest

    runner = _load_runner()
    manifest = parse_rskill_manifest(_REPO / "rskills" / skill / "rskill.yaml")
    assert runner.station_perception_wanted(manifest) is wanted


@pytest.fixture
def graph() -> Iterator[tuple[Any, Any, Any]]:
    """A runner-shaped node and a real lifecycle peer ``station_peer``, spinning."""
    import rclpy
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn

    ctx = rclpy.Context()
    rclpy.init(context=ctx)

    class Peer(LifecycleNode):  # type: ignore[misc]  # reason: rclpy untyped
        refuse_activate = False

        def on_activate(self, state: Any) -> Any:
            if self.refuse_activate:
                return TransitionCallbackReturn.FAILURE
            return super().on_activate(state)

    peer = Peer("station_peer", context=ctx)
    peer.trigger_configure()
    runner = rclpy.create_node("station_runner", context=ctx)
    runner.declare_parameter("station_perception_timeout_s", 5.0)
    runner._action_cb_group = ReentrantCallbackGroup()
    runner._station_clients = {}
    mod = _load_runner()
    for name in ("_station_transition", "_station_switch", "_station_call", "_station_service"):
        setattr(runner, name, MethodType(getattr(mod.RskillRunnerNode, name), runner))
    runner.declare_parameter("station_perception_service", "/station_peer/switch")
    from std_srvs.srv import SetBool

    peer.station_on = None

    def switch(req: Any, res: Any) -> Any:
        peer.station_on = bool(req.data)
        res.success = True
        return res

    peer.create_service(SetBool, "/station_peer/switch", switch)
    executor = MultiThreadedExecutor(context=ctx)
    executor.add_node(peer)
    executor.add_node(runner)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    try:
        yield runner, peer, executor
    finally:
        executor.shutdown()
        runner.destroy_node()
        peer.destroy_node()
        rclpy.shutdown(context=ctx)
        spin.join(timeout=5.0)


def _state(peer: Any) -> str:
    return str(peer._state_machine.current_state[1])


def test_activate_then_deactivate_a_real_lifecycle_peer(graph: tuple[Any, Any, Any]) -> None:
    runner, peer, _ = graph
    assert _state(peer) == "inactive"
    runner._station_transition("station_peer", active=True, raise_on_failure=True)
    assert _state(peer) == "active"
    runner._station_transition("station_peer", active=True, raise_on_failure=True)  # unchanged
    assert _state(peer) == "active"
    runner._station_transition("station_peer", active=False, raise_on_failure=True)
    assert _state(peer) == "inactive"


def test_a_failed_activation_refuses_the_goal_naming_the_node(
    graph: tuple[Any, Any, Any],
) -> None:
    from openral_core.exceptions import ROSPerceptionStale

    runner, peer, _ = graph
    peer.refuse_activate = True
    with pytest.raises(ROSPerceptionStale, match="station_peer"):
        runner._station_transition("station_peer", active=True, raise_on_failure=True)
    # After the goal ends the deactivation is best effort: logged, never raised.
    runner._station_transition("station_peer", active=False, raise_on_failure=False)


def test_a_node_that_is_not_on_the_graph_refuses_the_goal(graph: tuple[Any, Any, Any]) -> None:
    from openral_core.exceptions import ROSPerceptionStale

    runner, _, _ = graph
    runner.set_parameters(
        [__import__("rclpy").parameter.Parameter("station_perception_timeout_s", value=0.5)]
    )
    with pytest.raises(ROSPerceptionStale, match="no_such_tracker"):
        runner._station_transition("no_such_tracker", active=True, raise_on_failure=True)


@pytest.fixture(scope="module")
def launch_module() -> Any:
    """The real launch file, loaded by path (``test_exemption_provider_launch``'s pattern)."""
    import sys

    for dep in ("launch", "launch_ros", "lifecycle_msgs", "openral_foxglove_bringup"):
        pytest.importorskip(dep, reason=f"{dep} is a module-level import of the launch file")
    src = _REPO / "packages/openral_rskill_ros/launch/deploy_e2e.launch.py"
    spec = importlib.util.spec_from_file_location("deploy_e2e_launch_station", src)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["deploy_e2e_launch_station"] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("robot", "tracking", "segmenter", "nodes", "station"),
    [
        # The tracker is never among the lifecycle-toggled nodes: it stays ACTIVE so its held
        # payloads reach the kernel while the base drives (station mode switches it instead).
        ("openamrobot_v2", "tracking", True, ["openral_segmenter"], True),
        ("openamrobot_v2", "on", False, [], True),
        ("openamrobot_v2", "off", True, [], False),
        ("openarm", "tracking", True, [], False),  # a fixed-base cell: always on, as before
    ],
)
def test_only_a_mobile_base_with_tracking_launches_its_perception_inactive(
    launch_module: Any, robot: str, tracking: str, segmenter: bool, nodes: list[str], station: bool
) -> None:
    from openral_core import RobotDescription

    description = RobotDescription.from_yaml(str(_REPO / "robots" / robot / "robot.yaml"))
    got = launch_module._station_perception_nodes(description, tracking, segmenter=segmenter)
    assert got == nodes
    assert launch_module._station_mode(description, tracking) is station


def test_the_poll_autostart_can_stop_at_configured(launch_module: Any) -> None:
    cmd = launch_module._poll_autostart("openral_object_tracker", 60.0, target="inactive").cmd
    flat = ["".join(str(getattr(p, "text", p)) for p in part) for part in cmd]
    assert flat[flat.index("--target") + 1] == "inactive"
    cmd = launch_module._poll_autostart("openral_object_tracker", 60.0).cmd
    flat = ["".join(str(getattr(p, "text", p)) for p in part) for part in cmd]
    assert flat[flat.index("--target") + 1] == "active"


def test_the_tracker_switch_is_a_real_service_round_trip(graph: tuple[Any, Any, Any]) -> None:
    runner, peer, _ = graph
    runner._station_switch(True, raise_on_failure=True)
    assert peer.station_on is True
    runner._station_switch(False, raise_on_failure=True)
    assert peer.station_on is False


def test_a_missing_tracker_switch_refuses_the_goal(graph: tuple[Any, Any, Any]) -> None:
    import rclpy
    from openral_core.exceptions import ROSPerceptionStale

    runner, _, _ = graph
    runner.set_parameters(
        [
            rclpy.parameter.Parameter("station_perception_service", value="/no/such/switch"),
            rclpy.parameter.Parameter("station_perception_timeout_s", value=0.5),
        ]
    )
    with pytest.raises(ROSPerceptionStale, match="/no/such/switch"):
        runner._station_switch(True, raise_on_failure=True)
    runner._station_switch(False, raise_on_failure=False)  # after a goal: logged, not raised


def _zone(name: str, aliases: tuple[str, ...], frame: str, x: float) -> Any:
    from openral_core import PlaceRegion, PlaceZone, Pose6D

    pose = Pose6D(xyz=(x, 0.0, 0.5), quat_xyzw=(0, 0, 0, 1), frame_id=frame)
    box = PlaceRegion(frame_id=frame, pose=pose, half_extents=(0.1, 0.06, 0.06))
    return PlaceZone(name=name, aliases=aliases, region=box)


@pytest.mark.parametrize(
    ("robot", "zones", "prompt", "target"),
    [
        # The same slot at three map-frame shelves: every match, as a candidate set.
        (
            "openamrobot_v2",
            [_zone(f"s{k}_lower_slot_1", ("lower_slot_1",), "map", 2.0 * k) for k in (1, 2, 3)],
            "restock box x1 into lower slot 1",
            "zone:s1_lower_slot_1|s2_lower_slot_1|s3_lower_slot_1",
        ),
        # A fixed-base cell with unique names: exactly as before.
        (
            "openarm",
            [_zone("upper_slot_2", ("upper_2",), "openarm_base", 0.5)],
            "restock box x3 into upper slot 2",
            "zone:upper_slot_2",
        ),
        (
            "openarm",
            [_zone("upper_slot_2", (), "openarm_base", 0.5)],
            "restock box x3 onto the shelf",
            "surface",
        ),
    ],
)
def test_the_runner_names_one_zone_or_the_station_candidates(
    robot: str, zones: list[Any], prompt: str, target: str
) -> None:
    import rclpy
    from openral_core import RobotDescription
    from pydantic import TypeAdapter

    ctx = rclpy.Context()
    rclpy.init(context=ctx)
    node = rclpy.create_node("zone_runner", context=ctx)
    try:
        from openral_core import PlaceZone

        node.declare_parameter(
            "place_zones_json", TypeAdapter(list[PlaceZone]).dump_json(zones).decode()
        )
        node.declare_parameter("object_tracking", "tracking")
        node._description = RobotDescription.from_yaml(str(_REPO / "robots" / robot / "robot.yaml"))
        mod = _load_runner()
        decide = MethodType(mod.RskillRunnerNode._goal_scope_place_declaration, node)
        assert decide(60.0, prompt=prompt).target_id == target
    finally:
        node.destroy_node()
        rclpy.shutdown(context=ctx)
