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


def _shelf_zones() -> list[Any]:
    """The committed OpenArm shelf zones (``upper_slot_0..3``)."""
    import yaml
    from openral_core import PlaceZone
    from pydantic import TypeAdapter

    path = Path(__file__).resolve().parents[2] / (
        "scenes/deploy/place_zones/openarm_restock_shelf.yaml"
    )
    raw = yaml.safe_load(path.read_text())["place_zones"]
    return list(TypeAdapter(list[PlaceZone]).validate_python(raw))


# ponytail: the AMR aisle (six shelves) lives in the private Pro scene; three of its slots are
# built inline so CI keeps the candidate-set case instead of skipping it.
_AISLE = [_zone(f"s{k}_lower_slot_1", ("lower_slot_1",), "map", 2.0 * k) for k in (1, 2, 3)]
_AISLE_CANDIDATES = "zone:s1_lower_slot_1|s2_lower_slot_1|s3_lower_slot_1"
#: The fixed-cell twin's deploy-level placement queries: a DESTINATION from its graded task.
_WITH_DESTINATION = (
    '[{"label": "box", "count": "one", "role": "target"},'
    ' {"label": "shelf", "count": "one", "role": "destination"}]'
)


@pytest.mark.parametrize(
    ("robot", "zones", "prompt", "target", "mode", "queries"),
    [
        # The same slot at three map-frame shelves: every match, as a candidate set.
        (
            "openamrobot_v2",
            _AISLE,
            "put box x1 into lower slot 1",
            _AISLE_CANDIDATES,
            "tracking",
            "",
        ),
        # Pro `on`, reasoner-driven: no DESTINATION query anywhere, so the zone is named as
        # under `tracking` (AMR aisle 2026-11-16: no place allowance, the place stopped).
        ("openamrobot_v2", _AISLE, "put box x1 into lower slot 1", _AISLE_CANDIDATES, "on", ""),
        # `on` with a standing DESTINATION query: the tracked destination, no zone (as before).
        (
            "openamrobot_v2",
            _AISLE,
            "put box x1 into lower slot 1",
            "surface",
            "on",
            _WITH_DESTINATION,
        ),
        # `tracking` names the zone whatever the queries (unchanged).
        (
            "openamrobot_v2",
            _AISLE,
            "put box x1 into lower slot 1",
            _AISLE_CANDIDATES,
            "tracking",
            _WITH_DESTINATION,
        ),
        # A fixed-base cell with unique names: exactly as before.
        (
            "openarm",
            _shelf_zones(),
            "put box x3 into upper slot 2",
            "zone:upper_slot_2",
            "tracking",
            "",
        ),
        (
            "openarm",
            _shelf_zones(),
            "put box x3 onto the shelf",
            "surface",
            "tracking",
            "",
        ),
    ],
)
def test_the_runner_names_one_zone_or_the_station_candidates(
    robot: str, zones: list[Any], prompt: str, target: str, mode: str, queries: str
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
        node.declare_parameter("object_tracking", mode)
        node.declare_parameter("object_queries_json", queries)
        node._description = RobotDescription.from_yaml(str(_REPO / "robots" / robot / "robot.yaml"))
        mod = _load_runner()
        for name in ("_tracking_queries", "_object_tracking_mode"):
            setattr(node, name, MethodType(getattr(mod.RskillRunnerNode, name), node))
        decide = MethodType(mod.RskillRunnerNode._goal_scope_place_declaration, node)
        assert decide(60.0, prompt=prompt).target_id == target
    finally:
        node.destroy_node()
        rclpy.shutdown(context=ctx)


@pytest.mark.parametrize(
    ("tracking", "evicted"),
    [
        ("off", ["omdet_locator", "second_locator"]),  # no tracker: every locator is evictable
        ("tracking", ["second_locator"]),
        ("on", ["second_locator"]),
    ],
)
def test_the_trackers_locator_is_not_evicted_for_vram_while_tracking(
    launch_module: Any, tracking: str, evicted: list[str]
) -> None:
    """The tracker grounds a goal's object queries through its locator DURING the goal, after
    the reasoner evicted its VRAM peers: evicted, every query failed ``detector_inactive``
    (q5090, 2026-11-16)."""
    specs = [{"node": "omdet_locator"}, {"node": "second_locator"}]
    assert launch_module._vram_locator_peers(specs, tracking) == evicted


def test_the_launch_gives_the_reasoner_the_scenes_declared_grasping_hand(
    launch_module: Any,
) -> None:
    """``grasp_contact_links`` is the deploy scene's grasp declaration's hand (a real scene),
    and nothing without a scene declaration (the reasoner then refuses an unnamed hand on a
    multi-hand robot, as before)."""
    from openral_core import DeployScene

    scene = DeployScene.from_yaml(str(_REPO / "scenes/deploy/isaac_leisaac_pick_orange.yaml"))
    assert scene.grasp_declaration is not None
    raw = scene.grasp_declaration.model_dump_json()
    assert launch_module._declared_grasp_contact_links(raw) == ["moving_jaw"]
    assert launch_module._declared_grasp_contact_links("") == []


_TWO_SLOTS = [
    _zone(f"s{k}_{slot}", (slot,), "map", 2.0 * k)
    for k in (1, 2)
    for slot in ("lower_slot_1", "upper_slot_2")
]


@pytest.mark.parametrize(
    ("prompt", "base_frame", "names"),
    [
        # One slot at two map-frame stations: the candidate set.
        (
            "put box x1 into lower slot 1",
            "base_footprint",
            ["s1_lower_slot_1", "s2_lower_slot_1"],
        ),
        # A zone named by its own name wins over the alias every station shares.
        ("put box x1 into s2 lower slot 1", "base_footprint", ["s2_lower_slot_1"]),
        # Two slots named: ambiguous, never both.
        ("box into upper slot 2 and lower slot 1", "base_footprint", []),
        # Two zones named by name: ambiguous.
        ("s1 lower slot 1 or s2 lower slot 1", "base_footprint", []),
        # No robot description (no base frame): no candidate set.
        ("put box x1 into lower slot 1", "", []),
        # Several matches in the base frame are not stations: ambiguous.
        ("put box x1 into lower slot 1", "map", []),
    ],
)
def test_the_prompt_names_one_zone_one_slots_stations_or_nothing(
    prompt: str, base_frame: str, names: list[str]
) -> None:
    mod = _load_runner()
    got = mod.place_zones_for_prompt(prompt, _TWO_SLOTS, base_frame=base_frame)
    assert [z.name for z in got] == names


def test_a_base_only_skill_turns_the_perception_off_best_effort(
    graph: tuple[Any, Any, Any],
) -> None:
    """A crashed segmenter or a missing switch must not refuse every Nav2 goal: turning the
    perception off before a base-only skill is logged, never raised."""
    import rclpy
    from openral_rskill.loader import parse_rskill_manifest

    runner, _, _ = graph
    mod = _load_runner()
    runner.declare_parameter("station_perception_nodes", ["no_such_segmenter"])
    runner.set_parameters(
        [
            rclpy.parameter.Parameter("station_perception_service", value="/no/such/switch"),
            rclpy.parameter.Parameter("station_perception_timeout_s", value=0.3),
        ]
    )
    for name in ("_station_perception_for", "_station_nodes"):
        setattr(runner, name, MethodType(getattr(mod.RskillRunnerNode, name), runner))
    nav2 = parse_rskill_manifest(_REPO / "rskills/rskill-nav2-navigate-to-pose/rskill.yaml")
    runner._station_perception_for(rskill_id="nav2", manifest=nav2)  # no raise


@pytest.mark.parametrize(
    ("robot", "hal_mode", "backend", "tolerance"),
    [
        ("openamrobot_v2", "sim", "isaacsim", 0.08),
        ("openarm", "sim", "isaacsim", 0.08),
        ("openamrobot_v2", "sim", "mujoco", 0.0),
        ("openamrobot_v2", "real", None, 0.0),
        ("so101_follower", "sim", "isaacsim", 0.0),
    ],
)
def test_only_an_isaac_twin_widens_the_starting_pose_arrival(
    launch_module: Any, robot: str, hal_mode: str, backend: str | None, tolerance: float
) -> None:
    from openral_core import RobotDescription

    name = RobotDescription.from_yaml(str(_REPO / "robots" / robot / "robot.yaml")).name
    got = launch_module._runner_starting_pose_tolerance(name, hal_mode, backend)
    assert got == tolerance


def test_a_wider_arrival_tolerance_never_skips_the_ramp() -> None:
    """``starting_pose_tolerance`` (an Isaac twin's arrival allowance) widens the arrival
    check only: a joint off by more than the manifest's tolerance is still ramped, then
    accepted where its gains let it settle."""
    import rclpy
    from openral_core import RobotDescription

    ctx = rclpy.Context()
    rclpy.init(context=ctx)
    node = rclpy.create_node("ramp_runner", context=ctx)
    try:
        node.declare_parameter("starting_pose_max_delta_per_s", 0.0)
        node.declare_parameter("starting_pose_tolerance", 0.12)
        node._description = RobotDescription.from_yaml(
            str(_REPO / "robots/openamrobot_v2/robot.yaml")
        )
        mod = _load_runner()
        bounds = MethodType(mod.RskillRunnerNode._starting_pose_bounds, node)
        names = [j.name for j in node._description.joints]
        target = [0.0] * len(names)
        current = list(target)
        current[names.index("left_joint4")] = -0.1
        assert mod.starting_pose_ramp_steps(current, target, bounds(arrival=False), 30.0) > 0
        assert mod.starting_pose_reached(current, target, bounds())
    finally:
        node.destroy_node()
        rclpy.shutdown(context=ctx)
