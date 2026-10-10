"""Live: a reasoner-named grasp/place target reaches the ExecuteRskill goal, grounded.

The reasoner names, perception grounds, the producer measures (real pick-and-place design
§2.2-2.3). Real ``ReasonerNode`` (OpenArm manifest via ``robot_yaml``), real
``/openral/world_state_slow`` carrying lifted boxes (the wire keeps the lift's
``bbox_3d``), real ``ExecuteRskill`` ActionServer recording the goal it receives:

* a named ``box`` + an optional ``shelf`` surface reach the goal as a seed-only
  ``GraspDeclaration`` (``search_box_valid``, no region) and a region-less
  ``PlaceDeclaration`` whose ``search_box`` is the padded shelf box — nothing surveyed;
* with no place target (the normal case: the policy picks the spot, the producer measures
  the surface under the payload) the goal carries no place declaration at all, and the
  real runner (``place_approach_enabled``) arms its goal-scope one;
* a supplied place hint that does not ground refuses the goal — never dropped to "place
  anywhere" — and the refusal tells the LLM to omit it;
* two boxes and no ``object_id`` refuse the dispatch — no goal is sent;
* no ``contact_links`` on the bimanual OpenArm refuses too — defaulting would exempt both hands.

The only double is the LLM (``FakeToolUseClient``, never consulted: the tool call is fed
to the dispatch directly).

Gated on ``OPENRAL_TEST_ROS_LIVE=1`` (``just test-ros-live -k grounded_targets``).
"""

from __future__ import annotations

import os
import pathlib
import time
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENRAL_TEST_ROS_LIVE"),
    reason="live rclpy graph — set OPENRAL_TEST_ROS_LIVE=1 and source install/setup.bash.",
)

_REPO = pathlib.Path(__file__).resolve().parents[2]
_SKILL = "openral/test-pick"


def _spin_until(executor: Any, predicate: Any, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.05)
        if predicate():
            return True
    return False


def _box(y: float) -> Any:
    from openral_core import DetectedObject, Pose6D

    return DetectedObject(
        label="box",
        confidence=0.9,
        pose=Pose6D(xyz=(0.40, y, -0.10), quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="openarm_base"),
        bbox_3d=(0.37, y - 0.03, -0.13, 0.43, y + 0.03, -0.07),
    )


def _shelf() -> Any:
    from openral_core import DetectedObject, Pose6D

    return DetectedObject(
        label="shelf",
        confidence=0.8,
        pose=Pose6D(
            xyz=(0.45, 0.25, 0.10), quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="openarm_base"
        ),
        bbox_3d=(0.30, 0.10, 0.0, 0.60, 0.40, 0.20),
    )


_LEFT = ("openarm_left_finger_pair",)


def _run(
    objects: list[Any],
    contact_links: tuple[str, ...] = _LEFT,
    voxel_m: float | None = 0.02,
    place: bool = True,
) -> tuple[list[Any], Any]:
    """Publish ``objects`` as the lifted world state, dispatch one named-target call."""
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs.msg")
    from openral_core import (
        ExecuteRskillTool,
        GraspTargetRef,
        JointState,
        PlaceTargetRef,
        WaitTool,
        WorldState,
    )
    from openral_msgs.action import ExecuteRskill
    from openral_msgs.msg import WorldStateStamped
    from openral_reasoner import ToolPalette
    from openral_reasoner_ros import ReasonerNode
    from openral_world_state_ros.lifecycle_node import build_world_state_stamped_msg
    from rclpy.action import ActionServer
    from rclpy.action.server import GoalResponse
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.parameter import Parameter
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

    from tests.integration.fakes.fake_llm import FakeToolUseClient

    received: list[Any] = []
    rclpy.init()
    try:
        reasoner = ReasonerNode(
            client=FakeToolUseClient(responses=[WaitTool() for _ in range(64)]),
            palette=ToolPalette(execute_rskill_ids=frozenset({_SKILL})),
            tick_hz=0.2,
        )
        params = [Parameter("robot_yaml", value=str(_REPO / "robots" / "openarm" / "robot.yaml"))]
        if voxel_m is not None:  # what deploy_e2e passes: its octree resolution
            params.append(Parameter("grasp_target_voxel_m", value=voxel_m))
        reasoner.set_parameters(params)
        reasoner.trigger_configure()
        reasoner.trigger_activate()
        assert reasoner._robot_description is not None

        peer = rclpy.create_node("openral_test_grounded_targets_peer")

        def _execute(goal_handle: Any) -> Any:
            received.append(goal_handle.request)
            goal_handle.succeed()
            result = ExecuteRskill.Result()
            result.success = True
            return result

        server = ActionServer(
            peer,
            ExecuteRskill,
            "/openral/execute_rskill",
            execute_callback=_execute,
            goal_callback=lambda _g: GoalResponse.ACCEPT,
        )
        pub = peer.create_publisher(
            WorldStateStamped,
            "/openral/world_state_slow",
            QoSProfile(
                depth=1,
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
            ),
        )
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(reasoner)
        executor.add_node(peer)
        world = WorldState(
            stamp_ns=1,
            joint_state=JointState(name=[], position=[], velocity=[], effort=[], stamp_ns=1),
            detected_objects=objects,
        )

        def _seen() -> bool:
            pub.publish(build_world_state_stamped_msg(None, world))
            return reasoner._world_state_msg is not None

        assert _spin_until(executor, _seen, 5.0), "world state never reached the reasoner"

        reasoner._dispatch_execute_rskill(
            ExecuteRskillTool(
                rskill_id=_SKILL,
                prompt="put the box on the shelf",
                patience_s=30.0,
                grasp_target=GraspTargetRef(label="box", contact_links=list(contact_links)),
                place_target=PlaceTargetRef(label="shelf") if place else None,
            ),
            traceparent=None,
        )
        _spin_until(executor, lambda: bool(received) and not reasoner._rskill_inflight, 5.0)
        executions = list(reasoner._renderer._executions)
        last = executions[-1] if executions else None

        executor.remove_node(reasoner)
        executor.remove_node(peer)
        executor.shutdown()
        server.destroy()
        peer.destroy_node()
        reasoner.destroy_node()
        return received, last
    finally:
        rclpy.shutdown()


def test_named_targets_reach_the_goal_as_seed_only_declarations() -> None:
    received, _ = _run([_box(-0.15), _shelf()])
    assert len(received) == 1
    goal = received[0]
    assert goal.grasp_declaration_valid
    grasp = goal.grasp_declaration
    assert grasp.target_id == "obj:box"
    assert list(grasp.contact_links) == list(_LEFT)
    assert not grasp.region_valid, "dispatch never supplies a region"
    assert grasp.search_box_valid
    assert grasp.search_box.frame_id == "openarm_base"
    assert grasp.search_box.pose.position.y == pytest.approx(-0.15, abs=1e-6)
    # The lift box (0.03 m half-extent) padded by one 20 mm cell + 15 mm extrinsic bound.
    assert grasp.search_box.half_extents.y == pytest.approx(0.065, abs=1e-6)
    assert grasp.timeout_s == pytest.approx(40.0)
    assert goal.place_declaration_valid
    place = goal.place_declaration
    assert place.target_id == "surface:shelf"
    assert not place.region_valid, "dispatch never supplies a region"
    assert place.search_box_valid
    assert place.search_box.frame_id == "openarm_base"
    assert place.search_box.pose.position.y == pytest.approx(0.25, abs=1e-6)
    assert place.search_box.half_extents.y == pytest.approx(0.15 + 0.035, abs=1e-6)


def test_no_place_target_sends_no_place_declaration() -> None:
    """Where to place is the policy's: the producer measures under the payload itself."""
    received, _ = _run([_box(-0.15)], place=False)
    assert len(received) == 1
    assert received[0].grasp_declaration_valid
    assert not received[0].place_declaration_valid


def test_a_named_surface_perception_does_not_see_sends_no_goal() -> None:
    """Option A: an optional place hint that does not ground refuses the goal — it is never
    dropped to "place on whatever surface is measured"."""
    received, last = _run([_box(-0.15)])
    assert received == [], "an ungroundable place hint must not be dispatched"
    assert last is not None and last.outcome == "failed"
    assert "place hint could not be grounded" in last.summary
    assert "place_target 'shelf'" in last.summary
    assert "Omit place_target" in last.summary


def test_no_place_hint_dispatches_and_the_runner_arms_the_goal_scope_place() -> None:
    """The normal case end to end: the reasoner sends no place declaration and the real
    runner (``place_approach_enabled``) arms its goal-scope one — ``target_id="surface"``,
    no object, no search box — so the place leg measures under the payload."""
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs.msg")
    from openral_core import (
        Action,
        ControlMode,
        ExecuteRskillTool,
        GraspTargetRef,
        JointState,
        WaitTool,
        WorldState,
    )
    from openral_msgs.msg import ActionChunk, WorldStateStamped
    from openral_msgs.msg import PlaceDeclaration as PlaceDeclarationMsg
    from openral_reasoner import ToolPalette
    from openral_reasoner_ros import ReasonerNode
    from openral_rskill.base import rSkillBase
    from openral_rskill_ros.compose import compose_so100_runtime
    from openral_world_state_ros.lifecycle_node import build_world_state_stamped_msg
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.lifecycle import TransitionCallbackReturn
    from rclpy.parameter import Parameter
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
    from std_msgs.msg import UInt64

    from tests.integration.fakes.fake_llm import FakeToolUseClient

    class _Hold(rSkillBase):
        def __init__(self) -> None:
            super().__init__(name=_SKILL, version="0.1.0", role="s1", embodiment_tags=["so100"])

        def _configure_impl(self) -> None:
            pass

        def _activate_impl(self) -> None:
            pass

        def _deactivate_impl(self) -> None:
            pass

        def _shutdown_impl(self) -> None:
            pass

        def _step_impl(self, _world_state: Any) -> Action:
            return Action(
                control_mode=ControlMode.JOINT_POSITION, horizon=1, joint_targets=[[0.0] * 6]
            )

    def _resolver(*_a: Any, **_k: Any) -> Any:
        skill = _Hold()
        skill.configure()
        skill.activate()
        return skill

    rclpy.init()
    try:
        runtime = compose_so100_runtime(skill_resolver=_resolver)
        runtime.skill_runner_node.set_parameters(
            [
                Parameter("joint_state_staleness_limit_s", value=0.5),
                Parameter("place_approach_enabled", value=True),
            ]
        )
        reasoner = ReasonerNode(
            client=FakeToolUseClient(responses=[WaitTool() for _ in range(64)]),
            palette=ToolPalette(execute_rskill_ids=frozenset({_SKILL})),
            tick_hz=0.2,
        )
        reasoner.set_parameters(
            [
                Parameter("robot_yaml", value=str(_REPO / "robots" / "openarm" / "robot.yaml")),
                Parameter("grasp_target_voxel_m", value=0.02),
            ]
        )
        peer = rclpy.create_node("openral_test_goal_scope_place_peer")
        reliable = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        # Stand in for the HAL at the wire: ack every tick as applied (startup liveness).
        applied = peer.create_publisher(UInt64, "/openral/action_applied", reliable)
        peer.create_subscription(
            ActionChunk,
            "/openral/candidate_action",
            lambda m: applied.publish(UInt64(data=int(m.tick_index))),
            reliable,
        )
        declared: list[Any] = []
        peer.create_subscription(
            PlaceDeclarationMsg,
            "/openral/place_declaration",
            declared.append,
            QoSProfile(
                depth=10,
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )
        world_pub = peer.create_publisher(WorldStateStamped, "/openral/world_state_slow", reliable)
        executor = MultiThreadedExecutor(num_threads=4)
        for node in (runtime.world_state_node, runtime.skill_runner_node, reasoner, peer):
            executor.add_node(node)
        for node in (runtime.world_state_node, runtime.skill_runner_node, reasoner):
            assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
            assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS
        world = WorldState(
            stamp_ns=1,
            joint_state=JointState(name=[], position=[], velocity=[], effort=[], stamp_ns=1),
            detected_objects=[_box(-0.15)],
        )

        def _seen() -> bool:
            world_pub.publish(build_world_state_stamped_msg(None, world))
            return reasoner._world_state_msg is not None

        assert _spin_until(executor, _seen, 5.0), "world state never reached the reasoner"
        reasoner._dispatch_execute_rskill(
            ExecuteRskillTool(
                rskill_id=_SKILL,
                prompt="put the box down",
                patience_s=30.0,
                grasp_target=GraspTargetRef(label="box", contact_links=list(_LEFT)),
            ),
            traceparent=None,
        )
        assert _spin_until(executor, lambda: any(m.active for m in declared), 10.0), (
            "a goal with no place hint must still dispatch and arm the goal-scope place"
        )
        armed = next(m for m in declared if m.active)
        executor.shutdown()
        for node in (reasoner, runtime.skill_runner_node, runtime.world_state_node):
            node.trigger_deactivate()
        reasoner.destroy_node()
        peer.destroy_node()
        runtime.skill_runner_node.destroy_node()
        runtime.world_state_node.destroy_node()
    finally:
        rclpy.shutdown()
    assert armed.target_id == "surface"
    assert armed.object_id == ""
    assert not armed.search_box_valid and not armed.region_valid


def test_an_ambiguous_named_target_sends_no_goal() -> None:
    received, last = _run([_box(-0.15), _box(0.15)])
    assert received == [], "an ambiguous target must not be dispatched"
    assert last is not None and last.outcome == "failed"
    assert "ambiguous" in last.summary


def test_a_bimanual_target_naming_no_gripper_sends_no_goal() -> None:
    received, last = _run([_box(-0.15)], contact_links=())
    assert received == [], "an unnamed gripper on a two-gripper robot must not be dispatched"
    assert last is not None and last.outcome == "failed"
    assert "2 hands" in last.summary


def _configure_without_voxel(robot: str) -> Any:
    """Configure a real ReasonerNode on ``robot`` with grasp_target_voxel_m left unset."""
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs.msg")
    from openral_reasoner import ToolPalette
    from openral_reasoner_ros import ReasonerNode
    from rclpy.parameter import Parameter

    from tests.integration.fakes.fake_llm import FakeToolUseClient

    rclpy.init()
    try:
        reasoner = ReasonerNode(
            client=FakeToolUseClient(responses=[]),
            palette=ToolPalette(execute_rskill_ids=frozenset({_SKILL})),
            tick_hz=0.2,
        )
        reasoner.set_parameters(
            [Parameter("robot_yaml", value=str(_REPO / "robots" / robot / "robot.yaml"))]
        )
        result = reasoner.trigger_configure()
        reasoner.destroy_node()
        return result
    finally:
        rclpy.shutdown()


def test_a_gripper_reasoner_without_the_deploy_voxel_resolution_fails_configure() -> None:
    """No silent cell-specific fallback, and no per-call refusal the LLM would replan on."""
    from rclpy.lifecycle import TransitionCallbackReturn

    assert _configure_without_voxel("openarm") == TransitionCallbackReturn.FAILURE


def test_a_gripperless_reasoner_configures_without_the_voxel_resolution() -> None:
    """ur5e declares no gripper: it can never ground a grasp target, so it needs no voxel."""
    from rclpy.lifecycle import TransitionCallbackReturn

    assert _configure_without_voxel("ur5e") == TransitionCallbackReturn.SUCCESS
