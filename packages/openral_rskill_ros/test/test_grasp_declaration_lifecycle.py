"""The grasp-phase declaration's goal lifecycle (real pick-and-place design §2.1).

Sibling of ``test_place_declaration_lifecycle.py``: ``rskill_runner_node`` is
the goal lifecycle authority, so it arms the grasp declaration at goal start,
strips any region (dispatch never measures), and retracts it on every exit —
success, cancel, an exception escaping the executor, and E-stop. These tests
drive the real node through ``rclpy`` and read the real
``/openral/grasp_declaration`` topic.

Per CLAUDE.md §1.11 the runtime, action server, skill and topic are all real.
The harness, resolver and goal helper are reused from the place test module.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

_ROS2_AVAILABLE = bool(os.environ.get("ROS_DISTRO"))

pytestmark = pytest.mark.skipif(
    not _ROS2_AVAILABLE,
    reason="ROS_DISTRO not set — these tests require a sourced ROS 2 installation.",
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_place_declaration_lifecycle import (
    _constant_skill_resolver,
    _run_goal,
    _spin_for,
)

_TARGET = "cell:pick_box"
_FINGERS = ("openarm_left_finger_pair", "openarm_right_finger_pair")


def _scene_json(**update: Any) -> str:
    from openral_core import GraspDeclaration

    fields: dict[str, Any] = {"contact_links": _FINGERS, **update}
    return GraspDeclaration(
        target_id=_TARGET, timeout_s=70.0, stamp_ns=0, **fields
    ).model_dump_json()


def _measured_region() -> Any:
    from openral_core import PlaceRegion, Pose6D

    return PlaceRegion(
        frame_id="openarm_base",
        pose=Pose6D(xyz=(0.45, 0.0, 0.12), quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="openarm_base"),
        half_extents=(0.06, 0.05, 0.04),
        evidence_ref="dispatch_claim:not_a_measurement",
    )


@contextmanager
def _harness(
    grasp_declaration_json: str,
    *,
    skill_resolver: Any = None,
    approach: bool = False,
    params: dict[str, Any] | None = None,
) -> Iterator[tuple[Any, Any, list[Any]]]:
    """The place harness with the grasp parameter and topic instead."""
    import rclpy
    from openral_msgs.msg import ActionChunk
    from openral_msgs.msg import GraspDeclaration as GraspDeclarationMsg
    from openral_rskill_ros.compose import compose_so100_runtime
    from rclpy.lifecycle import TransitionCallbackReturn
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
    from std_msgs.msg import UInt64

    rclpy.init()
    runtime = compose_so100_runtime(skill_resolver=skill_resolver or _constant_skill_resolver())
    runtime.skill_runner_node.set_parameters(
        [
            rclpy.parameter.Parameter("grasp_declaration_json", value=grasp_declaration_json),
            rclpy.parameter.Parameter("joint_state_staleness_limit_s", value=0.5),
            rclpy.parameter.Parameter("grasp_approach_enabled", value=approach),
            *(rclpy.parameter.Parameter(k, value=v) for k, v in (params or {}).items()),
        ]
    )
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=4)
    executor.add_node(runtime.world_state_node)
    executor.add_node(runtime.skill_runner_node)
    helper = rclpy.create_node("openral_grasp_declaration_test_helper")
    executor.add_node(helper)
    chunk_qos = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.VOLATILE,
        depth=10,
    )
    applied_pub = helper.create_publisher(UInt64, "/openral/action_applied", chunk_qos)
    helper.create_subscription(
        ActionChunk,
        "/openral/candidate_action",
        lambda msg: applied_pub.publish(UInt64(data=int(msg.tick_index))),
        chunk_qos,
    )
    seen: list[Any] = []
    helper.create_subscription(
        GraspDeclarationMsg,
        "/openral/grasp_declaration",
        seen.append,
        QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=10,
        ),
    )
    try:
        for node in (runtime.world_state_node, runtime.skill_runner_node):
            assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
            assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS
        yield executor, runtime, seen
    finally:
        for node in (runtime.skill_runner_node, runtime.world_state_node):
            try:
                node.trigger_deactivate()
                node.trigger_cleanup()
                node.trigger_shutdown()
            except RuntimeError:
                pass
        executor.shutdown()
        helper.destroy_node()
        runtime.skill_runner_node.destroy_node()
        runtime.world_state_node.destroy_node()
        rclpy.shutdown()


class _NoGoalDeclaration:
    """The shape of "this goal carries no grasp declaration of its own"."""

    grasp_declaration_valid = False


def test_no_declaration_configured_puts_nothing_on_the_wire() -> None:
    with _harness("") as (executor, runtime, seen):
        _run_goal(executor, runtime.skill_runner_node)
        _spin_for(executor, 0.3)
    assert seen == []


def test_a_goal_arms_then_retracts_the_scene_declaration() -> None:
    with _harness(_scene_json()) as (executor, runtime, seen):
        _run_goal(executor, runtime.skill_runner_node)
        _spin_for(executor, 0.3)

    active = [msg for msg in seen if msg.active]
    retracted = [msg for msg in seen if not msg.active]
    assert len(active) == 1
    assert len(retracted) == 1, "the goal ended without retracting its grasp declaration"
    assert active[0].target_id == _TARGET
    assert tuple(active[0].contact_links) == _FINGERS
    assert active[0].rskill_id == "openral/test-place-declaration-skill"
    assert active[0].stamp_ns > 0
    assert not active[0].region_valid


def test_approach_mode_arms_a_goal_scope_declaration_naming_no_target() -> None:
    """No goal or scene declaration, approach on: every hand, no target, no box, retracted."""
    with _harness("", approach=True) as (executor, runtime, seen):
        _run_goal(executor, runtime.skill_runner_node, deadline_s=3.0)
        _spin_for(executor, 0.3)

    active = [msg for msg in seen if msg.active]
    assert len(active) == 1
    assert active[0].target_id == "approach"
    # compose_so100_runtime's SO-100 manifest: its one jaw is its one hand.
    assert tuple(active[0].contact_links) == ("gripper_finger",)
    assert not active[0].search_box_valid and not active[0].region_valid
    assert active[0].rskill_id == "openral/test-place-declaration-skill"
    assert active[0].stamp_ns > 0
    assert active[0].timeout_s == pytest.approx(3.0), "the backstop is the goal deadline"
    assert not seen[-1].active, "the goal ended without retracting its goal-scope declaration"


def test_a_scene_declaration_wins_over_approach_mode() -> None:
    with _harness(_scene_json(), approach=True) as (executor, runtime, seen):
        _run_goal(executor, runtime.skill_runner_node)
        _spin_for(executor, 0.3)
    assert [msg.target_id for msg in seen if msg.active] == [_TARGET]


def test_cancelling_a_goal_retracts_its_declaration() -> None:
    with _harness(_scene_json()) as (executor, runtime, seen):
        _run_goal(executor, runtime.skill_runner_node, deadline_s=3.0, cancel=True)
        _spin_for(executor, 0.3)
    assert any(msg.active for msg in seen)
    assert not seen[-1].active, "a cancelled goal left its grasp declaration live"


def test_an_estop_retracts_the_declaration_immediately() -> None:
    from std_msgs.msg import Empty

    with _harness(_scene_json()) as (executor, runtime, seen):
        _spin_for(executor, 0.2)
        runtime.skill_runner_node._arm_grasp_declaration(
            _NoGoalDeclaration(),
            rskill_id="openral/x",
            trace_id="t",
            invocation_id="11111111111111111111111111111111",
        )
        _spin_for(executor, 0.2)
        assert seen and seen[-1].active
        runtime.skill_runner_node._on_estop(Empty())
        _spin_for(executor, 0.3)
    assert not seen[-1].active, "the E-stop left the grasp declaration live"


def test_an_exception_escaping_the_executor_still_retracts() -> None:
    def _exploding_resolver(*_args: Any, **_kwargs: Any) -> Any:
        raise TypeError("resolver blew up")

    with _harness(_scene_json(), skill_resolver=_exploding_resolver) as (executor, runtime, seen):
        _run_goal(executor, runtime.skill_runner_node)
        _spin_for(executor, 0.3)
    assert any(msg.active for msg in seen)
    assert not seen[-1].active, "a goal that died mid-executor left its grasp declaration live"


def test_a_dispatch_supplied_region_never_leaves_this_node() -> None:
    """Goal-carried region (scene JSON is parsed bare, so DeployScene's refusal
    does not cover it either): stripped before publication."""
    from openral_core import GraspDeclaration
    from openral_msgs.action import ExecuteRskill

    with _harness(_scene_json(region=_measured_region())) as (executor, runtime, seen):
        _spin_for(executor, 0.2)
        # Scene parameter carrying a region.
        runtime.skill_runner_node._arm_grasp_declaration(
            _NoGoalDeclaration(),
            rskill_id="openral/z",
            trace_id="t3",
            invocation_id="33333333333333333333333333333333",
        )
        _spin_for(executor, 0.2)
        assert seen and seen[-1].active and not seen[-1].region_valid
        # Goal carrying a region — and the goal wins over the scene.
        request = ExecuteRskill.Goal()
        request.grasp_declaration_valid = True
        GraspDeclaration(
            target_id="cell:other_box",
            contact_links=("openarm_left_finger_pair",),
            timeout_s=30.0,
            stamp_ns=0,
            region=_measured_region(),
            search_box=_measured_region(),
        ).fill_idl(request.grasp_declaration)
        assert request.grasp_declaration.region_valid
        runtime.skill_runner_node._arm_grasp_declaration(
            request,
            rskill_id="openral/z",
            trace_id="t4",
            invocation_id="44444444444444444444444444444444",
        )
        _spin_for(executor, 0.3)

    active = [msg for msg in seen if msg.active]
    assert active[-1].target_id == "cell:other_box"
    assert active[-1].rskill_id == "openral/z"
    assert active[-1].trace_id == "t4"
    assert not active[-1].region_valid, "a dispatch-supplied region reached the kernel's path"
    assert active[-1].search_box_valid, "the search box only seeds perception; it must pass"


def test_a_malformed_scene_declaration_is_refused_not_guessed() -> None:
    """Active with no contact_links: refused, the goal runs with no declaration."""
    bad = '{"target_id": "cell:pick_box", "timeout_s": 70.0, "stamp_ns": 0}'
    with _harness(bad) as (executor, runtime, seen):
        _run_goal(executor, runtime.skill_runner_node)
        _spin_for(executor, 0.3)
    assert seen == []


def test_a_reasoner_grounded_goal_declaration_reaches_the_wire_with_its_search_box() -> None:
    """The reasoner path: no scene block, the goal carries what the reasoner grounded.

    The declaration is the real grounding output for one lifted detection; the runner
    arms it with the seed box intact and no region, then retracts it at goal end.
    """
    from openral_core import DetectedObject, GraspTargetRef, Pose6D
    from openral_reasoner.grounding import ground_grasp_target

    box = DetectedObject(
        label="box",
        confidence=0.9,
        pose=Pose6D(
            xyz=(0.40, -0.15, -0.10), quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="openarm_base"
        ),
        bbox_3d=(0.37, -0.18, -0.13, 0.43, -0.12, -0.07),
    )
    grounded = ground_grasp_target(
        GraspTargetRef(label="box", contact_links=[_FINGERS[0]]),
        live_objects=[box],
        scene_graph=None,
        base_frame="openarm_base",
        default_contact_links=_FINGERS,
        patience_s=0.4,
        pad_m=0.035,
    )

    def _fill(goal: Any) -> None:
        goal.grasp_declaration_valid = True
        grounded.fill_idl(goal.grasp_declaration)

    with _harness("") as (executor, runtime, seen):
        _run_goal(executor, runtime.skill_runner_node, fill_goal=_fill)
        _spin_for(executor, 0.3)

    active = [msg for msg in seen if msg.active]
    assert len(active) == 1
    assert active[0].target_id == "obj:box"
    assert active[0].rskill_id == "openral/test-place-declaration-skill"
    assert not active[0].region_valid, "only the producer measures a region"
    assert active[0].search_box_valid
    assert active[0].search_box.frame_id == "openarm_base"
    assert active[0].search_box.half_extents.x == pytest.approx(0.03 + 0.035)
    assert not seen[-1].active, "the goal ended without retracting its grasp declaration"


# ── Object tracking (issue #418): standing queries ride the declaration ──

#: What `deploy sim` derives for LeIsaac PickOrange ("Grab orange and place into plate").
_PICK_ORANGE_QUERIES = (
    '[{"label": "orange", "count": "all", "role": "target"},'
    ' {"label": "plate", "count": "one", "role": "destination"}]'
)
#: Tracking on with no tracker running: the goal waits out a short grounding wait.
_TRACKING_ON: dict[str, Any] = {
    "object_tracking": "on",
    "object_queries_json": _PICK_ORANGE_QUERIES,
    "grounding_wait_s": 0.2,
}


def test_tracking_on_attaches_the_target_queries_with_the_hand_hint() -> None:
    from openral_core import ObjectQuery, hand_joint

    with _harness(
        _scene_json(contact_links=("gripper_finger",)),
        params=_TRACKING_ON,
    ) as (executor, runtime, seen):
        _run_goal(executor, runtime.skill_runner_node)
        _spin_for(executor, 0.3)
        hand = hand_joint(runtime.skill_runner_node._description, ["gripper_finger"])

    active = [msg for msg in seen if msg.active]
    assert len(active) == 1
    assert hand is not None
    assert [ObjectQuery.from_idl(q) for q in active[0].target_queries] == [
        ObjectQuery(label="orange", count="all", role="target", hand=hand)
    ], "the destination query rides the place declaration, never target_queries"
    assert active[0].target_id == _TARGET  # the legacy fields are unchanged
    assert not seen[-1].active


def test_tracking_off_leaves_the_declaration_as_before() -> None:
    with _harness(
        _scene_json(),
        params={"object_tracking": "off", "object_queries_json": _PICK_ORANGE_QUERIES},
    ) as (executor, runtime, seen):
        _run_goal(executor, runtime.skill_runner_node)
        _spin_for(executor, 0.3)
    active = [msg for msg in seen if msg.active]
    assert len(active) == 1
    assert list(active[0].target_queries) == []


def test_a_reasoner_grounded_three_orange_goal_reaches_the_wire_as_one_count_all_query() -> None:
    """Three oranges: with tracking the grounding does not refuse; the goal carries the
    query and no search box, and the runner keeps the goal's own query."""
    from openral_core import DetectedObject, GraspTargetRef, ObjectQuery, Pose6D
    from openral_reasoner.grounding import ground_grasp_target

    oranges = [
        DetectedObject(
            label="orange",
            confidence=0.9,
            pose=Pose6D(
                xyz=(0.4, y, -0.1), quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="openarm_base"
            ),
            bbox_3d=(0.37, y - 0.03, -0.13, 0.43, y + 0.03, -0.07),
        )
        for y in (-0.2, 0.0, 0.2)
    ]
    grounded = ground_grasp_target(
        GraspTargetRef(label="orange", contact_links=[_FINGERS[0]]),
        live_objects=oranges,
        scene_graph=None,
        base_frame="openarm_base",
        default_contact_links=_FINGERS,
        patience_s=0.4,
        pad_m=0.035,
        object_tracking=True,
    )

    def _fill(goal: Any) -> None:
        from openral_msgs.msg import ObjectQuery as ObjectQueryMsg

        goal.grasp_declaration_valid = True
        grounded.fill_idl(goal.grasp_declaration, query_factory=ObjectQueryMsg)

    with _harness("", params=_TRACKING_ON) as (executor, runtime, seen):
        _run_goal(executor, runtime.skill_runner_node, fill_goal=_fill)
        _spin_for(executor, 0.3)

    active = [msg for msg in seen if msg.active]
    assert len(active) == 1
    assert active[0].target_id == "obj:orange"
    assert not active[0].search_box_valid
    assert [ObjectQuery.from_idl(q) for q in active[0].target_queries] == [
        ObjectQuery(label="orange", count="all", role="target")
    ]


# ── Grounding wait (issue #418 design step 1.4): grounding_wait_s / grounding_required ──


def _orange_set(key: str, *, status: str = "confirmed", stamp_ns: int = 1) -> Any:
    """A tracker's set holding one ``orange`` TARGET track keyed to ``key``."""
    from openral_core import (
        FitKind,
        ObjectRole,
        PlaceRegion,
        Pose6D,
        TrackedObject,
        TrackedObjectSet,
        TrackStatus,
    )
    from openral_msgs.msg import AttachedCollisionPrimitive
    from openral_msgs.msg import TrackedObject as ObjectMsg
    from openral_msgs.msg import TrackedObjectSet as SetMsg

    box = PlaceRegion(
        frame_id="base",
        pose=Pose6D(xyz=(0.07, -0.30, 0.05), quat_xyzw=(0, 0, 0, 1), frame_id="base"),
        half_extents=(0.03, 0.03, 0.03),
    )
    track = TrackedObject(
        object_id="trk:1",
        epoch=0,
        label="orange",
        role=ObjectRole.TARGET,
        declaration_key=key,
        status=TrackStatus(status),
        box=box,
        fit_kind=FitKind.SILHOUETTE_CONE,
        support_z=0.02,
        support_ref="map",
        floor_z=0.035,
        visible_cells=4,
        occupied_cells=4,
        last_fit_stamp_ns=stamp_ns,
        last_confirmed_stamp_ns=stamp_ns,
    )
    msg = SetMsg()
    tracked = TrackedObjectSet(
        frame_id="base", stamp_ns=stamp_ns, revision=stamp_ns, objects=[track]
    )
    tracked.fill_idl(msg, object_factory=ObjectMsg, primitive_factory=AttachedCollisionPrimitive)
    return msg


def _tracker_stand_in(executor: Any, *, status: str) -> list[Any]:
    """Publish what a tracker would, keyed to every armed declaration (``status`` track)."""
    import rclpy
    from openral_msgs.msg import GraspDeclaration as GraspDeclarationMsg
    from openral_msgs.msg import TrackedObjectSet as SetMsg
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

    node = rclpy.create_node("openral_grounding_tracker_stand_in")
    executor.add_node(node)
    pub = node.create_publisher(
        SetMsg,
        "/openral/tracked_objects",
        QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        ),
    )
    keys: list[str] = []

    def _on_decl(msg: Any) -> None:
        if msg.active:
            keys.append(f"{msg.target_id}@{msg.stamp_ns}")

    node.create_subscription(
        GraspDeclarationMsg,
        "/openral/grasp_declaration",
        _on_decl,
        QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=10,
        ),
    )
    node.create_timer(0.05, lambda: keys and pub.publish(_orange_set(keys[-1], status=status)))
    return keys


def test_grounding_required_refuses_a_goal_whose_target_never_grounds() -> None:
    import time

    from openral_msgs.action import ExecuteRskill

    params = {**_TRACKING_ON, "grounding_wait_s": 0.5, "grounding_required": True}
    with _harness(_scene_json(contact_links=("gripper_finger",)), params=params) as (
        executor,
        runtime,
        seen,
    ):
        t0 = time.monotonic()
        response = _run_goal(executor, runtime.skill_runner_node, deadline_s=3.0)
        waited = time.monotonic() - t0
        _spin_for(executor, 0.3)
    result = response.result
    assert not result.success
    assert result.failure_kind == ExecuteRskill.Result.FAILURE_PERCEPTION_STALE, (
        result.failure_reason
    )
    assert "ROSPerceptionStale" in result.failure_reason and "orange" in result.failure_reason
    assert waited >= 0.5, "refused before grounding_wait_s elapsed"
    assert not seen[-1].active, "the refused goal left its grasp declaration live"


def test_grounding_required_refuses_a_target_seen_only_tentatively() -> None:
    from openral_msgs.action import ExecuteRskill

    params = {**_TRACKING_ON, "grounding_wait_s": 0.5, "grounding_required": True}
    with _harness(_scene_json(contact_links=("gripper_finger",)), params=params) as (
        executor,
        runtime,
        _seen,
    ):
        _tracker_stand_in(executor, status="tentative")
        response = _run_goal(executor, runtime.skill_runner_node, deadline_s=3.0)
    assert response.result.failure_kind == ExecuteRskill.Result.FAILURE_PERCEPTION_STALE


def test_a_confirmed_target_track_grounds_a_required_goal() -> None:
    from openral_msgs.action import ExecuteRskill

    params = {**_TRACKING_ON, "grounding_wait_s": 2.0, "grounding_required": True}
    with _harness(_scene_json(contact_links=("gripper_finger",)), params=params) as (
        executor,
        runtime,
        _seen,
    ):
        keys = _tracker_stand_in(executor, status="confirmed")
        response = _run_goal(executor, runtime.skill_runner_node, deadline_s=0.4)
    assert keys, "no declaration was armed"
    assert response.result.failure_kind != ExecuteRskill.Result.FAILURE_PERCEPTION_STALE, (
        response.result.failure_reason
    )


def test_without_grounding_required_the_goal_proceeds_after_the_wait() -> None:
    import time

    from openral_msgs.action import ExecuteRskill

    params = {**_TRACKING_ON, "grounding_wait_s": 0.5, "grounding_required": False}
    with _harness(_scene_json(contact_links=("gripper_finger",)), params=params) as (
        executor,
        runtime,
        _seen,
    ):
        t0 = time.monotonic()
        response = _run_goal(executor, runtime.skill_runner_node, deadline_s=0.4)
        waited = time.monotonic() - t0
    assert response.result.failure_kind != ExecuteRskill.Result.FAILURE_PERCEPTION_STALE
    assert waited >= 0.5, "the goal never waited grounding_wait_s"


def test_tracking_on_without_a_grounding_wait_refuses_to_configure() -> None:
    import rclpy
    from openral_rskill_ros.compose import compose_so100_runtime
    from rclpy.lifecycle import TransitionCallbackReturn

    rclpy.init()
    try:
        runtime = compose_so100_runtime(skill_resolver=_constant_skill_resolver())
        node = runtime.skill_runner_node
        node.set_parameters(
            [
                rclpy.parameter.Parameter("object_tracking", value="on"),
                rclpy.parameter.Parameter("object_queries_json", value=_PICK_ORANGE_QUERIES),
            ]
        )
        assert runtime.world_state_node.trigger_configure() == TransitionCallbackReturn.SUCCESS
        assert node.trigger_configure() == TransitionCallbackReturn.FAILURE
        node.destroy_node()
        runtime.world_state_node.destroy_node()
    finally:
        rclpy.shutdown()
