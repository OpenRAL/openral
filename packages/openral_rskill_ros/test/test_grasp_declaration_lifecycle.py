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


def test_an_ungrounded_context_query_holds_the_wait_only_grounding_context_wait_s() -> None:
    """A grounded TARGET plus a never-seen neighbour: the goal starts after the short
    context wait, not the full ``grounding_wait_s``, and is not refused (Spark restock,
    2026-10-28: a hidden carton held every episode at the full 20 s)."""
    import time

    from openral_msgs.action import ExecuteRskill

    params = {
        **_TRACKING_ON,
        "object_queries_json": (
            '[{"label": "orange", "count": "all", "role": "target"},'
            ' {"label": "carton", "count": "all", "role": "context", "min_instances": 3}]'
        ),
        "grounding_wait_s": 6.0,
        "grounding_context_wait_s": 0.3,
        "grounding_required": True,
    }
    with _harness(_scene_json(contact_links=("gripper_finger",)), params=params) as (
        executor,
        runtime,
        _seen,
    ):
        _tracker_stand_in(executor, status="confirmed")
        t0 = time.monotonic()
        response = _run_goal(executor, runtime.skill_runner_node, deadline_s=0.4)
        waited = time.monotonic() - t0
    assert response.result.failure_kind != ExecuteRskill.Result.FAILURE_PERCEPTION_STALE, (
        response.result.failure_reason
    )
    assert waited < 4.0, f"the context query held the goal {waited:.1f} s"


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


# ── Per-goal object queries: grounding is for manipulation goals only ──

_REPO = Path(__file__).resolve().parents[3]
_NAV2 = "OpenRAL/rskill-nav2-mobile_base-navigate_to_pose"
_PI05 = "OpenRAL/rskill-pi05-makermods_metal-pick_place-bf16"
#: The Isaac shelf aisle deploy's standing queries (`deploy sim` from the graded task).
_SHELF_AISLE_QUERIES = (
    '[{"label": "stock object", "count": "one", "role": "target"},'
    ' {"label": "carton", "count": "all", "role": "context"}]'
)


def _in_tree_resolver() -> Any:
    """The production two-phase resolver over the in-tree ``rskills/`` (no Hub)."""
    from openral_rskill_ros.rskill_runner_node import ManifestSkillResolver

    return ManifestSkillResolver(search_paths=[_REPO / "rskills"], hub_fallback=False)


def _named(rskill_id: str, fill: Any = None) -> Any:
    def _fill(goal: Any) -> None:
        goal.rskill_id = rskill_id
        if fill is not None:
            fill(goal)

    return _fill


def _place_topic(executor: Any) -> list[Any]:
    import rclpy
    from openral_msgs.msg import PlaceDeclaration as PlaceDeclarationMsg
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

    node = rclpy.create_node("openral_place_declaration_listener")
    executor.add_node(node)
    seen: list[Any] = []
    node.create_subscription(
        PlaceDeclarationMsg,
        "/openral/place_declaration",
        seen.append,
        QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=10,
        ),
    )
    return seen


def test_a_base_only_goal_under_tracking_arms_nothing_and_waits_for_nothing() -> None:
    """Nav2 with the shelf aisle deploy's standing queries and grounding_required: no grasp or
    place declaration, no query, no grounding wait (Isaac run, 2026-11-16: refused twice,
    ``queries ['stock object', 'carton'] not grounded``). The goal then meets the manifest
    gate (Nav2 cannot run on the SO-100), which is past the grounding stage."""
    import time

    from openral_msgs.action import ExecuteRskill

    params = {
        "object_tracking": "tracking",
        "object_queries_json": _SHELF_AISLE_QUERIES,
        "grounding_wait_s": 5.0,
        "grounding_required": True,
        "place_approach_enabled": True,
    }
    with _harness(
        _scene_json(contact_links=("gripper_finger",)),
        skill_resolver=_in_tree_resolver(),
        params=params,
    ) as (executor, runtime, seen):
        places = _place_topic(executor)
        t0 = time.monotonic()
        response = _run_goal(executor, runtime.skill_runner_node, fill_goal=_named(_NAV2))
        waited = time.monotonic() - t0
        _spin_for(executor, 0.3)
    result = response.result
    assert result.failure_kind == ExecuteRskill.Result.FAILURE_CAPABILITY_MISMATCH, (
        result.failure_reason
    )
    assert waited < 3.0, f"a base-only goal waited {waited:.1f} s for grounding"
    assert not [m for m in seen if m.active], "a base-only goal armed a grasp declaration"
    assert not [m for m in places if m.active], "a base-only goal armed a place declaration"


def test_a_manipulation_goal_naming_no_object_under_tracking_is_refused_before_motion() -> None:
    """The scene's declaration has a search box but no query, the deploy has no standing
    query and the goal no grasp_target: refused (FAILURE_PLANNING_ERROR) before the
    declaration is published or anything moves (Isaac run, 2026-11-16: the box went
    untracked and its stand-in clipped the shelf)."""
    from openral_msgs.action import ExecuteRskill

    params = {
        "object_tracking": "tracking",
        "grounding_wait_s": 0.2,
        "place_approach_enabled": True,
    }
    with _harness(
        _scene_json(contact_links=("gripper_finger",)),
        skill_resolver=_in_tree_resolver(),
        params=params,
    ) as (executor, runtime, seen):
        places = _place_topic(executor)
        response = _run_goal(executor, runtime.skill_runner_node, fill_goal=_named(_PI05))
        _spin_for(executor, 0.3)
        chunks = runtime.skill_runner_node._chunks_published
    result = response.result
    assert not result.success
    assert result.failure_kind == ExecuteRskill.Result.FAILURE_PLANNING_ERROR, result.failure_reason
    assert result.failure_reason.startswith("ROSReasonerInvalidPlan: object tracking is on")
    assert "grasp_target" in result.failure_reason and _PI05 in result.failure_reason
    assert not [m for m in seen if m.active], "the refused goal published its declaration"
    assert not [m for m in places if m.active], "the refused goal published a place zone"
    assert chunks == 0


@pytest.mark.parametrize("scene", ["", "malformed"])
def test_a_grasping_goal_with_no_declaration_in_force_is_refused_too(scene: str) -> None:
    """Tracking on and nothing to grasp declared at all (no scene declaration, or one that
    does not validate, no approach, no grasp_target): refused, never run untracked."""
    from openral_msgs.action import ExecuteRskill

    raw = '{"target_id": "cell:pick_box", "timeout_s": 70.0, "stamp_ns": 0}' if scene else ""
    params = {"object_tracking": "tracking", "grounding_wait_s": 0.2}
    with _harness(raw, skill_resolver=_in_tree_resolver(), params=params) as (
        executor,
        runtime,
        seen,
    ):
        response = _run_goal(executor, runtime.skill_runner_node, fill_goal=_named(_PI05))
        _spin_for(executor, 0.3)
    assert response.result.failure_kind == ExecuteRskill.Result.FAILURE_PLANNING_ERROR, (
        response.result.failure_reason
    )
    assert not [m for m in seen if m.active]


def test_a_skill_that_grasps_nothing_is_not_asked_for_a_grasp_target() -> None:
    """A MoveIt look-at actuates the arm and grasps nothing: under tracking it is not refused
    for naming no object (it then meets the SO-100's manifest gate)."""
    from openral_msgs.action import ExecuteRskill

    params = {"object_tracking": "tracking", "grounding_wait_s": 0.2}
    with _harness(
        _scene_json(contact_links=("gripper_finger",)),
        skill_resolver=_in_tree_resolver(),
        params=params,
    ) as (executor, runtime, _seen):
        response = _run_goal(
            executor,
            runtime.skill_runner_node,
            fill_goal=_named("OpenRAL/rskill-moveit-multi-look_at"),
        )
    result = response.result
    assert result.failure_kind != ExecuteRskill.Result.FAILURE_PLANNING_ERROR, result.failure_reason
    assert "must name the object" not in result.failure_reason


def test_a_manipulation_goal_with_its_own_query_reaches_the_grounding_wait() -> None:
    """The reasoner named ``orange`` with no live detection: the query rides the goal
    (count one, no search box) and the goal grounds against the tracker's track."""
    from openral_core import GraspTargetRef, ObjectQuery
    from openral_msgs.action import ExecuteRskill
    from openral_reasoner.grounding import ground_grasp_target

    grounded = ground_grasp_target(
        GraspTargetRef(label="orange", contact_links=[_FINGERS[0]]),
        live_objects=[],
        scene_graph=None,
        base_frame="openarm_base",
        default_contact_links=_FINGERS,
        patience_s=0.4,
        pad_m=0.035,
        object_tracking=True,
    )
    assert grounded.search_box is None
    assert grounded.target_queries == (ObjectQuery(label="orange", count="one", role="target"),)

    def _fill(goal: Any) -> None:
        from openral_msgs.msg import ObjectQuery as ObjectQueryMsg

        goal.grasp_declaration_valid = True
        grounded.fill_idl(goal.grasp_declaration, query_factory=ObjectQueryMsg)

    params = {"object_tracking": "tracking", "grounding_wait_s": 2.0, "grounding_required": True}
    with _harness("", skill_resolver=_in_tree_resolver(), params=params) as (
        executor,
        runtime,
        seen,
    ):
        keys = _tracker_stand_in(executor, status="confirmed")
        response = _run_goal(executor, runtime.skill_runner_node, fill_goal=_named(_PI05, _fill))
    assert keys, "no declaration was armed"
    active = [m for m in seen if m.active]
    assert [ObjectQuery.from_idl(q) for q in active[0].target_queries] == [
        ObjectQuery(label="orange", count="one", role="target")
    ]
    # Grounded, then the manifest gate (pi0.5 metal cannot run on the SO-100).
    assert response.result.failure_kind == ExecuteRskill.Result.FAILURE_CAPABILITY_MISMATCH, (
        response.result.failure_reason
    )


def test_the_standing_query_preload_path_is_unchanged_for_a_manipulation_skill() -> None:
    """A fixed-base deploy's standing queries still ride the scene declaration of a
    manipulation goal (real pi0.5 manifest) and still gate it on grounding."""
    from openral_core import ObjectQuery
    from openral_msgs.action import ExecuteRskill

    params = {**_TRACKING_ON, "grounding_wait_s": 0.5, "grounding_required": True}
    with _harness(
        _scene_json(contact_links=("gripper_finger",)),
        skill_resolver=_in_tree_resolver(),
        params=params,
    ) as (executor, runtime, seen):
        response = _run_goal(
            executor, runtime.skill_runner_node, deadline_s=3.0, fill_goal=_named(_PI05)
        )
        _spin_for(executor, 0.3)
    active = [m for m in seen if m.active]
    assert len(active) == 1
    assert [ObjectQuery.from_idl(q).label for q in active[0].target_queries] == ["orange"]
    assert response.result.failure_kind == ExecuteRskill.Result.FAILURE_PERCEPTION_STALE, (
        response.result.failure_reason
    )


# ── The goal's declaration completed from the scene's (AMR aisle sim, 2026-11-16) ──

_SMOLVLA = "OpenRAL/rskill-smolvla-franka_panda-libero_spatial-bf16"  # max_execution_s 60
_LINGBOT = "OpenRAL/rskill-lingbot_va_a1-galaxea_a1-fruit_placement-bf16"  # 420
_LEFT, _RIGHT = ("openarm_left_ee_link1",), ("openarm_right_ee_link1",)


def _aisle_box(xyz: tuple[float, float, float], half: float, evidence: str) -> Any:
    from openral_core import PlaceRegion, Pose6D

    pose = Pose6D(xyz=xyz, quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="base_footprint")
    return PlaceRegion(
        frame_id="base_footprint",
        pose=pose,
        half_extents=(half, half, half),
        evidence_ref=evidence,
    )


def _aisle_scene(links: tuple[str, ...] = _LEFT) -> str:
    """The aisle scene's declaration: the tote the robot carries, in base_footprint."""
    from openral_core import GraspDeclaration

    return GraspDeclaration(
        target_id="cell:stock_object",
        contact_links=links,
        timeout_s=70.0,
        stamp_ns=0,
        search_box=_aisle_box(
            (0.504, 0.085, 0.457), 0.08, "reasoner_seed:detection 'stock_object'"
        ),
        workspace_box=_aisle_box((0.504, 0.085, 0.45), 0.2, "sim:tote"),
    ).model_dump_json()


def _reasoner_box_goal(links: tuple[str, ...] = (), declared_hand: tuple[str, ...] = _LEFT) -> Any:
    """What the reasoner sends: grasp_target 'box', no live detection, patience_s=0; by
    default naming no hand, so it takes the deployment's declared one (the scene's)."""
    from openral_core import GraspTargetRef
    from openral_reasoner.grounding import ground_grasp_target

    grounded = ground_grasp_target(
        GraspTargetRef(label="box", contact_links=list(links)),
        live_objects=[],
        scene_graph=None,
        base_frame="base_footprint",
        default_contact_links=(_LEFT, _RIGHT),
        patience_s=0.0,
        pad_m=0.035,
        object_tracking=True,
        declared_hand=declared_hand,
    )
    assert grounded.timeout_s == 10.0 and grounded.search_box is None

    def _fill(goal: Any) -> None:
        from openral_msgs.msg import ObjectQuery as ObjectQueryMsg

        goal.grasp_declaration_valid = True
        grounded.fill_idl(goal.grasp_declaration, query_factory=ObjectQueryMsg)

    return _fill


@pytest.mark.parametrize("mode", ["off", "tracking", "on"])
def test_the_goal_query_rides_the_scene_boxes_hand_and_lifetime(capfd: Any, mode: str) -> None:
    """A label-only grasp_target: the reasoner took the scene's hand (`grasp_contact_links`),
    so the merge takes the scene's boxes too (`hand=same`), in every tracking mode (the
    merge reads no mode; Pro `on` included)."""
    from openral_core import GraspDeclaration, ObjectQuery, PlaceRegion

    params = {} if mode == "off" else {"object_tracking": mode, "grounding_wait_s": 0.2}
    scene = GraspDeclaration.model_validate_json(_aisle_scene())
    with _harness(_aisle_scene(), skill_resolver=_in_tree_resolver(), params=params) as (
        executor,
        runtime,
        seen,
    ):
        _run_goal(
            executor, runtime.skill_runner_node, fill_goal=_named(_PI05, _reasoner_box_goal())
        )
        _spin_for(executor, 0.3)
    active = [m for m in seen if m.active]
    assert len(active) == 1
    armed = active[0]
    assert armed.target_id == "obj:box", "the goal's key: queries, tracks and kernel share it"
    assert tuple(armed.contact_links) == _LEFT
    assert armed.timeout_s == pytest.approx(70.0)
    assert armed.search_box_valid and armed.workspace_box_valid
    assert PlaceRegion.from_idl(armed.search_box) == scene.search_box
    assert PlaceRegion.from_idl(armed.workspace_box) == scene.workspace_box
    assert [ObjectQuery.from_idl(q) for q in armed.target_queries] == [
        ObjectQuery(label="box", count="one", role="target", seed=scene.search_box)
    ], "the scene's non-sim search box seeds the goal's TARGET query"
    err = capfd.readouterr().err
    assert "rskill_runner.grasp_declaration_merged" in err and "hand=same" in err


def test_a_goal_naming_the_other_hand_takes_no_scene_box(capfd: Any) -> None:
    """The LLM named the right hand, the scene grasps with the left: the goal keeps its own
    links and takes nothing from the scene, not even its lifetime (warning); a failure the
    hand can explain says so, for the retry."""
    from openral_msgs.action import ExecuteRskill

    with _harness(_aisle_scene(_LEFT), skill_resolver=_in_tree_resolver()) as (
        executor,
        runtime,
        seen,
    ):
        wrong = _run_goal(
            executor,
            runtime.skill_runner_node,
            fill_goal=_named(_PI05, _reasoner_box_goal(_RIGHT)),
        )
        _spin_for(executor, 0.3)
        armed = next(m for m in seen if m.active)
        # The wrong-hand goal's note on a failure the hand explains (a contact stop).
        stopped = ExecuteRskill.Result()
        stopped.failure_kind = ExecuteRskill.Result.FAILURE_SAFETY_ESTOP
        stopped.failure_reason = "safety_estop:collision"
        noted = runtime.skill_runner_node._note_grasp_hand(stopped).failure_reason
        right = _run_goal(
            executor, runtime.skill_runner_node, fill_goal=_named(_PI05, _reasoner_box_goal())
        )
    assert tuple(armed.contact_links) == _RIGHT
    assert not armed.search_box_valid and not armed.workspace_box_valid
    assert all(not q.seed_valid for q in armed.target_queries)
    assert armed.timeout_s == pytest.approx(10.4), "the goal's 0.4 s budget + margin"
    err = capfd.readouterr().err
    assert "rskill_runner.grasp_declaration_merged" in err and "hand=different" in err
    assert noted.endswith(
        "(note: the goal's grasp_target named the hand ['openarm_right_ee_link1'], but this "
        "deployment grasps with ['openarm_left_ee_link1']; leave grasp_target.contact_links "
        "unset to use the deployment's hand)"
    ), noted
    # Both fail at the SO-100's manifest gate, which says nothing about the hand: no note.
    assert wrong.result.failure_kind == ExecuteRskill.Result.FAILURE_CAPABILITY_MISMATCH
    assert "named the hand" not in wrong.result.failure_reason
    assert "named the hand" not in right.result.failure_reason


@pytest.mark.parametrize(
    ("rskill_id", "deadline_s", "timeout_s"),
    [
        (_PI05, 0.4, 10.4),  # the goal's own deadline + the margin
        (_SMOLVLA, 0.0, 70.0),  # max_execution_s 60 + 10
        (_LINGBOT, 0.0, 120.0),  # 420 + 10, capped at GraspDeclaration.MAX_TIMEOUT_S
    ],
)
def test_without_a_scene_declaration_only_the_lifetime_floor_changes(
    rskill_id: str, deadline_s: float, timeout_s: float
) -> None:
    from openral_core import ObjectQuery

    with _harness("", skill_resolver=_in_tree_resolver()) as (executor, runtime, seen):
        _run_goal(
            executor,
            runtime.skill_runner_node,
            deadline_s=deadline_s,
            fill_goal=_named(rskill_id, _reasoner_box_goal()),
        )
        _spin_for(executor, 0.3)
    armed = next(m for m in seen if m.active)
    assert armed.target_id == "obj:box"
    assert tuple(armed.contact_links) == _LEFT
    assert not armed.search_box_valid and not armed.workspace_box_valid
    assert [ObjectQuery.from_idl(q) for q in armed.target_queries] == [
        ObjectQuery(label="box", count="one", role="target")
    ]
    assert armed.timeout_s == pytest.approx(timeout_s)


def test_the_scene_declaration_alone_is_armed_unchanged() -> None:
    """No goal declaration (the fixed-base preload path): the scene's, as before."""
    from openral_core import GraspDeclaration, PlaceRegion

    scene = GraspDeclaration.model_validate_json(_aisle_scene())
    with _harness(_aisle_scene(), skill_resolver=_in_tree_resolver()) as (executor, runtime, seen):
        _run_goal(executor, runtime.skill_runner_node, fill_goal=_named(_PI05))
        _spin_for(executor, 0.3)
    armed = next(m for m in seen if m.active)
    assert armed.target_id == "cell:stock_object"
    assert armed.timeout_s == pytest.approx(70.0)
    assert PlaceRegion.from_idl(armed.search_box) == scene.search_box
    assert list(armed.target_queries) == []


def test_the_deploys_context_queries_ride_a_goal_that_names_its_own_target() -> None:
    """The goal names its target ('box'); the deploy's standing CONTEXT query ('carton') still
    rides the armed declaration, its TARGET query ('stock object') does not (Spark R273)."""
    from openral_core import ObjectQuery

    params = {
        "object_tracking": "tracking",
        "object_queries_json": _SHELF_AISLE_QUERIES,
        "grounding_wait_s": 0.2,
    }
    with _harness(_aisle_scene(), skill_resolver=_in_tree_resolver(), params=params) as (
        executor,
        runtime,
        seen,
    ):
        _run_goal(
            executor, runtime.skill_runner_node, fill_goal=_named(_PI05, _reasoner_box_goal())
        )
        _spin_for(executor, 0.3)
    armed = next(m for m in seen if m.active)
    got = [(q.label, q.role) for q in map(ObjectQuery.from_idl, armed.target_queries)]
    assert got == [("box", "target"), ("carton", "context")]


def test_a_goal_at_the_query_ceiling_keeps_its_own_and_takes_no_more() -> None:
    """A goal naming ``MAX_TARGET_QUERIES`` objects leaves no room for the deploy's CONTEXT
    query: the armed declaration carries the goal's eight, never nine (the tracker's decode
    would refuse them)."""
    from openral_core import GraspDeclaration, ObjectQuery

    ceiling = GraspDeclaration.MAX_TARGET_QUERIES
    goal = GraspDeclaration(
        target_id="cell:box",
        contact_links=_LEFT,
        timeout_s=10.0,
        stamp_ns=0,
        target_queries=tuple(
            ObjectQuery(label=f"box {k}", count="one", role="target") for k in range(ceiling)
        ),
    )

    def _fill(msg: Any) -> None:
        from openral_msgs.msg import ObjectQuery as ObjectQueryMsg

        msg.grasp_declaration_valid = True
        goal.fill_idl(msg.grasp_declaration, query_factory=ObjectQueryMsg)

    params = {
        "object_tracking": "tracking",
        "object_queries_json": _SHELF_AISLE_QUERIES,
        "grounding_wait_s": 0.2,
    }
    with _harness(_aisle_scene(), skill_resolver=_in_tree_resolver(), params=params) as (
        executor,
        runtime,
        seen,
    ):
        _run_goal(executor, runtime.skill_runner_node, fill_goal=_named(_PI05, _fill))
        _spin_for(executor, 0.3)
    armed = next(m for m in seen if m.active)
    got = [q.label for q in map(ObjectQuery.from_idl, armed.target_queries)]
    assert got == [f"box {k}" for k in range(ceiling)]


def test_a_grasping_goal_with_only_context_queries_is_refused() -> None:
    """The deploy's standing queries are CONTEXT only (neighbours) and the goal names no
    target: nothing names the object the policy grasps, so the goal is refused."""
    from openral_msgs.action import ExecuteRskill

    params = {
        "object_tracking": "tracking",
        "object_queries_json": '[{"label": "carton", "count": "all", "role": "context"}]',
        "grounding_wait_s": 0.2,
    }
    with _harness(
        _scene_json(contact_links=("gripper_finger",)),
        skill_resolver=_in_tree_resolver(),
        params=params,
    ) as (executor, runtime, seen):
        response = _run_goal(executor, runtime.skill_runner_node, fill_goal=_named(_PI05))
        _spin_for(executor, 0.3)
    result = response.result
    assert result.failure_kind == ExecuteRskill.Result.FAILURE_PLANNING_ERROR, result.failure_reason
    assert not [m for m in seen if m.active]


def test_the_query_cap_never_drops_the_deploys_target_behind_its_context_queries() -> None:
    """The deploy lists more CONTEXT queries than the ceiling holds, its TARGET last: the
    TARGET still rides (first), the CONTEXT queries fill the rest in their own order."""
    import json

    from openral_core import GraspDeclaration, ObjectQuery

    ceiling = GraspDeclaration.MAX_TARGET_QUERIES
    deploy = [{"label": f"carton {k}", "count": "all", "role": "context"} for k in range(ceiling)]
    deploy.append({"label": "orange", "count": "all", "role": "target"})
    params = {
        "object_tracking": "tracking",
        "object_queries_json": json.dumps(deploy),
        "grounding_wait_s": 0.2,
    }
    with _harness(
        _scene_json(contact_links=("gripper_finger",)),
        skill_resolver=_in_tree_resolver(),
        params=params,
    ) as (executor, runtime, seen):
        _run_goal(executor, runtime.skill_runner_node, fill_goal=_named(_PI05))
        _spin_for(executor, 0.3)
    armed = next(m for m in seen if m.active)
    got = [q.label for q in map(ObjectQuery.from_idl, armed.target_queries)]
    assert got == ["orange", *(f"carton {k}" for k in range(ceiling - 1))]


def test_a_scene_box_declared_for_another_object_is_not_lent(capfd: Any) -> None:
    """The scene's declaration names 'orange'; a goal for 'box' with the same hand takes the
    scene's hand but not its boxes (they bound another object's search and contact)."""
    from openral_core import GraspDeclaration, ObjectQuery

    scene = GraspDeclaration.model_validate_json(_aisle_scene()).model_copy(
        update={"target_queries": (ObjectQuery(label="orange", count="one", role="target"),)}
    )
    with _harness(scene.model_dump_json(), skill_resolver=_in_tree_resolver()) as (
        executor,
        runtime,
        seen,
    ):
        _run_goal(
            executor, runtime.skill_runner_node, fill_goal=_named(_PI05, _reasoner_box_goal())
        )
        _spin_for(executor, 0.3)
    armed = next(m for m in seen if m.active)
    assert armed.target_id == "obj:box"
    assert not armed.search_box_valid and not armed.workspace_box_valid
    assert "boxes=other_object" in capfd.readouterr().err


def test_a_goal_declaration_outlives_the_pre_execution_wait() -> None:
    """A goal-built declaration's lifetime (budget + margin) counted from arming; a 6 s
    grounding wait would leave it lapsing mid-run, so it is republished, same key, with the
    budget counted from execution start."""
    params = {"object_tracking": "tracking", "grounding_wait_s": 6.0}
    with _harness("", params=params) as (executor, runtime, seen):
        _run_goal(
            executor,
            runtime.skill_runner_node,
            fill_goal=_reasoner_box_goal(),
            result_timeout_s=12.0,
        )
        _spin_for(executor, 0.3)
    active = [m for m in seen if m.active]
    assert len(active) == 2, "armed, then extended once"
    first, extended = active
    assert (extended.target_id, extended.stamp_ns) == (first.target_id, first.stamp_ns)
    assert first.timeout_s == pytest.approx(10.4)
    assert extended.timeout_s > first.timeout_s + 5.0
    assert not seen[-1].active
