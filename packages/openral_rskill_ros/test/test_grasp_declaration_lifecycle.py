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

_TARGET = "cell:restock_box"
_FINGERS = ("openarm_left_finger_pair", "openarm_right_finger_pair")


def _scene_json(**update: Any) -> str:
    from openral_core import GraspDeclaration

    return GraspDeclaration(
        target_id=_TARGET, contact_links=_FINGERS, timeout_s=70.0, stamp_ns=0, **update
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
    grasp_declaration_json: str, *, skill_resolver: Any = None
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
            _NoGoalDeclaration(), rskill_id="openral/x", trace_id="t"
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
            _NoGoalDeclaration(), rskill_id="openral/z", trace_id="t3"
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
        ).fill_idl(request.grasp_declaration)
        assert request.grasp_declaration.region_valid
        runtime.skill_runner_node._arm_grasp_declaration(
            request, rskill_id="openral/z", trace_id="t4"
        )
        _spin_for(executor, 0.3)

    active = [msg for msg in seen if msg.active]
    assert active[-1].target_id == "cell:other_box"
    assert active[-1].rskill_id == "openral/z"
    assert active[-1].trace_id == "t4"
    assert not active[-1].region_valid, "a dispatch-supplied region reached the kernel's path"


def test_a_malformed_scene_declaration_is_refused_not_guessed() -> None:
    """Active with no contact_links: refused, the goal runs with no declaration."""
    bad = '{"target_id": "cell:restock_box", "timeout_s": 70.0, "stamp_ns": 0}'
    with _harness(bad) as (executor, runtime, seen):
        _run_goal(executor, runtime.skill_runner_node)
        _spin_for(executor, 0.3)
    assert seen == []
