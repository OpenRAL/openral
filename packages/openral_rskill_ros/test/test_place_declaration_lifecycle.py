"""The place-phase declaration's goal lifecycle (ADR-0097, HZ-0097-3).

The declaration is the only thing that lifts the pick witness's mid-carry
anti-scope, so a declaration that outlives the goal that justified it would arm
an exemption for a later, unrelated execution. `rskill_runner_node` is the goal
lifecycle authority — it accepts, succeeds, aborts, cancels, and hears
`/openral/estop` — so it is where the declaration is scoped, and these tests
drive the real node through `rclpy` and read the real
`/openral/place_declaration` topic.

Per CLAUDE.md §1.11: the runtime, the action server, the skill and the topic
are all real; nothing here is a double.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

_ROS2_AVAILABLE = bool(os.environ.get("ROS_DISTRO"))

pytestmark = pytest.mark.skipif(
    not _ROS2_AVAILABLE,
    reason="ROS_DISTRO not set — these tests require a sourced ROS 2 installation.",
)

_TARGET = "sim:cab_1_left_group_main"


def _constant_skill_resolver() -> Any:
    """A real one-chunk rSkill, resolved for any id."""
    from openral_core.schemas import Action, ControlMode
    from openral_rskill.base import rSkillBase

    class _ConstantSkill(rSkillBase):
        def __init__(self) -> None:
            super().__init__(
                name="openral/test-place-declaration-skill",
                version="0.1.0",
                role="s1",
                embodiment_tags=["so100_follower"],
            )

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
                control_mode=ControlMode.JOINT_POSITION,
                horizon=1,
                joint_targets=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            )

    def _resolver(*_args: Any, **_kwargs: Any) -> Any:
        skill = _ConstantSkill()
        skill.configure()
        skill.activate()
        return skill

    return _resolver


@contextmanager
def _harness(place_declaration_json: str) -> Iterator[tuple[Any, Any, list[Any]]]:
    """Compose the real runtime with a scene-committed declaration installed."""
    import rclpy
    from openral_msgs.msg import PlaceDeclaration as PlaceDeclarationMsg
    from openral_rskill_ros.compose import compose_so100_runtime
    from rclpy.lifecycle import TransitionCallbackReturn
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

    rclpy.init()
    runtime = compose_so100_runtime(skill_resolver=_constant_skill_resolver())
    runtime.skill_runner_node.set_parameters(
        [rclpy.parameter.Parameter("place_declaration_json", value=place_declaration_json)]
    )

    executor = rclpy.executors.MultiThreadedExecutor(num_threads=4)
    executor.add_node(runtime.world_state_node)
    executor.add_node(runtime.skill_runner_node)
    helper = rclpy.create_node("openral_place_declaration_test_helper")
    executor.add_node(helper)

    seen: list[Any] = []
    helper.create_subscription(
        PlaceDeclarationMsg,
        "/openral/place_declaration",
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
            except Exception:  # reason: best-effort teardown
                pass
        executor.shutdown()
        helper.destroy_node()
        runtime.skill_runner_node.destroy_node()
        runtime.world_state_node.destroy_node()
        rclpy.shutdown()


def _spin_for(executor: Any, duration_s: float) -> None:
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.02)


def _scene_declaration_json(*, timeout_s: float = 60.0, object_id: str = "") -> str:
    from openral_core import PlaceDeclaration

    return PlaceDeclaration(
        target_id=_TARGET,
        object_id=object_id,
        timeout_s=timeout_s,
        stamp_ns=0,
    ).model_dump_json()


def _run_goal(executor: Any, node: Any, *, deadline_s: float = 0.4, cancel: bool = False) -> None:
    """Dispatch one real ExecuteRskill goal and let it resolve (or cancel it)."""
    from openral_msgs.action import ExecuteRskill
    from rclpy.action import ActionClient

    client = ActionClient(node, ExecuteRskill, "/openral/execute_rskill")
    _spin_for(executor, 0.2)
    assert client.wait_for_server(timeout_sec=2.0), "ExecuteRskill action server not ready"
    goal = ExecuteRskill.Goal()
    goal.rskill_id = "openral/test-place-declaration-skill"
    goal.revision = ""
    goal.prompt = "put the baguette in the cabinet"
    goal.prompt_metadata_json = ""
    goal.deadline_s = deadline_s
    send_future = client.send_goal_async(goal)
    deadline = time.monotonic() + 3.0
    while not send_future.done() and time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.02)
    goal_handle = send_future.result()
    assert goal_handle is not None and goal_handle.accepted, "goal rejected"
    if cancel:
        _spin_for(executor, 0.1)
        goal_handle.cancel_goal_async()
    result_future = goal_handle.get_result_async()
    deadline = time.monotonic() + 5.0
    while not result_future.done() and time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.02)
    assert result_future.done(), "goal result timed out"


def test_no_declaration_configured_puts_nothing_on_the_wire() -> None:
    """Today's every scene. No declaration means the anti-scope stands."""
    with _harness("") as (executor, runtime, seen):
        _run_goal(executor, runtime.skill_runner_node)
        _spin_for(executor, 0.3)
    assert seen == []


def test_a_goal_arms_then_retracts_the_scene_declaration() -> None:
    """Arm on start, retract on end — and the retraction is not optional."""
    with _harness(_scene_declaration_json()) as (executor, runtime, seen):
        _run_goal(executor, runtime.skill_runner_node)
        _spin_for(executor, 0.3)

    active = [msg for msg in seen if msg.active]
    retracted = [msg for msg in seen if not msg.active]
    assert len(active) == 1, "exactly one declaration should have been armed"
    assert len(retracted) == 1, "the goal ended without retracting its declaration"
    assert active[0].target_id == _TARGET
    # Attributability (HZ-0097-2): stamped by the dispatcher, at dispatch, and
    # carrying the ids an incident review would look the goal up by.
    assert active[0].rskill_id == "openral/test-place-declaration-skill"
    assert active[0].stamp_ns > 0
    assert active[0].timeout_s == pytest.approx(60.0)


def test_cancelling_a_goal_retracts_its_declaration() -> None:
    """Cancel is one of HZ-0097-3's three required expiry transitions."""
    with _harness(_scene_declaration_json()) as (executor, runtime, seen):
        _run_goal(executor, runtime.skill_runner_node, deadline_s=3.0, cancel=True)
        _spin_for(executor, 0.3)

    assert any(msg.active for msg in seen), "the declaration never armed"
    assert any(not msg.active for msg in seen), "a cancelled goal left its declaration live"


def test_an_estop_retracts_the_declaration_immediately() -> None:
    """E-stop is the transition where a lingering exemption is least defensible."""
    from std_msgs.msg import Empty

    with _harness(_scene_declaration_json()) as (executor, runtime, seen):
        _run_goal(executor, runtime.skill_runner_node, deadline_s=3.0, cancel=True)
        _spin_for(executor, 0.2)
        seen.clear()
        # Re-arm, then stop.
        runtime.skill_runner_node._arm_place_declaration(
            _GoalRequestStub(), rskill_id="openral/x", trace_id="t"
        )
        _spin_for(executor, 0.2)
        assert any(msg.active for msg in seen)
        runtime.skill_runner_node._on_estop(Empty())
        _spin_for(executor, 0.3)

    assert not seen[-1].active, "the E-stop left the declaration live"


class _GoalRequestStub:
    """The two goal fields `_arm_place_declaration` reads, with neither set.

    Not a double for the action server — it is the exact shape of "this goal
    carries no declaration of its own", which routes resolution to the node's
    scene parameter. The action server itself is real in every test above.
    """

    place_declaration_valid = False


def test_the_goals_own_declaration_wins_over_the_scenes() -> None:
    """A reasoner grounds the target per goal; the scene is the direct-dispatch
    fallback, never an override of what the goal itself declared."""
    from openral_core import PlaceDeclaration
    from openral_msgs.action import ExecuteRskill

    with _harness(_scene_declaration_json()) as (executor, runtime, seen):
        _spin_for(executor, 0.2)
        request = ExecuteRskill.Goal()
        request.place_declaration_valid = True
        PlaceDeclaration(
            target_id="sim:sink_main",
            timeout_s=30.0,
            stamp_ns=0,
        ).fill_idl(request.place_declaration)
        runtime.skill_runner_node._arm_place_declaration(
            request, rskill_id="openral/y", trace_id="t2"
        )
        _spin_for(executor, 0.3)

    active = [msg for msg in seen if msg.active]
    assert active, "no declaration was armed"
    assert active[-1].target_id == "sim:sink_main"
    assert active[-1].timeout_s == pytest.approx(30.0)


def test_an_exception_escaping_the_executor_still_retracts() -> None:
    """A goal that dies without unwinding the node's per-goal state.

    An exception escaping the execute callback makes rclpy abort the goal
    without `_reset_active_goal` ever running. Found while writing this suite,
    with a real resolver failure: the declaration survived the dead goal and
    would have armed an exemption for whatever ran next.
    """
    import rclpy
    from openral_msgs.msg import PlaceDeclaration as PlaceDeclarationMsg
    from openral_rskill_ros.compose import compose_so100_runtime
    from rclpy.lifecycle import TransitionCallbackReturn
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

    def _exploding_resolver(*_args: Any, **_kwargs: Any) -> Any:
        raise TypeError("resolver blew up")

    rclpy.init()
    runtime = compose_so100_runtime(skill_resolver=_exploding_resolver)
    runtime.skill_runner_node.set_parameters(
        [rclpy.parameter.Parameter("place_declaration_json", value=_scene_declaration_json())]
    )
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=4)
    executor.add_node(runtime.world_state_node)
    executor.add_node(runtime.skill_runner_node)
    helper = rclpy.create_node("openral_place_declaration_explode_helper")
    executor.add_node(helper)
    seen: list[Any] = []
    helper.create_subscription(
        PlaceDeclarationMsg,
        "/openral/place_declaration",
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
        _run_goal(executor, runtime.skill_runner_node)
        _spin_for(executor, 0.3)
    finally:
        for node in (runtime.skill_runner_node, runtime.world_state_node):
            try:
                node.trigger_deactivate()
                node.trigger_cleanup()
                node.trigger_shutdown()
            except Exception:  # reason: best-effort teardown
                pass
        executor.shutdown()
        helper.destroy_node()
        runtime.skill_runner_node.destroy_node()
        runtime.world_state_node.destroy_node()
        rclpy.shutdown()

    assert any(msg.active for msg in seen), "the declaration never armed"
    assert not seen[-1].active, "a goal that died mid-executor left its declaration live"


def test_a_malformed_scene_declaration_is_refused_not_guessed() -> None:
    """Fail closed: an unparseable declaration arms nothing and the goal runs
    without one, which is the pre-ADR-0097 behaviour."""
    with _harness('{"target_id": "sim:x", "timeout_s": -1.0, "stamp_ns": 0}') as (
        executor,
        runtime,
        seen,
    ):
        _run_goal(executor, runtime.skill_runner_node)
        _spin_for(executor, 0.3)
    assert seen == []
