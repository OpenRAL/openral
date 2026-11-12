"""Real-rclpy tests: the watchdog nodes' parameters are fixed once configured (issue #546).

The deadman reads ``safe_action_deadline_s`` / ``first_chunk_deadline_s`` on
every tick, so before this guard a same-host graph peer could
``ros2 param set`` them to a day and silence the brake. Mirrors the C++
kernel's guard (``LifecycleKernelTest.SafetyLimitParametersAreRefusedAfterConfigure``
/ ``AFailedConfigureStillLocksTheParameters``): every change to an existing
parameter is refused from the first configure on, a failed configure latches
too, and a declaration of a new parameter still passes.

Driven through the real lifecycle transitions and the real
``/<node>/set_parameters`` service, as a peer would reach it. No mocks
(CLAUDE.md §1.11).
"""

from __future__ import annotations

from typing import Any

import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("openral_msgs")

from openral_safety_watchdog.deadman_watchdog_node import (
    DEFAULT_FIRST_CHUNK_DEADLINE_S,
    DeadmanWatchdogNode,
)
from openral_safety_watchdog.hardware_estop_node import HardwareEstopNode
from rclpy.executors import SingleThreadedExecutor
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.parameter import Parameter
from rclpy.parameter_client import AsyncParameterClient


@pytest.fixture
def ros_context() -> Any:
    """Spin up rclpy for the test; tear down after."""
    rclpy.init()
    try:
        yield
    finally:
        rclpy.shutdown()


def _double(node: Any, name: str) -> float:
    return float(node.get_parameter(name).get_parameter_value().double_value)


def _peer_set(executor: Any, client: Any, params: list[Parameter]) -> list[bool]:
    """Set ``params`` one by one over the real set_parameters service."""
    outcomes = []
    for param in params:
        future = client.set_parameters([param])
        executor.spin_until_future_complete(future, timeout_sec=5.0)
        assert future.done(), f"set_parameters({param.name}) never answered"
        outcomes.append(bool(future.result().results[0].successful))
    return outcomes


def test_a_peer_cannot_relax_the_deadman_deadlines_after_configure(ros_context: None) -> None:
    """``ros2 param set`` on a running deadman is refused; the deadlines stay put."""
    node = DeadmanWatchdogNode(node_name="deadman_param_guard")
    # Before configure the parameters are still settable (the existing tests do this).
    assert node.set_parameters([Parameter("safe_action_deadline_s", Parameter.Type.DOUBLE, 0.1)])[
        0
    ].successful
    peer = rclpy.create_node("deadman_param_guard_peer")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(peer)
    client = AsyncParameterClient(peer, "deadman_param_guard")
    try:
        assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
        assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS
        assert client.wait_for_services(timeout_sec=5.0)

        relaxed = [
            Parameter("safe_action_deadline_s", Parameter.Type.DOUBLE, 86400.0),
            Parameter("first_chunk_deadline_s", Parameter.Type.DOUBLE, 86400.0),
            Parameter("check_period_s", Parameter.Type.DOUBLE, 3600.0),
            Parameter("arm_status_topic", Parameter.Type.STRING, "/nobody/status"),
            Parameter("safety_status_topic", Parameter.Type.STRING, ""),
            Parameter("result_only_topic", Parameter.Type.STRING, ""),
            Parameter("result_only_grace_s", Parameter.Type.DOUBLE, 86400.0),
            Parameter("result_only_max_s", Parameter.Type.DOUBLE, 86400.0),
        ]
        assert _peer_set(executor, client, relaxed) == [False] * len(relaxed), (
            "a graph peer changed a deadman parameter after configure"
        )
        # The in-process path goes through the same callback.
        assert not node.set_parameters(
            [Parameter("safe_action_deadline_s", Parameter.Type.DOUBLE, 86400.0)]
        )[0].successful
        assert _double(node, "safe_action_deadline_s") == 0.1
        assert _double(node, "first_chunk_deadline_s") == DEFAULT_FIRST_CHUNK_DEADLINE_S

        # cleanup -> set -> configure must not relax them either.
        assert node.trigger_deactivate() == TransitionCallbackReturn.SUCCESS
        assert node.trigger_cleanup() == TransitionCallbackReturn.SUCCESS
        assert _peer_set(executor, client, relaxed[:1]) == [False]
        assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
        assert _double(node, "safe_action_deadline_s") == 0.1

        # A declaration adds a parameter and relaxes nothing: it still passes.
        node.declare_parameter("declared_after_configure", 1.0)
        assert _double(node, "declared_after_configure") == 1.0
        assert not node.set_parameters(
            [Parameter("declared_after_configure", Parameter.Type.DOUBLE, 9.0)]
        )[0].successful
    finally:
        executor.remove_node(peer)
        executor.remove_node(node)
        peer.destroy_node()
        node.destroy_node()


def test_a_failed_configure_still_locks_the_parameters(ros_context: None) -> None:
    """A configure that FAILS stays UNCONFIGURED without a cleanup; the guard must not reopen.

    Otherwise a peer could repoint ``device`` (or flip ``active_low``) between
    the failed attempt and the retry.
    """
    node = HardwareEstopNode(node_name="hardware_estop_param_guard_fail")
    try:
        assert node.set_parameters([Parameter("poll_rate_hz", Parameter.Type.DOUBLE, 50.0)])[
            0
        ].successful
        assert node.trigger_configure() == TransitionCallbackReturn.FAILURE  # no device
        for param in (
            Parameter("device", Parameter.Type.STRING, "/dev/null"),
            Parameter("active_low", Parameter.Type.BOOL, False),
            Parameter("poll_rate_hz", Parameter.Type.DOUBLE, 1.0),
            Parameter("channel_label", Parameter.Type.STRING, "spoofed"),
        ):
            assert not node.set_parameters([param])[0].successful, param.name
        assert node.get_parameter("device").get_parameter_value().string_value == ""
        assert _double(node, "poll_rate_hz") == 50.0
    finally:
        node.destroy_node()


def test_a_hardware_estop_parameter_is_refused_while_active(ros_context: None) -> None:
    """The pendant bridge reads ``channel_label`` live and its vendor subclasses ``active_low``."""
    node = HardwareEstopNode(node_name="hardware_estop_param_guard_active")
    node.read_pressed_hook = lambda: False
    node.set_parameters([Parameter("device", Parameter.Type.STRING, "/dev/null")])
    try:
        assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
        assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS
        assert not node.set_parameters([Parameter("active_low", Parameter.Type.BOOL, False)])[
            0
        ].successful
        assert node.get_parameter("active_low").get_parameter_value().bool_value is True
    finally:
        node.destroy_node()
