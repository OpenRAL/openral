# SPDX-License-Identifier: Apache-2.0
"""HAL + deploy-runtime spins end quietly on shutdown, even mid-publish; live errors propagate.

rclpy's SIGINT handler invalidates the context first; a timer callback already
dequeued (the sim bridge's depth publish) then publishes on the dead context, raises
``RCLError: publisher's context is invalid`` out of ``rclpy.spin`` and the HAL exited
1 — on Ctrl-C of an OpenArm ``deploy sim`` once its head depth camera published
(2026-10-04, ``lifecycle_node.py`` exit code 1). Both HAL mains spin through
``openral_hal.lifecycle.spin_until_shutdown``, which treats exactly that as teardown.

Real rclpy context, node, publisher and timer; the race is made deterministic by
shutting the context down inside the callback, before it publishes.

Gated on ``OPENRAL_TEST_ROS_LIVE=1``::

    source /opt/ros/jazzy/setup.bash && source install/setup.bash
    OPENRAL_TEST_ROS_LIVE=1 pytest tests/integration/test_hal_sigint_teardown_live.py
"""

from __future__ import annotations

import os

import pytest

_LIVE_ROS = bool(os.getenv("OPENRAL_TEST_ROS_LIVE"))
pytestmark = pytest.mark.skipif(
    not _LIVE_ROS,
    reason="live rclpy context — set OPENRAL_TEST_ROS_LIVE=1 and source install/setup.bash first.",
)


def test_a_publish_racing_the_shutdown_ends_the_spin_quietly() -> None:
    rclpy = pytest.importorskip("rclpy")
    from openral_hal.lifecycle import spin_until_shutdown
    from std_msgs.msg import Empty

    rclpy.init()
    node = rclpy.create_node("test_spin_until_shutdown_race")
    pub = node.create_publisher(Empty, "/test_spin_until_shutdown", 1)

    def tick() -> None:
        rclpy.shutdown()  # what the SIGINT handler does first
        pub.publish(Empty())  # the dequeued sensor publish: RCLError, context invalid

    node.create_timer(0.01, tick)
    try:
        spin_until_shutdown(node)  # must return, not raise
        assert not rclpy.ok()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def test_an_rcl_error_on_a_live_context_still_propagates() -> None:
    rclpy = pytest.importorskip("rclpy")
    from openral_hal.lifecycle import spin_until_shutdown
    from rclpy._rclpy_pybind11 import RCLError

    rclpy.init()
    node = rclpy.create_node("test_spin_until_shutdown_live_error")

    def tick() -> None:
        raise RCLError("a real failure with the context still up")

    node.create_timer(0.01, tick)
    try:
        with pytest.raises(RCLError, match="real failure"):
            spin_until_shutdown(node)
        assert rclpy.ok()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def test_the_deploy_runtime_executor_ends_quietly_on_the_same_race() -> None:
    """``runtime_node`` spins a ``MultiThreadedExecutor``; it exited 1 on this race too."""
    rclpy = pytest.importorskip("rclpy")
    from openral_rskill_ros.compose import spin_until_shutdown
    from rclpy.executors import MultiThreadedExecutor
    from std_msgs.msg import Empty

    rclpy.init()
    node = rclpy.create_node("test_runtime_spin_until_shutdown_race")
    pub = node.create_publisher(Empty, "/test_runtime_spin_until_shutdown", 1)

    def tick() -> None:
        rclpy.shutdown()
        pub.publish(Empty())

    node.create_timer(0.01, tick)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        spin_until_shutdown(executor)  # must return, not raise
        assert not rclpy.ok()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


def _unhandled_thread_errors() -> tuple[list[BaseException], object]:
    """Capture what would print as ``Exception in thread ...`` (restored by the caller)."""
    import threading

    errors: list[BaseException] = []
    previous = threading.excepthook

    def hook(args: threading.ExceptHookArgs) -> None:
        if args.exc_value is not None:
            errors.append(args.exc_value)

    threading.excepthook = hook
    return errors, previous


def test_an_executor_spin_thread_ends_quietly_when_the_context_goes_down() -> None:
    """The dashboard/runner spin threads: external shutdown must not escape the thread."""
    import threading

    rclpy = pytest.importorskip("rclpy")
    from openral_observability.rclpy_spin import spin_executor_until_shutdown
    from rclpy.executors import SingleThreadedExecutor

    rclpy.init()
    node = rclpy.create_node("test_spin_thread_shutdown")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    errors, previous = _unhandled_thread_errors()
    thread = threading.Thread(target=spin_executor_until_shutdown, args=(executor,), daemon=True)
    try:
        thread.start()
        rclpy.shutdown()  # what the SIGINT handler does, from outside the spin thread
        thread.join(timeout=5.0)
        assert not thread.is_alive()
        assert errors == []
    finally:
        threading.excepthook = previous  # type: ignore[assignment]  # reason: restoring the saved hook
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


def test_the_ros2_image_reader_spin_thread_ends_quietly_on_shutdown() -> None:
    """``Ros2ImageSensorReader`` (the Isaac deploy's camera leg) spins a private node."""
    import threading

    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("sensor_msgs")
    from openral_runner.backends.ros2_image import Ros2ImageSensorReader

    reader = Ros2ImageSensorReader(sensor_id="cam", topic="/test_spin_shutdown/image")
    errors, previous = _unhandled_thread_errors()
    try:
        reader.open()
        rclpy.shutdown()
        reader._spin_thread.join(timeout=5.0)  # reason: the thread under test
        assert not reader._spin_thread.is_alive()
        assert errors == []
    finally:
        threading.excepthook = previous  # type: ignore[assignment]  # reason: restoring the saved hook
        reader.close()
        rclpy.try_shutdown()


def test_a_node_spin_ends_quietly_on_a_take_racing_the_shutdown() -> None:
    """The segmenter's teardown crash: a bare ``RuntimeError`` once the context is down."""
    rclpy = pytest.importorskip("rclpy")
    from openral_observability.rclpy_spin import spin_node_until_shutdown
    from std_msgs.msg import Empty

    rclpy.init()
    node = rclpy.create_node("test_spin_node_take_race")
    pub = node.create_publisher(Empty, "/test_spin_node_take_race", 10)

    def on_msg(_: Empty) -> None:
        rclpy.shutdown()  # what the SIGINT handler does first, with messages still pending
        raise RuntimeError("Unable to convert call argument '0' to Python object")

    node.create_subscription(Empty, "/test_spin_node_take_race", on_msg, 10)
    node.create_timer(0.01, lambda: [pub.publish(Empty()) for _ in range(5)])
    try:
        spin_node_until_shutdown(node)  # must return, not raise
        assert not rclpy.ok()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def test_a_runtime_error_on_a_live_context_still_propagates_from_a_node_spin() -> None:
    rclpy = pytest.importorskip("rclpy")
    from openral_observability.rclpy_spin import spin_node_until_shutdown

    rclpy.init()
    node = rclpy.create_node("test_spin_node_live_error")

    def tick() -> None:
        raise RuntimeError("a real failure with the context still up")

    node.create_timer(0.01, tick)
    try:
        with pytest.raises(RuntimeError, match="real failure"):
            spin_node_until_shutdown(node)
        assert rclpy.ok()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def test_sigint_to_a_node_spinning_through_a_busy_subscription_exits_zero() -> None:
    """Real SIGINT, subscription saturated with pending messages: exit 0, no traceback."""
    import signal
    import subprocess
    import sys
    import time

    script = (
        "import rclpy\n"
        "from std_msgs.msg import Empty\n"
        "from openral_observability.rclpy_spin import spin_node_until_shutdown\n"
        "rclpy.init()\n"
        "node = rclpy.create_node('test_sigint_busy_spin')\n"
        "pub = node.create_publisher(Empty, '/test_sigint_busy_spin', 100)\n"
        "node.create_subscription(Empty, '/test_sigint_busy_spin', lambda m: None, 100)\n"
        "node.create_timer(0.001, lambda: [pub.publish(Empty()) for _ in range(50)])\n"
        "print('ready', flush=True)\n"
        "try:\n"
        "    spin_node_until_shutdown(node)\n"
        "finally:\n"
        "    node.destroy_node()\n"
        "    rclpy.try_shutdown()\n"
    )
    proc = subprocess.Popen(  # reason: fixed argv, test-owned script
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stdout is not None
        assert proc.stdout.readline().strip() == "ready"
        time.sleep(0.5)
        proc.send_signal(signal.SIGINT)
        _, err = proc.communicate(timeout=20)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == 0, err
    assert "Traceback" not in err, err
