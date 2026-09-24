"""``start_world_state_executor``: a raising callback reaches ``on_failure``; a stop does not.

A daemon thread has nothing above it, so before this a world-state callback
that raised ended ingestion silently while the skill runner kept executing.
"""

from __future__ import annotations

import time

import pytest


def test_a_raising_callback_is_handed_to_on_failure_and_a_stop_is_not() -> None:
    rclpy = pytest.importorskip("rclpy", reason="needs a sourced ROS 2 overlay")
    pytest.importorskip("rclpy.experimental", reason="needs rclpy's EventsExecutor")
    from openral_rskill_ros.compose import start_world_state_executor

    rclpy.init()
    try:
        failures: list[BaseException] = []
        node = rclpy.create_node("world_state_executor_probe")

        def _boom() -> None:
            raise RuntimeError("world-state callback failed")

        node.create_timer(0.05, _boom)
        stop = start_world_state_executor(node, on_failure=failures.append)
        assert stop is not None
        deadline = time.monotonic() + 5.0
        while not failures and time.monotonic() < deadline:
            time.sleep(0.02)
        assert failures and "world-state callback failed" in str(failures[0])
        stop()
        node.destroy_node()

        # A clean stop is not a failure.
        quiet: list[BaseException] = []
        node2 = rclpy.create_node("world_state_executor_probe_quiet")
        stop2 = start_world_state_executor(node2, on_failure=quiet.append)
        assert stop2 is not None
        time.sleep(0.2)
        stop2()
        node2.destroy_node()
        assert quiet == []
    finally:
        rclpy.try_shutdown()
