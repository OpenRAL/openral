"""A completion check that raises does not end the check worker (#421 review).

``_worker`` had only ``try/finally``: an exception out of ``_check`` (a failed service
future re-raises in ``_call``) ended the thread, and every later trigger queued one job
nobody consumed, so the goal silently ran to its deadline. The real
``CompletionCheckNode`` and its real worker thread; ``_check`` raises once, the way a
failed ``QueryScene`` future does, and the next job must still be checked.

Gated on ``OPENRAL_TEST_ROS_LIVE=1``; listed in ``scripts/ros_live_tests.sh``.
"""

from __future__ import annotations

import os
import time

import pytest


@pytest.mark.skipif(
    not os.getenv("OPENRAL_TEST_ROS_LIVE"),
    reason="live rclpy node: set OPENRAL_TEST_ROS_LIVE=1 and source install/",
)
def test_the_worker_survives_a_check_that_raises() -> None:
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs.srv")
    from openral_reasoner_ros.completion_check_node import CompletionCheckNode

    checked: list[str] = []

    class _FailsFirst(CompletionCheckNode):  # type: ignore[misc,unused-ignore]  # reason: rclpy Node base is untyped
        def _check(self, reason: str, task: str) -> None:
            checked.append(reason)
            if len(checked) == 1:
                raise RuntimeError("QueryScene future raised")

    rclpy.init()
    try:
        node = _FailsFirst()
        for reason in ("success", "release"):
            node._busy.set()  # what _offer does before it queues a job
            node._jobs.put_nowait((reason, "put the banana in the bowl"))
            end = time.monotonic() + 5.0
            while time.monotonic() < end and node._busy.is_set():
                time.sleep(0.02)
            assert not node._busy.is_set(), f"job {reason!r} was never checked"
        assert checked == ["success", "release"]
        node.destroy_node()
    finally:
        rclpy.shutdown()
