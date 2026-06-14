"""Lifecycle smoke test for ``openral_hal_h1``.

Drives the standard managed-lifecycle transition path against the
generic ``_HALLifecycleNode`` from :mod:`openral_hal.lifecycle` using
the real ``H1MujocoHAL`` factory. The HAL pulls its MJCF lazily (typically
from ``robot_descriptions``) and ships its canonical
:class:`openral_core.RobotDescription` (``H1_DESCRIPTION``).

The test is gated on ``rclpy``, ``openral_hal``, ``mujoco`` and
``robot_descriptions`` being importable; in lint-only environments
any missing piece causes a clean skip.
"""

from __future__ import annotations

import time

import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("openral_hal")
pytest.importorskip("mujoco")
pytest.importorskip("robot_descriptions")

from openral_hal import H1MujocoHAL
from openral_hal.lifecycle import _HALLifecycleNode  # type: ignore[attr-defined]
from rclpy.lifecycle import TransitionCallbackReturn
from sensor_msgs.msg import JointState as RosJointState

NODE_NAME = "openral_hal_h1"
_EXPECTED_DOF = 19


def test_lifecycle_transitions_full_cycle() -> None:
    """UC → INACTIVE → ACTIVE → INACTIVE → cleanup; publishes JointState when ACTIVE."""
    rclpy.init()
    try:
        node = _HALLifecycleNode(node_name=NODE_NAME, hal_factory=H1MujocoHAL)
        try:
            assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
            assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS

            received: list[RosJointState] = []
            helper = rclpy.create_node(f"{NODE_NAME}_test_helper")
            helper.create_subscription(RosJointState, "/joint_states", received.append, 10)
            deadline = time.time() + 3.0
            while time.time() < deadline and not received:
                rclpy.spin_once(node, timeout_sec=0.05)
                rclpy.spin_once(helper, timeout_sec=0.05)

            assert received, "no /joint_states published while ACTIVE"
            assert len(received[-1].position) == _EXPECTED_DOF, (
                f"expected {_EXPECTED_DOF} positions, got {len(received[-1].position)}"
            )

            assert node.trigger_deactivate() == TransitionCallbackReturn.SUCCESS
            assert node.trigger_cleanup() == TransitionCallbackReturn.SUCCESS
            helper.destroy_node()
        finally:
            node.destroy_node()
    finally:
        rclpy.shutdown()
