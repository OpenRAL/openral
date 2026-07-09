"""Lifecycle smoke test for ``openral_hal_ur10e``.

Drives the standard managed-lifecycle transition path against the generic
``_HALLifecycleNode`` from :mod:`openral_hal.lifecycle` using the real
``UR10eHAL`` factory.  The HAL pulls its MJCF lazily from the
``robot_descriptions`` package and ships its canonical
:class:`openral_core.RobotDescription` (``UR10e_DESCRIPTION``) — so this
smoke exercises the same RobotDescription wiring used at runtime, not a
stub.

Skips cleanly when ``rclpy`` / ``openral_hal`` / ``mujoco`` /
``robot_descriptions`` are unavailable (lint-only environments).
"""

from __future__ import annotations

import time

import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("openral_hal")
pytest.importorskip("mujoco")
pytest.importorskip("robot_descriptions")

from openral_hal import UR10eHAL
from openral_hal.lifecycle import _HALLifecycleNode  # type: ignore[attr-defined]
from rclpy.lifecycle import TransitionCallbackReturn
from sensor_msgs.msg import JointState as RosJointState

NODE_NAME = "openral_hal_ur10e"
N_JOINTS = 6


def _hal_factory() -> UR10eHAL:
    return UR10eHAL(gravity_enabled=False)


def test_lifecycle_smoke() -> None:
    """Drive the full configure→activate→deactivate→cleanup transition cycle."""
    rclpy.init()
    try:
        node = _HALLifecycleNode(NODE_NAME, _hal_factory)
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(node)

        def spin_for(seconds: float) -> None:
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                executor.spin_once(timeout_sec=0.02)

        try:
            assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
            assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS

            assert node._hal is not None  # type: ignore[attr-defined]
            desc = node._hal.description  # type: ignore[attr-defined]
            assert len(desc.joints) == N_JOINTS, (
                f"expected {N_JOINTS}-DoF RobotDescription on the live HAL, got {len(desc.joints)}"
            )

            helper = rclpy.create_node("test_lifecycle_subscriber")
            executor.add_node(helper)
            received: list[RosJointState] = []
            from rclpy.qos import (
                QoSDurabilityPolicy,
                QoSProfile,
                QoSReliabilityPolicy,
            )

            qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=10,
            )
            helper.create_subscription(
                RosJointState,
                f"/{NODE_NAME}/joint_states",
                received.append,
                qos,
            )

            spin_for(1.0)

            assert len(received) >= 1, (
                f"expected ≥1 joint_states message during active phase, got {len(received)}"
            )
            assert len(received[-1].position) == N_JOINTS

            assert node.trigger_deactivate() == TransitionCallbackReturn.SUCCESS
            assert node.trigger_cleanup() == TransitionCallbackReturn.SUCCESS
            assert node.trigger_shutdown() == TransitionCallbackReturn.SUCCESS

            executor.remove_node(helper)
            helper.destroy_node()
        finally:
            executor.remove_node(node)
            node.destroy_node()
    finally:
        rclpy.shutdown()
