"""Lifecycle smoke test for ``openral_hal_franka``.

Drives the standard managed-lifecycle transition path against the generic
``_HALLifecycleNode`` from :mod:`openral_hal.lifecycle` using the real
``FrankaPandaHAL`` factory.  The HAL pulls its MJCF lazily from the
``robot_descriptions`` package and ships its canonical
:class:`openral_core.RobotDescription` (``FRANKA_PANDA_DESCRIPTION``) —
so this smoke exercises the same RobotDescription wiring used at runtime,
not a stub.

Lifecycle phases exercised:

* ``unconfigured → configure → inactive → activate → active``
* joint-state publication while ``active``
* ``deactivate → cleanup → shutdown``

The test is gated on ``rclpy``, ``openral_hal``, ``mujoco`` and
``robot_descriptions`` being importable; in lint-only environments any
missing piece causes a clean skip.
"""

from __future__ import annotations

import time

import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("openral_hal")
pytest.importorskip("mujoco")
pytest.importorskip("robot_descriptions")

from openral_hal import FrankaPandaHAL
from openral_hal.lifecycle import _HALLifecycleNode  # type: ignore[attr-defined]
from rclpy.lifecycle import TransitionCallbackReturn
from sensor_msgs.msg import JointState as RosJointState

NODE_NAME = "openral_hal_franka"
_FRANKA_DOF = 8  # 7 arm joints + 1 gripper, matching FRANKA_PANDA_DESCRIPTION


def _hal_factory() -> FrankaPandaHAL:
    # gravity_enabled=False keeps the smoke deterministic without a controller.
    return FrankaPandaHAL(gravity_enabled=False)


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
            assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS, (
                "configure transition failed"
            )
            assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS, (
                "activate transition failed"
            )

            # The real HAL carries the canonical RobotDescription; verify the
            # smoke is talking to it (and not a stub) by checking joint count.
            assert node._hal is not None  # type: ignore[attr-defined]
            desc = node._hal.description  # type: ignore[attr-defined]
            assert len(desc.joints) == _FRANKA_DOF, (
                f"expected {_FRANKA_DOF}-DoF RobotDescription on the live HAL, "
                f"got {len(desc.joints)}"
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
            assert len(received[-1].position) == _FRANKA_DOF, (
                f"expected {_FRANKA_DOF}-DoF joint state from FrankaPandaHAL, "
                f"got {len(received[-1].position)}"
            )

            assert node.trigger_deactivate() == TransitionCallbackReturn.SUCCESS, (
                "deactivate transition failed"
            )
            assert node.trigger_cleanup() == TransitionCallbackReturn.SUCCESS, (
                "cleanup transition failed"
            )
            assert node.trigger_shutdown() == TransitionCallbackReturn.SUCCESS, (
                "shutdown transition failed"
            )

            executor.remove_node(helper)
            helper.destroy_node()
        finally:
            executor.remove_node(node)
            node.destroy_node()
    finally:
        rclpy.shutdown()
