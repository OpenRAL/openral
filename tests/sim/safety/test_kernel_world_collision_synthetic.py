"""ADR-0030 world phase — world-obstacle collision rejection through the real kernel.

A synthetic 1-link arm is lowered to kernel collision params and driven through
the **real safety_kernel_node** while world obstacles are published on
``/openral/world_collision``:

1. **Before any world is published** the kernel fails closed: a chunk it cannot
   verify against a fresh world is dropped (no ``safe_action``) but the kernel
   is NOT latched.
2. With a fresh world whose obstacle is **far**, the chunk passes through.
3. With a fresh world whose obstacle **overlaps** the arm link, the kernel drops
   it, fires ``/openral/estop``, and publishes
   ``FailureTrigger(KIND_COLLISION)`` with ``collision_kind="world"``.

World geometry is analytic (the obstacle is synthetic, not in any MJCF), so the
expected ``min_distance_m`` is computed by hand.

Gates: ROS_DISTRO + rclpy + openral_msgs, on a sourced + colcon-built workspace.
"""

from __future__ import annotations

import json
import os
import time
import uuid

import pytest

_ROS2_AVAILABLE = bool(os.environ.get("ROS_DISTRO"))
pytestmark = pytest.mark.skipif(
    not _ROS2_AVAILABLE, reason="ROS_DISTRO not set — requires a sourced ROS 2 install."
)
pytest.importorskip("rclpy")
pytest.importorskip("openral_msgs")

from openral_core import (  # noqa: E402
    CapsuleShape,
    ControlMode,
    EmbodimentKind,
    JointSpec,
    JointType,
    LinkCollisionGeometry,
    RobotCapabilities,
    RobotDescription,
    SafetyEnvelope,
)
from openral_safety.envelope_loader import collision_params_from_description  # noqa: E402

from tests.sim.safety._kernel_subprocess import (  # noqa: E402
    activate_kernel_node,
    isolated_domain_id,
    start_kernel,
    terminate_kernel,
)


def _one_link_arm() -> RobotDescription:
    """base → link1 (revolute Y); a capsule on link1 spanning z∈[0.02, 0.38] at rest."""
    return RobotDescription(
        name="synthetic_1link",
        embodiment_kind=EmbodimentKind.MANIPULATOR,
        joints=[
            JointSpec(
                name="j1",
                joint_type=JointType.REVOLUTE,
                parent_link="base",
                child_link="link1",
                axis_xyz=(0.0, 1.0, 0.0),
                origin_xyz=(0.0, 0.0, 0.0),
            )
        ],
        collision_geometry=[
            LinkCollisionGeometry(
                link_name="link1",
                shape=CapsuleShape(radius_m=0.05, length_m=0.36),
                origin_xyz_rpy=(0.0, 0.0, 0.2, 0.0, 0.0, 0.0),
            )
        ],
        allowed_collision_pairs=[],
        capabilities=RobotCapabilities(
            supported_control_modes=[ControlMode.JOINT_POSITION], embodiment_tags=["synthetic"]
        ),
        safety=SafetyEnvelope(),
    )


def _kernel_params() -> dict[str, object]:
    params: dict[str, object] = {
        "n_dof": 1,
        "robot_name": "synthetic_1link",
        "joint_position_min": [-6.5],
        "joint_position_max": [6.5],
        "joint_velocity_max": [100.0],
        "joint_torque_max": [100.0],
        "world_collision_enabled": True,
        "world_collision_margin_m": 0.0,
        "world_collision_deadline_ms": 5000.0,
        "world_collision_max_primitives": 16,
    }
    params.update(collision_params_from_description(_one_link_arm()))
    return params


def test_kernel_world_collision_fail_closed_then_rejects() -> None:
    """Fail-closed before a world; pass with a far obstacle; estop on overlap."""
    import rclpy
    from openral_msgs.msg import ActionChunk, FailureTrigger, WorldCollision
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from std_msgs.msg import Empty

    node_name = f"safety_kernel_world_{uuid.uuid4().hex[:8]}"
    proc = start_kernel(_kernel_params(), node_name, isolated_domain_id())
    try:
        time.sleep(1.5)
        rclpy.init()
        try:
            helper = rclpy.create_node("world_helper")
            assert activate_kernel_node(node_name, helper), "kernel activation failed"

            safe: dict[str, ActionChunk] = {}
            failures: list[FailureTrigger] = []
            estops: list[Empty] = []
            safe_sub = helper.create_subscription(
                ActionChunk, "/openral/safe_action", lambda m: safe.__setitem__(m.trace_id, m), 10
            )
            helper.create_subscription(
                FailureTrigger, "/openral/failure/safety", failures.append, 50
            )
            helper.create_subscription(Empty, "/openral/estop", estops.append, 10)
            pub = helper.create_publisher(ActionChunk, "/openral/candidate_action", 10)
            world_pub = helper.create_publisher(
                WorldCollision,
                "/openral/world_collision",
                QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE),
            )

            executor = SingleThreadedExecutor()
            executor.add_node(helper)
            deadline = time.time() + 5.0
            while time.time() < deadline:
                if pub.get_subscription_count() >= 1 and safe_sub.get_publisher_count() >= 1:
                    break
                executor.spin_once(timeout_sec=0.05)

            def send(trace: str) -> None:
                chunk = ActionChunk()
                chunk.control_mode = 0  # JOINT_POSITION
                chunk.horizon = 1
                chunk.n_dof = 1
                chunk.flat = [0.0]
                chunk.rskill_id = "openral/world-test"
                chunk.trace_id = trace
                pub.publish(chunk)
                end = time.time() + 1.2
                while time.time() < end:
                    executor.spin_once(timeout_sec=0.02)

            def publish_world(x: float) -> None:
                wc = WorldCollision()
                wc.header.frame_id = "base"
                wc.header.stamp = helper.get_clock().now().to_msg()
                wc.radius = [0.05]
                wc.half_length = [0.18]
                wc.origin_xyzrpy = [x, 0.0, 0.2, 0.0, 0.0, 0.0]
                wc.object_id = ["obstacle"]
                world_pub.publish(wc)
                end = time.time() + 0.5
                while time.time() < end:
                    executor.spin_once(timeout_sec=0.02)

            # 1. No world yet → fail-closed: dropped, but NOT latched/estopped.
            send("no-world")
            assert "no-world" not in safe, "chunk must be dropped when the world is unavailable"
            assert not estops, "world-unavailable is fail-closed but must not estop/latch"

            # 2. Fresh world, obstacle far away → passes through.
            publish_world(1.0)
            send("clear")
            assert "clear" in safe, "chunk should pass with a fresh, non-colliding world"
            assert not estops

            # 3. Fresh world, obstacle overlapping the arm link → estop + world collision.
            publish_world(0.05)  # capsule centre 0.05 m from link1's at z=0.2
            send("collide")
            assert "collide" not in safe, "world-colliding chunk must NOT reach safe_action"
            assert estops, "world collision must fire /openral/estop"
            assert failures, "world collision must publish a FailureTrigger"
            trigger = failures[-1]
            assert trigger.kind == FailureTrigger.KIND_COLLISION
            evidence = json.loads(trigger.evidence_json)
            assert evidence["collision_kind"] == "world"
            assert evidence["link_a"] == "link1"
            assert evidence["link_b_or_object"] == "obstacle"
            # centreline 0.05, radii 0.05 + 0.05 → -0.05 penetration.
            assert abs(evidence["min_distance_m"] - (-0.05)) < 1e-6
        finally:
            rclpy.shutdown()
    finally:
        terminate_kernel(proc)
