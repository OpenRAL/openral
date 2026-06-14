"""ADR-0030 voxel phase — dense occupancy-grid (octomap path) collision via the real kernel.

A synthetic 1-link arm is driven through the **real safety_kernel_node** while a
dense occupancy voxel grid is published on ``/openral/world_voxels`` (the
kernel-facing form an OctoMap bridge would produce):

1. **Before any grid is published** the kernel fails closed (drops, no latch).
2. With a fresh, all-free grid the chunk passes through.
3. With a fresh grid whose occupied voxel overlaps the arm link, the kernel
   drops it, fires ``/openral/estop``, and publishes
   ``FailureTrigger(KIND_COLLISION)`` with ``collision_kind="world"`` and a
   ``voxel_<idx>`` obstacle label.

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

# 5x5x5 grid, 0.1 m voxels, origin (-0.25, -0.25, 0.0) → voxel (2,2,2)'s centre
# is (0, 0, 0.25), inside the arm link's capsule (centred at (0,0,0.2)).
_SX = _SY = _SZ = 5
_OCC_INDEX = 2 + _SX * (2 + _SY * 2)  # linear index of voxel (2,2,2)


def _one_link_arm() -> RobotDescription:
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
        "world_voxel_enabled": True,
        "world_voxel_margin_m": 0.0,
        "world_voxel_deadline_ms": 5000.0,
        "world_voxel_max_cells": 4096,
    }
    params.update(collision_params_from_description(_one_link_arm()))
    return params


def test_kernel_voxel_collision_fail_closed_then_rejects() -> None:
    """Fail-closed before a grid; pass with an empty grid; estop on an occupied voxel."""
    import rclpy
    from geometry_msgs.msg import Point
    from openral_msgs.msg import ActionChunk, FailureTrigger, OccupancyVoxels
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from std_msgs.msg import Empty

    node_name = f"safety_kernel_voxel_{uuid.uuid4().hex[:8]}"
    proc = start_kernel(_kernel_params(), node_name, isolated_domain_id())
    try:
        time.sleep(1.5)
        rclpy.init()
        try:
            helper = rclpy.create_node("voxel_helper")
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
            voxel_pub = helper.create_publisher(
                OccupancyVoxels,
                "/openral/world_voxels",
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
                chunk.rskill_id = "openral/voxel-test"
                chunk.trace_id = trace
                pub.publish(chunk)
                end = time.time() + 1.2
                while time.time() < end:
                    executor.spin_once(timeout_sec=0.02)

            def publish_grid(occupied_index: int | None) -> None:
                grid = OccupancyVoxels()
                grid.header.frame_id = "base"
                grid.header.stamp = helper.get_clock().now().to_msg()
                grid.origin = Point(x=-0.25, y=-0.25, z=0.0)
                grid.resolution = 0.1
                grid.size_x = _SX
                grid.size_y = _SY
                grid.size_z = _SZ
                occ = [0] * (_SX * _SY * _SZ)
                if occupied_index is not None:
                    occ[occupied_index] = 1
                grid.occupancy = occ
                voxel_pub.publish(grid)
                end = time.time() + 0.5
                while time.time() < end:
                    executor.spin_once(timeout_sec=0.02)

            # 1. No grid yet → fail-closed: dropped, but not latched/estopped.
            send("no-grid")
            assert "no-grid" not in safe, "chunk must be dropped when the voxel grid is unavailable"
            assert not estops, "voxel-unavailable is fail-closed but must not estop/latch"

            # 2. Fresh, all-free grid → passes through.
            publish_grid(None)
            send("clear")
            assert "clear" in safe, "chunk should pass with a fresh, all-free voxel grid"
            assert not estops

            # 3. Fresh grid with an occupied voxel inside the arm link → estop.
            publish_grid(_OCC_INDEX)
            send("collide")
            assert "collide" not in safe, "voxel-colliding chunk must NOT reach safe_action"
            assert estops, "voxel collision must fire /openral/estop"
            assert failures, "voxel collision must publish a FailureTrigger"
            trigger = failures[-1]
            assert trigger.kind == FailureTrigger.KIND_COLLISION
            evidence = json.loads(trigger.evidence_json)
            assert evidence["collision_kind"] == "world"
            assert evidence["link_a"] == "link1"
            assert evidence["link_b_or_object"].startswith("voxel_")
            assert evidence["min_distance_m"] < 0.0
        finally:
            rclpy.shutdown()
    finally:
        terminate_kernel(proc)
