"""ADR-0040 — mobile-base world collision via the REAL kernel + REAL manifest.

End-to-end proof of the base-relative frame fix (HZ-0040-1) on the actual
``safety_kernel_node`` binary loaded from the real ``robots/panda_mobile``
manifest (the robocasa deploy-sim robot). It is the deploy-graph analogue of the
unit test ``LifecycleKernelTest.MobileBaseArmCaughtAgainstVoxelWall``:

1. The kernel is configured from the manifest exactly as ``sim_e2e.launch.py``
   does — envelope + collision model + ``collision_base_dofs`` (the planar base)
   + ``collision_joint_names`` + ``collision_ee_link_index``.
2. ``/joint_states`` seeds the BASE far out in the world (x=y=5 m) and the arm at
   home. A base-relative occupancy grid (filled occupied around ``base_link``)
   is published on ``/openral/world_voxels``.
3. A ``CARTESIAN_DELTA`` chunk (the robocasa arm mode) must be rejected + estop:
   the kernel zeroes the base dofs, evaluates the arm in the ``base_link`` frame
   the grid lives in, finds the arm inside an occupied voxel, and fires
   ``/openral/estop`` + ``FailureTrigger(KIND_COLLISION)``.

Without the base-dof fix the arm would be placed at the base's world pose (x~5 m,
outside the local grid) and nothing would be caught — so this fails closed-loop
on a regression. CLAUDE.md §1.11 — real kernel binary, real manifest, real DDS;
no mocks.

Gates: ROS_DISTRO + rclpy + openral_msgs on a sourced + colcon-built workspace.
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

from openral_core import RobotDescription  # noqa: E402
from openral_safety.envelope_loader import (  # noqa: E402
    collision_params_from_description,
    compute_intersection,
    ee_link_index_from_collision_params,
    kernel_params_from_envelope,
)

from tests.sim.safety._kernel_subprocess import (  # noqa: E402
    activate_kernel_node,
    isolated_domain_id,
    start_kernel,
    terminate_kernel,
)

_MANIFEST = "robots/panda_mobile/robot.yaml"

# Base-relative occupancy box around base_link, filled occupied: x,y in [-2,2],
# z in [-1,2] @ 0.25 m. The home arm (base-relative) is inside; the base seeded
# at (5,5) would put the arm outside without the base-dof zeroing fix.
_RES = 0.25
_ORIGIN = (-2.0, -2.0, -1.0)
_SX, _SY, _SZ = 16, 16, 12


def _kernel_params() -> dict[str, object]:
    desc = RobotDescription.from_yaml(_MANIFEST)
    collision = collision_params_from_description(desc)
    params: dict[str, object] = dict(kernel_params_from_envelope(compute_intersection(desc, None)))
    params.update(collision)
    params.update(
        {
            "world_voxel_enabled": True,
            "world_voxel_margin_m": 0.0,
            "world_voxel_deadline_ms": 5000.0,
            "world_voxel_max_cells": _SX * _SY * _SZ,
            # ADR-0040 — exactly what sim_e2e.launch.py emits for this robot.
            "collision_joint_names": [j.name for j in desc.joints],
            "collision_base_dofs": [
                i for i, j in enumerate(desc.joints) if j.name in set(desc.base_joints or [])
            ],
            "collision_ee_link_index": ee_link_index_from_collision_params(collision),
            "collision_seed_dt_s": 0.0,
            "collision_state_deadline_ms": 5000.0,
        }
    )
    return params


def test_real_kernel_mobile_base_world_collision_estops() -> None:
    """The real kernel + real panda_mobile manifest estops a CARTESIAN_DELTA chunk
    whose base-relative arm config sits inside an occupied voxel."""
    import rclpy
    from geometry_msgs.msg import Point
    from openral_msgs.msg import ActionChunk, FailureTrigger, OccupancyVoxels
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Empty

    desc = RobotDescription.from_yaml(_MANIFEST)
    joint_names = [j.name for j in desc.joints]
    base_dofs = {jn for jn in (desc.base_joints or [])}

    node_name = f"safety_kernel_mobile_{uuid.uuid4().hex[:8]}"
    proc = start_kernel(_kernel_params(), node_name, isolated_domain_id())
    try:
        time.sleep(1.5)
        rclpy.init()
        try:
            helper = rclpy.create_node("mobile_collision_helper")
            assert activate_kernel_node(node_name, helper), "kernel activation failed"

            safe: dict[str, ActionChunk] = {}
            failures: list[FailureTrigger] = []
            estops: list[Empty] = []
            helper.create_subscription(
                ActionChunk, "/openral/safe_action", lambda m: safe.__setitem__(m.trace_id, m), 10
            )
            helper.create_subscription(
                FailureTrigger, "/openral/failure/safety", failures.append, 50
            )
            helper.create_subscription(Empty, "/openral/estop", estops.append, 10)
            cand_pub = helper.create_publisher(ActionChunk, "/openral/candidate_action", 10)
            voxel_pub = helper.create_publisher(
                OccupancyVoxels,
                "/openral/world_voxels",
                QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE),
            )
            js_pub = helper.create_publisher(
                JointState,
                "/joint_states",
                QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
            )

            executor = SingleThreadedExecutor()
            executor.add_node(helper)
            deadline = time.time() + 5.0
            while time.time() < deadline:
                if cand_pub.get_subscription_count() >= 1:
                    break
                executor.spin_once(timeout_sec=0.05)

            # Seed: base FAR out in the world (x=y=5 m), arm + gripper at home.
            js = JointState()
            js.name = joint_names
            js.position = [5.0 if n in base_dofs else 0.0 for n in joint_names]

            # Base-relative occupancy box, all cells occupied.
            grid = OccupancyVoxels()
            grid.header.frame_id = "base_link"
            grid.origin = Point(x=_ORIGIN[0], y=_ORIGIN[1], z=_ORIGIN[2])
            grid.resolution = _RES
            grid.size_x = _SX
            grid.size_y = _SY
            grid.size_z = _SZ
            grid.occupancy = [1] * (_SX * _SY * _SZ)

            # Establish a fresh seed + grid first.
            warm = time.time() + 1.0
            while time.time() < warm:
                js.header.stamp = helper.get_clock().now().to_msg()
                grid.header.stamp = helper.get_clock().now().to_msg()
                js_pub.publish(js)
                voxel_pub.publish(grid)
                executor.spin_once(timeout_sec=0.02)

            # CARTESIAN_DELTA chunk (the robocasa arm mode), zero delta — the
            # REACTIVE check on the measured (base-relative) config must catch it.
            chunk = ActionChunk()
            chunk.control_mode = 5  # CARTESIAN_DELTA
            chunk.horizon = 1
            chunk.n_dof = 6
            chunk.flat = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            chunk.rskill_id = "openral/mobile-collision-test"
            chunk.trace_id = "collide"

            end = time.time() + 4.0
            while time.time() < end and not estops:
                js.header.stamp = helper.get_clock().now().to_msg()
                grid.header.stamp = helper.get_clock().now().to_msg()
                js_pub.publish(js)
                voxel_pub.publish(grid)
                cand_pub.publish(chunk)
                executor.spin_once(timeout_sec=0.02)

            assert "collide" not in safe, "colliding Cartesian chunk must NOT reach safe_action"
            assert estops, "mobile-base world collision must fire /openral/estop"
            assert failures, "mobile-base world collision must publish a FailureTrigger"
            trigger = failures[-1]
            assert trigger.kind == FailureTrigger.KIND_COLLISION
            evidence = json.loads(trigger.evidence_json)
            assert evidence["collision_kind"] == "world"
            assert evidence["link_b_or_object"].startswith("voxel_")
        finally:
            rclpy.shutdown()
    finally:
        terminate_kernel(proc)
