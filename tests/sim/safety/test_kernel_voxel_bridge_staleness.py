"""A stopped octree must reach the real kernel as ``DROP_VOXEL_UNAVAILABLE``.

Hazard log Entry 033. ``octomap_server`` publishes ``/octomap_binary`` only when
it inserts a cloud, so a dead camera is a silent octree. The octomap bridge used
to keep republishing its LAST octree as a fresh ``/openral/world_voxels`` grid
on its 10 Hz timer; the kernel times voxel freshness from receipt, so it kept
certifying motion against a frozen map (observed on Thor 2026-09-24: ZED driver
stopped, ``/octomap_binary`` silent, ``/openral/world_voxels`` still 7.1 Hz).

The chain here is all real: the ``octomap_voxel_bridge`` binary, the
``safety_kernel_node`` binary with the deploy's ``world_voxel_deadline_ms``, a
real ``octomap_msgs/Octomap`` in octomap's own binary encoding, a real static
TF. Octrees arrive at Thor's measured octomap cadence (~3.2 Hz); then they stop;
then they resume. The kernel must certify while they arrive, drop with
``DROP_VOXEL_UNAVAILABLE`` (no latch, no E-stop) once the bridge's
``max_octree_age_s`` plus the kernel's deadline have passed, and certify again
after the next octree.

Gates: ROS_DISTRO + rclpy + openral_msgs + octomap_msgs + tf2_ros, on a sourced
workspace with ``openral_safety_kernel`` and ``openral_octomap_bridge`` built.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
import uuid
from typing import Any

import pytest

_ROS2_AVAILABLE = bool(os.environ.get("ROS_DISTRO"))
pytestmark = pytest.mark.skipif(
    not _ROS2_AVAILABLE, reason="ROS_DISTRO not set — requires a sourced ROS 2 install."
)
pytest.importorskip("rclpy")
pytest.importorskip("openral_msgs")
pytest.importorskip("octomap_msgs")
pytest.importorskip("tf2_ros")

from tests.sim.safety._kernel_subprocess import (  # noqa: E402
    activate_kernel_node,
    isolated_domain_id,
    start_kernel,
    terminate_kernel,
)
from tests.sim.safety.test_kernel_voxel_collision_synthetic import (  # noqa: E402
    _kernel_params,
)

# The deploy's pair (`deploy_e2e.launch.py`): the kernel's deadline, and the
# bridge bound derived from it as half.
_WORLD_VOXEL_DEADLINE_MS = 1000.0
_MAX_OCTREE_AGE_S = 0.5
_OCTREE_PERIOD_S = 0.31  # Thor's measured octomap cadence, 3.2 Hz
_RESOLUTION = 0.05
_COVERAGE_RADIUS_M = 0.5  # (2*0.5/0.05 + 1)^3 = 9261 cells, under the cap below

# One OcTree node in octomap's binary encoding (`writeBinaryNode`): a root whose
# eight children are all FREE leaves (2 bits per child, `10` = free, low bit
# first → 0b01010101 per four children). A known, entirely free world.
_FREE_WORLD_OCTREE = [0x55, 0x55]


def _start_bridge(domain_id: int) -> Any:
    if shutil.which("ros2") is None:
        pytest.skip("ros2 binary not on PATH; source install/setup.bash first")
    return subprocess.Popen(
        [
            "ros2",
            "run",
            "openral_octomap_bridge",
            "octomap_voxel_bridge",
            "--ros-args",
            "-r",
            f"__node:=octomap_bridge_{uuid.uuid4().hex[:8]}",
            "-p",
            "base_frame:=base",
            "-p",
            f"resolution:={_RESOLUTION}",
            "-p",
            f"coverage_radius_m:={_COVERAGE_RADIUS_M}",
            "-p",
            f"max_octree_age_s:={_MAX_OCTREE_AGE_S}",
        ],
        env={**os.environ, "ROS_DOMAIN_ID": str(domain_id)},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def test_a_stopped_octree_drops_voxel_unavailable_and_resumes() -> None:
    import rclpy
    from geometry_msgs.msg import TransformStamped
    from octomap_msgs.msg import Octomap
    from openral_msgs.msg import ActionChunk, SafetyStatus
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import Empty
    from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

    domain_id = isolated_domain_id()
    params = {
        **_kernel_params(),
        "world_voxel_deadline_ms": _WORLD_VOXEL_DEADLINE_MS,
        "world_voxel_max_cells": 20000,
    }
    kernel_name = f"safety_kernel_bridge_staleness_{uuid.uuid4().hex[:8]}"
    kernel = start_kernel(params, kernel_name, domain_id)
    bridge = _start_bridge(domain_id)
    try:
        time.sleep(1.5)
        rclpy.init()
        try:
            helper = rclpy.create_node("bridge_staleness_helper")
            assert activate_kernel_node(kernel_name, helper), "kernel activation failed"

            tf = TransformStamped()
            tf.header.stamp = helper.get_clock().now().to_msg()
            tf.header.frame_id = "map"
            tf.child_frame_id = "base"
            tf.transform.rotation.w = 1.0
            broadcaster = StaticTransformBroadcaster(helper)
            broadcaster.sendTransform(tf)

            safe: set[str] = set()
            status: list[SafetyStatus] = []
            estops: list[Empty] = []
            safe_sub = helper.create_subscription(
                ActionChunk, "/openral/safe_action", lambda m: safe.add(m.trace_id), 10
            )
            helper.create_subscription(
                SafetyStatus,
                "/openral/safety_status",
                status.append,
                QoSProfile(
                    depth=1,
                    reliability=ReliabilityPolicy.RELIABLE,
                    durability=DurabilityPolicy.TRANSIENT_LOCAL,
                ),
            )
            helper.create_subscription(Empty, "/openral/estop", estops.append, 10)
            chunk_pub = helper.create_publisher(ActionChunk, "/openral/candidate_action", 10)
            octomap_pub = helper.create_publisher(
                Octomap,
                "/octomap_binary",
                QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE),
            )

            executor = SingleThreadedExecutor()
            executor.add_node(helper)
            deadline = time.time() + 10.0
            while time.time() < deadline:
                if (
                    chunk_pub.get_subscription_count() >= 1
                    and safe_sub.get_publisher_count() >= 1
                    and octomap_pub.get_subscription_count() >= 1
                ):
                    break
                executor.spin_once(timeout_sec=0.05)
            assert octomap_pub.get_subscription_count() >= 1, "bridge never subscribed"

            def octree() -> Octomap:
                msg = Octomap()
                msg.header.frame_id = "map"
                msg.header.stamp = helper.get_clock().now().to_msg()
                msg.binary = True
                msg.id = "OcTree"
                msg.resolution = _RESOLUTION
                msg.data = _FREE_WORLD_OCTREE
                return msg

            def run(seconds: float, *, octrees: bool) -> None:
                end = time.time() + seconds
                next_octree = time.time()
                while time.time() < end:
                    if octrees and time.time() >= next_octree:
                        octomap_pub.publish(octree())
                        next_octree += _OCTREE_PERIOD_S
                    executor.spin_once(timeout_sec=0.02)

            def send(trace: str) -> None:
                chunk = ActionChunk()
                chunk.control_mode = 0  # JOINT_POSITION
                chunk.horizon = 1
                chunk.n_dof = 1
                chunk.flat = [0.0]
                chunk.rskill_id = "openral/bridge-staleness-test"
                chunk.trace_id = trace
                chunk_pub.publish(chunk)

            # 1. Octrees arriving: the kernel certifies.
            run(2.0, octrees=True)
            send("fresh")
            run(0.6, octrees=True)
            assert "fresh" in safe, "a chunk must pass while octrees arrive"

            # 2. Octrees stop (the camera died). Past the bridge bound plus the
            #    kernel's deadline, with margin, the kernel must fail closed.
            #    Pre-fix the bridge kept the grid arriving and this chunk passed.
            run(_MAX_OCTREE_AGE_S + _WORLD_VOXEL_DEADLINE_MS / 1000.0 + 0.5, octrees=False)
            send("stale")
            run(0.6, octrees=False)
            assert "stale" not in safe, (
                "the kernel certified motion against an octree that stopped arriving"
            )
            assert status, "the kernel published no SafetyStatus"
            assert status[-1].drop_reason == SafetyStatus.DROP_VOXEL_UNAVAILABLE, status[-1]
            assert not status[-1].latched, "an unavailable map is a drop, not a latch"
            assert not estops, "an unavailable map must not E-stop"

            # 3. The next octree resumes certification with no operator action.
            run(1.0, octrees=True)
            send("resumed")
            run(0.6, octrees=True)
            assert "resumed" in safe, "a chunk must pass again once octrees resume"
        finally:
            rclpy.shutdown()
    finally:
        terminate_kernel(bridge)
        terminate_kernel(kernel)
