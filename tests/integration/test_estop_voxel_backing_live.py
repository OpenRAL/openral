# SPDX-License-Identifier: Apache-2.0
"""Live-ROS: a world-voxel stop must be able to say what backs its cell.

The pure classification is unit-tested against compiled ``MjModel``s in
``tests/unit/test_sim_estop_voxel_backing.py``. What only a live graph can pin
is the wiring that gets the question asked at all, because it spans two topics
that meet nowhere else:

* ``/openral/failure/safety`` names the cell as an **index** — ``b=voxel_76001``
  in the field logs — and nothing more;
* ``/openral/world_voxels`` carries the grid that index addresses, and the
  kernel never republishes it alongside the evidence.

Until they are joined, an evidence index is not a position and the record
cannot look at the map at all — which is exactly how the 2026-08-22 round
adjudicated two stops as false positives on ground truth that had never
examined the cell.

This is the live half: the production ``ManifestHALLifecycleNode`` on a real
``SimAttachedHAL`` (the real ``tabletop_push`` rollout, a real compiled
``MjModel``), its real ``SimSensorBridge``, and real ``openral_msgs`` on the
wire. No mocks (CLAUDE.md §1.11).

Gated on ``OPENRAL_TEST_ROS_LIVE=1`` and listed in ``scripts/ros_live_tests.sh``.
Locally::

    source /opt/ros/jazzy/setup.bash && just ros2-build
    source install/setup.bash
    just test-ros-live -k estop_voxel_backing
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

_LIVE_ROS = bool(os.getenv("OPENRAL_TEST_ROS_LIVE"))
_LIVE_ROS_REASON = (
    "live rclpy node + colcon openral_msgs overlay — set OPENRAL_TEST_ROS_LIVE=1 "
    "and source install/setup.bash first."
)

pytestmark = pytest.mark.skipif(not _LIVE_ROS, reason=_LIVE_ROS_REASON)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ROBOT_YAML = _REPO_ROOT / "robots" / "so101_follower" / "robot.yaml"
_SCENE_YAML = _REPO_ROOT / "scenes" / "sim" / "tabletop_cube_push.yaml"
_DEADLINE_S = 10.0

# A real bounded local grid of the shape the octomap bridge publishes: base
# frame, x fastest, `origin` the minimum corner of cell (0,0,0).
_GRID_ORIGIN = (-0.8, -0.8, -0.3)
_GRID_RES = 0.025
_GRID_DIMS = (64, 64, 64)
# The cell the 2026-08-22 drawer_utensil stop named, verbatim. Its decode is the
# thing under test: index 76001 on this lattice is (33, 35, 18), whose centre is
# base-frame (0.0375, 0.0875, 0.1625) — the position quoted in that round's
# notes, so a regression in the index convention shows up as a wrong point
# rather than as a plausible one.
_EVIDENCE_INDEX = 76001
_EVIDENCE_IJK = (33, 35, 18)
_EVIDENCE_BASE_XYZ = (0.0375, 0.0875, 0.1625)


def _wait_until(predicate: Any, *, timeout_s: float = _DEADLINE_S) -> bool:
    """Spin-wait on a live graph predicate (the executor runs on its own thread)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_an_evidence_voxel_index_becomes_a_position_on_the_live_graph() -> None:
    """The two topics a world-voxel stop needs are joined into one located cell."""
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("mujoco")
    pytest.importorskip("openral_msgs")

    from openral_core import CollisionEvidence
    from openral_hal.lifecycle import ManifestHALLifecycleNode
    from openral_msgs.msg import FailureTrigger, OccupancyVoxels
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

    rclpy.init()
    node: Any = ManifestHALLifecycleNode("test_estop_voxel_backing_live")
    node.set_parameters(
        [
            Parameter("robot_yaml", value=str(_ROBOT_YAML)),
            Parameter("hal_mode", value="sim"),
            Parameter("sim_env_yaml", value=str(_SCENE_YAML)),
            Parameter("publish_rate_hz", value=20.0),
            Parameter("viewer_enabled", value=False),
        ]
    )
    peer = Node("test_estop_voxel_backing_peer")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(peer)

    volatile = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.VOLATILE,
        depth=1,
    )
    bus = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.VOLATILE,
        depth=50,
    )
    voxel_pub = peer.create_publisher(OccupancyVoxels, "/openral/world_voxels", volatile)
    failure_pub = peer.create_publisher(FailureTrigger, "/openral/failure/safety", bus)

    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    try:
        assert str(node.trigger_configure()).endswith("SUCCESS"), "configure failed"
        assert str(node.trigger_activate()).endswith("SUCCESS"), "activate failed"
        bridge = node._bridge
        assert bridge is not None, "the manifest node wires the SimSensorBridge at activate"

        # Before either message, there is nothing to locate — and the record
        # says so rather than inventing a cell.
        assert bridge._evidence_voxel() is None

        grid = OccupancyVoxels()
        grid.header.stamp = peer.get_clock().now().to_msg()
        grid.header.frame_id = "base_link"
        grid.origin.x, grid.origin.y, grid.origin.z = _GRID_ORIGIN
        grid.resolution = _GRID_RES
        grid.size_x, grid.size_y, grid.size_z = _GRID_DIMS
        grid.occupancy = bytes(_GRID_DIMS[0] * _GRID_DIMS[1] * _GRID_DIMS[2])
        assert _wait_until(lambda: voxel_pub.get_subscription_count() > 0), (
            "the bridge must subscribe /openral/world_voxels for the stop record"
        )
        voxel_pub.publish(grid)
        assert _wait_until(lambda: bridge._last_voxel_grid is not None), (
            "the grid geometry must be retained when the message lands"
        )

        # A grid alone still is not a stop: no evidence, no cell.
        assert bridge._evidence_voxel() is None

        trigger = FailureTrigger()
        trigger.header.stamp = peer.get_clock().now().to_msg()
        trigger.kind = FailureTrigger.KIND_COLLISION
        trigger.rskill_id = "OpenRAL/rskill-xr1-panda_mobile-robocasa365-nf4"
        trigger.trace_id = ""
        trigger.evidence_json = CollisionEvidence(
            collision_kind="world",
            horizon_step=-1,
            link_a="panda_link1",
            link_b_or_object=f"voxel_{_EVIDENCE_INDEX}",
            min_distance_m=-0.0172764,
        ).model_dump_json()
        assert _wait_until(lambda: failure_pub.get_subscription_count() > 0)
        failure_pub.publish(trigger)
        assert _wait_until(lambda: bridge._evidence_voxel() is not None), (
            "evidence + grid must join into a located cell"
        )

        located = bridge._evidence_voxel()
        assert located["index"] == _EVIDENCE_INDEX
        assert located["resolution"] == pytest.approx(_GRID_RES)
        assert tuple(located["size"]) == _GRID_DIMS
        assert tuple(located["origin"]) == pytest.approx(_GRID_ORIGIN)

        # And the located cell decodes to the position the field notes quote.
        from openral_hal.sim_sensor_bridge import voxel_backing_record

        model, data = node._hal.mujoco_handles()
        record = voxel_backing_record(
            model,
            data,
            voxel_index=int(located["index"]),
            grid_origin=list(located["origin"]),
            grid_resolution=float(located["resolution"]),
            grid_size=list(located["size"]),
            robot_body_ids=bridge._depth_self_bodies,
        )
        assert tuple(record["voxel_ijk"]) == _EVIDENCE_IJK
        assert tuple(record["base_xyz"]) == pytest.approx(_EVIDENCE_BASE_XYZ)
        # The cell is empty in this scene, and "unbacked" is attested by how
        # hard the probe looked — never by silence.
        assert record["verdict"] == "unbacked"
        assert int(record["rays_cast"]) > 0
    finally:
        with suppress(Exception):
            node.trigger_deactivate()
        with suppress(Exception):
            node.trigger_cleanup()
        executor.shutdown()
        spin.join(timeout=5.0)
        with suppress(Exception):
            peer.destroy_node()
        with suppress(Exception):
            node.destroy_node()
        rclpy.shutdown()
