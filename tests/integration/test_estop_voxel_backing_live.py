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

The second half of this test is the **ordering**, which is where the join was
actually being lost. The kernel publishes the failure trigger and the E-stop on
different topics with no guaranteed delivery order, and the snapshot is
deliberately never delayed for the evidence. So when the evidence loses the
race the located cell does not exist yet, and the map-side half of the record
used to be dropped without a word — 14 of the 15 stops in the 2026-08-26
five-round battery recorded no backing at all. The record must instead say
nothing while it knows nothing (never attribute the *previous* stop's cell to
this one) and then emit the backing on the late line.

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

import json
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


def _logged(err: str, marker: str) -> dict[str, Any]:
    """The JSON payload of the last ``marker`` line the node logged to stderr."""
    lines = [line for line in err.splitlines() if marker in line]
    assert lines, f"no {marker} line was logged"
    return dict(json.loads(lines[-1].split(marker, 1)[1].strip()))


def test_an_evidence_voxel_index_becomes_a_position_on_the_live_graph(capfd: Any) -> None:
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
    from std_msgs.msg import Empty

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
    estop_qos = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.VOLATILE,
        depth=10,
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

        # --- the ordering half: the evidence loses the race ---------------
        #
        # Let the cached evidence age past the freshness window, so the next
        # stop is the real "E-stop first, evidence second" case the kernel's
        # two topics produce.
        from openral_hal.sim_sensor_bridge import _ESTOP_EVIDENCE_WINDOW_NS

        time.sleep(_ESTOP_EVIDENCE_WINDOW_NS / 1e9 + 0.1)
        estop_pub = peer.create_publisher(Empty, "/openral/estop", estop_qos)
        assert _wait_until(lambda: estop_pub.get_subscription_count() > 0)
        capfd.readouterr()
        estop_pub.publish(Empty())
        assert _wait_until(lambda: bridge._estop_awaiting_evidence), (
            "a stop whose evidence has not arrived must record that it is waiting"
        )
        snapshot = _logged(capfd.readouterr().err, "sim.estop_ground_truth_snapshot")

        # Nothing is known about the cell yet, and the record says exactly
        # that. The stale evidence still cached from above addresses a
        # DIFFERENT stop, and attributing its cell here would put a confident,
        # wrong backing next to a null ``collision_evidence``.
        assert snapshot["collision_evidence"] is None
        assert snapshot["evidence_voxel_backing"] is None

        # Now the evidence lands. The map-side half of the record is what the
        # late line exists to carry; without it this stop could never be
        # classified as backed by real geometry, by decoration, or by nothing.
        trigger.header.stamp = peer.get_clock().now().to_msg()
        failure_pub.publish(trigger)
        assert _wait_until(lambda: not bridge._estop_awaiting_evidence)
        late = _logged(capfd.readouterr().err, "sim.estop_ground_truth_evidence")

        assert late["stop_seq"] == snapshot["stop_seq"], "the two lines join on stop_seq"
        assert late["collision_evidence"] is not None
        backing = late["evidence_voxel_backing"]
        assert isinstance(backing, dict), "the late line must carry the backing, not just evidence"
        assert tuple(backing["voxel_ijk"]) == _EVIDENCE_IJK
        assert tuple(backing["base_xyz"]) == pytest.approx(_EVIDENCE_BASE_XYZ)
        assert backing["verdict"] == "unbacked"
        # How stale the probe is, so a reader can discount a backing body that
        # could have moved between the stop and the probe.
        assert int(late["backing_after_snapshot_ns"]) >= 0
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
