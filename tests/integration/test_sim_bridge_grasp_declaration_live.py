# SPDX-License-Identifier: Apache-2.0
"""Live-ROS: the sim bridge subscribes a grasp declaration, measures it, and publishes it.

The producer end of the grasp-target wire (real pick-and-place design §2.1;
``docs/reference/real-pick-place-adr-drafts.md``). Dispatch publishes a region-less
``GraspDeclaration`` on ``/openral/grasp_declaration``; the sim attachment producer
measures the named target's box in the base frame and rides it on every
``/openral/attachment_state`` envelope, without churning the attachment revision,
until the declaration is retracted. ``tests/integration/test_world_state_stamped_objects.py``
covers the next hop (AttachmentState -> WorldStateStamped).

Real ``SimSensorBridge`` on a real rclpy node, real ``SimAttachedHAL`` over the compiled
place-phase rig (``tests/unit/test_sim_place_phase_witness.py``: a real ``MjModel`` with
a free ``cup`` body), served through the ``SimRollout`` boundary recorder
``tests/unit/fakes/fake_sim_env.FakeSimEnv``; real ``openral_msgs`` on the wire. No
mocks (CLAUDE.md §1.11).

Gated on ``OPENRAL_TEST_ROS_LIVE=1``, listed in ``scripts/ros_live_tests.sh``. Locally::

    source /opt/ros/jazzy/setup.bash && just ros2-build
    source install/setup.bash
    just test-ros-live -k grasp_declaration_live
"""

from __future__ import annotations

import os
import time
from typing import Any

import pytest

_LIVE_ROS = bool(os.getenv("OPENRAL_TEST_ROS_LIVE"))
_LIVE_ROS_REASON = (
    "live rclpy node + colcon openral_msgs overlay — set OPENRAL_TEST_ROS_LIVE=1 "
    "and source install/setup.bash first."
)

pytestmark = pytest.mark.skipif(not _LIVE_ROS, reason=_LIVE_ROS_REASON)

_DEADLINE_S = 5.0


def _spin_until(rclpy: Any, node: Any, predicate: Any) -> bool:
    deadline = time.monotonic() + _DEADLINE_S
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        if predicate():
            return True
    return False


def test_the_bridge_measures_a_grasp_declaration_onto_every_envelope() -> None:
    rclpy = pytest.importorskip("rclpy")
    mujoco = pytest.importorskip("mujoco")
    pytest.importorskip("openral_msgs")

    from openral_core import GraspDeclaration
    from openral_hal.sim_attached import SimAttachedHAL
    from openral_hal.sim_sensor_bridge import SimSensorBridge
    from openral_msgs.msg import AttachmentState, OccupancyVoxels
    from openral_msgs.msg import GraspDeclaration as GraspDeclarationMsg
    from rclpy.node import Node
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

    from tests.unit.fakes.fake_sim_env import FakeSimEnv
    from tests.unit.test_sim_place_phase_witness import _MJCF, _description

    model = mujoco.MjModel.from_xml_string(_MJCF)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    description = _description()
    hal = SimAttachedHAL(FakeSimEnv(handles=(model, data)), description)

    latched = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        depth=1,
    )
    rclpy.init()
    try:
        node = Node("test_sim_bridge_grasp_declaration_live")
        try:
            received: list[AttachmentState] = []
            node.create_subscription(
                AttachmentState, "/openral/attachment_state", received.append, latched
            )
            declare = node.create_publisher(
                GraspDeclarationMsg, "/openral/grasp_declaration", latched
            )
            grids = node.create_publisher(
                OccupancyVoxels,
                "/openral/world_voxels",
                QoSProfile(
                    reliability=QoSReliabilityPolicy.RELIABLE,
                    durability=QoSDurabilityPolicy.VOLATILE,
                    depth=1,
                ),
            )
            hal.connect()
            bridge = SimSensorBridge(node, hal, description, viewer_enabled=False)
            try:
                bridge.setup()
                assert bridge._attachment_tracker is not None, "the rig must arm the producer"
                assert bridge._grasp_declaration_sub is not None

                # Dispatch's copy: no region, ever.
                declaration = GraspDeclaration(
                    target_id="sim:cup",
                    contact_links=("left_finger", "right_finger"),
                    rskill_id="openral/pi05-robocasa",
                    trace_id="4bf92f3577b34da6a3ce929d0e0e4736",
                    timeout_s=60.0,
                    stamp_ns=int(node.get_clock().now().nanoseconds),
                )
                msg = GraspDeclarationMsg()
                declaration.fill_idl(msg)
                assert msg.region_valid is False
                declare.publish(msg)

                def declared() -> list[AttachmentState]:
                    return [m for m in received if m.grasp_declaration_valid]

                def measured() -> list[AttachmentState]:
                    return [m for m in declared() if m.grasp_declaration.region_valid]

                # No grid yet: the region's one-voxel lift off the support needs the
                # lattice's cell edge, so the declaration rides region-less (fail closed).
                # The heartbeat runs at 5 Hz: wait for three envelopes.
                assert _spin_until(rclpy, node, lambda: len(declared()) >= 3), (
                    "no grasp declaration rode /openral/attachment_state"
                )
                assert measured() == [], "no grid seen: the region must be withheld"

                # A grid arrives; a fresh declaration re-measures (the rig's cup attached
                # under the first one, which froze its region-less state).
                grid = OccupancyVoxels()
                grid.header.frame_id = "base_link"
                grid.orientation.w = 1.0
                grid.resolution = 0.02
                grid.size_x = grid.size_y = grid.size_z = 1
                grid.occupancy = [0]
                assert _spin_until(
                    rclpy,
                    node,
                    lambda: grids.publish(grid) or bridge._last_voxel_grid is not None,
                ), "the bridge never saw the grid"
                declaration = declaration.model_copy(
                    update={"stamp_ns": int(node.get_clock().now().nanoseconds)}
                )
                msg = GraspDeclarationMsg()
                declaration.fill_idl(msg)
                declare.publish(msg)
                assert _spin_until(rclpy, node, lambda: len(measured()) >= 3), (
                    "no measured grasp declaration rode /openral/attachment_state"
                )
                envelopes = measured()
                assert {m.revision for m in declared()} == {received[0].revision}, (
                    "the envelope field must not churn the attachment revision"
                )
                for envelope in envelopes:
                    wire = envelope.grasp_declaration
                    assert wire.target_id == "sim:cup"
                    assert list(wire.contact_links) == ["left_finger", "right_finger"]
                    assert wire.region_valid is True
                    assert wire.region.evidence_ref == "mujoco_body_subtree:cup"
                    assert wire.region.frame_id == "base_link"
                    assert list(wire.region.geometry) == []
                    assert wire.region.half_extents.x == pytest.approx(0.0401, abs=1e-6)
                    # Lower face lifted one 20 mm voxel off the support.
                    assert wire.region.half_extents.z == pytest.approx(0.0301, abs=1e-6)
                    decoded = GraspDeclaration.from_idl(wire)
                    assert decoded.region is not None
                    assert decoded.model_dump(exclude={"region"}) == declaration.model_dump(
                        exclude={"region"}
                    )

                # Retraction takes the declaration, region and all, off the envelope.
                retract = GraspDeclarationMsg()
                declaration.model_copy(update={"active": False}).fill_idl(retract)
                declare.publish(retract)
                count = len(received)
                assert _spin_until(
                    rclpy,
                    node,
                    lambda: any(not m.grasp_declaration_valid for m in received[count:]),
                ), "a retracted grasp declaration must leave the envelope"
            finally:
                bridge.teardown()
                hal.disconnect()
        finally:
            node.destroy_node()
    finally:
        rclpy.try_shutdown()
