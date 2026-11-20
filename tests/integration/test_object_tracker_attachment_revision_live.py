# SPDX-License-Identifier: Apache-2.0
"""Live-ROS: the object-tracker node's attachment revision follows the attachment set only.

The rig (``_tracker_rig``): real rclpy, real ``openral_msgs`` on the wire, the real
``object_tracker_node`` (mode ``tracking``) with a declared place zone, the SO-101 PickOrange
``top_depth`` camera ray-cast with an orange at the hold.

With a grasp declaration and a place declaration naming the zone, every new grid re-expresses
the zone's box, and ``/openral/attachment_state`` is republished for it under ONE revision:
the kernel reads a revision edge with no payload as a detach and retired the armed grasp
region on each one (AMR aisle, ``tracking``, 2026-11-20: 65-70 retirements a run, before
contact). A real attachment change (an ATTACH at the hold) still bumps the revision.

Gated on ``OPENRAL_TEST_ROS_LIVE=1``. Locally::

    source /opt/ros/jazzy/setup.bash && source install/setup.bash
    PYTHONPATH=$PWD/packages/world_state:$PWD:$PYTHONPATH OPENRAL_TEST_ROS_LIVE=1 \\
        pytest tests/integration/test_object_tracker_attachment_revision_live.py
"""

from __future__ import annotations

import json
import os
import time

import pytest

from tests.integration._tracker_rig import tracker_rig, wait

_LIVE_ROS = bool(os.getenv("OPENRAL_TEST_ROS_LIVE"))
pytestmark = pytest.mark.skipif(
    not _LIVE_ROS,
    reason="live rclpy nodes + colcon openral_msgs overlay — set OPENRAL_TEST_ROS_LIVE=1 and "
    "source install/setup.bash first.",
)


def test_a_zone_moving_with_each_grid_republishes_under_one_revision() -> None:
    pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs")
    from openral_core import PlaceDeclaration, PlaceRegion, PlaceZone, Pose6D
    from openral_msgs.msg import PlaceDeclaration as PlaceDeclarationMsg

    with tracker_rig() as rig:
        pinch = tuple(float(v) for v in rig.pinch)
        # A slot beside the hold, in the fixed base's own frame (allowed for a manipulator).
        slot = PlaceZone(
            name="slot",
            region=PlaceRegion(
                frame_id="base",
                pose=Pose6D(
                    xyz=(pinch[0], pinch[1] + 0.15, pinch[2]),
                    quat_xyzw=(0, 0, 0, 1),
                    frame_id="base",
                ),
                half_extents=(0.05, 0.05, 0.04),
            ),
        )
        rig.tracker("tracking", place_zones_json=json.dumps([slot.model_dump(mode="json")]))
        rig.declare()
        place_pub = rig.helper.create_publisher(
            PlaceDeclarationMsg, "/openral/place_declaration", rig.latched
        )
        place = PlaceDeclaration(
            target_id=slot.target_id,
            timeout_s=120.0,
            stamp_ns=rig.helper.get_clock().now().nanoseconds,
        )
        msg = PlaceDeclarationMsg()
        place.fill_idl(msg)
        place_pub.publish(msg)
        assert rig.ground(), [[o.status for o in s.objects] for s in rig.tracked[-3:]]
        assert wait(lambda: bool(rig.attach) and rig.attach[-1].place_declaration_valid, 10.0)

        # Fresh grids: each re-expresses the zone, so the state is republished, never bumped.
        start = len(rig.attach)
        for _ in range(8):
            rig.frame()
            time.sleep(0.3)
        window = rig.attach[start:]
        assert len(window) >= 3, f"republished per grid: {len(window)} messages"
        revisions = {m.revision for m in window}
        assert len(revisions) == 1, f"one revision while nothing is attached: {sorted(revisions)}"
        assert all(not m.objects and m.place_declaration_valid for m in window)

        # A real attachment change still bumps it.
        (revision,) = revisions
        start = len(rig.attach)

        def bumped() -> bool:
            rig.publish_hold()
            rig.attach_event()
            rig.frame()
            time.sleep(0.2)
            return any(m.objects and m.revision > revision for m in rig.attach[start:])

        assert wait(bumped), [(m.revision, len(m.objects)) for m in rig.attach[start:]]
