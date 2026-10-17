# SPDX-License-Identifier: Apache-2.0
"""Live-ROS: the #418 object-tracker node grounds, relays, binds and survives a restart.

The rig (``_tracker_rig``): real rclpy, real ``openral_msgs`` on the wire, the real
``object_tracker_node`` (mode ``bind``) and the real World State lifecycle node with a
real ``WorldStateAggregator``, fed the SO-101 PickOrange ``top_depth`` camera ray-cast
with a ∅4.8 cm orange in the SO-101's pinch volume at a measured hold.

Asserts: the set reaches CONFIRMED on ``/openral/tracked_objects`` with a revision seeded
from wall time; ``world_state_fast`` carries it with ``tracked_objects_valid``; an ATTACH
at the hold publishes ``/openral/attachment_state`` with a ``TRACKED_OBJECT`` binding of
that id and the set shows it BOUND; with the tracker deactivated the relayed set goes
invalid within the aggregator's staleness window; a fresh tracker node's first revision is
above every revision before the restart (the kernel refuses a regressing one).

Gated on ``OPENRAL_TEST_ROS_LIVE=1``. Locally::

    source /opt/ros/jazzy/setup.bash && source install/setup.bash
    PYTHONPATH=$PWD/packages/world_state:$PWD:$PYTHONPATH OPENRAL_TEST_ROS_LIVE=1 \\
        pytest tests/integration/test_object_tracker_node_live.py
"""

from __future__ import annotations

import os
import time
from typing import Any

import pytest

from tests.integration._tracker_rig import tracker_rig, wait

_LIVE_ROS = bool(os.getenv("OPENRAL_TEST_ROS_LIVE"))
pytestmark = pytest.mark.skipif(
    not _LIVE_ROS,
    reason="live rclpy nodes + colcon openral_msgs overlay — set OPENRAL_TEST_ROS_LIVE=1 and "
    "source install/setup.bash first.",
)


def test_tracker_node_grounds_relays_binds_and_never_regresses() -> None:
    pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs")
    from openral_core import TrackStatus

    with tracker_rig() as rig:
        first: Any = None
        try:
            activated_ns = time.time_ns()
            first = rig.tracker("bind")
            stamp = rig.declare()
            tracked, attach, fast = rig.tracked, rig.attach, rig.fast
            assert rig.ground(), [[o.status for o in s.objects] for s in tracked[-3:]]
            [obj] = tracked[-1].objects
            assert obj.object_id == "trk:1" and obj.declaration_key == f"orange@{stamp}"
            assert all(s.revision >= activated_ns for s in tracked)
            assert wait(lambda: bool(fast) and fast[-1].tracked_objects_valid, 5.0)
            assert [o.object_id for o in fast[-1].tracked_objects.objects] == ["trk:1"]

            # ATTACH at the measured hold: the tracker owns attachment_state in `bind`.
            rig.publish_hold()
            assert wait(lambda: bool(attach), 5.0)  # the empty, fresh set first
            rig.attach_event()
            assert wait(lambda: bool(attach[-1].objects), 5.0)
            [held] = attach[-1].objects
            assert held.object_id == "trk:1" and held.evidence_kind == "tracked_object"
            assert (
                attach[-1].grasp_declaration_valid and not attach[-1].grasp_declaration.region_valid
            )
            assert wait(lambda: tracked[-1].objects[0].status == TrackStatus.BOUND.value, 5.0)
            last_revision = max(s.revision for s in tracked)

            # The tracker stops: world_state_fast drops the set within the staleness window.
            first.trigger_deactivate()
            assert wait(lambda: not fast[-1].tracked_objects_valid, 4.0)
            first.trigger_cleanup()
            rig.executor.remove_node(first)
            first.destroy_node()
            first = None

            # A restarted tracker never regresses the revision the kernel last accepted.
            n_before = len(tracked)
            first = rig.tracker("bind")
            assert wait(lambda: len(tracked) > n_before, 5.0)
            assert tracked[n_before].revision > last_revision
        finally:
            if first is not None:
                first.destroy_node()
