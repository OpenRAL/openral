# SPDX-License-Identifier: Apache-2.0
"""Live-ROS: the #418 object-tracker node grounds, relays, binds and survives a restart.

The rig (``_tracker_rig``): real rclpy, real ``openral_msgs`` on the wire, the real
``object_tracker_node`` (mode ``tracking``) and the real World State lifecycle node with a
real ``WorldStateAggregator``, fed the SO-101 PickOrange ``top_depth`` camera ray-cast
with a ∅4.8 cm orange in the SO-101's pinch volume at a measured hold.

Asserts: the set reaches CONFIRMED on ``/openral/tracked_objects`` with a revision seeded
from wall time; ``world_state_fast`` carries it with ``tracked_objects_valid``; an ATTACH
at the hold publishes ``/openral/attachment_state`` with a ``TRACKED_OBJECT`` binding of
that id and the set shows it BOUND; the grasp declaration carries the target's box as its
region while the orange is grounded and unheld (``tracking``), none once it is held, and none
ever under ``on``; with the tracker deactivated the relayed set goes
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
            first = rig.tracker("tracking")
            stamp = rig.declare()
            tracked, attach, fast = rig.tracked, rig.attach, rig.fast
            assert rig.ground(), [[o.status for o in s.objects] for s in tracked[-3:]]
            [obj] = tracked[-1].objects
            assert obj.object_id == "trk:1" and obj.declaration_key == f"orange@{stamp}"
            assert all(s.revision >= activated_ns for s in tracked)
            assert wait(lambda: bool(fast) and fast[-1].tracked_objects_valid, 5.0)
            assert [o.object_id for o in fast[-1].tracked_objects.objects] == ["trk:1"]

            # The grounded, unheld orange's box is the grasp region (`tracking`).
            def region_valid() -> bool:
                rig.publish_hold()  # the heartbeat runs while joint states are fresh
                return bool(attach) and attach[-1].grasp_declaration.region_valid

            assert wait(region_valid, 5.0)
            region = attach[-1].grasp_declaration.region
            assert region.evidence_ref.startswith("tracked:trk:1|epoch=")
            assert region.frame_id == tracked[-1].frame_id
            # ATTACH at the measured hold: the tracker owns attachment_state.
            rig.attach_event()
            assert wait(lambda: bool(attach[-1].objects), 5.0)
            [held] = attach[-1].objects
            assert held.object_id == "trk:1" and held.evidence_kind == "tracked_object"
            # Held: the region rides on (the kernel latches it at handover, ADR-0115).
            assert attach[-1].grasp_declaration_valid and attach[-1].grasp_declaration.region_valid
            assert attach[-1].grasp_declaration.region.evidence_ref.startswith("tracked:trk:1|")
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
            first = rig.tracker("tracking")
            assert wait(lambda: len(tracked) > n_before, 5.0)
            assert tracked[n_before].revision > last_revision
        finally:
            if first is not None:
                first.destroy_node()


def test_a_tracker_without_hal_mode_refuses_to_configure() -> None:
    """Unset ``hal_mode`` means a real cell: the node must not track (fail closed)."""
    pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs")
    from openral_world_state_ros.object_tracker_node import make_object_tracker_node
    from rclpy.lifecycle import TransitionCallbackReturn
    from rclpy.parameter import Parameter

    from tests.integration._tracker_rig import ROBOT

    with tracker_rig():
        node = make_object_tracker_node()
        try:
            assert node.get_parameter("hal_mode").value == "real"
            node.set_parameters(
                [
                    Parameter("robot_yaml", value=ROBOT),
                    Parameter("mode", value="tracking"),
                    Parameter("cameras", value=["top"]),
                    Parameter("grounding_wait_s", value=5.0),
                ]
            )
            assert node.trigger_configure() != TransitionCallbackReturn.SUCCESS
        finally:
            node.destroy_node()


def test_on_publishes_no_grasp_region() -> None:
    """``on``: the exemption provider keys on track ids; the declaration carries no region."""
    pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs")

    with tracker_rig() as rig:
        node: Any = None
        try:
            node = rig.tracker("on")
            rig.declare()
            assert rig.ground()
            attach = rig.attach
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                rig.publish_hold()
                time.sleep(0.1)
            assert attach and all(m.grasp_declaration_valid for m in attach[-5:])
            assert not any(m.grasp_declaration.region_valid for m in attach)
        finally:
            if node is not None:
                node.destroy_node()


def test_tracker_fill_survives_frozen_sim_clock_without_refreshing_evidence() -> None:
    pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs")
    from rclpy.parameter import Parameter
    from rosgraph_msgs.msg import Clock

    with tracker_rig() as rig:
        node = rig.tracker("tracking")
        try:
            rig.declare()
            assert rig.ground()
            pub = rig.helper.create_publisher(Clock, "/clock", 10)
            stamp = node.get_clock().now().to_msg()
            node.set_parameters([Parameter("use_sim_time", value=True)])
            assert wait(lambda: pub.get_subscription_count() > 0, 3.0)
            msg = Clock(clock=stamp)
            pub.publish(msg)
            assert wait(lambda: node.get_clock().now().to_msg() == stamp, 3.0)
            # Drain callbacks already ready when ROS time froze; no new camera/joint input.
            time.sleep(0.6)
            before = len(rig.tracked)
            assert wait(lambda: len(rig.tracked) >= before + 2, 1.4)
            snapshots = rig.tracked[before:]
            assert snapshots[0].objects
            assert all(s == snapshots[0] for s in snapshots)
            # A hung producer still stops filling after the existing five-second bound.
            time.sleep(5.2)
            before = len(rig.tracked)
            time.sleep(0.7)
            assert len(rig.tracked) == before
            assert wait(lambda: not rig.fast[-1].tracked_objects_valid, 3.0)
        finally:
            node.destroy_node()


def test_service_answers_are_fed_on_the_process_timer_never_from_their_callback() -> None:
    """Spark R49/R62 stack dumps: two segmenter answers, fed from their futures' done
    callbacks (executor tasks outside every callback group), ran the tracker concurrently on
    both executor threads, so neither the heartbeat nor the starvation fill could run and the
    kernel lost the tracked set for 1.6 s of sim time. An answer is only queued; the next
    `_process` tick, on the node's one default group, feeds it."""
    pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs")
    import numpy as np
    from openral_core import IntrinsicsPinhole
    from openral_world_state.object_tracker import CameraFrame, TrackerResult

    with tracker_rig() as rig:
        node = rig.tracker("tracking")
        try:
            k = IntrinsicsPinhole(fx=1.0, fy=1.0, cx=0.0, cy=0.0, width=2, height=2)
            frame = CameraFrame("cam", 1, np.zeros((2, 2)), k, None, None)
            node._answer(frame, "cam_optical", None, TrackerResult(10**9))  # an unknown request
            assert len(node._answers) == 1  # queued, not fed on this (callback) thread
            assert wait(lambda: not node._answers, 2.0)  # the process timer drained it
        finally:
            node.destroy_node()


def test_an_abandoned_service_call_is_removed_from_its_client_at_the_next_issue() -> None:
    """Review finding (#421): the scheduler abandons a request past
    ``segment_budget_wall_s`` and issues the next, but the abandoned call's future stayed in
    its client's pending table, holding its done callback and the ``CameraFrame`` it closes
    over, and a late reply would feed a request already dropped. The next issue removes it
    (here a call to a service nobody serves, so it can never answer)."""
    pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs")
    import numpy as np
    from openral_core import IntrinsicsPinhole
    from openral_msgs.srv import SegmentInView
    from openral_world_state.object_tracker import CameraFrame
    from openral_world_state.object_tracker.scheduler import TrackerRequest

    with tracker_rig() as rig:
        node = rig.tracker("tracking")
        try:
            client = node.create_client(SegmentInView, "/openral/test/never_served")
            future = client.call_async(SegmentInView.Request())
            assert future in client._pending_requests.values()
            node._pending_call = (client, future)
            k = IntrinsicsPinhole(fx=1.0, fy=1.0, cx=0.0, cy=0.0, width=2, height=2)
            frame = CameraFrame("cam", 0, np.zeros((2, 2)), k, None, None)
            # A stamp-less frame is refused before any call: only the drop runs.
            node._issue(
                TrackerRequest("segment_in_view", "cam", 0, 5, 0, "orange", request_id=7),
                frame,
                "cam_optical",
                None,
            )
            assert future not in client._pending_requests.values()
            assert not future.done() and node._pending_call is None
            node.destroy_client(client)
        finally:
            node.destroy_node()


def _entities(node: Any) -> tuple[list[str], list[str], int, int]:
    return (
        sorted(p.topic_name for p in node.publishers),
        sorted(s.topic_name for s in node.subscriptions),
        len(list(node.timers)),
        len(list(node.clients)),
    )


def test_cleanup_destroys_every_entity_configure_created() -> None:
    """Review finding (#426): ``on_cleanup`` left ``hands_clear``, ``/diagnostics`` and the
    TF listener's ``/tf`` / ``/tf_static`` subscriptions alive, so a cleanup + configure (a
    supported lifecycle path) doubled them. After cleanup the node owns exactly what it owned
    before configure, and a second configure creates the same set as the first."""
    pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs")

    with tracker_rig() as rig:
        node = rig.tracker("tracking")
        try:
            configured = _entities(node)
            assert {"/openral/object_tracker/hands_clear", "/diagnostics"} <= set(configured[0])
            assert {"/tf", "/tf_static"} <= set(configured[1])
            node.trigger_deactivate()
            node.trigger_cleanup()
            after = _entities(node)
            for topic in (
                "/openral/tracked_objects",
                "/openral/attachment_state",
                "/openral/object_tracker/hands_clear",
                "/diagnostics",
            ):
                assert topic not in after[0], after[0]
            assert not after[1], after[1]  # every subscription was configure's
            assert after[2:] == (0, 0), after
            node.trigger_configure()
            node.trigger_activate()
            assert _entities(node) == configured  # no second copies
        finally:
            node.destroy_node()


def test_an_invalid_declaration_is_rejected_without_escaping_its_callback() -> None:
    """Review finding (#426): ``GraspDeclaration.from_idl`` / ``PlaceDeclaration.from_idl``
    validate, so a malformed latched declaration (here the all-default message: ``timeout_s``
    0) raised out of the subscription callback and stopped the executor's spin. It is now
    logged and dropped; the session keeps the declaration it had."""
    pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs")
    from openral_msgs.msg import GraspDeclaration as GraspDeclarationMsg
    from openral_msgs.msg import PlaceDeclaration as PlaceDeclarationMsg

    with tracker_rig() as rig:
        node = rig.tracker("tracking")
        try:
            node._on_grasp_declaration(GraspDeclarationMsg())
            node._on_place_declaration(PlaceDeclarationMsg())
            # The node still grounds a valid declaration on the same executor.
            rig.declare()
            assert rig.ground()
        finally:
            node.destroy_node()
