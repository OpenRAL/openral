# SPDX-License-Identifier: Apache-2.0
"""Live-ROS regression: a place-witness revision must release the action barrier.

Round 6 (``robocasa_sink_cup``) aborted a goal whose place had *succeeded*. The
shape, from the field log:

* attachment revision 2 published at ``1786728108.893`` — the ADR-0097 place
  witness, which re-publishes the SAME carried object under a bumped revision
  with a ``support_contact`` attestation attached;
* ``deferring action_applied tick=180 for attachment perception`` — the HAL
  lifecycle node held the completed tick because
  ``SimSensorBridge.attachment_action_ack_ready()`` was False while that
  revision sat in ``_attachment_pending``;
* no ``attachment perception barrier waiting for N ... depth frames`` line ever
  — the bridge's applied-revision handler tested *object-id addition*, the
  witness added no object id, so nothing armed and nothing released;
* ``action group tick 180 was not applied within 8 s`` → goal abort.

The barrier exists so motion waits until perception reflects **new masked
geometry**. A witness-only re-publish masks nothing new, so the fix releases it
immediately (guarded by ``attachment_action_ack_ready`` so an earlier
revision's outstanding depth/voxel frames are never skipped).

This is the live half: the production ``ManifestHALLifecycleNode`` on a real
``SimAttachedHAL`` (the real ``tabletop_push`` rollout — a real compiled
``MjModel`` with a free ``cube`` body), its real ``SimSensorBridge``, real
``openral_msgs`` on the wire, and the release observed as a real
``/openral/action_applied`` message. No mocks (CLAUDE.md §1.11).

Gated on ``OPENRAL_TEST_ROS_LIVE=1`` and listed in ``scripts/ros_live_tests.sh``.
Locally::

    source /opt/ros/jazzy/setup.bash && just ros2-build
    source install/setup.bash
    just test-ros-live -k attachment_barrier
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
# The scene's own free body — a real payload identity, not a placeholder.
_PAYLOAD_BODY = "cube"
_PAYLOAD_ID = f"sim:{_PAYLOAD_BODY}"
_TICK = 180
_DEADLINE_S = 10.0


def _wait_until(predicate: Any, *, timeout_s: float = _DEADLINE_S) -> bool:
    """Spin-wait on a live graph predicate (the executor runs on its own thread)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _payload(description: Any, *, stamp_ns: int, witnessed: bool) -> Any:
    """One real ``AttachedCollisionObject`` for the scene's cube.

    ``witnessed=True`` is the ADR-0097 place attestation: same ``object_id``,
    same ``evidence_ref`` (so the perception mask is byte-identical), plus a
    real :class:`~openral_core.SupportContactWitness`.
    """
    from openral_core import (
        AttachedCollisionObject,
        AttachedCollisionPrimitive,
        AttachmentEvidenceKind,
        BoxShape,
        Pose6D,
        SupportContactWitness,
    )

    attach_link = description.joints[-1].child_link
    witness = (
        SupportContactWitness(
            support_id="sim:table",
            contact_point_in_object=(0.0, 0.0, -0.025),
            contact_normal_in_object=(0.0, 0.0, 1.0),
            patch_radius_m=0.05,
            max_penetration_m=0.001,
            confidence=1.0,
            evidence_kind=AttachmentEvidenceKind.SIM_GEOM_DISTANCE,
            stamp_ns=stamp_ns,
        )
        if witnessed
        else None
    )
    return AttachedCollisionObject(
        object_id=_PAYLOAD_ID,
        attach_link=attach_link,
        touch_links=[attach_link],
        primitives=[
            AttachedCollisionPrimitive(
                shape=BoxShape(half_extents_m=(0.025, 0.025, 0.025)),
                pose_in_object=Pose6D(
                    xyz=(0.0, 0.0, 0.0),
                    quat_xyzw=(0.0, 0.0, 0.0, 1.0),
                    frame_id=_PAYLOAD_ID,
                ),
            )
        ],
        pose_in_link=Pose6D(
            xyz=(0.0, 0.0, 0.03),
            quat_xyzw=(0.0, 0.0, 0.0, 1.0),
            frame_id=attach_link,
        ),
        confidence=1.0,
        evidence_kind=AttachmentEvidenceKind.SIM_CONTACT,
        evidence_ref=f"mujoco_body:{_PAYLOAD_BODY}",
        stamp_ns=stamp_ns,
        support_contact=witness,
    )


def test_place_witness_revision_releases_the_deferred_action_tick() -> None:
    """A bumped revision on an unchanged payload releases the deferred tick."""
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("mujoco")
    pytest.importorskip("openral_msgs")

    from openral_core import Action, ControlMode, RobotDescription
    from openral_hal.lifecycle import ManifestHALLifecycleNode
    from openral_msgs.msg import (
        AttachedCollisionObject as AttachedCollisionObjectMsg,
    )
    from openral_msgs.msg import (
        AttachedCollisionPrimitive as AttachedCollisionPrimitiveMsg,
    )
    from openral_msgs.msg import AttachmentState
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
    from std_msgs.msg import UInt64

    description = RobotDescription.from_yaml(str(_ROBOT_YAML))

    rclpy.init()
    node: Any = ManifestHALLifecycleNode("test_attachment_barrier_live")
    node.set_parameters(
        [
            Parameter("robot_yaml", value=str(_ROBOT_YAML)),
            Parameter("hal_mode", value="sim"),
            Parameter("sim_env_yaml", value=str(_SCENE_YAML)),
            Parameter("publish_rate_hz", value=20.0),
            Parameter("viewer_enabled", value=False),
        ]
    )
    peer = Node("test_attachment_barrier_peer")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(peer)

    latched = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        depth=1,
    )
    control = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.VOLATILE,
        depth=1,
    )
    state_pub = peer.create_publisher(AttachmentState, "/openral/attachment_state", latched)
    applied_pub = peer.create_publisher(UInt64, "/openral/attachment_state_applied", latched)
    acknowledged: list[int] = []
    peer.create_subscription(
        UInt64,
        "/openral/action_applied",
        lambda msg: acknowledged.append(int(msg.data)),
        control,
    )

    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    try:
        assert str(node.trigger_configure()).endswith("SUCCESS"), "configure failed"
        assert str(node.trigger_activate()).endswith("SUCCESS"), "activate failed"
        bridge = node._bridge
        assert bridge is not None, "the manifest node wires the SimSensorBridge at activate"
        hal = node._hal

        def publish_revision(revision: int, *, witnessed: bool) -> None:
            """Publish one real atomic attachment snapshot on the live topic."""
            msg = AttachmentState()
            msg.header.stamp = peer.get_clock().now().to_msg()
            msg.revision = revision
            item = AttachedCollisionObjectMsg()
            _payload(description, stamp_ns=revision, witnessed=witnessed).fill_idl(
                item, primitive_factory=AttachedCollisionPrimitiveMsg
            )
            msg.objects.append(item)
            state_pub.publish(msg)

        # ── 1. Grasp: the payload enters the perception mask for the first time.
        publish_revision(1, witnessed=False)
        assert _wait_until(lambda: bridge._attachment_revision >= 1), (
            "the bridge never received the first attachment revision"
        )
        applied_pub.publish(UInt64(data=1))
        assert _wait_until(bridge.attachment_action_ack_ready), (
            "the grasp revision never settled — the barrier is stuck before the "
            "case under test even starts"
        )

        # ── 2. Place witness: SAME object id, SAME evidence_ref, bumped revision.
        publish_revision(2, witnessed=True)
        assert _wait_until(lambda: bridge._attachment_pending is not None), (
            "the bridge never staged the place-witness revision"
        )
        assert not bridge.attachment_action_ack_ready()

        # ── 3. A complete action group lands while that revision is pending.
        #      Two slots (arm + gripper) is the shape the runner emits per
        #      inference tick, and the shape whose group ack the barrier holds.
        arm_dim = sum(1 for joint in description.joints if joint.role != "gripper")
        for slot in (
            Action(
                control_mode=ControlMode.JOINT_POSITION,
                horizon=1,
                joint_targets=[[0.0] * arm_dim],
                tick_index=_TICK,
                tick_group_size=2,
                stamp_ns=1,
            ),
            Action(
                control_mode=ControlMode.GRIPPER_POSITION,
                horizon=1,
                gripper=[0.0],
                tick_index=_TICK,
                tick_group_size=2,
                stamp_ns=1,
            ),
        ):
            hal.send_action(slot)
        assert hal.last_committed_tick == _TICK, "the action group never committed"

        node._publish_action_applied_if_complete(slot)
        assert node._deferred_action_applied_tick == _TICK, (
            "the node must defer the tick while an attachment revision is pending"
        )
        assert not acknowledged, "the tick was acknowledged before perception settled"

        # ── 4. The kernel accepts the witness revision. THIS is the regression:
        #      no geometry entered the mask, so the barrier must release now.
        applied_pub.publish(UInt64(data=2))
        assert _wait_until(lambda: acknowledged == [_TICK]), (
            "the place-witness revision never released the deferred action tick — "
            "the goal would abort 8 s later on a place that succeeded; "
            f"observed acknowledgements: {acknowledged}"
        )
        assert bridge.attachment_action_ack_ready()
    finally:
        with suppress(Exception):
            node.trigger_deactivate()
            node.trigger_cleanup()
        executor.shutdown()
        spin.join(timeout=5.0)
        peer.destroy_node()
        node.destroy_node()
        rclpy.shutdown()
